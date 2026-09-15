#!/usr/bin/env python3

import os
import pickle
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

hep.style.use("CMS")

# config
REPROCESS = True

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023", "2023BPix"]
TAG = "13Feb2026_Data_Standard_Cuts_QCD_Inclusive_v15_scouting_zbb"
YEAR_DIRS = [f"{SKIMMER_DIR}/{TAG}/{y}" for y in YEARS]

OUTDIR = "/eos/user/z/zima/MASS_DECORRELATION_EFFORTS/NoSmoothing/outputs"
os.makedirs(OUTDIR, exist_ok=True)
FILENAME = f"{OUTDIR}/second_bins_TXBBPRECUT0p3.pkl"

ALPHAS = [0.95, 0.995]
extra_txbb_cut = 0.3 # additional cut on txbb

USE_SMOOTHING = True
SMOOTH_SIGMA = (1.0, 1.0) 

# Binning
PT_LIMS  = (300, 1800)
RHO_LIMS = (-7.0, -1.0) # -2

N_PT   = 50
N_RHO  = 50
N_TXBB = 400

TXBB_EDGES = np.linspace(0.0, 1.0, N_TXBB + 1)

# utilities

def get_qcd_files(year_dirs=YEAR_DIRS):
    qcd_filelist = []
    for year_dir in year_dirs:
        for root, _, files in os.walk(year_dir):
            for file in files:
                f = os.path.join(root, file)
                if "QCD" in f and f.endswith(".root"):
                    qcd_filelist.append(f)

    print(f"Length of QCD filelist: {len(qcd_filelist)}")
    return qcd_filelist


def compute_rho(m, pt):
    return np.log((m * m) / (pt * pt))

def build_binned_histograms(files):

    pt_edges  = np.linspace(*PT_LIMS,  N_PT + 1)
    rho_edges = np.linspace(*RHO_LIMS, N_RHO + 1)

    hists  = np.zeros((N_PT, N_RHO, N_TXBB), dtype=np.float64)
    counts = np.zeros((N_PT, N_RHO), dtype=np.int64)

    for f in tqdm(files, desc="Binning QCD"):
        with uproot.open(f) as file:
            if "Events" not in file:
                continue

            ev = file["Events"]

            txbb = ev["bbFatJetScoutParTTXbb0"].array(library="np")
            pt   = ev["bbFatJetPt0"].array(library="np")
            m    = ev["bbFatJetScoutParTmassCorrectedX2p0"].array(library="np")
            w    = ev["weight"].array(library="np")

            mask = (pt > 0) & (m > 0) & np.isfinite(txbb)
            txbb, pt, m, w = txbb[mask], pt[mask], m[mask], w[mask]

            if extra_txbb_cut != 0.0:
                pre = txbb > extra_txbb_cut # extra selection to match data selections made earlier
                txbb, pt, m, w = txbb[pre], pt[pre], m[pre], w[pre] 

            rho = compute_rho(m, pt)

            ipt  = np.digitize(pt,  pt_edges)  - 1
            irho = np.digitize(rho, rho_edges) - 1
            it   = np.digitize(txbb, TXBB_EDGES) - 1

            valid = (
                (ipt  >= 0) & (ipt  < N_PT) &
                (irho >= 0) & (irho < N_RHO) &
                (it   >= 0) & (it   < N_TXBB)
            )

            for i, j, k, ww in zip(ipt[valid], irho[valid], it[valid], w[valid]):
                hists[i, j, k] += ww
                counts[i, j]  += 1

    return {
        "hists": hists,
        "counts": counts,
        "pt_edges": pt_edges,
        "rho_edges": rho_edges,
        "pt_centers": 0.5 * (pt_edges[:-1] + pt_edges[1:]),
        "rho_centers": 0.5 * (rho_edges[:-1] + rho_edges[1:]),
    }

def compute_quantile_surface(bins, alpha):

    hists = bins["hists"]
    qsurf = np.full((N_PT, N_RHO), np.nan)

    for i in range(N_PT):
        for j in range(N_RHO):
            hist = hists[i, j]
            tot = hist.sum()
            if tot <= 0:
                continue

            cdf = np.cumsum(hist) / tot
            k = np.searchsorted(cdf, alpha)

            if k < len(TXBB_EDGES) - 1:
                qsurf[i, j] = TXBB_EDGES[k]

    return qsurf


def plot_raw_surface(qsurf, bins, alpha):

    fig, ax = plt.subplots(figsize=(10, 9))
    m = ax.pcolormesh(
        bins["rho_centers"],
        bins["pt_centers"],
        qsurf,
        shading="auto",
        cmap="viridis"
    )

    fig.colorbar(m, ax=ax, label=r"$T_{Xbb}^\alpha$")
    ax.set_xlabel(r"$\rho = \ln(m^2/p_T^2)$")
    ax.set_ylabel(r"$p_T$ [GeV]")

    ax.text(
        0.03, 0.97, fr"$\alpha = {alpha}$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=40, fontweight="bold"#, color = "white"
    )

    hep.cms.label(ax=ax, year="2023", com="13.6")
    fig.savefig(f"{OUTDIR}/raw_quantile_alpha_{alpha}.pdf")
    plt.close(fig)

def smooth_quantile_surface(qsurf, sigma=(1.0, 1.0)):
    """
    Gaussian smoothing that preserves NaNs.
    """
    mask = np.isfinite(qsurf)
    filled = np.where(mask, qsurf, 0.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    norm = gaussian_filter(mask.astype(float), sigma=sigma)

    out = np.divide(
        smoothed, norm,
        out=np.full_like(qsurf, np.nan),
        where=norm > 0
    )
    return out

def get_quantile_surface(bins, alpha, smooth=False):
    q = compute_quantile_surface(bins, alpha)
    if smooth:
        q = smooth_quantile_surface(q, SMOOTH_SIGMA)
    return q

def compute_decorrelated_efficiency(files, qsurf, bins):

    pt_edges  = bins["pt_edges"]
    rho_edges = bins["rho_edges"]

    passed = np.zeros((N_PT, N_RHO), dtype=np.int64)
    total  = np.zeros((N_PT, N_RHO), dtype=np.int64)

    for f in tqdm(files, desc="Decorrelating QCD (direct lookup)"):
        with uproot.open(f) as file:
            if "Events" not in file:
                continue

            ev = file["Events"]

            txbb = ev["bbFatJetScoutParTTXbb0"].array(library="np")
            pt   = ev["bbFatJetPt0"].array(library="np")
            m    = ev["bbFatJetScoutParTmassCorrectedX2p0"].array(library="np")

            mask = (pt > 0) & (m > 0) & np.isfinite(txbb)
            if not np.any(mask):
                continue

            txbb = txbb[mask]
            pt   = pt[mask]
            m    = m[mask]

            rho = compute_rho(m, pt)

            ipt  = np.digitize(pt,  pt_edges)  - 1
            irho = np.digitize(rho, rho_edges) - 1

            valid = (
                (ipt  >= 0) & (ipt  < N_PT) &
                (irho >= 0) & (irho < N_RHO)
            )

            ipt  = ipt[valid]
            irho = irho[valid]
            txbb = txbb[valid]

            q = qsurf[ipt, irho]
            ok_q = np.isfinite(q)

            ipt  = ipt[ok_q]
            irho = irho[ok_q]
            tdec = txbb[ok_q] - q[ok_q]

            np.add.at(total,  (ipt, irho), 1)
            np.add.at(passed, (ipt[tdec > 0], irho[tdec > 0]), 1)

    eff = np.divide(
        passed, total,
        out=np.zeros_like(passed, dtype=float),
        where=total > 0
    )

    return eff


def plot_efficiency(eff, bins, alpha):

    fig, ax = plt.subplots(figsize=(10, 10))
    m = ax.pcolormesh(
        bins["rho_centers"],
        bins["pt_centers"],
        eff,
        shading="auto",
        cmap="viridis",
        vmin = 0.0,
        vmax = 0.1
    )

    fig.colorbar(m, ax=ax, label="QCD efficiency")
    ax.set_xlabel(r"$\rho = \ln(m^2/p_T^2)$")
    ax.set_ylabel(r"$p_T$ [GeV]")

    ax.text(
        0.03, 0.97,
        r"$T_{Xbb} - T_{Xbb}^\alpha > 0$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold"
    )

    hep.cms.label(ax=ax, year="2023", com="13.6")
    fig.savefig(f"{OUTDIR}/decorrelated_efficiency_alpha_{alpha}.pdf")
    plt.close(fig)

def compute_decorrelated_efficiency_weighted(files, qsurf, bins):

    pt_edges  = bins["pt_edges"]
    rho_edges = bins["rho_edges"]

    passed_w = np.zeros((N_PT, N_RHO), dtype=np.float64)
    total_w  = np.zeros((N_PT, N_RHO), dtype=np.float64)

    for f in tqdm(files, desc="Decorrelating QCD (weighted)"):
        with uproot.open(f) as file:
            if "Events" not in file:
                continue

            ev = file["Events"]

            txbb = ev["bbFatJetScoutParTTXbb0"].array(library="np")
            pt   = ev["bbFatJetPt0"].array(library="np")
            m    = ev["bbFatJetScoutParTmassCorrectedX2p0"].array(library="np")
            w    = ev["weight"].array(library="np")

            mask = (pt > 0) & (m > 0) & np.isfinite(txbb) & np.isfinite(w)
            if not np.any(mask):
                continue

            txbb = txbb[mask]
            pt   = pt[mask]
            m    = m[mask]
            w    = w[mask]

            rho = compute_rho(m, pt)

            ipt  = np.digitize(pt,  pt_edges)  - 1
            irho = np.digitize(rho, rho_edges) - 1

            valid = (
                (ipt  >= 0) & (ipt  < N_PT) &
                (irho >= 0) & (irho < N_RHO)
            )

            ipt  = ipt[valid]
            irho = irho[valid]
            txbb = txbb[valid]
            w    = w[valid]

            q = qsurf[ipt, irho]
            ok_q = np.isfinite(q)

            ipt  = ipt[ok_q]
            irho = irho[ok_q]
            txbb = txbb[ok_q]
            w    = w[ok_q]
            q    = q[ok_q]

            t_decorr = txbb - q
            passed = t_decorr > 0

            np.add.at(total_w,  (ipt, irho), w)
            np.add.at(passed_w, (ipt[passed], irho[passed]), w[passed])

    eff = np.divide(
        passed_w,
        total_w,
        out=np.zeros_like(passed_w),
        where=total_w > 0
    )

    return eff


def plot_quantile_efficiency_matrix(
    alphas,
    qsurfs,
    effs,
    bins,
    outname,
):
    n = len(alphas)

    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n + 1.5, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    qmin = np.nanmin([q.min() for q in qsurfs])
    qmax = np.nanmax([q.max() for q in qsurfs])

    # emin, emax = 0.0, 0.1

    qmappable = None
    emappable = None

    for i, alpha in enumerate(alphas):
        axq = axes[0, i]
        qmappable = axq.pcolormesh(
            bins["rho_centers"],
            bins["pt_centers"],
            qsurfs[i],
            shading="auto",
            cmap="viridis",
            vmin=qmin,
            vmax=qmax,
        )
        axq.set_title(fr"$\alpha = {alpha}$", fontsize=16)
        if i == 0:
            axq.set_ylabel(r"$p_T$ [GeV]")

        axe = axes[1, i]

        from matplotlib.colors import LogNorm

        eff_norm = LogNorm(vmin=1e-4, vmax=1e-1)

        eff_plot = np.ma.masked_less_equal(effs[i], 0.0)

        emappable = axe.pcolormesh(
            bins["rho_centers"],
            bins["pt_centers"],
            eff_plot,
            shading="auto",
            cmap="hsv",
            norm=eff_norm,
        )

        if i == 0:
            axe.set_ylabel(r"$p_T$ [GeV]")
        axe.set_xlabel(r"$\rho = \ln(m^2/p_T^2)$")


    cbar_q = fig.colorbar(
        qmappable,
        ax=axes[0, :],
        orientation="horizontal",
        # fraction=0.05,
        # pad=0.08,
        label=r"$T_{Xbb}^\alpha$"
    )

    cbar_e = fig.colorbar(
        emappable,
        ax=axes[1, :],
        orientation="horizontal",
        # fraction=0.05,
        # pad=0.08,
        label="QCD efficiency"
    )

    hep.cms.label(ax=axes[0, 0], year="2023", com="13.6")
    fig.savefig(outname)
    plt.close(fig)

def plot_quantile_efficiency_matrix_2(
    alphas,
    qsurfs,
    effs,
    bins,
    outname,
):

    n = len(alphas)

    fig = plt.figure(
        figsize=(4.5 * n + 1.2, 9),
        constrained_layout=True,
    )

    gs = gridspec.GridSpec(
        2, n + 1,
        figure=fig,
        width_ratios=[1] * n + [0.05], 
        height_ratios=[1, 1],
        wspace=0.05,
        hspace=0.05,
    )

    axes = np.empty((2, n), dtype=object)
    for i in range(n):
        axes[0, i] = fig.add_subplot(gs[0, i])
        axes[1, i] = fig.add_subplot(gs[1, i], sharex=axes[0, i], sharey=axes[0, i])

    cax_q = fig.add_subplot(gs[0, -1])
    cax_e = fig.add_subplot(gs[1, -1])

    qmin, qmax = 0.0, 1.0  # FIXED TXbb range, might want to change

    qmappable = None
    emappable = None

    for i, alpha in enumerate(alphas):
        axq = axes[0, i]
        qmappable = axq.pcolormesh(
            bins["rho_centers"],
            bins["pt_centers"],
            qsurfs[i],
            shading="auto",
            cmap="viridis",
            vmin=qmin,
            vmax=qmax,
        )

        axq.text(
            0.05, 0.95,
            fr"$\alpha = {alpha}$",
            transform=axq.transAxes, 
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

        axe = axes[1, i]
        eff_plot = np.ma.masked_less_equal(effs[i], 0.0)
        emappable = axe.pcolormesh(
            bins["rho_centers"],
            bins["pt_centers"],
            eff_plot,
            shading="auto",
            cmap="hsv",
            norm=LogNorm(vmin=1e-4, vmax=1e-1),
        )

        axq.set_box_aspect(1)
        axe.set_box_aspect(1)

    fig.colorbar(
        qmappable,
        cax=cax_q,
        label=r"$T_{Xbb}^\alpha$",
    )

    fig.colorbar(
        emappable,
        cax=cax_e,
        label="QCD efficiency",
    )

    fig.supxlabel(r"$\rho = \ln(m^2/p_T^2)$", fontsize=28)
    fig.supylabel(r"$p_T$ [GeV]", fontsize=28)

    # couldnt get these to work
    # hep.cms.text(
    #     "Simulation",
    #     loc=0,
    #     ax=None,
    #     # fig=fig,
    #     # x=0.01,
    #     # y=0.995,
    # )

    # fig.text(
    #     0.99, 0.995,
    #     "2023 (13.6 TeV)",
    #     ha="right",
    #     va="top",
    #     fontsize=16,
    # )

    fig.savefig(outname)
    plt.close(fig)


def save_results(obj, out_file):
    with open(out_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(in_file):
    with open(in_file, "rb") as f:
        return pickle.load(f)

def main():

    if REPROCESS:
        files = get_qcd_files()
        bins = build_binned_histograms(files)
        save_results(bins, FILENAME)
    else:
        bins = load_results(FILENAME)
        files = get_qcd_files()

    qsurfs = []
    effs   = []

    for alpha in ALPHAS:
        qsurf = get_quantile_surface(
            bins,
            alpha,
            smooth=USE_SMOOTHING
        )

        mvar = "mX2p"

        save_results((qsurf, bins["pt_edges"], bins["rho_edges"]), f"{OUTDIR}/TXBBPRECUT0p3_qsurf_{str(alpha).replace('.', 'p')}_{mvar}.pkl")

        eff = compute_decorrelated_efficiency_weighted(
            files, qsurf, bins
        )

        save_results((eff, bins["pt_edges"], bins["rho_edges"]), f"{OUTDIR}/TXBBPRECUT0p3_eff_{str(alpha).replace('.', 'p')}_{mvar}.pkl")

        qsurfs.append(qsurf)
        effs.append(eff)

    tag = "Smoothed" if USE_SMOOTHING else "Raw"

    plot_quantile_efficiency_matrix_2(
        ALPHAS,
        qsurfs,
        effs,
        bins,
        f"{OUTDIR}/quantile_efficiency_matrix_{tag}_TXBBPRECUT0p3.pdf"
    )


if __name__ == "__main__":
    main()
