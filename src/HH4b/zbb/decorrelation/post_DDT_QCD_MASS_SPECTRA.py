from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import uproot
import pickle

from tqdm import tqdm
import mplhep as hep

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

mpl.rcParams.update({
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
})

QDICT = {
    0.5: "TXBBPRECUT0p3_qsurf_0p5_mX2p.pkl",
    0.7: "TXBBPRECUT0p3_qsurf_0p7_mX2p.pkl",
    0.9: "TXBBPRECUT0p3_qsurf_0p9_mX2p.pkl",
    0.99: "TXBBPRECUT0p3_qsurf_0p99_mX2p.pkl",
    0.995: "TXBBPRECUT0p3_qsurf_0p995_mX2p.pkl",
    0.997: "TXBBPRECUT0p3_qsurf_0p997_mX2p.pkl",
    # 0.999: "TXBBPRECUT0p3_qsurf_0p999_mX2p.pkl",
}

ALPHAS = sorted(QDICT.keys())
REFERENCE_ALPHA = ALPHAS[0]

extra_txbb_cut = 0.3

SCOUTING: bool = True

offline_details = {
    "mass_var": "bbFatJetParT3massGeneric0",
    "txbb_var": "bbFatJetParT3TXbb0",
}

scouting_details = {
    "mass_var": "bbFatJetMsd0",
    "txbb_var": "bbFatJetScoutParTTXbb0",
}

details = scouting_details if SCOUTING else offline_details
mass_variable = details["mass_var"]
txbb_variable = details["txbb_var"]

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023BPix", "2023"]

TAG = (
    "13Feb2026_Data_Standard_Cuts_QCD_Inclusive_v15_scouting_zbb"
    if SCOUTING
    else "06Feb2025_QCD_Offline_Inclusive_PTl_300_etaS_HT_1k_v15_scouting_zbb"
)

YEAR_DIR = [f"{SKIMMER_DIR}/{TAG}/{YEAR}" for YEAR in YEARS]

SAVE_TO = "/eos/user/z/zima/QCD_SPECTRA"
RESULTS_DIR = f"{SAVE_TO}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_QCD_files(year_dirs: list[str] = YEAR_DIR) -> list[str]:
    qcd_files = []
    for year_dir in year_dirs:
        for root, _, files in os.walk(year_dir):
            for file in files:
                f = os.path.join(root, file)
                if "QCD" in f and f.endswith(".root"):
                    qcd_files.append(f)
    print(f"Found {len(qcd_files)} QCD files")
    return qcd_files


def load_results(in_file):
    with open(in_file, "rb") as f:
        return pickle.load(f)


def compute_rho(m, pt):
    return np.log((m * m) / (pt * pt))


def lookup_quantile(pt, rho, qsurf, pt_edges, rho_edges):
    ipt = np.digitize(pt, pt_edges) - 1
    irho = np.digitize(rho, rho_edges) - 1

    valid = (
        (ipt >= 0) & (ipt < qsurf.shape[0]) &
        (irho >= 0) & (irho < qsurf.shape[1])
    )

    q = np.full(pt.shape, np.nan)
    q[valid] = qsurf[ipt[valid], irho[valid]]
    return q

def aggregate_QCD_mass_inclusive(
    qcd_files,
    pt_regions,
    mass_range=(30, 200),
    mass_bin_width=10.0,
):

    n_mass_bins = int((mass_range[1] - mass_range[0]) / mass_bin_width)
    mass_bins = np.linspace(mass_range[0], mass_range[1], n_mass_bins + 1)

    hist = np.zeros((len(pt_regions), n_mass_bins))
    hist_w2 = np.zeros_like(hist)

    pt_bounds = np.array([p[0] for p in pt_regions] + [pt_regions[-1][1]])

    branches = [
        txbb_variable,
        "bbFatJetPt0",
        mass_variable,
        "weight",
    ]

    for file in tqdm(qcd_files, desc="Inclusive mass aggregation", leave=False):
        with uproot.open(file) as f:
            if "Events" not in f:
                continue

            ev = f["Events"].arrays(branches, library="np")

            txbb = ev[txbb_variable]
            pt   = ev["bbFatJetPt0"]
            mass = ev[mass_variable]
            w    = ev["weight"]

            mask = np.isfinite(pt) & np.isfinite(mass) & np.isfinite(w)
            if not np.any(mask):
                continue

            pt, mass, w = pt[mask], mass[mask], w[mask]

            pre = txbb > 0.3 # extra selection to match data selections made earlier
            txbb, pt, mass, w = txbb[pre], pt[pre], mass[pre], w[pre] 

            pt_idx = np.digitize(pt, pt_bounds) - 1
            valid = (pt_idx >= 0) & (pt_idx < len(pt_regions))

            pt_idx, mass, w = pt_idx[valid], mass[valid], w[valid]

            for i in range(len(pt_regions)):
                sel = pt_idx == i
                if not np.any(sel):
                    continue

                h, _  = np.histogram(mass[sel], bins=mass_bins, weights=w[sel])
                h2, _ = np.histogram(mass[sel], bins=mass_bins, weights=w[sel] ** 2)

                hist[i]    += h
                hist_w2[i] += h2

    return hist, hist_w2, mass_bins


def aggregate_QCD_mass_DDT(
    qcd_files,
    qsurf,
    pt_edges,
    rho_edges,
    pt_regions,
    mass_range=(30, 200),
    mass_bin_width=10.0,
):

    n_mass_bins = int((mass_range[1] - mass_range[0]) / mass_bin_width)
    mass_bins = np.linspace(mass_range[0], mass_range[1], n_mass_bins + 1)

    hist = np.zeros((len(pt_regions), n_mass_bins))
    hist_w2 = np.zeros_like(hist)

    pt_bounds = np.array([p[0] for p in pt_regions] + [pt_regions[-1][1]])

    branches = [
        txbb_variable,
        "bbFatJetPt0",
        mass_variable,
        "weight",
    ]

    for file in tqdm(qcd_files, leave=False):
        with uproot.open(file) as f:
            if "Events" not in f:
                continue

            ev = f["Events"].arrays(branches, library="np")

            txbb = ev[txbb_variable]
            pt = ev["bbFatJetPt0"]
            mass = ev[mass_variable]
            w = ev["weight"]

            mask = np.isfinite(txbb) & np.isfinite(pt) & np.isfinite(mass) & np.isfinite(w)
            if not np.any(mask):
                continue

            txbb, pt, mass, w = txbb[mask], pt[mask], mass[mask], w[mask]

            if extra_txbb_cut != 0.0:
                pre = extra_txbb_cut > 0.3 # extra selection to match data selections made earlier
                txbb, pt, mass, w = txbb[pre], pt[pre], mass[pre], w[pre] 

            rho = compute_rho(mass, pt)
            q = lookup_quantile(pt, rho, qsurf, pt_edges, rho_edges)

            passed = (txbb - q) > 0
            if not np.any(passed):
                continue

            pt, mass, w = pt[passed], mass[passed], w[passed]

            pt_idx = np.digitize(pt, pt_bounds) - 1
            valid = (pt_idx >= 0) & (pt_idx < len(pt_regions))

            pt_idx, mass, w = pt_idx[valid], mass[valid], w[valid]

            for i in range(len(pt_regions)):
                sel = pt_idx == i
                if not np.any(sel):
                    continue

                h, _ = np.histogram(mass[sel], bins=mass_bins, weights=w[sel])
                h2, _ = np.histogram(mass[sel], bins=mass_bins, weights=w[sel] ** 2)

                hist[i] += h
                hist_w2[i] += h2

    return hist, hist_w2, mass_bins

def plot_all_alphas_for_pt(
    aggregate,
    pt_region,
    reference_alpha,
    outfile,
):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 12),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        sharex=True,
    )

    inc = aggregate[("inclusive", pt_region)]
    inc_hist = inc["hist"]
    inc_err  = inc["err"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(ALPHAS)))

    # ref_hist = aggregate[(reference_alpha, pt_region)]["hist"]
    # ref_err = aggregate[(reference_alpha, pt_region)]["err"]
    ref_hist = inc_hist
    ref_err  = inc_err
    bins = aggregate[(reference_alpha, pt_region)]["bins"]
    centers = 0.5 * (bins[:-1] + bins[1:])

    ax1.step(
        centers,
        inc_hist,
        where="mid",
        lw=3,
        color="black",
        ls="--",
        label="Inclusive (no DDT)",
    )
    ax1.errorbar(
        centers,
        inc_hist,
        yerr=inc_err,
        fmt="none",
        color="black",
    )

    for color, alpha in zip(colors, ALPHAS):
        d = aggregate[(alpha, pt_region)]

        ax1.step(centers, d["hist"], where="mid", lw=3, color=color,
                 label=fr"$\alpha={alpha}$")
        ax1.errorbar(centers, d["hist"], yerr=d["err"],
                     fmt="none", color=color)

        # if alpha == reference_alpha:
        #     continue

        ratio = d["hist"] / ref_hist
        ratio_err = ratio * np.sqrt(
            (d["err"] / d["hist"]) ** 2 +
            (ref_err / ref_hist) ** 2
        )

        ax2.step(centers, ratio, where="mid", lw=3, color=color)
        ax2.errorbar(centers, ratio, yerr=ratio_err,
                     fmt="none", color=color)

    ax1.set_ylabel("A.U.")
    ax1.legend()

    ax2.axhline(1.0, color="black", ls="--")
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel(r"$m_{\mathrm{X2p}}$ [GeV]")

    hep.cms.label(data=False, year="2023", com="13.6", ax=ax1)

    fig.savefig(outfile)
    plt.close(fig)

def main():
    qcd_files = get_QCD_files()

    pt_regions = [
        (300, 450),
        (450, 550),
        (550, 10000),
    ]

    aggregate = {}

    print("\nProcessing inclusive (no DDT)")

    hist_inc, hist_inc_w2, bins = aggregate_QCD_mass_inclusive(
        qcd_files,
        pt_regions,
    )

    for alpha in ALPHAS:
        print(f"\nProcessing alpha = {alpha}")
        qsurf, pt_edges, rho_edges = load_results(QDICT[alpha])

        hist, hist_w2, bins = aggregate_QCD_mass_DDT(
            qcd_files,
            qsurf,
            pt_edges,
            rho_edges,
            pt_regions,
        )

        for i, pt in enumerate(pt_regions):
            h = hist[i]
            h2 = hist_w2[i]
            norm = h.sum()

            aggregate[(alpha, pt)] = {
                "hist": h / norm,
                "err": np.sqrt(h2) / norm,
                "bins": bins,
            }

    for i, pt in enumerate(pt_regions):
        h  = hist_inc[i]
        h2 = hist_inc_w2[i]
        norm = h.sum()

        aggregate[("inclusive", pt)] = {
            "hist": h / norm,
            "err":  np.sqrt(h2) / norm,
            "bins": bins,
        }


    for pt in pt_regions:
        plot_all_alphas_for_pt(
            aggregate,
            pt_region=pt,
            reference_alpha=REFERENCE_ALPHA,
            outfile=f"{RESULTS_DIR}/DDT_mass_allAlpha_pT{pt[0]}to{pt[1]}.pdf",
        )


if __name__ == "__main__":
    main()
