from __future__ import annotations

import os
import pickle
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import mplhep as hep

from tqdm import tqdm

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

SCOUTING = True
BLIND_REGION = (100, 150)

offline_details = {
    "model": "GloParT-v3",
    "mass_var": "bbFatJetParT3massCorrectedX2p0",
    "txbb_var": "bbFatJetParT3TXbb0",
}

scouting_details = {
    "model": "Scouting GloParT (22–23)",
    "mass_var": "bbFatJetScoutParTmassCorrectedX2p0",
    "txbb_var": "bbFatJetScoutParTTXbb0",
}

details = scouting_details if SCOUTING else offline_details

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023", "2023BPix"]
TAG = (
    "14Feb2026_VJets_TT_Only_v15_scouting_zbb"
    if SCOUTING
    else "07Feb2026_PTl450TXbb0p3_PTsl200_HT1000_v15_scouting_zbb"
)
YEAR_DIRS = [f"{SKIMMER_DIR}/{TAG}/{YEAR}" for YEAR in YEARS]

SAVE_TO = "/eos/user/z/zima/DATA_SPECTRA"
RESULTS_DIR = f"{SAVE_TO}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

mpl.rcParams["font.size"] = 30
mpl.rcParams["figure.dpi"] = 400

def apply_mass_blinding(hist, bins, blind_region):
    if blind_region is None:
        return hist.copy(), np.ones_like(hist, dtype=bool)

    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = (centers < blind_region[0]) | (centers > blind_region[1])

    out = hist.copy().astype(float)
    out[~mask] = np.nan
    return out, mask


def get_DATA_files(year_dirs: list(str) = YEAR_DIRS) -> list:
    """Get all DATA root files from the directory structure"""
    data_filelist = []
    for year_dir in year_dirs:
        for root, dirs, files in os.walk(year_dir):
            for file in files:
                f = os.path.join(root, file)
                if ("Run2023C" in f or "Run2023D" in f) and f.endswith(".root"):
                    data_filelist.append(f)
    
    print(f"Length of DATA filelist: {len(data_filelist)}")
    return data_filelist


def lookup_quantile(pt, rho, qsurf, pt_edges, rho_edges):
    ipt = np.digitize(pt, pt_edges) - 1
    irho = np.digitize(rho, rho_edges) - 1

    valid = (
        (ipt >= 0) & (ipt < qsurf.shape[0]) &
        (irho >= 0) & (irho < qsurf.shape[1])
    )

    q = np.full_like(pt, np.nan, dtype=float)
    q[valid] = qsurf[ipt[valid], irho[valid]]
    return q


def aggregate_DATA_DDT_regions(
    alphas,
    qsurf_dict,
    pt_regions,
    data_files,
    mass_variable,
    txbb_variable,
    mass_range=(20, 200),
    mass_bin_width=5.0,
):

    bins = np.arange(mass_range[0], mass_range[1] + mass_bin_width, mass_bin_width)
    n_bins = len(bins) - 1

    out = {}

    for a in alphas:
        for pt in pt_regions:
            key = f"DDT α={a}, pT=[{pt[0]}, {pt[1]}]"
            out[key] = dict(
                hist=np.zeros(n_bins),
                bins=bins,
                alpha=a,
                pt_region=pt,
                total_events=0,
            )

    for f in tqdm(data_files, desc="Processing DATA"):
        with uproot.open(f) as file:
            if "Events" not in file:
                continue

            ev = file["Events"]
            try:
                txbb = ev[txbb_variable].array().to_numpy()
                pt = ev["bbFatJetPt0"].array().to_numpy()
                mass = ev[mass_variable].array().to_numpy()
            except KeyError:
                continue

            mask = np.isfinite(txbb) & np.isfinite(pt) & np.isfinite(mass)
            if not np.any(mask):
                continue

            txbb, pt, mass = txbb[mask], pt[mask], mass[mask]
            rho = np.log((mass * mass) / (pt * pt))

            for a in alphas:
                qsurf, pt_edges, rho_edges = qsurf_dict[a]
                q = lookup_quantile(pt, rho, qsurf, pt_edges, rho_edges)

                passed = (txbb - q) > 0
                if not np.any(passed):
                    continue

                for pt_reg in pt_regions:
                    m = (pt[passed] >= pt_reg[0]) & (pt[passed] < pt_reg[1])
                    if not np.any(m):
                        continue

                    h, _ = np.histogram(mass[passed][m], bins=bins)
                    key = f"DDT α={a}, pT=[{pt_reg[0]}, {pt_reg[1]}]"
                    out[key]["hist"] += h
                    out[key]["total_events"] += m.sum()

    return out


def normalize_histograms(agg, blind_region):
    out = {}
    for k, d in agg.items():
        h, mask = apply_mass_blinding(d["hist"], d["bins"], blind_region)
        norm = np.nansum(h)
        out[k] = dict(
            hist=h / norm if norm > 0 else h,
            bins=d["bins"],
            alpha=d["alpha"],
            pt_region=d["pt_region"],
            total_events=d["total_events"],
        )
    return out

def compare_shapes_in_pt_region(aggregate_dict, reference_key, pt_region, blind_region):

    norm = normalize_histograms(aggregate_dict, blind_region)

    keys = [
        k for k in norm
        if f"pT=[{pt_region[0]}, {pt_region[1]}]" in k
    ]

    if reference_key not in keys:
        raise KeyError(f"Reference key not found: {reference_key}")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(keys)))
    color_map = dict(zip(keys, colors))

    ref_hist = norm[reference_key]["hist"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 12),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        sharex=True,
    )

    plot_text_lines = [
    "HLT Scouting Data",
    f"Model: Scouting GloParT-DDT (22-23)",
    rf"$p_T \in ({pt_region[0]}, {pt_region[1]})$",
    r"$T_{{\mathrm{{Xbb}}}}(\rho, p_T) > T_{{\mathrm{{Xbb}}}}^\alpha(\rho, p_T)$"
    ]

    # text_x_position = 0.25
    # text_y_start = 0.85
    text_x_position = 0.2
    text_y_start = 0.85

    for i, line in enumerate(plot_text_lines):
        fig.text(text_x_position, text_y_start - i*0.04, line,
                 fontsize=20, fontweight='bold',
                 verticalalignment='top',
                 horizontalalignment='left')

    # ---- Shapes
    for k in keys:
        d = norm[k]
        bins = d["bins"]
        centers = 0.5 * (bins[:-1] + bins[1:])

        raw = aggregate_dict[k]["hist"]
        raw_b, mask = apply_mass_blinding(raw, bins, blind_region)
        err = np.sqrt(raw_b)
        s = np.nansum(raw_b)
        err = err / s if s > 0 else err
        err[~mask] = np.nan

        ax1.step(centers, d["hist"], where="mid",
                 color=color_map[k], linewidth=2,
                 label=rf"$\alpha = {d['alpha']}$")
        ax1.errorbar(centers, d["hist"], yerr=err,
                     fmt="none", color=color_map[k])

    ax1.legend(fontsize=28)
    ax1.set_ylabel("A.U.")

    # ---- Ratios
    for k in keys:
        if k == reference_key:
            continue

        d = norm[k]
        bins = d["bins"]
        centers = 0.5 * (bins[:-1] + bins[1:])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ref_hist > 0, d["hist"] / ref_hist, np.nan)

        ax2.step(centers, ratio, where="mid",
                 color=color_map[k], linewidth=2)

    ax2.axhline(1.0, color="black", linestyle="--")
    ax2.set_ylim(0.5, 2.0)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel(r"$m_{\mathrm{X2p}}$ [GeV]")

    for ax in (ax1, ax2):
        ax.axvspan(*blind_region, color="gray", alpha=0.3, hatch="//")

    hep.cms.label("Internal", data=True, year="2023",
                  com="13.6", lumi=28, ax=ax1)

    return fig

def main():
    data_files = get_DATA_files()

    qsurf_dict = {
        a: pickle.load(open(QDICT[a], "rb"))
        for a in ALPHAS
    }

    pt_regions = [(300, 450), (450, 550), (550, 10000)]

    agg = aggregate_DATA_DDT_regions(
        ALPHAS,
        qsurf_dict,
        pt_regions,
        data_files,
        details["mass_var"],
        details["txbb_var"],
    )

    for pt in pt_regions:
        ref = f"DDT α={ALPHAS[0]}, pT=[{pt[0]}, {pt[1]}]"
        fig = compare_shapes_in_pt_region(agg, ref, pt, BLIND_REGION)
        fig.savefig(f"{RESULTS_DIR}/pT{pt[0]}to{pt[1]}_DDT_TXBBPRE0p3.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
