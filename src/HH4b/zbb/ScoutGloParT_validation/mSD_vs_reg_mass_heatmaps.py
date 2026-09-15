from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
import os

import mplhep as hep
hep.style.use("CMS")

plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

SCOUTING = True
LOAD_FROM_CACHE = False
CACHE_FILE = "qcd_mass_heatmaps_scouting.npz"


offline_details = {
    "tag": "06Feb2025_QCD_Offline_Inclusive_PTl_300_etaS_HT_1k_v15_scouting_zbb",
    "model": "GloParT-v3",
    "cuts": r"$H_T > 1000$ GeV, $|\eta| < 2.4$",
    "msd": "bbFatJetMsd0",
    "mx2p": "bbFatJetParT3massCorrectedX2p0",    
    "mgeneric": "bbFatJetParT3massGeneric0",
    "txbb_var": "bbFatJetParT3TXbb0",
}

scouting_details = {
    "tag": "13Feb2026_Data_Standard_Cuts_QCD_Inclusive_v15_scouting_zbb", #"06Feb2026_QCD_Scouting_Inclusive_v15_scouting_zbb",
    "model": "Scouting GloParT (22-23)",
    "cuts": r"$H_T > 600$ GeV, $|\eta| < 2.4$",
    "msd": "bbFatJetMsd0",
    "mx2p": "bbFatJetScoutParTmassCorrectedX2p0",    
    "mgeneric": "bbFatJetScoutParTmassGeneric0",
    "txbb_var": "bbFatJetScoutParTTXbb0",
}

if SCOUTING:
    details = scouting_details
else:
    details = offline_details

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023", "2023BPix"]
TAG = details["tag"]
YEAR_DIRS = [f"{SKIMMER_DIR}/{TAG}/{YEAR}" for YEAR in YEARS]

def get_QCD_files(year_dirs: list[str] = YEAR_DIRS):
    """Get all QCD root files from the directory structure"""
    qcd_filelist = []
    for year_dir in year_dirs:
        if not os.path.exists(year_dir):
            print(f"Warning: {year_dir} does not exist, skipping")
            continue
        for root, dirs, files in os.walk(year_dir):
            for file in files:
                f = os.path.join(root, file)
                if "QCD" in f and f.endswith(".root"):
                    qcd_filelist.append(f)
    
    print(f"Length of QCD filelist: {len(qcd_filelist)}")
    return qcd_filelist

def stream_hist2d(
    files,
    details,
    x_key,
    y_key,
    bins=(40, 40),
    ranges=((10, 300), (10, 300)),
    step_size="100 MB",
):
    """
    Stream once over files and fill a 2D histogram.
    Returns histogram and bin edges.
    """

    x_edges = np.linspace(*ranges[0], bins[0] + 1)
    y_edges = np.linspace(*ranges[1], bins[1] + 1)

    hist2d = np.zeros((bins[0], bins[1]), dtype=np.float64)

    x_branch = details[x_key]
    y_branch = details[y_key]

    files_with_tree = [f"{f}:Events" for f in files]

    for arrays in tqdm(
        uproot.iterate(
            files_with_tree,
            expressions=[x_branch, y_branch],
            library="np",
            step_size=step_size,
            allow_missing=True,
        ),
        desc=f"Streaming {x_key} vs {y_key}",
    ):

        if x_branch not in arrays or y_branch not in arrays:
            continue

        x = arrays[x_branch]
        y = arrays[y_branch]

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        h_chunk, _, _ = np.histogram2d(
            x, y, bins=(x_edges, y_edges)
        )
        hist2d += h_chunk

    return hist2d, x_edges, y_edges


def stream_hist2d_filewise(
    files,
    details,
    x_key,
    y_key,
    bins=(40, 40),
    ranges=((10, 300), (10, 300)),
):
    """
    Loop file-by-file and fill a 2D histogram.
    """

    x_edges = np.linspace(*ranges[0], bins[0] + 1)
    y_edges = np.linspace(*ranges[1], bins[1] + 1)

    hist2d = np.zeros((bins[0], bins[1]), dtype=np.float64)

    x_branch = details[x_key]
    y_branch = details[y_key]

    for f in tqdm(files, desc=f"Processing {x_key} vs {y_key}"):

        try:
            with uproot.open(f) as file:
                if "Events" not in file:
                    continue

                tree = file["Events"]

                # Load only needed branches
                arrays = tree.arrays(
                    [x_branch, y_branch],
                    library="np",
                )

        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        if x_branch not in arrays or y_branch not in arrays:
            continue

        x = arrays[x_branch]
        y = arrays[y_branch]

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        h_chunk, _, _ = np.histogram2d(
            x, y,
            bins=(x_edges, y_edges),
        )

        hist2d += h_chunk

    return hist2d, x_edges, y_edges

def stream_hist2d_multi_filewise(
    files,
    details,
    bins=(40, 40),
    ranges=((10, 300), (10, 300)),
):
    """
    Loop once over files and fill:
      - msd vs x2p
      - msd vs generic
    """

    x_edges = np.linspace(*ranges[0], bins[0] + 1)
    y_edges = np.linspace(*ranges[1], bins[1] + 1)

    h_msd_x2p = np.zeros((bins[0], bins[1]), dtype=np.float64)
    h_msd_gen = np.zeros((bins[0], bins[1]), dtype=np.float64)

    msd_branch = details["msd"]
    x2p_branch = details["mx2p"]
    gen_branch = details["mgeneric"]

    branches = [msd_branch, x2p_branch, gen_branch]

    for f in tqdm(files, desc="Processing files (all masses)"):

        try:
            with uproot.open(f) as file:
                if "Events" not in file:
                    continue

                tree = file["Events"]
                arrays = tree.arrays(branches, library="np")

        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        if not all(b in arrays for b in branches):
            continue

        msd = arrays[msd_branch]
        x2p = arrays[x2p_branch]
        gen = arrays[gen_branch]

        # Common finite mask
        base_mask = np.isfinite(msd)

        # --- msd vs x2p ---
        mask_x2p = base_mask & np.isfinite(x2p)
        if np.any(mask_x2p):
            h_chunk, _, _ = np.histogram2d(
                msd[mask_x2p],
                x2p[mask_x2p],
                bins=(x_edges, y_edges),
            )
            h_msd_x2p += h_chunk

        # --- msd vs generic ---
        mask_gen = base_mask & np.isfinite(gen)
        if np.any(mask_gen):
            h_chunk, _, _ = np.histogram2d(
                msd[mask_gen],
                gen[mask_gen],
                bins=(x_edges, y_edges),
            )
            h_msd_gen += h_chunk

    return (h_msd_x2p, h_msd_gen), x_edges, y_edges



def plot_heatmap(
    hist2d,
    x_edges,
    y_edges,
    xlabel,
    ylabel,
    # title,
    details,
    fname,
):
    fig, ax = plt.subplots(figsize=(9, 8))

    # Avoid log(0) issues
    eps = 1e-6
    hist2d = np.where(hist2d > 0, hist2d, eps)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist2d.T,
        norm=plt.matplotlib.colors.LogNorm(),
        cmap="viridis",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Events")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.text(
        0.05, 0.95,
        f"QCD Multijet\nModel: {details['model']}\n{details['cuts']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    hep.cms.label(
        year="2023",
        data=False,
        com="13.6",
        ax=ax,
        fontsize=18,
    )

    # ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def save_histograms(
    fname,
    h_msd_x2p,
    h_msd_gen,
    x_edges,
    y_edges,
    metadata=None,
):
    """
    Save histograms + binning (+ optional metadata).
    """
    np.savez_compressed(
        fname,
        h_msd_x2p=h_msd_x2p,
        h_msd_gen=h_msd_gen,
        x_edges=x_edges,
        y_edges=y_edges,
        metadata=metadata,
    )


def load_histograms(fname):
    """
    Load histograms + binning.
    """
    data = np.load(fname, allow_pickle=True)
    return (
        data["h_msd_x2p"],
        data["h_msd_gen"],
        data["x_edges"],
        data["y_edges"],
        data.get("metadata", None),
    )


def main():

    bins = (30, 30)
    ranges = ((0, 300), (0, 300))

    if LOAD_FROM_CACHE and os.path.exists(CACHE_FILE):

        print(f"Loading cached histograms from {CACHE_FILE}")
        h_msd_x2p, h_msd_gen, x_edges, y_edges, metadata = load_histograms(
            CACHE_FILE
        )

    else:
        files = get_QCD_files(YEAR_DIRS)
        if not files:
            print("No files found!")
            return

        (h_msd_x2p, h_msd_gen), x_edges, y_edges = (
            stream_hist2d_multi_filewise(
                files,
                details,
                bins=bins,
                ranges=ranges,
            )
        )

        metadata = {
            "bins": bins,
            "ranges": ranges,
            "tag": details["tag"],
            "model": details["model"],
            "cuts": details["cuts"],
        }

        save_histograms(
            CACHE_FILE,
            h_msd_x2p,
            h_msd_gen,
            x_edges,
            y_edges,
            metadata,
        )

    # ---- Plotting (fast, repeatable) ----

    plot_heatmap(
        h_msd_x2p,
        x_edges,
        y_edges,
        xlabel=r"$m_\mathrm{SD}$ [GeV]",
        ylabel=r"$m_\mathrm{X2p}$ [GeV]",
        details=details,
        fname="heatmap_msd_vs_x2p_scouting.pdf",
    )

    plot_heatmap(
        h_msd_gen,
        x_edges,
        y_edges,
        xlabel=r"$m_\mathrm{SD}$ [GeV]",
        ylabel=r"$m_\mathrm{generic}$ [GeV]",
        details=details,
        fname="heatmap_msd_vs_generic_scouting.pdf",
    )



if __name__ == "__main__":
    main()
