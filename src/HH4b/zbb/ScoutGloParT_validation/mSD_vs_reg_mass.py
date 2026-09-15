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

def stream_multiple_hist1d(files, details, mass_keys,
                           bins=100, mass_range=(0, 300),
                           step_size="300 MB"):
    """
    Stream once over files and fill multiple 1D histograms.
    Returns dict of normalized histograms and common bin edges.
    """

    edges = np.linspace(*mass_range, bins + 1)

    # Initialize histogram accumulators
    hists = {key: np.zeros(bins, dtype=np.float64) for key in mass_keys}

    branches = [details[key] for key in mass_keys]
    files_with_tree = [f"{f}:Events" for f in files]

    for arrays in tqdm(
        uproot.iterate(
            files_with_tree,
            expressions=branches,
            library="np",
            step_size=step_size,
            allow_missing=True,
            # num_workers=4,  # enable on lxplus if desired
        ),
        desc="Streaming mass branches",
    ):

        for key in mass_keys:
            branch = details[key]

            if branch not in arrays:
                continue

            x = arrays[branch]
            mask = np.isfinite(x)
            x = x[mask]

            h_chunk, _ = np.histogram(x, bins=edges)
            hists[key] += h_chunk

    # Normalize all histograms to unit area
    for key in mass_keys:
        total = hists[key].sum()
        if total > 0:
            hists[key] /= total

    return hists, edges

def plot_normalized_masses(hists, edges, details):

    fig, ax = plt.subplots(figsize=(10, 8))
    centers = 0.5 * (edges[1:] + edges[:-1])

    for key, hist in hists.items():
        if key == "mx2p":
            label = r"$m_\text{X2p}$"
        elif key == "mgeneric":
            label = r"$m_\text{generic}$"
        elif key == "msd":
            label = r"$m_\text{SD}$"
        else:
            label = key
        ax.step(centers, hist, where="mid", linewidth=2, label=label)

    ax.set_xlabel("Mass [GeV]")
    ax.set_yscale("log")
    ax.set_ylabel("A.U.")

    model = details["model"]
    cuts = details["cuts"]
    ax.text(
        0.07, 0.93,
        f"QCD Multijet\nModel: {model}\n{cuts}",
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

    # ax.set_title(f"{details['model']}\n{details['cuts']}")
    ax.legend()
    ax.set_xlim(edges[0], edges[-1])

    return fig, ax

def main():

    files = get_QCD_files(YEAR_DIRS)
    if not files:
        print("No files found! Check your paths.")
        return

    mass_keys = ["msd", "mx2p", "mgeneric"]

    hists, edges = stream_multiple_hist1d(
        files,
        details,
        mass_keys,
        bins=58,
        mass_range=(10, 300),
        step_size="300 MB", 
    )

    fig, ax = plot_normalized_masses(
        hists,
        edges,
        details,
    )

    plt.savefig("normalized_mass_comparison_scouting.pdf")

if __name__ == "__main__":
    main()
