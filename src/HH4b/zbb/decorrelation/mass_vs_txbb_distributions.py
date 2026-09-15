#!/usr/bin/env python3

import os
import pickle
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

from tqdm import tqdm
from matplotlib.colors import LogNorm

hep.style.use("CMS")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPROCESS = True

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023", "2023BPix"]
TAG = "13Feb2026_Data_Standard_Cuts_QCD_Inclusive_v15_scouting_zbb"

YEAR_DIRS = [f"{SKIMMER_DIR}/{TAG}/{y}" for y in YEARS]

NBINS_M = 60
NBINS_TXBB = 60

M_LIMS = (20, 200)     # GeV
TXBB_LIMS = (0.8, 1.0)

OUTFILE = "mGeneric_vs_TXbb_weighted_normed.pdf"
HIST_OUTFILE = "hist2d_mGeneric.pkl"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_qcd_files(year_dirs):
    files = []
    for year_dir in year_dirs:
        for root, _, fnames in os.walk(year_dir):
            for fname in fnames:
                f = os.path.join(root, fname)
                if "QCD" in f and f.endswith(".root"):
                    files.append(f)

    print(f"Found {len(files)} QCD files")
    return files


def build_weighted_hist2d(files):
    hist = np.zeros((NBINS_M, NBINS_TXBB), dtype=np.float64)

    m_edges = np.linspace(*M_LIMS, NBINS_M + 1)
    txbb_edges = np.linspace(*TXBB_LIMS, NBINS_TXBB + 1)

    for f in tqdm(files, desc="Reading QCD"):
        with uproot.open(f) as file:
            if "Events" not in file:
                continue

            ev = file["Events"]

            m = ev["bbFatJetScoutParTmassGeneric0"].array(library="np")
            txbb = ev["bbFatJetScoutParTTXbb0"].array(library="np")
            w = ev["weight"].array(library="np")

            mask = (m > 0) & np.isfinite(txbb) & np.isfinite(w)
            if not np.any(mask):
                continue

            h, _, _ = np.histogram2d(
                m[mask],
                txbb[mask],
                bins=[m_edges, txbb_edges],
                weights=w[mask]
            )

            hist += h

    # Normalize to unity (joint PDF)
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist, m_edges, txbb_edges


def plot_hist2d(hist, m_edges, txbb_edges, outfile):
    fig, ax = plt.subplots(figsize=(10, 9))

    mesh = ax.pcolormesh(
        m_edges,
        txbb_edges,
        hist.T,
        shading="auto",
        cmap="hsv",
        norm=LogNorm()
    )

    fig.colorbar(mesh, ax=ax, label="Normalized density")

    ax.set_xlabel(r"$m_{\mathrm{generic}}$ [GeV]")
    ax.set_ylabel(r"$T_{Xbb}$")

    hep.cms.label(ax=ax, data=False, year="2023", com="13.6")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def save_results(obj, out_file):
    with open(out_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(in_file):
    with open(in_file, "rb") as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    files = get_qcd_files(YEAR_DIRS)

    if REPROCESS:
        hist, m_edges, txbb_edges = build_weighted_hist2d(files)
        save_results((hist, m_edges, txbb_edges), HIST_OUTFILE)
    else:
        hist, m_edges, txbb_edges = load_results(HIST_OUTFILE)

    plot_hist2d(hist, m_edges, txbb_edges, OUTFILE)


if __name__ == "__main__":
    main()
