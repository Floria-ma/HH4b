from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import subprocess
import uproot
import numpy as np
import awkward as ak
import pickle

import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import matplotlib.ticker as mticker
import mplhep as hep
from hist import Hist
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from tqdm import tqdm

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

SCOUTING: bool = False
BLIND_REGION = (100, 150)

offline_details = {
    "model": "GloParT-v3",
    "mass_var": "bbFatJetParT3massGeneric0", #"bbFatJetMsd0", #"bbFatJetParT3massGeneric0", #"bbFatJetParT3massCorrectedX2p0",
    "txbb_var": "bbFatJetParT3TXbb0",
}

scouting_details = {
    "model": "Scouting GloParT (22-23)",
    "mass_var": "bbFatJetMsd0", #"bbFatJetScoutParTmassGeneric0", #"bbFatJetScoutParTmassCorrectedX2p0",
    "txbb_var": "bbFatJetScoutParTTXbb0",
}

if SCOUTING:
    details = scouting_details
else:
    details = offline_details

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

mpl.rcParams["font.size"] = 30
mpl.rcParams["lines.linewidth"] = 2
# mpl.rcParams["grid.color"] = "#CCCCCC"
# mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["figure.edgecolor"] = "none"

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023", "2023BPix"]
TAG =  "07Feb2026_PTl450TXbb0p3_PTsl200_HT1000_v15_scouting_zbb" if not SCOUTING else "14Feb2026_VJets_TT_Only_v15_scouting_zbb"
YEAR_DIRS = [f"{SKIMMER_DIR}/{TAG}/{YEAR}" for YEAR in YEARS]
REPROCESS = True

SAVE_TO = "/eos/user/z/zima/DATA_SPECTRA"
SAVE_TO_FILE = f"{SAVE_TO}/saved_aggregate_dict.pkl"
RESULTS_DIR = f"{SAVE_TO}/results"
os.makedirs(RESULTS_DIR, exist_ok = True)

print(f"{SAVE_TO_FILE} exists {os.path.isfile(SAVE_TO_FILE)}")

def apply_mass_blinding(hist, bins, blind_region=(100.0, 150.0)):
    """
    Zero histogram bins inside the blinded mass region.

    Parameters
    ----------
    hist : np.ndarray
        Histogram values
    bins : np.ndarray
        Bin edges
    blind_region : tuple or None
        (min, max) mass region to blind. If None, no blinding.

    Returns
    -------
    blinded_hist : np.ndarray
        Histogram with blinded bins set to zero
    blind_mask : np.ndarray (bool)
        True for bins that are kept (unblinded)
    """
    if blind_region is None:
        return hist.copy(), np.ones_like(hist, dtype=bool)

    m_low, m_high = blind_region
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    blind_mask = (bin_centers < m_low) | (bin_centers > m_high)

    blinded_hist = hist.copy()
    blinded_hist[~blind_mask] = np.nan

    return blinded_hist, blind_mask


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

def aggregate_DATA_TXbb_regions(
    txbb_regions: list = None,
    pt_regions: list = None,
    data_filelist: list = None,
    mass_variable: str = "bbFatJetScoutParTmassCorrectedX2p0",
    mass_range: tuple = (20, 200),
    mass_bin_width: float = 10.0
):
    """
    Aggregate DATA events into histograms for each pt and TXbb region combination.
    
    Parameters:
    -----------
    txbb_regions : list of tuples
        List of (txbb_min, txbb_max) for TXbb regions
    pt_regions : list of tuples
        List of (pt_min, pt_max) for pt regions
    data_filelist : list
        List of DATA root files to process
    mass_variable : str
        Name of the mass variable in the ntuples
    mass_range : tuple
        (min, max) for mass histogram
    mass_bin_width : float
        Width of mass bins
    
    Returns:
    --------
    dict : Dictionary with region combinations as keys and (hist, bins) as values
    """
    
    if txbb_regions is None:
        raise Exception("TXbb regions not defined")
    
    if pt_regions is None:
        raise Exception("pT regions not defined")
    
    if data_filelist is None:
        data_filelist = get_DATA_files()
    
    print(f"Processing {len(data_filelist)} DATA files...")
    
    n_bins = int((mass_range[1] - mass_range[0]) / mass_bin_width)
    mass_bins = np.linspace(mass_range[0], mass_range[1], n_bins + 1)
    
    aggregate_dictionary = {}

    for txbb_region in txbb_regions:
        for pt_region in pt_regions:
            key = f"TXbb = [{txbb_region[0]}, {txbb_region[1]}], pT = [{pt_region[0]}, {pt_region[1]}]"
            aggregate_dictionary[key] = {
                'hist': np.zeros(n_bins),
                'bins': mass_bins,
                'txbb_region': txbb_region,
                'pt_region': pt_region,
                'total_events': 0
            }
    
    for i, file in enumerate(data_filelist):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(data_filelist)}: {os.path.basename(file)}")
        
        try:
            with uproot.open(file) as f:
                if "Events" not in f:
                    print(f"Warning: No 'Events' tree in {file}")
                    continue
                    
                events = f["Events"]
                
                try:
                    txbb_values = events[details["txbb_var"]].array().to_numpy()
                    pt_values = events["bbFatJetPt0"].array().to_numpy()
                    mass_values = events[mass_variable].array().to_numpy()

                except KeyError as e:
                    print(f"Warning: Missing variable {e} in {file}")
                    continue
                
                for txbb_region in txbb_regions:
                    txbb_mask = (txbb_values >= txbb_region[0]) & (txbb_values < txbb_region[1])
                    
                    for pt_region in pt_regions:
                        pt_mask = (pt_values >= pt_region[0]) & (pt_values < pt_region[1])
                        
                        combined_mask = txbb_mask & pt_mask
                        
                        if combined_mask.any():
                            selected_masses = mass_values[combined_mask]
                            
                            hist, _ = np.histogram(selected_masses, bins=mass_bins)
                            
                            key = f"TXbb = [{txbb_region[0]}, {txbb_region[1]}], pT = [{pt_region[0]}, {pt_region[1]}]"
                            # key = f"txbb_{txbb_region[0]}_{txbb_region[1]}_pt_{pt_region[0]}_{pt_region[1]}"
                            aggregate_dictionary[key]['hist'] += hist
                            aggregate_dictionary[key]['total_events'] += len(selected_masses)
                            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return aggregate_dictionary

def normalize_histograms(
    aggregate_dict,
    blind_region: tuple = (100.0, 150.0)
):
    """
    Normalize histograms to unity *after* applying mass blinding.
    """
    normalized_dict = {}

    for key, region_data in aggregate_dict.items():
        hist = region_data['hist']
        bins = region_data['bins']

        blinded_hist, blind_mask = apply_mass_blinding(
            hist, bins, blind_region
        )

        norm = np.nansum(blinded_hist)
        if norm > 0:
            hist_normalized = blinded_hist / norm
        else:
            hist_normalized = blinded_hist

        normalized_dict[key] = {
            'hist': hist_normalized,
            'bins': bins,
            'txbb_region': region_data['txbb_region'],
            'pt_region': region_data['pt_region'],
            'total_events': region_data['total_events'],
            'blind_region': blind_region,
            'norm_after_blinding': norm
        }

    return normalized_dict


def compare_shapes_in_pt_region(
    aggregate_dict: dict, 
    reference_key: str, 
    pt_region: tuple = (300, 450),
    blind_region: tuple = (100, 150)
    ):
    """
    Compare normalized shapes between different regions.
    
    Parameters:
    -----------
    aggregate_dict : dict
        Dictionary with region histograms
    reference_key : str
        Key for reference region. If None, use first region.
    """
    # plt.rcParams.update({
        # "font.size": 30,
        # "text.usetex": True,  # Use LaTeX to render text
        # "font.family": "serif",
        # Optional: Specify LaTeX preamble for packages
        # "text.latex.preamble": r"\usepackage{amsmath}"  # For better math symbols
    # })
    
    normalized_dict = normalize_histograms(aggregate_dict, blind_region = blind_region)
    
    keys = [k for k in aggregate_dict.keys() if f"pT = [{pt_region[0]}, {pt_region[1]}]" in k]
    print(keys)
    
    ref_data = normalized_dict[reference_key]
    ref_hist = ref_data['hist']
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 16), 
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
                           sharex=True)
    
    ax1 = axes[0]
    ax2 = axes[1]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(normalized_dict)))
    
    for i, (key, region_data) in enumerate(normalized_dict.items()):
        if f"pT = [{pt_region[0]}, {pt_region[1]}]" not in key:
            continue
        
        bins = region_data['bins']
        bin_edges = bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        label = fr"$T_{{\mathrm{{Xbb}}}} > {region_data['txbb_region'][0]}$"
        
        orig_hist = aggregate_dict[key]['hist']
        
        orig_hist_blinded, blind_mask = apply_mass_blinding(
            orig_hist, bins, blind_region
        )

        orig_errors = np.sqrt(orig_hist_blinded)

        norm = np.nansum(orig_hist_blinded)

        if norm > 0:
            scaled_errors = orig_errors / norm
        else:
            scaled_errors = np.zeros_like(orig_errors)

        scaled_errors[~blind_mask] = np.nan
        
        ax1.step(bin_centers, region_data['hist'], where='mid', 
                color=colors[i], linewidth=2, label=label)
        
        ax1.errorbar(bin_centers, region_data['hist'], 
                    yerr=scaled_errors, 
                    fmt='none',  
                    color=colors[i], 
                    # alpha=0.5,  
                    # capsize=3,  
                    # capthick=1,
                    linewidth=1)
    
    ax1.set_ylabel('A.U.')
    # ax1.set_title('Normalized Mass Spectra', y=1.08)
    
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        
        ax1.legend(handles, labels, loc='upper right', fontsize=30)
    
    ax1.set_xlabel("")
    # ax1.grid(True)

    model = details["model"]
    plot_text_lines = [
        "Offline JetMET Data",
        f"Model: {model}",
        rf"$p_T \in ({pt_region[0]}, {pt_region[1]})$",
        r"$T_{{\mathrm{{Xbb}}}} = \frac{P(X \to b\bar{b})}{P(X \to b\bar{b}) + P(\text{QCD})}$"
    ]

    text_x_position = 0.25  
    text_y_start = 0.85  
    
    for i, line in enumerate(plot_text_lines):
        fig.text(text_x_position, text_y_start - i*0.04, line,
                fontsize=25, fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left')

    for i, (key, region_data) in enumerate(normalized_dict.items()):
        if key == reference_key:
            continue
        if f"pT = [{pt_region[0]}, {pt_region[1]}" not in key:
            continue
            
        bins = region_data['bins']
        bin_edges = bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        orig_hist_current = aggregate_dict[key]['hist']
        orig_hist_ref = aggregate_dict[reference_key]['hist']

        orig_hist_current, blind_mask = apply_mass_blinding(
            orig_hist_current, bins, blind_region
        )
        orig_hist_ref, _ = apply_mass_blinding(
            orig_hist_ref, bins, blind_region
        )
        
        orig_errors_current = np.sqrt(orig_hist_current)
        orig_errors_ref = np.sqrt(orig_hist_ref)
        
        total_current = np.nansum(orig_hist_current)
        total_ref = np.nansum(orig_hist_ref)
        
        if total_current > 0 and total_ref > 0:
            norm_factor_current = 1.0 / total_current
            norm_factor_ref = 1.0 / total_ref
            
            scaled_errors_current = orig_errors_current * norm_factor_current
            scaled_errors_ref = orig_errors_ref * norm_factor_ref
            
            with np.errstate(divide='ignore', invalid='ignore'):
                valid = (~np.isnan(ref_hist)) & (ref_hist > 0)

                ratio = np.full_like(ref_hist, np.nan)
                ratio[valid] = region_data['hist'][valid] / ref_hist[valid]

                rel_err_current = np.zeros_like(scaled_errors_current)
                rel_err_ref = np.zeros_like(scaled_errors_ref)
                
                current_mask = region_data['hist'] > 0
                ref_mask = ref_hist > 0
                
                rel_err_current[current_mask] = scaled_errors_current[current_mask] / region_data['hist'][current_mask]
                rel_err_ref[ref_mask] = scaled_errors_ref[ref_mask] / ref_hist[ref_mask]
                
                rel_err_ratio = np.sqrt(rel_err_current**2 + rel_err_ref**2)
                
                ratio_errors = ratio * rel_err_ratio
                
                # Replace infinities and NaNs with 0
                ratio_errors = np.nan_to_num(ratio_errors, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            ratio = np.ones_like(region_data['hist'])
            ratio_errors = np.zeros_like(region_data['hist'])
        
        ax2.step(bin_centers, ratio, where='mid', 
                color=colors[i], linewidth=2)
        
        ax2.errorbar(bin_centers, ratio, 
                    yerr=ratio_errors, 
                    fmt='none',
                    color=colors[i], 
                    # alpha=0.5,
                    # capsize=3,
                    # capthick=1,
                    linewidth=1)
    
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax2.set_ylim((0.5, 2.0))
    
    ax2.set_xlabel(r'$m_{{\mathrm{{generic}}}}$ [GeV]')
    ax2.set_ylabel(f'Ratio')
    
    # ax2.grid(axis='y', linestyle='-', linewidth=2, which='both')
    minor_locator = mticker.AutoMinorLocator(2)
    ax2.yaxis.set_minor_locator(minor_locator)

    if blind_region is not None:
        ax1.axvspan(
            blind_region[0], blind_region[1],
            color="gray", alpha=0.3, hatch="//", label="Blinded"
        )
        ax2.axvspan(
            blind_region[0], blind_region[1],
            color="gray", alpha=0.3, hatch="//", label=None
        )
    
    with mpl.rc_context({"text.usetex": False}):
        hep.cms.label(
            "Internal",
            data=True,
            year="2023",
            ax=ax1,
            com="13.6",
            lumi=28
        )
    
    return fig


def save_results(aggregate_dict, out_file="/eos/user/z/zima/DATA_SPECTRA/saved_aggregate_dict.pkl"):
    """Save histograms and summary to files"""
    
    with open(out_file, 'wb') as f:
        pickle.dump(aggregate_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(in_file):
    with open(in_file, 'rb') as f:
        aggregate_dict = pickle.load(f)
    return aggregate_dict

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


def main():
    if REPROCESS or not os.path.isfile(SAVE_TO_FILE):
        data_files = get_DATA_files()
        print(f"Found {len(data_files)} DATA files")
    
    txbb_regions = [
        (0.3, 1.0),
        (0.6, 1.0),
        #(0.7, 1.0),
        # (0.8, 1.0),
        (0.9, 1.0),
        #(0.92, 1.0),
        (0.94, 1.0),
        # (0.95, 1.0),
        (0.96, 1.0),
        # (0.97, 1.0),
        (0.98, 1.0),
        #(0.99, 1.0)
    ]
    
    pt_regions = [
        (300, 450),
        (450, 550), 
        (550, 10000)
    ]
    
    if REPROCESS or not os.path.isfile(SAVE_TO_FILE):
        print("Aggregating DATA events by region...")
        aggregate_dict = aggregate_DATA_TXbb_regions(
            txbb_regions=txbb_regions,
            pt_regions=pt_regions,
            data_filelist=data_files,
            mass_range=(20, 200),
            mass_bin_width=5.0,
            mass_variable = details["mass_var"]
        )
        save_results(aggregate_dict)
    else:
        print(f"Loading aggregated dictionary from {SAVE_TO_FILE}")
        aggregate_dict = load_results(SAVE_TO_FILE)

    
    print("Region Summary:")
    for key, region_data in aggregate_dict.items():
        print(f"{key}: {region_data['total_events']} events")
    
    print("Creating plots...")
    mass_variable_name = details["mass_var"]
    for pt_region in pt_regions:
        fig = compare_shapes_in_pt_region(
            aggregate_dict, 
            reference_key = f"TXbb = [{txbb_regions[0][0]}, {txbb_regions[0][1]}], pT = [{pt_region[0]}, {pt_region[1]}]",
            pt_region = pt_region
            )
        fig.savefig(f"{RESULTS_DIR}/pT{pt_region[0]}to{pt_region[1]}_{mass_variable_name}.pdf")

if __name__ == "__main__":
    main()
