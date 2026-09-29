from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import uproot
import numpy as np
import pickle

from tqdm import tqdm
import mplhep as hep
import matplotlib.ticker as mticker
from HH4b.zbb import nanotrees_loader

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

mpl.rcParams.update({
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
})

# Input ntuples: "nanotrees" reads the VH NanoTrees "had" trees in NANOTREES_DIR,
# "scouting" / "offline" the bbbbSkimmer outputs in SKIMMER_DIR
INPUT = "nanotrees"

offline_details = {
    "model": "GloParT-v3",
    #"mass_var": "bbFatJetParT3massCorrectedX2p0",
    #"mass_var": "bbFatJetMsd0", # This will be a good check (granted QCD is trained to target M_SD so not sure how good, might be better for data check)
    "mass_var": "bbFatJetParT3massGeneric0",
    "txbb_var": "bbFatJetParT3TXbb0",
    "mass_label": r"$m_{\mathrm{generic}}$ [GeV]",
    "year": "2023",
    "filename_suffix": "offline"
}

scouting_details = {
    "model": "Scouting GloParT (22-23)",
    #"mass_var": "bbFatJetScoutParTmassCorrectedX2p0",
    "mass_var": "bbFatJetMsd0",
    #"mass_var": "bbFatJetScoutParTmassGeneric0",
    "txbb_var": "bbFatJetScoutParTTXbb0",
    "mass_label": r"$m_{\mathrm{SD}}$ [GeV]",
    "year": "2023",
    "filename_suffix": "scouting"
}

# nanotrees_loader columns of the highest-TXbb jet: ak8_ParT_resonanceMass and ak8_ParT_HbbVsQCD
nanotrees_details = {
    "model": "Scouting GloParT (24)",
    "mass_var": "bbFatJetScoutParTmassCorrResonance",
    "txbb_var": "bbFatJetScoutParTTXbb",
    "mass_label": r"$m_{\mathrm{res}}$ [GeV]",
    "year": "2024",
    "filename_suffix": "nanotrees"
}

details = {"offline": offline_details, "scouting": scouting_details, "nanotrees": nanotrees_details}[INPUT]

filename_suffix = details["filename_suffix"]
mass_variable = details["mass_var"]

SKIMMER_DIR = "/eos/user/z/zima/bbbb/skimmer"
YEARS = ["2023BPix", "2023"]
TAG = "13Feb2026_Data_Standard_Cuts_QCD_Inclusive_v15_scouting_zbb" if INPUT == "scouting" else "06Feb2025_QCD_Offline_Inclusive_PTl_300_etaS_HT_1k_v15_scouting_zbb"
YEAR_DIR = [f"{SKIMMER_DIR}/{TAG}/{YEAR}" for YEAR in YEARS]

#NANOTREES_DIR = "/eos/user/b/bribeiro/HadronicVH/treesScouting_Jun2_2024_had"
#NANOTREES_DIR = "/eos/user/z/zima/private_production/vhcc/test_2024_had"
NANOTREES_DIR = "/eos/user/z/zima/HadronicVH/tagger_Cali_2024_had"
#NANOTREES_DIR = "/eos/user/b/bribeiro/HadronicVH/treesScouting_Sep24_2024_had/"
# QCD files relative to NANOTREES_DIR (None: nanotrees_loader.SAMPLE_FILES, the merged mc/qcd*_tree.root).
# The merge of this production stopped at qcd1 (HT 100-800), so read the HT-binned parts, here slimmed by
# skim_nanotrees.py to the branches nanotrees_loader reads (all events kept). The originals are in mc/parts
NANOTREES_QCD_FILES = "mc/skim/qcd*_tree.root"
# Lowest TXbb region, also replacing the TXbb > 0.3 cut of nanotrees_loader.DEFAULT_SELECTION (the scouting Zbb
# selection of the templates). Reprocess after changing.
# tagger_Cali_2024_had has no tagger preselection, so TXbb > 0.0 is inclusive. Productions run with
# tagger_threshold_bb/cc = 0.3 (e.g. treesScouting_Jun2_2024_had) only keep events with an AK8 jet passing
# HbbVsQCD > 0.3 or HccVsQCD > 0.3, so there below TXbb = 0.3 is only Hcc-tagged QCD
NANOTREES_MIN_TXBB = 0.0
# common selection of the VH had analysis (vh_had.py), i.e. no pT/HT cut but mass windows on both jets
NANOTREES_SELECTION = {**nanotrees_loader.VH_HAD_SELECTION, "txbb_lead": NANOTREES_MIN_TXBB}

REPROCESS = False

SAVE_TO = "/eos/user/z/zima/QCD_SPECTRA"
SAVE_TO_FILE = f"{SAVE_TO}/saved_aggregate_dict_{filename_suffix}_{mass_variable}.pkl"
RESULTS_DIR = f"{SAVE_TO}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_QCD_files(year_dirs: list[str] = YEAR_DIR) -> list[str]:
    qcd_filelist = []
    for year_dir in year_dirs:
        for root, dirs, files in os.walk(year_dir):
            for file in files:
                f = os.path.join(root, file)
                if "QCD" in f and f.endswith(".root"):
                    qcd_filelist.append(f)
    print(f"Length of QCD filelist: {len(qcd_filelist)}")
    return qcd_filelist


def iterate_QCD(qcd_filelist: list = None, mass_variable: str = mass_variable):
    """Yields (TXbb, pT, mass, weight) of the highest-TXbb jet in QCD, per skimmer file or NanoTrees chunk"""
    if INPUT == "nanotrees":
        chunks = nanotrees_loader.iterate_nanotrees(
            NANOTREES_DIR, "QCD", selection=NANOTREES_SELECTION, file_pattern=NANOTREES_QCD_FILES
        )
        for _, chunk in tqdm(chunks, desc="Aggregating QCD TXbb regions", unit=" chunks"):
            if chunk is None:
                continue
            yield (
                chunk[details["txbb_var"]][:, 0],
                chunk["bbFatJetPt"][:, 0],
                chunk[mass_variable][:, 0],
                chunk["weight"][:, 0],
            )
        return

    if qcd_filelist is None:
        qcd_filelist = get_QCD_files()

    print(f"Processing {len(qcd_filelist)} QCD files...")

    for file in tqdm(qcd_filelist, desc="Aggregating QCD TXbb regions"):
        with uproot.open(file) as f:
            if "Events" not in f:
                continue

            events = f["Events"]
            yield (
                events[details["txbb_var"]].array(library="np"),
                events["bbFatJetPt0"].array(library="np"),
                events[mass_variable].array(library="np"),
                events["weight"].array(library="np"),
            )


def aggregate_QCD_TXbb_regions(
    txbb_regions: list = None,
    pt_regions: list = None,
    qcd_filelist: list = None,
    mass_variable: str = "bbFatJetScoutParTmassCorrectedX2p0",
    mass_range: tuple = (20, 200),
    mass_bin_width: float = 12
):
    if txbb_regions is None:
        raise Exception("TXbb regions not defined")
    if pt_regions is None:
        raise Exception("pT regions not defined")

    pt_bins = np.array([pt[0] for pt in pt_regions] + [pt_regions[-1][1]])
    txbb_bins = np.linspace(0.0, 1.0, 101)  # 100 bins, did 400 in other
    n_mass_bins = int((mass_range[1] - mass_range[0]) / mass_bin_width)
    mass_bins = np.linspace(mass_range[0], mass_range[1], n_mass_bins + 1)

    hist = np.zeros((len(pt_bins)-1, len(txbb_bins)-1, len(mass_bins)-1))
    hist_w2 = np.zeros_like(hist)

    for txbb_values, pt_values, mass_values, weight_values in iterate_QCD(qcd_filelist, mass_variable):
        mask = np.isfinite(txbb_values) & np.isfinite(pt_values) & np.isfinite(mass_values)
        txbb_values = txbb_values[mask]
        pt_values = pt_values[mask]
        mass_values = mass_values[mask]
        weight_values = weight_values[mask]

        i_pt = np.digitize(pt_values, pt_bins) - 1
        j_txbb = np.digitize(txbb_values, txbb_bins) - 1
        k_mass = np.digitize(mass_values, mass_bins) - 1

        valid = (i_pt >= 0) & (i_pt < len(pt_bins)-1) & \
                (j_txbb >= 0) & (j_txbb < len(txbb_bins)-1) & \
                (k_mass >= 0) & (k_mass < len(mass_bins)-1)

        i_pt_v = i_pt[valid]
        j_txbb_v = j_txbb[valid]
        k_mass_v = k_mass[valid]
        w_v = weight_values[valid]

        np.add.at(hist, (i_pt_v, j_txbb_v, k_mass_v), w_v)
        np.add.at(hist_w2, (i_pt_v, j_txbb_v, k_mass_v), w_v**2)

    return {
        "hist": hist,
        "hist_w2": hist_w2,
        "pt_bins": pt_bins,
        "txbb_bins": txbb_bins,
        "mass_bins": mass_bins,
        "pt_regions": pt_regions,
        "txbb_regions": txbb_regions
    }


def build_region_dict_from_3D(hist_dict):
    hist = hist_dict["hist"]
    hist_w2 = hist_dict["hist_w2"]
    pt_bins = hist_dict["pt_bins"]
    txbb_bins = hist_dict["txbb_bins"]
    mass_bins = hist_dict["mass_bins"]
    pt_regions = hist_dict["pt_regions"]
    txbb_regions = hist_dict["txbb_regions"]

    out = {}

    for pt_min, pt_max in pt_regions:
        ipt_min = np.searchsorted(pt_bins, pt_min, side="left")
        ipt_max = np.searchsorted(pt_bins, pt_max, side="right")

        for txbb_min, txbb_max in txbb_regions:
            itxbb_min = np.searchsorted(txbb_bins, txbb_min, side="left")
            itxbb_max = np.searchsorted(txbb_bins, txbb_max, side="right")

            key = f"TXbb = [{txbb_min}, {txbb_max}], pT = [{pt_min}, {pt_max}]"

            h = hist[ipt_min:ipt_max, itxbb_min:itxbb_max, :].sum(axis=(0, 1))
            h2 = hist_w2[ipt_min:ipt_max, itxbb_min:itxbb_max, :].sum(axis=(0, 1))

            out[key] = {
                "hist": h,
                "hist_w2": h2,
                "bins": mass_bins,
                "txbb_region": (txbb_min, txbb_max),
                "pt_region": (pt_min, pt_max),
                "total_weight": float(h.sum()),
                "total_w2": float(h2.sum())
            }

    return out


def calculate_weighted_histogram_errors(hist_w2, total_weight):
    unnormalized_errors = np.sqrt(hist_w2)
    normalized_errors = unnormalized_errors / total_weight
    return normalized_errors


def normalize_histograms(aggregate_dict):
    normalized_dict = {}
    for key, region_data in aggregate_dict.items():
        hist = region_data['hist'].copy()
        hist_w2 = region_data['hist_w2'].copy()
        total_weight = region_data['total_weight']

        if total_weight <= 0:
            hist_normalized = np.zeros_like(hist)
            hist_errors = np.zeros_like(hist)
        else:
            hist_normalized = hist / total_weight
            hist_errors = calculate_weighted_histogram_errors(hist_w2, total_weight)

        normalized_dict[key] = {
            'hist': hist_normalized,
            'hist_errors': hist_errors,
            'bins': region_data['bins'],
            'txbb_region': region_data['txbb_region'],
            'pt_region': region_data['pt_region'],
            'total_weight': total_weight,
            'total_w2': region_data['total_w2']
        }

    return normalized_dict


def calculate_ratio_with_errors(hist_A, err_A, hist_B, err_B):
    ratio = np.ones_like(hist_A)
    ratio_err = np.zeros_like(hist_A)

    valid_mask = (hist_B > 0) & (err_B >= 0) & (err_A >= 0)
    if not np.any(valid_mask):
        return ratio, ratio_err

    ratio[valid_mask] = hist_A[valid_mask] / hist_B[valid_mask]

    rel_err_A = np.zeros_like(err_A)
    rel_err_B = np.zeros_like(err_B)

    a_nonzero = hist_A > 0
    b_nonzero = hist_B > 0

    rel_err_A[a_nonzero] = err_A[a_nonzero] / hist_A[a_nonzero]
    rel_err_B[b_nonzero] = err_B[b_nonzero] / hist_B[b_nonzero]

    rel_err_combined = np.sqrt(rel_err_A**2 + rel_err_B**2)
    ratio_err = np.abs(ratio) * rel_err_combined

    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    ratio_err = np.nan_to_num(ratio_err, nan=0.0, posinf=0.0, neginf=0.0)

    return ratio, ratio_err


def compare_shapes_in_pt_region(aggregate_dict, reference_key, pt_region=(300, 450)):
    # plt.rcParams.update({
    #     "font.size": 30,
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "text.latex.preamble": r"\usepackage{amsmath}"
    # })

    normalized_dict = normalize_histograms(aggregate_dict)
    ref_data = normalized_dict[reference_key]
    ref_hist = ref_data['hist']
    ref_errors = ref_data['hist_errors']

    fig, axes = plt.subplots(2, 1, figsize=(12, 12),
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
                           sharex=True)
    ax1, ax2 = axes

    model = details["model"]
    plot_text_lines = [
        "QCD Multijet",
        f"Model: {model}",
        rf"$p_T \in ({pt_region[0]}, {pt_region[1]}), |\eta| < 2.4$",
        r"$T_{{\mathrm{{Xbb}}}} = \frac{P(X \to b\bar{b})}{P(X \to b\bar{b}) + P(QCD)}$"
    ]

    # text_x_position = 0.25
    # text_y_start = 0.85
    text_x_position = 0.2
    text_y_start = 0.83

    for i, line in enumerate(plot_text_lines):
        fig.text(text_x_position, text_y_start - i*0.04, line,
                 fontsize=20, #fontweight='bold',
                 verticalalignment='top',
                 horizontalalignment='left')

    pt_keys = [
        k for k in aggregate_dict.keys()
        if f"pT = [{pt_region[0]}, {pt_region[1]}]" in k
    ]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(pt_keys)))    

    # for color_idx, (key, region_data) in enumerate(normalized_dict.items()):
    #     if f"pT = [{pt_region[0]}, {pt_region[1]}]" not in key:
    #         continue

    #     bins = region_data['bins']
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     hist = region_data['hist']
    #     errors = region_data['hist_errors']

    #     txbb_min = region_data['txbb_region'][0]
    #     label = fr"$T_{{\mathrm{{Xbb}}}} > {txbb_min}$"

    #     ax1.step(bin_centers, hist, where='mid',
    #              color=colors[color_idx], linewidth=2, label=label)
    #     ax1.errorbar(bin_centers, hist, yerr=errors,
    #                  fmt='none', color=colors[color_idx], linewidth=1)
    for color_idx, key in enumerate(pt_keys):
        region_data = normalized_dict[key]

        bins = region_data['bins']
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist = region_data['hist']
        errors = region_data['hist_errors']

        txbb_min = region_data['txbb_region'][0]
        label = fr"$T_{{\mathrm{{Xbb}}}} > {txbb_min}$" if txbb_min != 0.0 else fr"$Inclusive$"

        ax1.step(
            bin_centers, hist,
            where='mid',
            color=colors[color_idx],
            linewidth=3,
            label=label
        )
        ax1.errorbar(
            bin_centers, hist,
            yerr=errors,
            fmt='none',
            color=colors[color_idx],
            linewidth=3
        )


    ax1.set_ylabel('A.U.')
    #ax1.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='upper right')

    # for color_idx, (key, region_data) in enumerate(normalized_dict.items()):
    #     if key == reference_key or f"pT = [{pt_region[0]}, {pt_region[1]}]" not in key:
    #         continue

    #     bins = region_data['bins']
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     hist = region_data['hist']
    #     errors = region_data['hist_errors']

    #     ratio, ratio_errors = calculate_ratio_with_errors(
    #         hist, errors, ref_hist, ref_errors
    #     )

    #     ax2.step(bin_centers, ratio, where='mid',
    #              color=colors[color_idx], linewidth=2)
    #     ax2.errorbar(bin_centers, ratio, yerr=ratio_errors,
    #                  fmt='none', color=colors[color_idx], linewidth=1)

    for color_idx, key in enumerate(pt_keys):
        if key == reference_key:
            continue

        region_data = normalized_dict[key]

        bins = region_data['bins']
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist = region_data['hist']
        errors = region_data['hist_errors']

        ratio, ratio_errors = calculate_ratio_with_errors(
            hist, errors, ref_hist, ref_errors
        )

        ax2.step(
            bin_centers, ratio,
            where='mid',
            color=colors[color_idx],
            linewidth=3
        )
        ax2.errorbar(
            bin_centers, ratio,
            yerr=ratio_errors,
            fmt='none',
            color=colors[color_idx],
            linewidth=3
        )

    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2)
    ax2.set_ylim(1/3.0, 3.0)
    ax2.set_xlabel(details["mass_label"])
    ax2.set_ylabel('Ratio')
    #ax2.grid(axis='y', linestyle='-', linewidth=1, which='both')

    hep.cms.label(data=False, year=details["year"], ax=ax1, com="13.6")

    return fig

def extract_inclusive_TXbb(hist_dict, normalize=True):
    """
    Integrate over all pT and mass bins and return TXbb histogram.
    """
    hist = hist_dict["hist"]
    hist_w2 = hist_dict["hist_w2"]
    txbb_bins = hist_dict["txbb_bins"]

    txbb_hist = hist.sum(axis=(0, 2))
    txbb_w2 = hist_w2.sum(axis=(0, 2))

    if normalize and txbb_hist.sum() > 0:
        norm = txbb_hist.sum()
        txbb_hist /= norm
        txbb_err = np.sqrt(txbb_w2) / norm
    else:
        txbb_err = np.sqrt(txbb_w2)

    txbb_centers = 0.5 * (txbb_bins[:-1] + txbb_bins[1:])

    return txbb_centers, txbb_hist, txbb_err

def plot_TXbb_comparison(
    datasets,
    logy=True,
    normalize=True
):
    """datasets: {model label: hist dict of aggregate_QCD_TXbb_regions}"""
    print("Plotting comparisons")
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["tab:blue", "tab:red", "tab:green"]

    for color, (label, hdict) in zip(colors, datasets.items()):
        x, y, yerr = extract_inclusive_TXbb(hdict, normalize=normalize)

        ax.step(
            x, y,
            where="mid",
            linewidth=3,
            color=color,
            label=label
        )
        ax.errorbar(
            x, y,
            yerr=yerr,
            fmt="none",
            linewidth=2,
            color=color
        )

    ax.set_xlabel(r"$T_{\mathrm{Xbb}}$")
    ax.set_ylabel("A.U." if normalize else "Events")
    ax.set_xlim(0.0, 1.0)

    plot_text_lines = [
        "QCD Multijet",
        # f"Model: {model}",
        rf"$p_T > 300\, GeV, |\eta| < 2.4$",
        r"$T_{{\mathrm{{Xbb}}}} = \frac{P(X \to b\bar{b})}{P(X \to b\bar{b}) + P(QCD)}$"
    ]

    text_x_position = 0.2
    text_y_start = 0.83

    for i, line in enumerate(plot_text_lines):
        fig.text(text_x_position, text_y_start - i*0.06, line,
                 fontsize=20, #fontweight='bold',
                 verticalalignment='top',
                 horizontalalignment='left')

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs="auto"))

    ax.legend(fontsize = 20)

    hep.cms.label(
        data=False,
        year=details["year"],
        ax=ax,
        com="13.6"
    )

    return fig





def save_results(aggregate_dict, out_file=SAVE_TO_FILE):
    with open(out_file, 'wb') as f:
        pickle.dump(aggregate_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(in_file):
    with open(in_file, 'rb') as f:
        aggregate_dict = pickle.load(f)
    return aggregate_dict


def main():
    txbb_regions = [
        (0.0, 1.0),
        (0.3, 1.0),
        # (0.4, 1.0),
        (0.6, 1.0),
        # (0.8, 1.0),
        (0.9, 1.0),
        # (0.92, 1.0),
        (0.94, 1.0),
        (0.96, 1.0),
        (0.98, 1.0),
        #(0.99, 1.0),
    ]
    if INPUT == "nanotrees":
        txbb_regions = [region for region in txbb_regions if region[0] >= NANOTREES_MIN_TXBB]

    '''
    pt_regions = [
        (300, 450),
        (450, 550),
        (550, 10000),
        # (300, 10000),
        # (450, 10000)
    ]
    '''
    pt_regions = [
        (100, 500),
        (500,1300 ), 
        #(550, 10000)
    ]

    # inputs of the saved hists: a saved file made with other ones is reprocessed instead of silently reused
    if INPUT == "nanotrees":
        inputs = {"dir": NANOTREES_DIR, "files": NANOTREES_QCD_FILES, "selection": NANOTREES_SELECTION}
    else:
        inputs = {"dirs": YEAR_DIR}
    config = {"inputs": inputs, "mass_variable": mass_variable, "pt_regions": pt_regions,
              "mass_range": (20, 200), "mass_bin_width": 12.0}

    hist_dict = None
    if not REPROCESS and os.path.isfile(SAVE_TO_FILE):
        hist_dict = load_results(SAVE_TO_FILE)
        if hist_dict.get("config") != config:
            print(f"{SAVE_TO_FILE} was made with {hist_dict.get('config')}, reprocessing for {config}")
            hist_dict = None

    if hist_dict is None:
        hist_dict = aggregate_QCD_TXbb_regions(
            txbb_regions=txbb_regions,
            pt_regions=pt_regions,
            mass_range=config["mass_range"],
            mass_bin_width=config["mass_bin_width"],
            mass_variable=details["mass_var"]
        )
        hist_dict["config"] = config
        save_results(hist_dict)
    else:
        # the 3D hist spans the full TXbb range, so the TXbb regions can change without reprocessing (unlike the pT ones)
        hist_dict["txbb_regions"] = txbb_regions

    aggregate_dict = build_region_dict_from_3D(hist_dict)

    if INPUT == "scouting":
        datasets = {
            scouting_details["model"]: hist_dict,
            offline_details["model"]: load_results(f"{SAVE_TO}/saved_aggregate_dict_offline.pkl"),
        }
        txbb_plot_name = "comparison"
    elif INPUT == "offline":
        datasets = {
            scouting_details["model"]: load_results(f"{SAVE_TO}/saved_aggregate_dict_scouting_{mass_variable}.pkl"),
            offline_details["model"]: hist_dict,
        }
        txbb_plot_name = "comparison"
    else:
        datasets = {details["model"]: hist_dict}
        txbb_plot_name = filename_suffix
    fig = plot_TXbb_comparison(datasets)
    fig.savefig(f"{RESULTS_DIR}/TXbb_full_distribution_{txbb_plot_name}.pdf")

    print("Region Summary:")
    for key, region_data in aggregate_dict.items():
        print(f"{key}: {region_data['total_weight']} events")

    print("Creating plots...")
    for pt_region in pt_regions:
        fig = compare_shapes_in_pt_region(
            aggregate_dict,
            reference_key=f"TXbb = [{txbb_regions[0][0]}, {txbb_regions[0][1]}], pT = [{pt_region[0]}, {pt_region[1]}]",
            pt_region=pt_region
        )
        fig.savefig(f"{RESULTS_DIR}/pT{pt_region[0]}to{pt_region[1]}_{filename_suffix}_{mass_variable}.pdf")


if __name__ == "__main__":
    main()
