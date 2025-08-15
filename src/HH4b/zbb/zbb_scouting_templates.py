from __future__ import annotations

import argparse
import itertools
from collections import OrderedDict
from pathlib import Path
import correctionlib

import numpy as np
import pandas as pd

from HH4b import postprocessing, utils

YEARS = ["2023", "2023BPix"]
YEARS_COMBINED_DICT = {
    "2023All": ["2023", "2023BPix"]
    }

TAG = "13Aug2025_ScoutingDataQCDWto2QZto2Q_Full2023_wDST_v15_scouting_zbb"
DATA_DIR = f"/eos/user/e/eheikkil/bbbb/skimmer/{TAG}"
OUTDIR = f"/eos/user/e/eheikkil/scouting/templates/{TAG}"
PERSISTENT_PATH = Path(f"/eos/user/e/eheikkil/scouting/templates/{TAG}/scouting_events.pkl")
PERSISTENT_PATH.parent.mkdir(parents=True, exist_ok=True)

TXBB_BINS = [0.95, 0.975, 0.99, 1.0]
PT_BINS = [200, 450, 550, 10000]
MASS_BIN_SIZE = 5
M_LOW = 50.0
M_HIGH = 150.0

SAMPLES_DICT = {
    "data": ["Data"], # It is important that this d in the "data" key be lower case
    "Zto2Q": ["Zto2Q-2Jets"],
    "Wto2Q": ["Wto2Q-2Jets"],
    "qcd": ["QCD"] # Same with lowercase q here
}
MC_SAMPLES_LIST = [k for k in SAMPLES_DICT if k != "data"]

APPLY_Zto2Q_CORR = False # TIODO: Can't use right now because 2023BPix is implemented correctly but 2023 might not work? Need loop over years and clarification with combined events dict

def categorize_zto2q(events_dict):
    for year in events_dict:
        if "Zto2Q" not in events_dict[year]:
            continue
        df = events_dict[year]["Zto2Q"]
        matched = df[("bbFatJetVQQMatch", 0)] == 1
        is_ZBB = df[("GenZBB", 0)]
        is_ZCC = df[("GenZCC", 0)]
        is_ZQQ = ~(is_ZBB | is_ZCC)

        events_dict[year]["Zto2Q_BB"] = df[is_ZBB & matched]
        events_dict[year]["Zto2Q_CC"] = df[is_ZCC & matched]
        events_dict[year]["Zto2Q_QQ"] = df[is_ZQQ & matched]
        events_dict[year]["Zto2Q_unmatched"] = df[~matched]

def create_templates(events_dict):
    pt_branch = "bbFatJetPt0"
    mass_branch = "bbFatJetScoutParTmassGeneric0"
    txbb_branch = "bbFatJetScoutParTTXbb0"

    selection_regions = {}
    for (txbb_low, txbb_high), (pt_low, pt_high) in itertools.product(
        zip(TXBB_BINS[:-1], TXBB_BINS[1:]), zip(PT_BINS[:-1], PT_BINS[1:])
    ):
        txbb_low_str = str(txbb_low).replace(".", "p")
        txbb_high_str = str(txbb_high).replace(".", "p")
        pt_low_str = str(pt_low)
        pt_high_str = str(pt_high)
        region_key = (
            f"pass_TXbb{txbb_low_str}to{txbb_high_str}_pT{pt_low_str}to{pt_high_str}"
        )
        selection_regions[region_key] = postprocessing.Region(
            cuts={
                pt_branch: [pt_low, pt_high],
                mass_branch: [M_LOW, M_HIGH],
                txbb_branch: [txbb_low, txbb_high]
            },
            label=region_key
        )

    selection_regions["fail"] = postprocessing.Region(
        cuts={
            pt_branch: [PT_BINS[0], PT_BINS[-1]],
            mass_branch: [M_LOW, M_HIGH],
            txbb_branch: [0.1, TXBB_BINS[0]]
        },
        label="fail"
    )

    shape_var = postprocessing.ShapeVar(
        mass_branch,
        r"$m_\mathrm{reg}$ (GeV)",
        [int((M_HIGH - M_LOW) / MASS_BIN_SIZE), M_LOW, M_HIGH],
        reg=True,
    )

    bg_keys = [
        "Zto2Q_BB",
        "Zto2Q_CC",
        "Zto2Q_QQ",
        "Zto2Q_unmatched",
        "Wto2Q",
        "qcd",
    ]
    bg_order = list(reversed(bg_keys))

    for year in events_dict:
        bg_keys_filtered = [k for k in bg_keys if k in events_dict[year] and not events_dict[year][k].empty] # hot fix

        plot_dir=Path(OUTDIR) / "plots" / year
        plot_dir.mkdir(parents=True, exist_ok=True)

        templates = postprocessing.get_templates(
            events_dict=events_dict[year],
            year=year,
            sig_keys=[],
            selection_regions=selection_regions,
            shape_vars=[shape_var],
            systematics={},
            template_dir=Path(OUTDIR),
            bg_keys=bg_keys_filtered,
            bg_order=bg_order,
            bg_err_mcstat=False,
            plot_dir=plot_dir,
            weight_key="finalWeight",
            weight_shifts=None,
            blind=False,
        )

        # Save to pickle
        out_pkl = Path(OUTDIR) / f"templates_{year}.pkl"
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with out_pkl.open("wb") as f:
            pd.to_pickle(templates, f)
        print(f"Saved templates for {year} to {out_pkl}")

def main():
    events_dict = {}
    common_columns = [
        ("bbFatJetPt", 2),
        ("bbFatJetEta", 2),
        ("bbFatJetMsd", 2),
        ("bbFatJetMass", 2),
        ("bbFatJetScoutParTmassGeneric", 2),
        ("bbFatJetScoutParTmassCorrectedX2p", 2),
        ("bbFatJetScoutParTmassCorrectedW2p", 2), 
        ("bbFatJetScoutParTTXbb", 2),
        # ("bbFatJetParT3massGeneric", 2), # Comment in / out if using scouting vs. offline variables NOTE: Offline vars wont work with scouting data, only MC where we added noprmal reco to scouting MC as well
        # ("bbFatJetParT3massCorrectedX2p", 2),
        # ("bbFatJetParT3massCorrectedW2p", 2), 
        # ("bbFatJetParT3TXbb", 2),
        ("weight", 1),
    ]

    zto2q_columns = [
        ("GenZPt", 1),
        ("bbFatJetVQQMatch", 2),
        ("GenZBB", 1),
        ("GenZCC", 1),
    ]

    mc_columns = {
        "Zto2Q": zto2q_columns,
        "Wto2Q": [],
        "qcd": []
    }

    if not PERSISTENT_PATH.exists():
        for year in YEARS:
            print(f"Loading samples for {year}...")
            events_dict[year] = {}

            for sample, sample_list in SAMPLES_DICT.items():
                is_data = sample.lower() == "data"

                # Use appropriate columns
                columns = common_columns.copy()
                if not is_data:
                    columns += mc_columns[sample]

                print(f"Loading sample {sample}")

                try:
                    loaded = utils.load_samples(
                        data_dir=DATA_DIR,
                        samples={sample: sample_list},  # only load one sample type at a time
                        year=year,
                        columns=utils.format_columns(columns),
                        variations=False,
                        weight_shifts=None,
                        load_weight_noxsec=True,
                    )
                    
                    if sample not in loaded or loaded[sample].empty:
                        print(f"No data loaded for {sample} in year {year}. Skipping")
                        continue

                    events_dict[year].update(loaded)  # merge into events_dict
                except Exception as e:
                    print(f"Error loading sample {sample} for year {year}: {e}")

        # Combine events from different years into a single dictionary
        print("Combining events from different years...")
        events_combined = {year: {} for year in YEARS_COMBINED_DICT}
        for sample in SAMPLES_DICT:
            for combined_year, year_list in YEARS_COMBINED_DICT.items():
                events_combined[combined_year][sample] = pd.concat(
                    [events_dict[year][sample] for year in year_list if sample in events_dict[year]]
                )

        with PERSISTENT_PATH.open("wb") as f:
            pd.to_pickle(events_combined, f)
        print(f"Saved processed events to {PERSISTENT_PATH}")
    else:
        print(f"Loading pre-processed events from {PERSISTENT_PATH}")
        with PERSISTENT_PATH.open("rb") as f:
            events_combined = pd.read_pickle(f)

    if APPLY_Zto2Q_CORR: # TODO: Scouting analysis needs this redone? TODO: This also needs to be redone when multiple years are used, e.g. 2023 and 2023BPix
        print("Applying Z->2Q corrections from ZMuMu measurement...")
        corr_dir = Path("ZMuMu_corrs")
        corr_dict = {}

        # removed loop over years
        corr_file = corr_dir / f"corr_2023.json"
        if not corr_file.exists():
            raise FileNotFoundError(f"Correction file {corr_file} does not exist.")

        # Load the correction
        corr = correctionlib.CorrectionSet.from_file(str(corr_file))
        corr_dict["2023BPix"] = corr
        print(f"Loaded correction for 2023BPix from {corr_file}")

        print("Applying Zto2Q corrections...")

        corr = corr_dict["2023BPix"]["GenZPtWeight"]
        GenZ_pt = events_dict["2023BPix"]["Zto2Q"]["GenZPt"].to_numpy()[:, 0] # uhhhh this should be events dict or combined dict?
        sf_nom = corr.evaluate(GenZ_pt, "nominal")
        events_dict["2023BPix"]["Zto2Q"]["SF_GenZPt"] = sf_nom

        # apply the scale factors to the final weight
        weight = events_dict["2023BPix"]["Zto2Q"]["finalWeight"]
        events_dict["2023BPix"]["Zto2Q"]["finalWeight"] = weight * sf_nom

        print("Zto2Q corrections applied")
    else:
        corr_dict = None
        print("Z->2Q corrections are not applied.")
    


    categorize_zto2q(events_combined)
    create_templates(events_combined)

if __name__ == "__main__":
    main()
