import pandas as pd
import numpy as np

from pathlib import Path

from HH4b import utils
from HH4b import postprocessing
from HH4b.hh_vars import LUMI
import itertools
import correctionlib
from collections import OrderedDict
import uproot
import os
import shutil
import tempfile

tag = "02Nov2026_2024_v15_scouting_zbb" # 2024 Zto2Q, original selection (jet veto applied); data from DATA_TAG #"06Nov2026_Zto2Q_jetvetocol_v15_scouting_zbb" # 2023 + 2024 Zto2Q and 2024 QCD HT 800-1500, jet veto saved as a column #"05Nov2026_Zto2Q_lepclean_v15_scouting_zbb" (2023 + 2024 Zto2Q re-skim with bbFatJetLepClean) #"5Nov2026_Scouting_zbb_2023_2024" (symlinks to zqq_2023_full (2023) and 02Nov2026_2024_v15_scouting_zbb (2024), Zto2Q only) #"5Nov2026_Scouting_zbb_2024" #"12Feb2026_Scouting_Fixed_v15_scouting_zbb" #"07Feb2026_PTl300TXbb0p3_PTsl300_HT600_v15_scouting_zbb" #"11Dec2025_v15_scouting_zbb"

# Pass and fail regions
txbb_bins = [0.75, 0.93, 1.0]
min_txbb = txbb_bins[0]
# pT bins
pt_bins = [350, 450, 550, 1800]

txbb_bins = list(zip(txbb_bins[:-1], txbb_bins[1:]))
pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

# Mass window of the VH NanoTrees producer (vhTreeProducer mass_range), emulated on both bbFatJets: a jet passes
# if its msd or its m_res is in [low, high). None: off
PRODUCER_MASS_RANGE = None
# lepton-jet cleaning of the VH NanoTrees producer on both bbFatJets (bbFatJetLepClean of bbbbSkimmer: no lepton with
# pT > 20 within dR < 0.8). Needs skims with that column (and REPROCESS = True for the event caches made before it)
PRODUCER_LEP_CLEANING: bool = False
# AK4 jet veto map, the first cut of every region (the ak4_jetveto column of bbbbSkimmer: the zbb skims made after
# 6Nov2026 save it instead of cutting on it). False for skims without the column (which already have the veto applied)
APPLY_JETVETO: bool = False
JETVETO_CUT = {"ak4_jetveto": [0.5, 1.5]} if APPLY_JETVETO else {}

# Mass bins
# probe-jet mass of the regions and templates: "bbFatJetScoutParTmassCorrectedX2p" (m_X2p, 40-180) or
# "bbFatJetScoutParTmassCorrResonance" (m_res, 30-200, as the NanoTrees templates)
MASS_VAR = "bbFatJetScoutParTmassCorrectedX2p"
m_low, m_high = (30, 200) if MASS_VAR == "bbFatJetScoutParTmassCorrResonance" else (45, 180)
bin_width = 5
n_mass_bins = int((m_high - m_low) / bin_width)

STORAGE_PROJ_DIR = Path("/eos/user/z/zima/bbbb/")
DATA_DIR = STORAGE_PROJ_DIR / f"skimmer/{tag}"
# data from a separate skim (same skimmer): the full Run2024C and Run2024E
DATA_TAG = "06Nov2026_2024data_v15_scouting_zbb"
SAMPLE_DATA_DIR = {  # skim dir per sample (default DATA_DIR)
    "data": STORAGE_PROJ_DIR / f"skimmer/{DATA_TAG}",
    "QCD": STORAGE_PROJ_DIR / "skimmer/10Nov2026_2024QCD_v15_scouting_zbb",  # all 2024 QCD-4Jets HT bins
}
PROCESSED_DIR = STORAGE_PROJ_DIR / f"scouting/templates/{DATA_TAG}_QCDMC"
# PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

REPROCESS: bool = False  # if True, reprocess from the skimmed ntuples
APPLY_Z_RECOIL_CORR: bool = False # if True, apply the Z->2Q recoil corrections derived from ZMuMu measurement
DO_JESR: bool = False # can't
DO_JMSR: bool = True
ZQQ_BB_CUTFLOW: bool = True  # if True, also save the cutflows of Zto2Q split into Z->bb with the probe jet matched to the Z, and the rest
tagger_branch = "bbFatJetScoutParTTXbb"


YEARS = [
    # "2022",
    # "2022EE",
    #  "2023",
    # "2023BPix",  # no 2023BPix Zto2Q skim
    "2024"
    ]
YEARS_COMBINED_DICT = {
    # "2022All": [
    #         "2022",
    #         "2022EE"
    #         ],
    #  "2023All": [
    #          "2023",
    #         "2023BPix"
    #          ],
    # "2023": ["2023"],
    "2024": ["2024"]
}
# MC normalised to the lumi of the data runs used (None: full year). Applied to all the MC weights after loading,
# so that the templates (and the MC subtraction of qcd_estimation.py) are at the data lumi
MC_LUMI_SCALE = (LUMI["2024C"] + LUMI["2024E"]) / LUMI["2024"]
RUNS = {
    "2023": ["Run2023C"],
    "2023BPix": ["Run2023D"],
    "2024": [
        "Run2024C", 
        "Run2024D",
        "Run2024E",
        "Run2024F",
        "Run2024G",
        "Run2024H",
        "Run2024I"
    ]
}

SAMPLES_DICT = {
    "data": ["Run2024C", "Run2024E"],  # from DATA_TAG
    # "data": [
    #     #2023
    #     "Run2023C", 

    #     # 2023BPix
    #     "Run2023D",

    #     # 2024
    #     "Run2024C", 
    #     "Run2024D",
    #     "Run2024E",
    #     "Run2024F",
    #     "Run2024G",
    #     "Run2024H",
    #     "Run2024I"
    # ],
    # # QCD from data in CR
    # "ttbar": ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"],
    "Zto2Q": ["Zto2Q-4Jets"],
    # "Wto2Q": ["Wto2Q-3Jets"],
    # 2024 QCD HT 800-1500 (= NanoTrees qcd2 + qcd3), for the skimmer vs NanoTrees QCD cutflow comparison
    # "qcd": ["QCD-4Jets"],
    # MC QCD ("QCD"; "qcd" is the data-driven estimate of qcd_estimation.py), scaled by MC_LUMI_SCALE as the other MC
    "QCD": ["QCD-4Jets"],
}
# years each sample is skimmed for (all YEARS if not listed)
SAMPLE_YEARS = {"qcd": ["2024"], "QCD": ["2024"]}
MC_SAMPLES_LIST = [sample for sample in SAMPLES_DICT.keys() if sample != "data"]

# Columns to load from the ntuples
sys_vars = [
    "FSRPartonShower", 
    "ISRPartonShower", 
    "pileup"
    ]

weight_shifts = sys_vars + [
                        "pdf"
                        # "scale_weights"
                        ]

fatjet_vars = [
    "bbFatJetPt",
    "bbFatJetEta",
    "bbFatJetMsd",
]
mass_vars = [
    "bbFatJetMsd",
]
scoutpart_mass_vars = [
    "bbFatJetScoutParTmassCorrectedX2p",
    "bbFatJetScoutParTmassGeneric",
    "bbFatJetScoutParTmassCorrResonance"
]

scoutpart_txbbb_vars = [
    "bbFatJetScoutParTTXbb",
]

fatjet_vars += scoutpart_mass_vars + scoutpart_txbbb_vars
mass_vars += scoutpart_mass_vars

pt_variations = []
if DO_JESR:
    for jesr, ud in itertools.product(["JES", "JER"], ["up", "down"]):
        pt_variations.append(f"bbFatJetPt_{jesr}_{ud}")

mass_variations = []
if DO_JMSR:
    for jmsr, ud in itertools.product(["JMS", "JMR"], ["up", "down"]):
        for var in mass_vars:
            if var == "bbFatJetScoutParTmassCorrResonance":
                continue  # no JMS/JMR variations of it in the skims
            mass_variations.append(f"{var}_{jmsr}_{ud}")


base_columns = [(var, 2) for var in fatjet_vars] + [("weight", 1)]
if PRODUCER_LEP_CLEANING:
    base_columns += [("bbFatJetLepClean", 2)]
if APPLY_JETVETO:
    base_columns += [("ak4_jetveto", 1)]

load_columns_pt_var = []
for pt_var in pt_variations:
    load_columns_pt_var.append((pt_var, 2))

load_columns_mass_var = []
for mass_var in mass_variations:
    load_columns_mass_var.append((mass_var, 2))

load_weight_shifts = []
for var, ud in itertools.product(sys_vars, ["Up", "Down"]):
    load_weight_shifts.append((f"weight_{var}{ud}", 1))

MC_common_extra_columns = load_columns_mass_var + load_columns_pt_var + load_weight_shifts

ZQQ_extra_columns = [("GenZPt", 1), ("GenZBB", 1), ("GenZCC", 1), ("bbFatJetVQQMatch", 2)]
WQQ_extra_columns = [("GenWPt", 1), ("GenWCS", 1), ("GenWUD", 1), ("bbFatJetVQQMatch", 2)]

extra_columns_dict = {
    "data": [],
    # "qcd": load_weight_shifts, # Don't bother loading QCD, just a waste of time since it is estimted from data anyway
    "qcd": load_columns_pt_var + load_weight_shifts,  # the QCD skims have no JMS/JMR mass columns
    "QCD": load_columns_pt_var + load_weight_shifts,
    "ttbar": MC_common_extra_columns,
    # "hbb": MC_common_extra_columns, # Don't have hbb samples
    "Zto2Q": MC_common_extra_columns + ZQQ_extra_columns,
    "Wto2Q": MC_common_extra_columns + WQQ_extra_columns,
}

# if True, apply the Z->2Q corrections from ZMuMu measurement
if APPLY_Z_RECOIL_CORR:
    corr_dir = Path("ZMuMu_corrs")
    corr_dict = {}

    for year in ["2022", "2023", "2024"]:
        corr_file = corr_dir / f"corr_{year}.json"
        if not corr_file.exists():
            raise FileNotFoundError(f"Correction file {corr_file} does not exist.")

        # Load the correction
        corr = correctionlib.CorrectionSet.from_file(str(corr_file))
        corr_dict[year] = corr
        print(f"Loaded correction for {year} from {corr_file}")
else:
    corr_dict = None
    print("Z->2Q corrections are not applied.")


def get_era_path(year):
    return PROCESSED_DIR / f"Zbb_events_{year}.pkl"

def get_combined_path(combined_year):
    return PROCESSED_DIR / f"Zbb_events_{combined_year}.pkl"


# Check if all combined years exist
all_combined_exist = all(
    get_combined_path(combined_year).exists() for combined_year in YEARS_COMBINED_DICT.keys()
)

if not REPROCESS and all_combined_exist:
    # Load all combined years directly
    print("Loading all combined years...")
    events_combined = {}
    for combined_year in YEARS_COMBINED_DICT.keys():
        combined_path = get_combined_path(combined_year)
        print(f"Loading combined year {combined_year}...")
        with combined_path.open("rb") as f:
            events_combined[combined_year] = pd.read_pickle(f)
    print("All combined years loaded!")

else:
    # ============================================================================
    # STEP 1: Process each era individually
    # ============================================================================

    for year in YEARS:
        era_path = get_era_path(year)

        if REPROCESS or not era_path.exists():
            print(f"Processing era: {year}")
            events_era = {}

            # Process each sample for this era
            for sample, sample_list in SAMPLES_DICT.items():
                if year not in SAMPLE_YEARS.get(sample, [year]):
                    continue
                print(f"Loading {sample} for {year}...")

                columns = base_columns + extra_columns_dict.get(sample, [])
                dataframes = {
                    **utils.load_samples(
                        data_dir=str(SAMPLE_DATA_DIR.get(sample, DATA_DIR)),
                        samples={sample: sample_list},
                        year=year,
                        columns=utils.format_columns(columns),
                        variations=True,
                        weight_shifts=weight_shifts,
                    )
                }

                # Process and concatenate dataframes for this sample
                sample_dfs = []
                for key, df in dataframes.items():
                    # Handle pT variations
                    for pt_var in ["bbFatJetPt"] + pt_variations:
                        if pt_var not in df.columns:
                            for i in range(2):
                                df[f"{pt_var}{i}"] = df[("bbFatJetPt", i)].copy()

                    # Handle mass variations
                    for mass_var in mass_vars + mass_variations:
                        if mass_var not in df.columns:
                            for i in range(2):
                                df[f"{mass_var}{i}"] = df[(mass_var.split("_")[0], i)].copy()

                    sample_dfs.append(df)

                # Concatenate all dataframes for this sample
                events_era[sample] = pd.concat(sample_dfs, ignore_index=True)
                print(f"  {sample}: {len(events_era[sample])} events")

                # Clear intermediate dataframes to free memory
                del dataframes, sample_dfs

            # Save this era's data
            with era_path.open("wb") as f:
                pd.to_pickle(events_era, f)
            print(f"Era {year} saved to {era_path}")

            # Clear era data to free memory
            del events_era
        else:
            print(f"Era {year} already processed at {era_path}")

    print("Individual era processing complete!")

    # ============================================================================
    # STEP 2: Combine eras into combined years
    # ============================================================================

    events_combined = {}

    for combined_year, year_list in YEARS_COMBINED_DICT.items():
        combined_path = get_combined_path(combined_year)

        if REPROCESS or not combined_path.exists():
            print(f"\nCombining eras for {combined_year}: {year_list}")

            # Load each era
            era_data = {}
            for year in year_list:
                era_path = get_era_path(year)
                if era_path.exists():
                    print(f"Loading era {year}...")
                    with era_path.open("rb") as f:
                        era_data[year] = pd.read_pickle(f)
                else:
                    print(f"Warning: Era file {era_path} not found!")

            # Combine samples across eras
            events_combined[combined_year] = {}
            for sample in SAMPLES_DICT.keys():
                sample_dfs = []
                for year in year_list:
                    if year in era_data and sample in era_data[year]:
                        sample_dfs.append(era_data[year][sample])

                if sample_dfs:
                    events_combined[combined_year][sample] = pd.concat(
                        sample_dfs, ignore_index=True
                    )
                    print(f"  {sample}: {len(events_combined[combined_year][sample])} events")

            # Save combined year
            with combined_path.open("wb") as f:
                pd.to_pickle(events_combined[combined_year], f)
            print(f"Combined year {combined_year} saved to {combined_path}")

            # Clear era data to free memory for next iteration
            del era_data
        else:
            # If combined year already exists, load it
            print(f"Combined year {combined_year} already exists, loading...")
            with combined_path.open("rb") as f:
                events_combined[combined_year] = pd.read_pickle(f)

    print("\nAll processing complete!")


# apply ZQQ corrections if needed
if APPLY_Z_RECOIL_CORR:
    print("Applying Zto2Q corrections...")
    for year in YEARS_COMBINED_DICT.keys():
        # apply corrections to the events
        corr = corr_dict[year.replace("All", "")]["GenZPtWeight"]
        GenZ_pt = events_combined[year]["Zto2Q"]["GenZPt"].values[:, 0]
        sf_nom = corr.evaluate(GenZ_pt, "nominal")
        sf_up = corr.evaluate(GenZ_pt, "stat_up")
        sf_down = corr.evaluate(GenZ_pt, "stat_dn")
        events_combined[year]["Zto2Q"]["SF_GenZPt"] = sf_nom
        events_combined[year]["Zto2Q"]["SF_GenZPt_up"] = sf_up
        events_combined[year]["Zto2Q"]["SF_GenZPt_down"] = sf_down

        # apply the scale factors to the final weight
        weight = events_combined[year]["Zto2Q"]["finalWeight"]
        events_combined[year]["Zto2Q"]["finalWeight"] = weight * sf_nom
        events_combined[year]["Zto2Q"]["weight_GenZPtUp"] = weight * sf_up
        events_combined[year]["Zto2Q"]["weight_GenZPtDown"] = weight * sf_down

        # Applied nominal SF to the other systematic variations
        for sys_var, up_down in itertools.product(sys_vars, ["Up", "Down"]):
            weight_name = f"weight_{sys_var}{up_down}"
            weight = events_combined[year]["Zto2Q"][weight_name].values[:, 0]
            events_combined[year]["Zto2Q"][weight_name] = weight * sf_nom

    print("Zto2Q corrections applied")

if MC_LUMI_SCALE is not None:
    print(f"MC weights scaled by {MC_LUMI_SCALE:.4f} (processed data lumi / full year lumi)")
    for year in YEARS_COMBINED_DICT.keys():
        for sample, df in events_combined[year].items():
            if sample == "data":
                continue
            wcols = [
                c for c in df.columns
                if (c[0] if isinstance(c, tuple) else c).startswith(("weight", "finalWeight", "pdf_weights", "scale_weights"))
            ]
            df[wcols] = df[wcols] * MC_LUMI_SCALE

# further split Zto2Q and Wto2Q events into different categories
for year in YEARS_COMBINED_DICT.keys():
    Zto2Q = events_combined[year]["Zto2Q"]
    matched = Zto2Q[("bbFatJetVQQMatch", 0)] == 1
    is_ZBB = Zto2Q[("GenZBB", 0)]
    is_ZCC = Zto2Q[("GenZCC", 0)]
    is_ZQQ = ~(is_ZBB | is_ZCC)  # u, d, s quarks
    ZtoBB = is_ZBB & matched
    ZtoCC = is_ZCC & matched
    ZtoQQ = is_ZQQ & matched
    Z_unmatched = ~matched
    events_combined[year]["Zto2Q_BB"] = Zto2Q[ZtoBB]
    events_combined[year]["Zto2Q_CC"] = Zto2Q[ZtoCC]
    events_combined[year]["Zto2Q_QQ"] = Zto2Q[ZtoQQ]
    # leave the unmatched to bkg fits
    # events_combined[year]["Zto2Q_unmatched"] = Zto2Q[Z_unmatched]

# MC_SAMPLES_FINAL_LIST = MC_SAMPLES_LIST + ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched"]
MC_SAMPLES_FINAL_LIST = MC_SAMPLES_LIST + ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ"]

def save_to_root(outfile: Path, templates: dict):
    # written on local disk, then copied: uproot writing directly to EOS (FUSE) can corrupt the file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / Path(outfile).name
        with uproot.recreate(str(tmpfile)) as f_out:
            for category in templates.keys():
                hist = templates[category]
                categories, _ = hist.axes
                for sample in list(categories):
                    h = templates[category][{"Sample": sample}]
                    f_out[f"{sample}_{category}"] = h
        shutil.copyfile(tmpfile, f"{outfile}.tmp")
        os.replace(f"{outfile}.tmp", outfile)

def apply_jmsr_smearing_in_templates(
    df: pd.DataFrame,
    mass_branch_base: str,
    jet_index: int = 0,
    seed: int = 42,
    jms_vals=(0.95, 1.0, 1.05), # down, nom, up
    jmr_vals=(1.0, 1.0, 1.1), # nom, down, up, here there really isnt a down variation
):
    """
    Rebuild JMS/JMR mass variations from the nominal mass branch.
    Overwrites existing *_JMS_* and *_JMR_* columns.
    """

    rng = np.random.default_rng(seed)

    mass = df[(mass_branch_base, jet_index)].to_numpy()

    smearing = rng.normal(size=mass.shape)
    jms_down, jms_nom, jms_up = jms_vals

    print("jms_down", jms_down)
    print("jms_nom", jms_nom)
    print("jms_up", jms_up)

    jmr_nom, jmr_down, jmr_up = ((smearing * max(jmr_vals[i] - 1, 0) + 1) for i in range(3))

    # Build shifted masses
    mass_nom = mass * jms_nom * jmr_nom

    mass_JMS_down = mass * jms_down * jmr_nom
    mass_JMS_up = mass * jms_up * jmr_nom

    mass_JMR_down = mass * jms_nom * jmr_down
    mass_JMR_up = mass * jms_nom * jmr_up

    df[f"{mass_branch_base}_JMS_down", jet_index] = mass_JMS_down
    df[f"{mass_branch_base}_JMS_up", jet_index] = mass_JMS_up
    df[f"{mass_branch_base}_JMR_down", jet_index] = mass_JMR_down
    df[f"{mass_branch_base}_JMR_up", jet_index] = mass_JMR_up

    df[(mass_branch_base, jet_index)] = mass_nom


# bkg_keys = ["Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched", "Wto2Q", "hbb", "ttbar", "qcd"]
# sig_keys = ["Zto2Q_BB"]
# use this if you want to include Zto2Q_BB in the stack plot
# bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched", "Wto2Q", "hbb", "ttbar", "qcd"]
# bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched", "Wto2Q", "ttbar", "qcd"]
# bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar", "qcd"]
# bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar"]
bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "QCD"]  # Zto2Q and MC QCD
sig_keys = []
bg_order = list(reversed(bkg_keys))

jshift_keys = [""]
for var, ud in itertools.product([
                                # "JES", 
                                # "JER", 
                                "JMS", 
                                "JMR"
                                ], [
                                    "up", 
                                    "down"
                                    ]):
    jshift_keys.append(f"{var}_{ud}")

weight_shifts = {
    "pileup": postprocessing.Syst(
        samples=MC_SAMPLES_FINAL_LIST, label="Pileup", years=list(YEARS_COMBINED_DICT.keys())
    ),
    "pdf": postprocessing.Syst(samples=sig_keys, label="PDFAcc", years=list(YEARS_COMBINED_DICT.keys())),
    "ISRPartonShower": postprocessing.Syst(
        samples=MC_SAMPLES_FINAL_LIST,
        label="ISR Parton Shower",
        years=list(YEARS_COMBINED_DICT.keys()),
    ),
    "FSRPartonShower": postprocessing.Syst(
        samples=MC_SAMPLES_FINAL_LIST,
        label="FSR Parton Shower",
        years=list(YEARS_COMBINED_DICT.keys()),
    ),
    # "pdf": postprocessing.Syst(
    #     samples=MC_SAMPLES_FINAL_LIST,
    #     label="PDF weights",
    #     years=list(YEARS_COMBINED_DICT.keys()),
    # ),
    # "scale": postprocessing.Syst(
    #     samples=MC_SAMPLES_FINAL_LIST,
    #     label="Scale weights",
    #     years=list(YEARS_COMBINED_DICT.keys()),
    # ),

    # What is scale systematic @ Zichun?
}

if APPLY_Z_RECOIL_CORR:
    weight_shifts["GenZPt"] = postprocessing.Syst(
        samples=["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ"],
        label="Gen Z pT correction derived from ZMuMu",
        years=list(YEARS_COMBINED_DICT.keys()),
    )

for year in YEARS_COMBINED_DICT:
    out_dir = PROCESSED_DIR  # was /eos/user/e/eheikkil/scouting/templates/{tag}/, not writable
    if PRODUCER_MASS_RANGE is not None:
        out_dir = out_dir / f"producerMass{PRODUCER_MASS_RANGE[0]}to{PRODUCER_MASS_RANGE[1]}"
    if PRODUCER_LEP_CLEANING:
        out_dir = Path(f"{out_dir}_lepClean") if PRODUCER_MASS_RANGE is not None else out_dir / "lepClean"
    if MASS_VAR == "bbFatJetScoutParTmassCorrResonance":
        out_dir = out_dir / f"mres{m_low}to{m_high}"
    # one directory per set of TXbb working points, e.g. TXbb0p75_0p93_1p0 (the event cache stays in PROCESSED_DIR)
    txbb_edges = [txbb_bins[0][0]] + [high for _, high in txbb_bins]
    out_dir = out_dir / ("TXbb" + "_".join(str(x).replace(".", "p") for x in txbb_edges))
    out_dir.mkdir(parents=True, exist_ok=True)

    template_dir = out_dir
    template_dir.mkdir(parents=True, exist_ok=True)

    for pt_low, pt_high in pt_bins:
        pt_low_str = str(pt_low)
        pt_high_str = str(pt_high)
        pt_bin_key = f"pT{pt_low_str}to{pt_high_str}"

        cutflows_dir = Path(f"{out_dir}/cutflows/{year}")
        cutflows_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = Path(f"{out_dir}/plots/{year}/{pt_bin_key}")
        plot_dir.mkdir(parents=True, exist_ok=True)

        templates = {}

        events = events_combined[year]

        # no JMS/JMR smearing of the MC mass for now (as decorrelated_scouting_templates.py). Note: inside this pT loop
        # it re-smeared the already smeared nominal mass once per pT bin
        # for sample_name, df in events.items():
        #     if sample_name != "data":
        #         print(sample_name)
        #         apply_jmsr_smearing_in_templates(
        #                 df,
        #                 mass_branch_base=MASS_VAR, #"bbFatJetParT3massCorrectedX2p",
        #                 jet_index=0,
        #                 seed=42,
        #                 jms_vals=(0.9, 1.0, 1.1), # these are in nonsensical order right now, change to work with dict
        #                 jmr_vals=(1.1, 1.0, 1.2)
        #             )

        # Determine the pt and mass variations
        for jshift in jshift_keys:
            pt_branch = "bbFatJetPt0"
            mass_branch = f"{MASS_VAR}0"
            #mass_branch = "bbFatJetScoutParTmassGeneric0"
            #mass_branch = "bbFatJetMsd0"
            if jshift == "":
                pt_branch = "bbFatJetPt0"
                mass_branch = f"{MASS_VAR}0"
                # mass_branch = "bbFatJetScoutParTmassGeneric0"
                #mass_branch = "bbFatJetMsd0"
            elif jshift.startswith("JES") or jshift.startswith("JER"):
                pt_branch = f"bbFatJetPt_{jshift}0"
                mass_branch = f"{MASS_VAR}0"
                # mass_branch = "bbFatJetScoutParTmassGeneric0"
                #mass_branch = "bbFatJetMsd0"
            elif jshift.startswith("JMS") or jshift.startswith("JMR"):
                pt_branch = "bbFatJetPt0"
                mass_branch = f"{MASS_VAR}_{jshift}0"
                # mass_branch = f"bbFatJetScoutParTmassGeneric_{jshift}0"
                #mass_branch = f"bbFatJetMsd_{jshift}0"
            # producer mass window on both jets, OR of msd and m_res ("a+b" cuts of utils.make_selection)
            producer_mass_cuts = {}
            if PRODUCER_MASS_RANGE is not None:
                for i in range(2):
                    producer_mass_cuts[f"bbFatJetMsd{i}+bbFatJetScoutParTmassCorrResonance{i}"] = [
                        list(PRODUCER_MASS_RANGE),
                        list(PRODUCER_MASS_RANGE),
                    ]
            # then the producer lepton-jet cleaning on both jets (1: kept), the last steps of the cutflow
            if PRODUCER_LEP_CLEANING:
                for i in range(2):
                    producer_mass_cuts[f"bbFatJetLepClean{i}"] = [0.5, 1.5]

            # Different pass regions based on TXbb and pT bins
            selection_regions = {}
            for txbb_low, txbb_high in txbb_bins:
                # Convert to strings
                txbb_low_str = str(txbb_low).replace(".", "p")
                txbb_high_str = str(txbb_high).replace(".", "p")
                region_key = f"pass_TXbb{txbb_low_str}to{txbb_high_str}_{pt_bin_key}"

                cutflows = {}
                
                for sample in events:
                    cutflows[sample] = OrderedDict()
                    cutflows[sample]["Skimmer Preselection"] = events_combined[year][sample][
                        "finalWeight"
                    ].sum()
                    # cutflows[sample]["HLT"] = events[sample]["finalWeight"].sum()
                cutflows = pd.DataFrame.from_dict(cutflows).transpose()

                # Create a region
                selection_regions[region_key] = postprocessing.Region(
                    cuts={
                        **JETVETO_CUT,
                        pt_branch: [pt_low, pt_high],
                        mass_branch: [m_low, m_high],
                        f"{tagger_branch}0": [txbb_low, txbb_high],
                        **producer_mass_cuts,
                    },
                    label=region_key,
                )

            selection_regions["fail"] = postprocessing.Region(
                cuts={
                    **JETVETO_CUT,
                    pt_branch: [pt_low, pt_high],
                    mass_branch: [m_low, m_high],
                    **producer_mass_cuts,
                    f"{tagger_branch}0": [0.3, min(0.9, min_txbb)], # Having tagger score selection at 0.9 leaks Zbb to fail
                },
                label="fail",
            )
            print(f"Selection regions for {year} with jshift {jshift}: {selection_regions.keys()}")

            fit_shape_var = postprocessing.ShapeVar(
                mass_branch,
                r"$m_\mathrm{res}$ (GeV)" if MASS_VAR == "bbFatJetScoutParTmassCorrResonance" else r"$m_\mathrm{X2p}$ (GeV)",
                [n_mass_bins, m_low, m_high],
                reg=True,
            )

            ttemps = postprocessing.get_templates(
                events,
                year=year,
                sig_keys=sig_keys,
                plot_sig_keys=sig_keys,
                selection_regions=selection_regions,
                shape_vars=[fit_shape_var],
                systematics={},
                template_dir=out_dir,
                bg_keys=bkg_keys,
                bg_order=bg_order,
                bg_err_mcstat=False,
                plot_dir=plot_dir if "data" in events else "",  # the data/MC plots need data
                prev_cutflow=cutflows,
                weight_key="finalWeight",
                weight_shifts=weight_shifts,
                plot_shifts=False,
                show=False,
                energy=13.6,
                jshift=jshift,
                blind=False,
            )
            templates = {**templates, **ttemps}

            # Zto2Q cutflow split into Z->bb with the probe jet (highest TXbb) matched to the Z (as Zto2Q_BB) and the
            # rest (nominal only), saved next to the get_templates cutflows
            if ZQQ_BB_CUTFLOW and jshift == "" and "Zto2Q" in events:
                is_zbb = events["Zto2Q"][("GenZBB", 0)].astype(bool) & (
                    events["Zto2Q"][("bbFatJetVQQMatch", 0)] == 1
                )
                zqq_split = {"Zto2Q_ZBB": events["Zto2Q"][is_zbb], "Zto2Q_nonZBB": events["Zto2Q"][~is_zbb]}
                zqq_prev = pd.DataFrame.from_dict(
                    {s: {"Skimmer Preselection": df["finalWeight"].to_numpy().sum()} for s, df in zqq_split.items()}
                ).transpose()
                for rname, region in selection_regions.items():
                    _, zqq_cf = utils.make_selection(
                        region.cuts, zqq_split, weight_key="finalWeight", prev_cutflow=zqq_prev
                    )
                    zqq_cf.loc["ZBB/nonZBB"] = zqq_cf.loc["Zto2Q_ZBB"] / zqq_cf.loc["Zto2Q_nonZBB"]
                    print(f"Zto2Q ZBB / non-ZBB cutflow {rname}\n{zqq_cf.round(3)}")
                    zqq_cf.round(3).to_csv(out_dir / "cutflows" / year / f"{rname}_cutflow_ZBBsplit.csv")

        # Save the templates to a file
        outfile = template_dir / f"templates_{year}_{pt_bin_key}.root"
        save_to_root(outfile, templates)
        # Save as a pickle file
        outfile_pickle = template_dir / f"templates_{year}_{pt_bin_key}.pkl"
        with outfile_pickle.open("wb") as f:
            pd.to_pickle(templates, f)