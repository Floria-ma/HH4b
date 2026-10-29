import pandas as pd
import numpy as np

from pathlib import Path

from HH4b import utils
from HH4b import postprocessing
import itertools
import correctionlib
from collections import OrderedDict
import uproot

import nanotrees_loader

# Input ntuples: "skimmer" reads the bbbbSkimmer parquet in DATA_DIR, "nanotrees" the VH NanoTrees "had" trees in NANOTREES_DIR
INPUT_FORMAT = "nanotrees"
NANOTREES_DIR = "/eos/user/z/zima/HadronicVH/tagger_Cali_2024_had"
# Luminosity (fb^-1) to normalise the NanoTrees MC to. None keeps the full-2024 lumiwgt of the trees,
# so set this to the lumi of the data eras in NANOTREES_DIR (test_2024_had has Run2024C+D only)
NANOTREES_MC_LUMI = None
# Files of each sample relative to NANOTREES_DIR, overriding nanotrees_loader.SAMPLE_FILES. tagger_Cali_2024_had
# has no merged trees, only the skim_nanotrees.py skims (as in QCD/Data_mass_decorrelation_plots.py)
NANOTREES_FILES = {
    "data": "data/skim/*/ScoutingPFRun3_tree_*.root",
    "Zto2Q": "mc/skim/z2qq_tree.root",
    "Wto2Q": "mc/skim/w2qq_tree.root",
    "ttbar": "mc/skim/ttbar_tree.root",
    "QCD": "mc/skim/qcd*_tree.root",
    "singletop": "mc/skim/singletop_tree.root",
    "hbb": "mc/skim/*h2bb_tree.root",  # ggH, ggZH, WH (W to lnu, qq) and ZH, H to bb
}
# processes reading the NanoTrees files of a sample in parallel (each holds a ~500 MB chunk, a few GB at its peak)
NANOTREES_N_WORKERS = 6

# NanoTrees event selection: "zbb" the scouting Zbb one of bbbbSkimmer, "vh" the VH had channel one
SELECTION = "zbb"
NANOTREES_SELECTIONS = {
    "zbb": nanotrees_loader.DEFAULT_SELECTION,
    "vh": nanotrees_loader.VH_HAD_SELECTION,
}
# the VH had channel weights its MC with the trigger SFs instead of an HT cut on the trigger plateau
NANOTREES_APPLY_TRIG_SF = SELECTION == "zbb"

tag = "14Feb2026_VJets_TT_Only_v15_scouting_zbb" #"12Feb2026_Scouting_Fixed_v15_scouting_zbb" #"07Feb2026_PTl300TXbb0p3_PTsl300_HT600_v15_scouting_zbb" #"11Dec2025_v15_scouting_zbb"
if INPUT_FORMAT == "nanotrees":
    tag = f"nanotrees_{Path(NANOTREES_DIR).name}_{SELECTION}"

# Pass and fail regions
txbb_bins = [0.75, 0.93, 1.0] #0.75 #0.93
min_txbb = txbb_bins[0]
# pT bins
#pt_bins = [170, 225, 250, 275, 300, 350, 500, 900, 1300]
pt_bins = [170, 250, 350, 400, 450, 500, 550, 600, 900, 10000]

txbb_bins = list(zip(txbb_bins[:-1], txbb_bins[1:]))
pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

# Mass bins
m_low, m_high = 30, 200
bin_width = 5
n_mass_bins = int((m_high - m_low) / bin_width)

STORAGE_PROJ_DIR = Path("/eos/user/z/zima/bbbb/")
DATA_DIR = STORAGE_PROJ_DIR / f"skimmer/{tag}"
PROCESSED_DIR = STORAGE_PROJ_DIR / f"scouting/templates/{tag}"
# PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

REPROCESS: bool = True  # if True, reprocess from the skimmed ntuples
APPLY_Z_RECOIL_CORR: bool = False # if True, apply the Z->2Q recoil corrections derived from ZMuMu measurement
DO_JESR: bool = False # can't
DO_JMSR: bool = True
tagger_branch = "bbFatJetScoutParTTXbb"
# Regressed mass to fit. The NanoTrees only have the resonance mass, skimmer outputs also "bbFatJetScoutParTmassCorrectedX2p"
MASS_VAR = "bbFatJetScoutParTmassCorrResonance"
MASS_LABEL = r"$m_\mathrm{res}$ (GeV)"
# H signal region of the VH analysis, whose selection these events share: data is zeroed there in the pass regions
# (templates and plots). The QCD estimate and the fit have to skip these bins
BLIND_WINDOW = [100, 150]
# fail region of the QCD estimate (qcd_estimation.py): the fail TXbb range with this lower bound. The "fail" region of
# the fit (TXbb from 0) is kept as is
QCD_FAIL_TXBB_MIN = 0.4
QCD_FAIL_REGION = f"fail_TXbb{str(QCD_FAIL_TXBB_MIN).replace('.', 'p')}to{str(min(0.9, min_txbb)).replace('.', 'p')}"


YEARS = [
    # "2022",
    # "2022EE",
    # "2023",
    # "2023BPix",
    "2024"
    ]
YEARS_COMBINED_DICT = {
    # "2022All": [
    #         "2022",
    #         "2022EE"
    #         ],
    # "2023All": [
    #         "2023",
    #         "2023BPix"
    #         ],
    "2024All": ["2024"]
}
# ZMuMu GenZPt correction applied to each combined year. None has been derived for 2024 yet, so 2024 reuses 2023
Z_RECOIL_CORR_YEAR = {"2022All": "2022", "2023All": "2023", "2024All": "2023"}
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
    "data": [
        #2023
        "Run2023C", 

        # 2023BPix
        "Run2023D",

        # 2024
        "Run2024C", 
        "Run2024D",
        "Run2024E",
        "Run2024F",
        "Run2024G",
        "Run2024H",
        "Run2024I"
    ],
    # QCD MC, only to compare with the QCD estimated from data (qcd_estimation.py writes that one as "qcd").
    # Named "QCD" so that its templates QCD_{region} don't clash with the estimate's qcd_{region}
    "QCD": ["QCD"],
    "ttbar": ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"],
    "singletop": ["singletop"],
    "Zto2Q": ["Zto2Q-4Jets"],
    "Wto2Q": ["Wto2Q-3Jets"],
    # "hbb", the key of the H to bb colour and label in plotting.py
    "hbb": ["ggh-h2bb", "ggzh-z2qq-h2bb", "wh-w2ln-h2bb", "wh-w2qq-h2bb", "zh-z2qq-h2bb"],
}
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
    MASS_VAR,
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
            mass_variations.append(f"{var}_{jmsr}_{ud}")


base_columns = [(var, 2) for var in fatjet_vars] + [("weight", 1)]

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
    "QCD": load_weight_shifts,
    "ttbar": MC_common_extra_columns,
    "singletop": MC_common_extra_columns,
    "hbb": MC_common_extra_columns,
    "Zto2Q": MC_common_extra_columns + ZQQ_extra_columns,
    "Wto2Q": MC_common_extra_columns + WQQ_extra_columns,
}

# if True, apply the Z->2Q corrections from ZMuMu measurement
if APPLY_Z_RECOIL_CORR:
    corr_dir = Path("ZMuMu_corrs")
    corr_dict = {}

    for year in ["2022", "2023"]:
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
    # a combined year of a single era is that era's file, instead of a second copy of it
    year_list = YEARS_COMBINED_DICT[combined_year]
    if len(year_list) == 1:
        return get_era_path(year_list[0])
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
                print(f"Loading {sample} for {year}...")

                columns = base_columns + extra_columns_dict.get(sample, [])
                if INPUT_FORMAT == "nanotrees":
                    dataframes = nanotrees_loader.load_nanotrees(
                        NANOTREES_DIR,
                        [sample],
                        selection=NANOTREES_SELECTIONS[SELECTION],
                        apply_trig_sf=NANOTREES_APPLY_TRIG_SF,
                        jmsr_vars=mass_vars if DO_JMSR else [],
                        mc_lumi=NANOTREES_MC_LUMI,
                        file_patterns=NANOTREES_FILES,
                        # the templates only use the highest-TXbb jet: its pT, TXbb and mass (and the mass_vars
                        # for the MC JMS/JMR variations). Keeping only these is what fits data (~2x10^8 events) in memory
                        jet_columns={
                            s: [tagger_branch, "bbFatJetPt", MASS_VAR] + ([] if s == "data" else mass_vars)
                            for s in SAMPLES_DICT
                        },
                        first_jet_only=list(SAMPLES_DICT),
                        n_workers=NANOTREES_N_WORKERS,
                    )
                else:
                    dataframes = {
                        **utils.load_samples(
                            data_dir=str(DATA_DIR),
                            samples={sample: sample_list},
                            year=year,
                            columns=utils.format_columns(columns),
                            variations=True,
                            weight_shifts=weight_shifts,
                        )
                    }

                # Process and concatenate dataframes for this sample
                # the pT and mass variations missing in a sample (e.g. data) are added per pT bin below, as copies
                # of the nominal for the events of that bin only
                sample_dfs = list(dataframes.values())

                # Concatenate all dataframes for this sample (concat would copy a single one too)
                events_era[sample] = (
                    sample_dfs[0] if len(sample_dfs) == 1 else pd.concat(sample_dfs, ignore_index=True)
                )
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

        if len(year_list) == 1:
            # nothing to combine, the era file written above is the combined year
            print(f"\nLoading {combined_year} from {combined_path}...")
            with combined_path.open("rb") as f:
                events_combined[combined_year] = pd.read_pickle(f)
        elif REPROCESS or not combined_path.exists():
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
        corr = corr_dict[Z_RECOIL_CORR_YEAR[year]]["GenZPtWeight"]
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
    # explicit copies, the JMS/JMR smearing below writes to them
    events_combined[year]["Zto2Q_BB"] = Zto2Q[ZtoBB].copy()
    events_combined[year]["Zto2Q_CC"] = Zto2Q[ZtoCC].copy()
    events_combined[year]["Zto2Q_QQ"] = Zto2Q[ZtoQQ].copy()
    # leave the unmatched to bkg fits
    # events_combined[year]["Zto2Q_unmatched"] = Zto2Q[Z_unmatched]

# MC_SAMPLES_FINAL_LIST = MC_SAMPLES_LIST + ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Zto2Q_unmatched"]
MC_SAMPLES_FINAL_LIST = MC_SAMPLES_LIST + ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ"]

def save_to_root(outfile: Path, templates: dict):
    with uproot.recreate(str(outfile)) as f_out:
        for category in templates.keys():
            hist = templates[category]
            categories, _ = hist.axes
            for sample in list(categories):
                h = templates[category][{"Sample": sample}]
                f_out[f"{sample}_{category}"] = h

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
bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar", "singletop", "QCD"]
# H to bb is plotted as an unstacked step histogram, scaled by SIG_SCALE in the pass regions
sig_keys = ["hbb"]
SIG_SCALE = 50
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
    # the NanoTrees have no PDF weights
    "pdf": postprocessing.Syst(
        samples=sig_keys if INPUT_FORMAT == "skimmer" else [], label="PDFAcc", years=list(YEARS_COMBINED_DICT.keys())
    ),
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
    out_dir = Path(f"/eos/user/z/zima/scouting/templates/{tag}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    template_dir = out_dir
    template_dir.mkdir(parents=True, exist_ok=True)

    events = events_combined[year]

    # memory (~2x10^8 data events): the templates weight with finalWeight, so drop its copy "weight", and sort each
    # sample by pT so that each pT bin below is a slice of the rows (a view) instead of a copy
    for sample_name in list(events):
        df = events[sample_name]
        if "weight" in df.columns.get_level_values(0):
            del df["weight"]
        # sorted one column at a time, popping each from the unsorted frame, so that the peak is the sample plus one
        # column (iloc with the sort order held a full second copy of the sample)
        order = np.argsort(df[("bbFatJetPt", 0)].to_numpy(), kind="stable")
        events[sample_name] = pd.DataFrame({col: df.pop(col).to_numpy()[order] for col in list(df.columns)}, copy=False)
        del df, order

    # smear once per year: it modifies the nominal mass in place, so inside the pT bin loop it compounded per bin
    for sample_name, df in events.items():
        if sample_name != "data":
            print(sample_name)
            apply_jmsr_smearing_in_templates(
                    df,
                    mass_branch_base=MASS_VAR,
                    jet_index=0,
                    seed=42,
                    jms_vals=(0.9, 1.0, 1.1), # these are in nonsensical order right now, change to work with dict
                    jmr_vals=(1.1, 1.0, 1.2)
                )

    for pt_low, pt_high in pt_bins:
        pt_low_str = str(pt_low)
        pt_high_str = str(pt_high)
        pt_bin_key = f"pT{pt_low_str}to{pt_high_str}"

        # the events of this pT bin, so that the selections of each region and JMS/JMR shift do not run over all
        # events. Same cut as the pT cut of the regions (low <= pT < high); only valid without the JES/JER pT shifts
        assert not DO_JESR, "the pT bin slicing does not account for the JES/JER shifts of the pT"
        events_bin = {}
        for sample_name, df in events.items():
            # the events are sorted by pT (NaN last), so low <= pT < high is a slice
            pt = df[("bbFatJetPt", 0)].to_numpy()
            i_low, i_high = np.searchsorted(pt, [pt_low, pt_high], side="left")
            df = df.iloc[i_low:i_high]

            # the selections and shapes of each jshift use the shifted variable for every sample, so the samples
            # without the variations (data) get copies of the nominal. New columns of the slice only, so the
            # chained assignment warning does not apply
            with pd.option_context("mode.chained_assignment", None):
                for pt_var in pt_variations:
                    if pt_var not in df.columns:
                        df[f"{pt_var}0"] = df[("bbFatJetPt", 0)]
                for mass_var in mass_variations:
                    if mass_var.startswith(MASS_VAR) and mass_var not in df.columns:
                        df[f"{mass_var}0"] = df[(MASS_VAR, 0)]

            events_bin[sample_name] = df

        cutflows_dir = Path(f"{out_dir}/cutflows/{year}")
        cutflows_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = Path(f"{out_dir}/plots/{year}/{pt_bin_key}")
        plot_dir.mkdir(parents=True, exist_ok=True)

        templates = {}

        # Determine the pt and mass variations
        for jshift in jshift_keys:
            pt_branch = "bbFatJetPt0"
            mass_branch = f"{MASS_VAR}0"
            if jshift.startswith("JES") or jshift.startswith("JER"):
                pt_branch = f"bbFatJetPt_{jshift}0"
            elif jshift.startswith("JMS") or jshift.startswith("JMR"):
                mass_branch = f"{MASS_VAR}_{jshift}0"
            # Different pass regions based on TXbb and pT bins
            selection_regions = {}
            for txbb_low, txbb_high in txbb_bins:
                # Convert to strings
                txbb_low_str = str(txbb_low).replace(".", "p")
                txbb_high_str = str(txbb_high).replace(".", "p")
                region_key = f"pass_TXbb{txbb_low_str}to{txbb_high_str}_{pt_bin_key}"

                cutflows = {}
                
                for sample in events_bin:
                    cutflows[sample] = OrderedDict()
                    cutflows[sample]["Skimmer Preselection"] = events_combined[year][sample][
                        "finalWeight"
                    ].sum()
                    # cutflows[sample]["HLT"] = events[sample]["finalWeight"].sum()
                cutflows = pd.DataFrame.from_dict(cutflows).transpose()

                # Create a region
                selection_regions[region_key] = postprocessing.Region(
                    cuts={
                        pt_branch: [pt_low, pt_high],
                        mass_branch: [m_low, m_high],
                        f"{tagger_branch}0": [txbb_low, txbb_high],
                    },
                    label=region_key,
                )

            selection_regions["fail"] = postprocessing.Region(
                cuts={
                    pt_branch: [pt_low, pt_high],
                    mass_branch: [m_low, m_high],
                    f"{tagger_branch}0": [0.0, min(0.9, min_txbb)], # Having tagger score selection at 0.9 leaks Zbb to fail
                },
                label="fail",
            )
            selection_regions[QCD_FAIL_REGION] = postprocessing.Region(
                cuts={
                    pt_branch: [pt_low, pt_high],
                    mass_branch: [m_low, m_high],
                    f"{tagger_branch}0": [QCD_FAIL_TXBB_MIN, min(0.9, min_txbb)],
                },
                label=QCD_FAIL_REGION,
            )
            print(f"Selection regions for {year} with jshift {jshift}: {selection_regions.keys()}")

            fit_shape_var = postprocessing.ShapeVar(
                mass_branch,
                MASS_LABEL,
                [n_mass_bins, m_low, m_high],
                reg=True,
                blind_window=BLIND_WINDOW,
            )

            ttemps = postprocessing.get_templates(
                events_bin,
                year=year,
                sig_keys=sig_keys,
                plot_sig_keys=sig_keys,
                sig_scale_dict={key: SIG_SCALE for key in sig_keys},
                selection_regions=selection_regions,
                shape_vars=[fit_shape_var],
                systematics={},
                template_dir=out_dir,
                bg_keys=bkg_keys,
                bg_order=bg_order,
                bg_err_mcstat=False,
                plot_dir=plot_dir,
                prev_cutflow=cutflows,
                weight_key="finalWeight",
                weight_shifts=weight_shifts,
                plot_shifts=False,
                show=False,
                energy=13.6,
                jshift=jshift,
                blind=True,
            )
            templates = {**templates, **ttemps}

        # Save the templates to a file
        outfile = template_dir / f"templates_{year}_{pt_bin_key}.root"
        save_to_root(outfile, templates)
        # Save as a pickle file
        outfile_pickle = template_dir / f"templates_{year}_{pt_bin_key}.pkl"
        with outfile_pickle.open("wb") as f:
            pd.to_pickle(templates, f)
