import pandas as pd
import numpy as np

from pathlib import Path
import pickle

from HH4b import utils
from HH4b import postprocessing
import itertools
import correctionlib
from collections import OrderedDict
import uproot

# Pass and fail regions
txbb_bins = [0.96, 1.0]
min_txbb = txbb_bins[0]
# pT bins
pt_bins = [350, 450, 550]

txbb_bins = list(zip(txbb_bins[:-1], txbb_bins[1:]))
pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

# Mass bins
m_low, m_high = 40, 180
bin_width = 5
n_mass_bins = int((m_high - m_low) / bin_width)

tag = "14Feb2026_VJets_TT_Only_v15_scouting_zbb" #"12Feb2026_Scouting_Fixed_v15_scouting_zbb" #"07Feb2026_PTl300TXbb0p3_PTsl300_HT600_v15_scouting_zbb" #"11Dec2025_v15_scouting_zbb"


STORAGE_PROJ_DIR = Path("/eos/user/z/zima/bbbb/")
DATA_DIR = STORAGE_PROJ_DIR / f"skimmer/{tag}"
PROCESSED_DIR = STORAGE_PROJ_DIR / f"scouting/templates/{tag}"
# PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

REPROCESS: bool = False  # if True, reprocess from the skimmed ntuples
APPLY_Z_RECOIL_CORR: bool = True
DO_JESR: bool = False
DO_JMSR: bool = True
tagger_branch = "bbFatJetScoutParTTXbb"

SAMPLES_DICT = {
    "data": [
        #2023
        "Run2023C", 

        # 2023BPix
        "Run2023D",

        # 2024
        # "Run2024C", 
        # "Run2024D",
        # "Run2024E",
        # "Run2024F",
        # "Run2024G",
        # "Run2024H",
        # "Run2024I"
    ],
    # QCD from data in CR
    "ttbar": ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"],
    "Zto2Q": ["Zto2Q-4Jets"],
    "Wto2Q": ["Wto2Q-3Jets"],
}
MC_SAMPLES_LIST = [sample for sample in SAMPLES_DICT.keys() if sample != "data"]


def load_qsurf(alpha: float, mvar: str, qsurf_dir: str):
    """
    A function for loading in a quantile surface map derived from QCD with rejection rate alpha (float).

    In general the returned map is a function of pT and rho = ln(m^2/pT^2). The used mass variable is specified in the filename,
    and must be one of "mSD", "mX2p", "mGeneric". Returned alongside the quantile surface are also the bin edges in rho and pT.

    The quantile surfaces are saved in the format:

    """

    if mvar not in ["mSD", "mX2p", "mGeneric"]:
        # raise Exception(f"Mass variable {mvar} not recognized. Has to be one of [mSD, mX2p, mGeneric]")
        pass

    alpha_str = str(alpha).replace(".","p")
    with open(f"{qsurf_dir}/TXBBPRECUT0p3_qsurf_{alpha_str}_{mvar}.pkl", "rb") as f: # add an underscofe between alpha and mvar!
        qsurf, pt_edges, rho_edges = pickle.load(f)

    # qsurf = bins["qsurf"]
    # pt_edges  = bins["pt_edges"]
    # rho_edges = bins["rho_edges"]
    print("Quantile surface loaded!")

    return qsurf, pt_edges, rho_edges

qsurf, pt_edges, rho_edges = load_qsurf(alpha = 0.995, mvar = "mX2p", qsurf_dir = "/eos/user/z/zima/MASS_DECORRELATION_EFFORTS/NoSmoothing/outputs") # OUTDIR of decorrelation/ScoutGloParT_decorrelation.py

# del qsurf, pt_edges, rho_edges

def add_TXbb_DDT_branch(df, qsurf, pt_edges, rho_edges):
    """
    Function adds the TXbb_DDT = TXbb - TXbb^alpha(rho, pT) branch to the df.
    """
    m = df[("bbFatJetScoutParTmassCorrectedX2p", 0)].to_numpy(dtype=np.float32, copy=False)
    pt = df[("bbFatJetPt", 0)].to_numpy(dtype=np.float32, copy=False)
    txbb = df[("bbFatJetScoutParTTXbb", 0)].to_numpy(dtype=np.float32, copy=False)

    rho = np.empty_like(m)
    np.divide(m, pt, out=rho)
    np.log(rho, out=rho)
    rho *= 2.0

    ipt = np.digitize(pt, pt_edges) - 1
    irho = np.digitize(rho, rho_edges) - 1

    valid = (
        (ipt >= 0) & (ipt < qsurf.shape[0]) &
        (irho >= 0) & (irho < qsurf.shape[1])
    )

    q = np.full(txbb.shape, np.nan, dtype=np.float32)
    q[valid] = qsurf[ipt[valid], irho[valid]]

    TXbb_DDT = txbb.copy()
    TXbb_DDT -= q

    df["TXbb_DDT0"] = TXbb_DDT

YEARS = [
    # "2022", 
    # "2022EE", 
     "2023", 
    "2023BPix",
    #"2024"
    ]
YEARS_COMBINED_DICT = {
    # "2022All": [
    #         "2022", 
    #         "2022EE"
    #         ],
     "2023All": [
             "2023", 
            "2023BPix"
             ],
    #"2024": ["2024"]
}

# RUNS = {
#     "2023": ["Run2023C"],
#     "2023BPix": ["Run2023D"],
#     # "2024": [
#     #     "Run2024C", 
#     #     "Run2024D",
#     #     "Run2024E",
#     #     "Run2024F",
#     #     "Run2024G",
#     #     "Run2024H",
#     #     "Run2024I"
#     # ]
# }

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
    # "bbFatJetMsd",
]
mass_vars = [
    # "bbFatJetMsd",
]
scoutpart_mass_vars = [
    "bbFatJetScoutParTmassCorrectedX2p",
    # "bbFatJetScoutParTmassGeneric"
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
    # "qcd": load_weight_shifts, # Don't bother loading QCD, just a waste of time since it is estimted from data anyway
    "ttbar": MC_common_extra_columns,
    # "hbb": MC_common_extra_columns, # Don't have hbb samples
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

                    add_TXbb_DDT_branch(df, qsurf, pt_edges, rho_edges)

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

# print(events_combined["2023All"]["data"]["TXbb_DDT"])
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
    jms_vals=(1.0, 1.05, 1.1), # down, nom, up
    jmr_vals=(1.0, 1.1, 1.2),
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
    print("jmr_nom", jmr_vals[0])
    print("jmr_down", jmr_vals[1])
    print("jmr_up", jmr_vals[2])

    # jmr_nom, jmr_down, jmr_up = ((smearing * max(jmr_vals[i] - 1, 0) + 1) for i in range(3))
    jmr_nom = smearing * max(jmr_vals[0] - 1, 0) + 1
    jmr_down = smearing * max(jmr_vals[1] - 1, 0) + 1
    jmr_up = smearing * max(jmr_vals[2] - 1, 0) + 1


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
bkg_keys = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar"]
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
    out_dir = Path(f"/eos/user/z/zima/scouting/templates/{tag}/")
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

        # for sample_name, df in events.items():
            # if sample_name != "data":
            #     print(sample_name)
            #     apply_jmsr_smearing_in_templates(
            #             df,
            #             mass_branch_base= "bbFatJetScoutParTmassCorrectedX2p", #"bbFatJetMsd", #"bbFatJetParT3massCorrectedX2p",
            #             jet_index=0,
            #             seed=42,
            #             jms_vals=(0.9, 1.0, 1.1), # nom, down, up
            #             jmr_vals=(1.025, 1.0, 1.05), # down, nom, up
            #         )

        # Determine the pt and mass variations
        for jshift in jshift_keys:
            pt_branch = "bbFatJetPt0"
            mass_branch = "bbFatJetScoutParTmassCorrectedX2p0"
            #mass_branch = "bbFatJetScoutParTmassGeneric0"
            # mass_branch = "bbFatJetMsd0"
            if jshift == "":
                pt_branch = "bbFatJetPt0"
                mass_branch = "bbFatJetScoutParTmassCorrectedX2p0"
                # mass_branch = "bbFatJetScoutParTmassGeneric0"
                # mass_branch = "bbFatJetMsd0"
            elif jshift.startswith("JES") or jshift.startswith("JER"):
                pt_branch = f"bbFatJetPt_{jshift}0"
                mass_branch = "bbFatJetScoutParTmassCorrectedX2p0"
                # mass_branch = "bbFatJetScoutParTmassGeneric0"
                # mass_branch = "bbFatJetMsd0"
            elif jshift.startswith("JMS") or jshift.startswith("JMR"):
                pt_branch = "bbFatJetPt0"
                mass_branch = f"bbFatJetScoutParTmassCorrectedX2p_{jshift}0"
                # mass_branch = f"bbFatJetScoutParTmassGeneric_{jshift}0"
                # mass_branch = f"bbFatJetMsd_{jshift}0"
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
                        pt_branch: [pt_low, pt_high],
                        mass_branch: [m_low, m_high],
                        "TXbb_DDT0": [0.0, 1.0],
                        # f"{tagger_branch}0": [txbb_low, txbb_high],
                    },
                    label=region_key,
                )

            selection_regions["fail"] = postprocessing.Region(
                cuts={
                    pt_branch: [pt_low, pt_high],
                    mass_branch: [m_low, m_high],
                    "TXbb_DDT0": [-1.0, 0.0], #testing
                    # f"{tagger_branch}0": [0.3, min(0.9, min_txbb)], # Having tagger score selection at 0.9 leaks Zbb to fail
                },
                label="fail",
            )
            print(f"Selection regions for {year} with jshift {jshift}: {selection_regions.keys()}")

            fit_shape_var = postprocessing.ShapeVar(
                mass_branch,
                r"$m_\mathrm{X2p}$ (GeV)",
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
                plot_dir=plot_dir,
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

        # Save the templates to a file
        outfile = template_dir / f"templates_{year}_{pt_bin_key}.root"
        save_to_root(outfile, templates)
        # Save as a pickle file
        outfile_pickle = template_dir / f"templates_{year}_{pt_bin_key}.pkl"
        with outfile_pickle.open("wb") as f:
            pd.to_pickle(templates, f)
