import pandas as pd
import numpy as np

from pathlib import Path

from HH4b import utils
from HH4b import postprocessing
import itertools
import correctionlib
from collections import OrderedDict
import uproot

# Pass and fail regions
txbb_bins = [0.95, 0.975, 0.99, 1.0]
# txbb_bins = [0.94, 0.96, 0.98, 1.0]
min_txbb = txbb_bins[0]
# pT bins
pt_bins = [
    450, 
    550,
    10000
    ]

txbb_bins = list(zip(txbb_bins[:-1], txbb_bins[1:]))
pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

# Mass bins
m_low, m_high = 40, 180
bins = 5
n_mass_bins = int((m_high - m_low) / bins)

YEARS = [
    # "2022", 
    # "2022EE", 
    "2023", 
    "2023BPix"
    ]
YEARS_COMBINED_DICT = {
    # "2022All": ["2022", "2022EE"],
    "2023All": [
        "2023", 
        "2023BPix"
        ], # Removed ALL
}

# tag = "ZbbHT25July22_v14_25v2_zbb"
# tag = "ZbbHT25July31_v14_25v2_zbb"
tag = "11Feb2026_StandardOfflineCuts_v15_scouting_zbb" #"07Feb2026_PTl450TXbb0p3_PTsl200_HT1000_v15_scouting_zbb" #"05Feb2026_offline_v15_scouting_zbb"

STORAGE_PROJ_DIR = Path("/eos/user/z/zima/bbbb/")
DATA_DIR = STORAGE_PROJ_DIR / f"skimmer/{tag}"
PROCESSED_DIR = STORAGE_PROJ_DIR / f"scouting/templates/{tag}"
# PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

REPROCESS: bool = False  # if True, reprocess from the skimmed ntuples
APPLY_Z_RECOIL_CORR: bool = True
APPLY_TRIGGER_SF: bool = True

SAMPLES_DICT = {
    "data": ["Run2023C", "Run2023D"],
    # QCD from data in CR
    "ttbar": ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"],
    "Zto2Q": ["Zto2Q-4Jets"],
    "Wto2Q": ["Wto2Q-3Jets"],
}
MC_SAMPLES_LIST = [sample for sample in SAMPLES_DICT.keys() if sample != "data"]

# Columns to load from the ntuples
sys_vars = ["FSRPartonShower", "ISRPartonShower", "pileup"]
weight_shifts = sys_vars + ["pdf_weights", "scale_weights"]

fatjet_vars = [
    "bbFatJetPt",
    "bbFatJetEta",
    "bbFatJetMsd",
    # "bbFatJetPNetMassLegacy",
]
mass_vars = [
    "bbFatJetMsd",
    # "bbFatJetPNetMassLegacy",
]
glopart_mass_vars = [
    # "bbFatJetParT2massVis",
    # "bbFatJetParT2massRes",
    "bbFatJetParT3massGeneric",
    "bbFatJetParT3massCorrectedX2p",
]
glopart_txbb_vars = [
    # "bbFatJetParT2TXbb",
    "bbFatJetParT3TXbb",
]
fatjet_vars += glopart_mass_vars + glopart_txbb_vars
mass_vars += glopart_mass_vars

pt_variations = []
for jesr, ud in itertools.product(["JES", "JER"], ["up", "down"]):
    pt_variations.append(f"bbFatJetPt_{jesr}_{ud}")

mass_variations = []
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
    # "qcd": load_weight_shifts,
    "ttbar": MC_common_extra_columns,
    "hbb": MC_common_extra_columns,
    "Zto2Q": MC_common_extra_columns + ZQQ_extra_columns,
    "Wto2Q": MC_common_extra_columns + WQQ_extra_columns,
}

triggers = {
    "2022": [
        "AK8PFJet500",
        "AK8PFJet420_MassSD30",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
    ],
    "2022EE": [
        "AK8PFJet500",
        "AK8PFJet420_MassSD30",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
    ],
    "2023": [
        "AK8PFJet500",
        "AK8PFJet420_MassSD30",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
    ],
    "2023BPix": [
        "AK8PFJet500",
        "AK8PFJet420_MassSD30",
        "AK8PFJet425_SoftDropMass40",
        "AK8PFJet230_SoftDropMass40_PNetBB0p06",
    ],
}

# Trigger efficiency corrections
trigger_sf_dir = Path("../corrections/data/trigger_sfs").resolve()
trigger_eff_txbb = {
    year: correctionlib.CorrectionSet.from_file(
        str(trigger_sf_dir / f"fatjet_triggereff_{year}_txbbGloParT_QCD.json")
    )
    for year in YEARS
}
trigger_eff_ptmsd = {
    year: correctionlib.CorrectionSet.from_file(
        str(trigger_sf_dir / f"fatjet_triggereff_{year}_ptmsd_QCD.json")
    )
    for year in YEARS
}

def _compute_SF(mc_eff_set, data_eff_set, *args):
    """Helper function to compute scale factor and error for a given efficiency set."""
    # Evaluate MC efficiencies
    mc_eff_nom = mc_eff_set.evaluate(*args, "nominal")
    mc_eff_err_up = mc_eff_set.evaluate(*args, "stat_up")
    mc_eff_err_down = mc_eff_set.evaluate(*args, "stat_dn")
    mc_eff_err = np.maximum(np.abs(mc_eff_err_up), np.abs(mc_eff_err_down))

    # Evaluate data efficiencies
    data_eff_nom = data_eff_set.evaluate(*args, "nominal")
    data_eff_up = data_eff_set.evaluate(*args, "stat_up")
    data_eff_down = data_eff_set.evaluate(*args, "stat_dn")
    data_eff_err = np.maximum(np.abs(data_eff_up), np.abs(data_eff_down))

    # Compute scale factor and propagate errors
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_nom = data_eff_nom / mc_eff_nom
        sf_err = sf_nom * np.sqrt(
            (data_eff_err / data_eff_nom) ** 2 + (mc_eff_err / mc_eff_nom) ** 2
        )

    # set sf to 1 if mc_eff_nom is zero to avoid division by zero
    sf_nom = np.where(mc_eff_nom == 0, 1.0, sf_nom)
    sf_err = np.where(mc_eff_nom == 0, 0.0, sf_err)
    sf_err = np.where(data_eff_nom == 0, 0.0, sf_err)
    # sf_nom = np.where(sf_nom > 2.0, 1.0, sf_nom)  # restrict scale factor to a maximum of 2.0
    # sf_err = np.where(sf_nom > 2.0, 0.0, sf_err)  # restrict scale factor error to a maximum of 2.0

    return sf_nom, sf_err

def eval_trigger_sf(
    txbb: np.ndarray, pt: np.ndarray, msd: np.ndarray, year: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate trigger scale factors with error propagation."""

    # txbb scale factor
    mc_eff_set = trigger_eff_txbb[year][f"fatjet_triggereffmc_{year}_txbbGloParT"]
    data_eff_set = trigger_eff_txbb[year][f"fatjet_triggereffdata_{year}_txbbGloParT"]
    sf_txbb_nom, sf_txbb_err = _compute_SF(mc_eff_set, data_eff_set, txbb)
    zero_sf_mask = sf_txbb_nom == 0
    sf_txbb_nom[zero_sf_mask] = 1.0
    sf_txbb_err[zero_sf_mask] = 0.0

    # ptmsd scale factor
    mc_eff_set = trigger_eff_ptmsd[year][f"fatjet_triggereffmc_{year}_ptmsd"]
    data_eff_set = trigger_eff_ptmsd[year][f"fatjet_triggereffdata_{year}_ptmsd"]
    sf_ptmsd_nom, sf_ptmsd_err = _compute_SF(mc_eff_set, data_eff_set, pt, msd)
    zero_sf_mask = sf_ptmsd_nom == 0
    sf_ptmsd_nom[zero_sf_mask] = 1.0
    sf_ptmsd_err[zero_sf_mask] = 0.0

    # Combine scale factors
    sf = sf_txbb_nom * sf_ptmsd_nom
    sf_err = sf * np.sqrt((sf_txbb_err / sf_txbb_nom) ** 2 + (sf_ptmsd_err / sf_ptmsd_nom) ** 2)

    sf_up = sf + sf_err
    sf_down = sf - sf_err

    # Ensure scale factors are not negative
    extreme_val_mask = sf_down < 0
    sf[extreme_val_mask] = 1.0
    sf_up[extreme_val_mask] = 1.0
    sf_down[extreme_val_mask] = 1.0

    return sf, sf_up, sf_down

# For my own fuckup: didn't remember to apply jms/r in skimmer
def apply_jmsr_smearing_in_templates(
    df: pd.DataFrame,
    mass_branch_base: str,
    jet_index: int = 0,
    seed: int = 42,
    jms_vals=(0.9, 1.0, 1.1),
    jmr_vals=(1.095, 1.052, 1.139),
):
    """
    Rebuild JMS/JMR mass variations from the nominal mass branch.
    Overwrites existing *_JMS_* and *_JMR_* columns.
    """

    rng = np.random.default_rng(seed)

    mass = df[(mass_branch_base, jet_index)].to_numpy()

    smearing = rng.normal(size=mass.shape)
    jms_down, jms_nom, jms_up = jms_vals

    jmr_down = 1 + smearing * (jmr_vals[0] - 1)
    jmr_nom  = 1 + smearing * (jmr_vals[1] - 1)
    jmr_up   = 1 + smearing * (jmr_vals[2] - 1)
    
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
                triggers_cols = [(trigger, 1) for trigger in triggers[year]]

                columns = triggers_cols + base_columns + extra_columns_dict.get(sample, [])
                dataframes = {
                    **utils.load_samples(
                        data_dir=str(DATA_DIR),
                        samples={sample: sample_list},
                        year=year,
                        columns=utils.format_columns(columns),
                        variations=True,
                        weight_shifts=weight_shifts,
                        # lumi_dict=actual_lumis,
                    )
                }

                # Process and concatenate dataframes for this sample
                sample_dfs = []
                for key, df in dataframes.items():
                    # Handle pT variations
                    for pt_var in pt_variations:
                        for i in range(2):
                            if (pt_var, i) not in df.columns:
                                df[(pt_var, i)] = df[("bbFatJetPt", i)].copy()


                    # Handle mass variations
                    for mass_var in mass_vars + mass_variations:
                        for i in range(2):
                            if (mass_var, i) not in df.columns:
                                df[f"{mass_var}{i}"] = df[(mass_var.split("_")[0], i)].copy()

                    if sample != "data":
                        # Evaluate trigger scale factors
                        sf, sf_up, sf_down = eval_trigger_sf(
                            txbb=df[("bbFatJetParT3TXbb", 0)].values,
                            pt=df[("bbFatJetPt", 0)].values,
                            msd=df[("bbFatJetMsd", 0)].values,
                            year=year,
                        )
                        df["SF_trigger"] = sf
                        df["SF_trigger_up"] = sf_up
                        df["SF_trigger_down"] = sf_down

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

def select_triggers(events, trigger_list):
    print(f"Selecting events with triggers: {trigger_list}")
    events_filtered = {}

    for year in events.keys():
        events_filtered[year] = {}
        for sample in events[year].keys():
            df = events[year][sample].copy()
            mask = np.zeros(len(df), dtype=bool)
            for trigger in trigger_list:
                if trigger in df.columns:
                    mask = mask | (df[trigger].values.reshape(-1) == 1)
            events_filtered[year][sample] = df[mask].copy()
            num_sel = mask.sum()
            num_total = len(df)
            print(
                f"Year: {year}, Sample: {sample}, Selected: {num_sel}, Total: {num_total}, Efficiency: {num_sel / num_total:.2%}"
            )
    return events_filtered


trigger_list_high_pt = [
    "AK8PFJet500",
    "AK8PFJet420_MassSD30",
    "AK8PFJet425_SoftDropMass40",
]

trigger_list_PNet = [
    "AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35",
    "AK8PFJet230_SoftDropMass40_PNetBB0p06",
]


events_high_pt = select_triggers(events_combined, trigger_list_high_pt)
events_PNet = select_triggers(events_combined, trigger_list_PNet)
del events_combined  # free memory


# apply trigger sf to events_PNet
if APPLY_TRIGGER_SF:
    print("Applying trigger scale factors to events_PNet...")
    for year in events_PNet.keys():
        for sample in events_PNet[year].keys():
            df = events_PNet[year][sample]
            if sample != "data":
                sf_nom = df["SF_trigger"]
                sf_up = df["SF_trigger_up"]
                sf_down = df["SF_trigger_down"]

                # apply the scale factors to the final weight
                weight = df["finalWeight"]
                df["finalWeight"] = weight * sf_nom
                df["weight_TriggerUp"] = weight * sf_up
                df["weight_TriggerDown"] = weight * sf_down

                # apply the nominal SF to other weights
                for sys_var, up_down in itertools.product(sys_vars, ["Up", "Down"]):
                    weight_name = f"weight_{sys_var}{up_down}"

                    if sys_var == "GenZPt" and weight_name not in df.columns:
                        # skip for samples without this weight
                        continue

                    weight = df[weight_name].values[:, 0]
                    df[weight_name] = weight * sf_nom

            events_PNet[year][sample] = df

    # do the same for the high pT events, but sf=1
    print("Applying trigger scale factors to events_high_pt with 1s")
    for year in events_high_pt.keys():
        for sample in events_high_pt[year].keys():
            df = events_high_pt[year][sample]
            if sample != "data":
                sf_nom = 1

                weight = df["finalWeight"]
                df["finalWeight"] = df["finalWeight"] * sf_nom
                df["weight_TriggerUp"] = 1
                df["weight_TriggerDown"] = 1

                # apply the nominal SF to other weights
                for sys_var, up_down in itertools.product(sys_vars, ["Up", "Down"]):
                    weight_name = f"weight_{sys_var}{up_down}"

                    if sys_var == "GenZPt" and weight_name not in df.columns:
                        # skip for samples without this weight
                        continue

                    weight = df[weight_name].values[:, 0]
                    df[weight_name] = weight * sf_nom

            events_high_pt[year][sample] = df
else:
    print("Trigger scale factors are not applied to events_PNet.")

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
    "JES", 
    "JER", 
    "JMS", 
    "JMR"
    ], ["up", "down"]):
    jshift_keys.append(f"{var}_{ud}")

weight_shifts = {
    "pileup": postprocessing.Syst(
        samples=MC_SAMPLES_FINAL_LIST, label="Pileup", years=list(YEARS_COMBINED_DICT.keys())
    ),
    # "pdf": postprocessing.Syst(samples=sig_keys, label="PDFAcc", years=list(YEARS_COMBINED_DICT.keys())),
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
}

if APPLY_Z_RECOIL_CORR:
    weight_shifts["GenZPt"] = postprocessing.Syst(
        samples=["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ"],
        label="Gen Z pT correction derived from ZMuMu",
        years=list(YEARS_COMBINED_DICT.keys()),
    )

if APPLY_TRIGGER_SF:
    weight_shifts["Trigger"] = postprocessing.Syst(
        samples=MC_SAMPLES_FINAL_LIST,
        label="Trigger SF of the PNet trigger",
        years=list(YEARS_COMBINED_DICT.keys()),
    )

GLOPART_VERSION: int = 3
# GLOPART_VERSION: int = 3

if GLOPART_VERSION == 2:
    tagger_branch = "bbFatJetParT2TXbb"
elif GLOPART_VERSION == 3:
    tagger_branch = "bbFatJetParT3TXbb"
else:
    raise ValueError(f"Invalid GLOPART_VERSION: {GLOPART_VERSION}. Must be 2 or 3.")


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

        if pt_low < 450:
            events = events_PNet[year]
        else:
            events = events_high_pt[year]
        
        for sample_name, df in events.items():
            if sample_name != "data":
                print(sample_name)
                apply_jmsr_smearing_in_templates(
                        df,
                        mass_branch_base= "bbFatJetParT3massCorrectedX2p", #"bbFatJetParT3massGeneric", #"bbFatJetParT3massCorrectedX2p",
                        jet_index=0,
                        seed=42,
                        jms_vals=(0.95, 1.0, 1.05),  # -10%, nominal, +10%
                        jmr_vals=(1.0, 1.05, 1.1),  # -10%, nominal, +10%
                    )

        # Determine the pt and mass variations
        for jshift in jshift_keys:
            if jshift == "":
                pt_branch = "bbFatJetPt0"
                if GLOPART_VERSION == 2:
                    mass_branch = "bbFatJetParT2massVis0"
                elif GLOPART_VERSION == 3:
                    # mass_branch = "bbFatJetParT3massX2p0"
                    mass_branch = "bbFatJetParT3massCorrectedX2p0" #"bbFatJetParT3massCorrectedX2p0"
            elif jshift.startswith("JES") or jshift.startswith("JER"):
                pt_branch = f"bbFatJetPt_{jshift}0"
                # mass_branch = "bbFatJetParTmassVis0"
                if GLOPART_VERSION == 2:
                    mass_branch = "bbFatJetParT2massVis0"
                elif GLOPART_VERSION == 3:
                    # mass_branch = "bbFatJetParT3massX2p0"
                    mass_branch = "bbFatJetParT3massCorrectedX2p0" #"bbFatJetParT3massCorrectedX2p0"
            elif jshift.startswith("JMS") or jshift.startswith("JMR"):
                pt_branch = "bbFatJetPt0"
                if GLOPART_VERSION == 2:
                    mass_branch = f"bbFatJetParT2massVis_{jshift}0"
                elif GLOPART_VERSION == 3:
                    # mass_branch = f"bbFatJetParT3massX2p_{jshift}0"
                    # mass_branch = f"bbFatJetParT3massCorrectedX2p_{jshift}0"
                    mass_branch = f"bbFatJetParT3massCorrectedX2p_{jshift}0"
            else:
                raise ValueError(f"Unknown jshift: {jshift}")

            # Different different pass regions based on TXbb and pT bins
            selection_regions = {}
            for txbb_low, txbb_high in txbb_bins:
                # Convert to strings
                txbb_low_str = str(txbb_low).replace(".", "p")
                txbb_high_str = str(txbb_high).replace(".", "p")
                region_key = f"pass_TXbb{txbb_low_str}to{txbb_high_str}_{pt_bin_key}"

                cutflows = {}
                for sample in events:
                    cutflows[sample] = OrderedDict()
                    cutflows[sample]["Skimmer Preselection"] = events[sample]["finalWeight"].sum()
                    cutflows[sample]["HLT"] = events[sample]["finalWeight"].sum()
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
                    f"{tagger_branch}0": [0.3, min(0.9, min_txbb)],
                },
                label="fail",
            )
            print(f"Selection regions for {year} with jshift {jshift}: {selection_regions.keys()}")

            fit_shape_var = postprocessing.ShapeVar(
                mass_branch,
                r"$m_\mathrm{reg}$ (GeV)",
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
