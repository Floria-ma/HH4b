
# load the VHcc had channel samples (NanoTrees)
# The NanoTrees store the two leading-pT AK8 jets (ak8_*, ak8_2_*). Here they are reordered by TXbb like the
# bbFatJets in the original skimmer, and the scouting Zbb selection of bbbbSkimmer is applied on top of the NanoTrees
# "had" preselection (>= 2 AK8 jets, pT > 200 / 170, |dphi| > 2.2, MET < 150, no leptons, golden JSON).


from __future__ import annotations

import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from HH4b.hh_vars import jmsr_shifts

# NanoTrees output files of each template sample, relative to the NanoTrees output directory
SAMPLE_FILES = {
    "data": "data/parts/*.root",
    "Zto2Q": "mc/z2qq_tree.root",
    "Wto2Q": "mc/w2qq_tree.root",
    "ttbar": "mc/ttbar_tree.root",
    "QCD": "mc/qcd*_tree.root",  # HT-binned, merged into qcd1-5 (mc/parts has the unmerged low-HT bins, don't add)
}

# skimmer bbFatJet column: (leading-pT jet branch, subleading-pT jet branch)
JET_BRANCHES = {
    "bbFatJetPt": ("ak8_pt", "ak8_2_pt"),
    "bbFatJetEta": ("ak8_eta", "ak8_2_eta"),
    "bbFatJetPhi": ("ak8_phi", "ak8_2_phi"),
    "bbFatJetMass": ("ak8_mass", "ak8_2_mass"),
    "bbFatJetMsd": ("ak8_sdmass", "ak8_2_sdmass"),
    "bbFatJetScoutParTTXbb": ("ak8_ParT_HbbVsQCD", "ak8_2_ParT_HbbVsQCD"),  # Xbb / (Xbb + QCD)
    "bbFatJetScoutParTmassCorrResonance": ("ak8_ParT_resonanceMass", "ak8_2_ParT_resonanceMass"),
}

# skimmer gen column: NanoTrees branch. "dr_vdaus" is the max dR(jet, V daughter) of each jet, used for bbFatJetVQQMatch
GEN_BRANCHES = {
    "Zto2Q": {
        "dr_vdaus": ("dr_ak8_zdaus", "dr_ak8_2_zdaus"),
        "GenZPt": "genZ_pt",
        "GenZBB": "z2bb",
        "GenZCC": "z2cc",
    },
    "Wto2Q": {
        "dr_vdaus": ("dr_ak8_wdaus", "dr_ak8_2_wdaus"),
        "GenWPt": "genW_pt",
        "GenWCS": "w2cq",
        "GenWUD": "w2qq",
    },
}

MC_WEIGHT_BRANCHES = [
    "xsecWeight",  # xsec * 1000 / sum(genWeight), i.e. per fb^-1
    "genWeight",
    "lumiwgt",  # fb^-1
    "puWeight",
    "puWeightUp",
    "puWeightDown",
    "PSWeight",
    "PSWeightNorm",
    "trigSF",
]

# scouting Zbb selection of bbbbSkimmer. The cuts below it are off by default (None), see VH_HAD_SELECTION
DEFAULT_SELECTION = {
    "pt_lead": 300,  # highest-TXbb jet
    "txbb_lead": 0.3,  # highest-TXbb jet
    "pt_subl": 200,  # both jets
    "dphi": np.pi / 2,  # NanoTrees already require > 2.2
    "ht": 600,  # NanoTrees ht sums all scouting AK4 jets, the skimmer only the good_ak4jets
    "met": None,  # needs the met branch, which the skims of skim_nanotrees.py do not keep
    "mreg": None,  # (min, max) regressed mass of both jets
    "msd": None,  # (min, max) soft drop mass of both jets
    "msd_mreg_ratio": None,  # (min, max) of (msd + msd_mreg_shift) / regressed mass, for both jets
    "msd_mreg_shift": 10.0,
}

# common_selections of the VH(bb/cc) hadronic analysis:
# PlotTools/VH/cards/rDataframeScripts/vh_had.py. Its cuts are the same for the H and the V jet, so they do
# not need the Hidx assignment of the analysis (H: the jet of the H/V pairing with the larger
# H_ParT_HVsQCD * V_ParT_VVsQCD) and are applied to both jets here.
# Left out: met < 150, already applied to the had channel by the NanoTrees producer (the trees have met < 150
# throughout), so that the selection also runs on skims without the met branch.
# The analysis applies no pT or HT cut in its common selection (its categories cut on H_pt at 500 GeV)
VH_HAD_SELECTION = {
    "pt_lead": 0,
    "txbb_lead": 0.0,
    "pt_subl": 0,
    "ht": 0,
    "dphi": 2.7,
    "mreg": None,
    "msd": (30, 190),
    "msd_mreg_ratio": (0.8, 1.2),
}

# same as bbFatJetVQQMatch in GenSelection.py: both V daughters within dR < 0.8 of the jet
VQQ_MATCH_DR = 0.8


def _to_np(arrays, branch: str) -> np.ndarray:
    return ak.to_numpy(arrays[branch])


def _jet_pair(arrays, branches: tuple[str, str]) -> np.ndarray:
    return np.stack([_to_np(arrays, branches[0]), _to_np(arrays, branches[1])], axis=1)


def _ps_weights(arrays) -> np.ndarray:
    """Normalised PS weights as [ISR up, FSR up, ISR down, FSR down], 1 for events without the 4 weights"""
    ps = arrays["PSWeight"] * arrays["PSWeightNorm"]
    has_4 = ak.to_numpy(ak.num(ps) == 4)
    ps = ak.to_numpy(ak.fill_none(ak.pad_none(ps, 4, clip=True), 1.0))
    return np.where(has_4[:, None], ps, 1.0)


def _select_chunk(
    arrays, cuts: dict, is_data: bool, gen: dict, mc_lumi: float | None, apply_trig_sf: bool
) -> dict[str, np.ndarray] | None:
    """Reorders the jets by TXbb, applies the selection and returns the skimmer-style columns of the passing events"""
    jets = {col: _jet_pair(arrays, branches) for col, branches in JET_BRANCHES.items()}

    # order the two jets by TXbb like the bbFatJets in the skimmer
    order = np.argsort(jets["bbFatJetScoutParTTXbb"], axis=1)[:, ::-1]
    jets = {col: np.take_along_axis(vals, order, axis=1) for col, vals in jets.items()}

    pt = jets["bbFatJetPt"]
    sel = (
        _to_np(arrays, "passTrigPFHTScouting")
        & (pt[:, 0] >= cuts["pt_lead"])
        & (jets["bbFatJetScoutParTTXbb"][:, 0] >= cuts["txbb_lead"])
        & np.all(pt >= cuts["pt_subl"], axis=1)
        & (_to_np(arrays, "dphi_ak8_ak8") >= cuts["dphi"])
        & (_to_np(arrays, "ht") >= cuts["ht"])
    )

    mreg = jets["bbFatJetScoutParTmassCorrResonance"]
    msd = jets["bbFatJetMsd"]
    if cuts.get("met") is not None:
        sel = sel & (_to_np(arrays, "met") < cuts["met"])
    if cuts.get("mreg") is not None:
        sel = sel & np.all((mreg > cuts["mreg"][0]) & (mreg < cuts["mreg"][1]), axis=1)
    if cuts.get("msd") is not None:
        sel = sel & np.all((msd > cuts["msd"][0]) & (msd < cuts["msd"][1]), axis=1)
    if cuts.get("msd_mreg_ratio") is not None:
        shifted = msd + cuts["msd_mreg_shift"]
        sel = sel & np.all(
            (shifted > cuts["msd_mreg_ratio"][0] * mreg) & (shifted < cuts["msd_mreg_ratio"][1] * mreg),
            axis=1,
        )

    if not np.any(sel):
        return None

    out = {col: vals[sel] for col, vals in jets.items()}

    if is_data:
        out["weight"] = np.ones((np.sum(sel), 1))
        return out

    arrays = arrays[sel]
    order = order[sel]

    lumi = mc_lumi if mc_lumi is not None else _to_np(arrays, "lumiwgt")
    base = _to_np(arrays, "xsecWeight").astype(np.float64) * _to_np(arrays, "genWeight") * lumi
    if apply_trig_sf:
        base = base * _to_np(arrays, "trigSF")

    weight = base * _to_np(arrays, "puWeight")
    ps = _ps_weights(arrays)
    shifted = {
        "weight": weight,
        "weight_pileupUp": base * _to_np(arrays, "puWeightUp"),
        "weight_pileupDown": base * _to_np(arrays, "puWeightDown"),
        "weight_ISRPartonShowerUp": weight * ps[:, 0],
        "weight_FSRPartonShowerUp": weight * ps[:, 1],
        "weight_ISRPartonShowerDown": weight * ps[:, 2],
        "weight_FSRPartonShowerDown": weight * ps[:, 3],
    }
    for col, vals in shifted.items():
        out[col] = vals[:, None]

    for col, branch in gen.items():
        if col == "dr_vdaus":
            dr = np.take_along_axis(_jet_pair(arrays, branch), order, axis=1)
            out["bbFatJetVQQMatch"] = dr < VQQ_MATCH_DR
        else:
            out[col] = _to_np(arrays, branch)[:, None]

    return out


def xrootd_url(path) -> str:
    """EOS user files through xrootd: reading a few branches through the fuse mount is latency bound, ~5x slower"""
    return f"root://eosuser.cern.ch/{path}" if str(path).startswith("/eos/user/") else str(path)


def read_branches(sample: str) -> list[str]:
    """NanoTrees branches read for a sample (also the ones skim_nanotrees.py keeps)"""
    branches = [b for pair in JET_BRANCHES.values() for b in pair]
    branches += ["ht", "dphi_ak8_ak8", "passTrigPFHTScouting"]
    if sample != "data":
        branches += MC_WEIGHT_BRANCHES
        for branch in GEN_BRANCHES.get(sample, {}).values():
            branches += list(branch) if isinstance(branch, tuple) else [branch]
    return branches


def iterate_nanotrees(
    nanotrees_dir: str,
    sample: str,
    selection: dict | None = None,
    mc_lumi: float | None = None,
    apply_trig_sf: bool = False,
    step_size: str = "500 MB",
    file_pattern: str | None = None,
):
    """
    Reads the NanoTrees of a sample one chunk at a time, for samples too large to hold in memory (e.g. QCD).
    Arguments as in load_nanotrees, plus file_pattern: glob of the sample files relative to nanotrees_dir,
    overriding SAMPLE_FILES (e.g. to read mc/parts of a production whose merged files are incomplete).

    Yields:
        (number of events read, skimmer-style columns of the passing events or None if none pass) per chunk
    """
    cuts = {**DEFAULT_SELECTION, **(selection or {})}

    if file_pattern is None:
        file_pattern = SAMPLE_FILES[sample]
    files = sorted(Path(nanotrees_dir).glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No NanoTrees for {sample} in {nanotrees_dir}/{file_pattern}")

    is_data = sample == "data"
    gen = {} if is_data else GEN_BRANCHES.get(sample, {})

    branches = read_branches(sample)
    if cuts.get("met") is not None:
        branches = [*branches, "met"]

    print(f"Loading {sample} from {len(files)} NanoTrees file(s)")
    for f in files:
        for arrays in _iterate_file(xrootd_url(f), branches, step_size):
            yield len(arrays), _select_chunk(arrays, cuts, is_data, gen, mc_lumi, apply_trig_sf)


def _iterate_file(path: str, branches: list[str], step_size: str, retries: int = 5):
    """Iterates over the Events of a file, resuming from the last chunk read after transient (xrootd) read errors"""
    entry_start = 0
    for attempt in range(retries + 1):
        try:
            # uproot's default xrootd timeout of 30 s expires on large chunks when EOS is busy
            with uproot.open(path, timeout=300) as f:
                for arrays in f["Events"].iterate(
                    branches, entry_start=entry_start, step_size=step_size, library="ak"
                ):
                    entry_start += len(arrays)
                    yield arrays
            return
        except OSError as e:
            if attempt == retries:
                raise
            warnings.warn(f"{e}\nretrying {path} from entry {entry_start}", stacklevel=1)


def load_nanotrees(
    nanotrees_dir: str,
    samples: list[str],
    selection: dict | None = None,
    jmsr_vars: list[str] = (),
    mc_lumi: float | None = None,
    apply_trig_sf: bool = False,
    step_size: str = "500 MB",
) -> dict[str, pd.DataFrame]:
    """
    Args:
        nanotrees_dir: NanoTrees output directory, containing data/ and mc/
        samples: template samples to load (keys of SAMPLE_FILES)
        selection: cuts overriding DEFAULT_SELECTION
        jmsr_vars: mass columns to add JMS/JMR variations for in MC, as copies of the nominal to be
            filled by the template script's smearing (the NanoTrees have no mass variations)
        mc_lumi: luminosity (fb^-1) to normalise MC to. None uses lumiwgt of the trees (full 2024),
            which is only right if the data covers all of 2024
        apply_trig_sf: multiply MC weights by the NanoTrees trigSF (bbbbSkimmer applies none for scouting)

    Returns:
        {sample: DataFrame} with (column, index) multi-index columns like utils.load_samples,
        and MC weights already normalised to xsec * lumi.
    """
    events_dict = {}

    for sample in samples:
        is_data = sample == "data"

        chunks = []
        n_read = 0
        for n_chunk, chunk in iterate_nanotrees(
            nanotrees_dir, sample, selection, mc_lumi, apply_trig_sf, step_size
        ):
            n_read += n_chunk
            if chunk is not None:
                chunks.append(chunk)

        if not chunks:
            warnings.warn(f"No {sample} events pass the selection!", stacklevel=1)
            continue

        columns = {col: np.concatenate([chunk[col] for chunk in chunks]) for col in chunks[0]}
        if not is_data:
            for var in jmsr_vars:
                for shift in jmsr_shifts:
                    columns[f"{var}_{shift}"] = columns[var].copy()

        # same layout as SkimmerABC.to_pandas
        events = pd.concat(
            [pd.DataFrame(vals) for vals in columns.values()], axis=1, keys=list(columns.keys())
        )
        events["finalWeight"] = events["weight"]

        print(
            f"  {sample}: {len(events)} / {n_read} events pass, sum of weights {events['finalWeight'].to_numpy().sum():.1f}"
        )
        events_dict[sample] = events

    return events_dict
