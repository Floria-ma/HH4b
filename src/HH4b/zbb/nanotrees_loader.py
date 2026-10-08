
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
    "data": "data/skim/parts*/*.root",
    "Zto2Q": "mc/skim/z2qq_tree.root",
    "Wto2Q": "mc/skim/w2qq_tree.root",
    "ttbar": "mc/skim/ttbar_tree.root",
    "QCD": "mc/skim/qcd*_tree.root",  # HT-binned, merged into qcd1-5 (mc/parts has the unmerged low-HT bins, don't add)
}

# skimmer bbFatJet column, to match the naming convention
JET_BRANCHES = {
    "bbFatJetPt": ("ak8_pt", "ak8_2_pt"),
    "bbFatJetEta": ("ak8_eta", "ak8_2_eta"),
    "bbFatJetPhi": ("ak8_phi", "ak8_2_phi"),
    "bbFatJetMass": ("ak8_mass", "ak8_2_mass"),
    "bbFatJetMsd": ("ak8_sdmass", "ak8_2_sdmass"),
    "bbFatJetScoutParTTXbb": ("ak8_ParT_HbbVsQCD", "ak8_2_ParT_HbbVsQCD"),  # Xbb / (Xbb + QCD)
    "bbFatJetScoutParTmassCorrResonance": ("ak8_ParT_resonanceMass", "ak8_2_ParT_resonanceMass"),
}
# bbFatJet columns the selection uses, always read
SELECTION_JET_COLUMNS = ["bbFatJetPt", "bbFatJetScoutParTTXbb", "bbFatJetMsd", "bbFatJetScoutParTmassCorrResonance"]

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

# scouting Zbb selection of bbbbSkimmer. The cuts below it are off by default (None)
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
    # Normalised PS weights as [ISR up, FSR up, ISR down, FSR down], 1 for events without the 4 weights
    # the same as the original skimmer's
    ps = arrays["PSWeight"] * arrays["PSWeightNorm"]
    has_4 = ak.to_numpy(ak.num(ps) == 4)
    ps = ak.to_numpy(ak.fill_none(ak.pad_none(ps, 4, clip=True), 1.0))
    return np.where(has_4[:, None], ps, 1.0)


def _select_chunk(
    arrays,
    cuts: dict,
    is_data: bool,
    gen: dict,
    mc_lumi: float | None,
    apply_trig_sf: bool,
    jet_columns: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Reorders the jets by TXbb, applies the selection and returns the skimmer-style columns of the passing events"""
    # (leading-pT jet, subleading-pT jet) values. The cuts are on both jets or on the highest-TXbb one, so only
    # the passing events are reordered (the reorder of every event dominated the run time on large data chunks)
    jets = {col: (_to_np(arrays, b0), _to_np(arrays, b1)) for col, (b0, b1) in _jet_branches(jet_columns).items()}

    # order the two jets by TXbb like the bbFatJets in the skimmer. Same as a descending argsort,
    # i.e. the subleading-pT jet first on ties or if its TXbb is NaN
    txbb0, txbb1 = jets["bbFatJetScoutParTTXbb"]
    swap = np.isnan(txbb1) | (txbb1 >= txbb0)

    pt0, pt1 = jets["bbFatJetPt"]
    sel = (
        _to_np(arrays, "passTrigPFHTScouting")
        & (np.where(swap, pt1, pt0) >= cuts["pt_lead"])
        & (np.where(swap, txbb1, txbb0) >= cuts["txbb_lead"])
        & (pt0 >= cuts["pt_subl"])
        & (pt1 >= cuts["pt_subl"])
        & (_to_np(arrays, "dphi_ak8_ak8") >= cuts["dphi"])
        & (_to_np(arrays, "ht") >= cuts["ht"])
    )

    mreg = jets["bbFatJetScoutParTmassCorrResonance"]
    msd = jets["bbFatJetMsd"]
    if cuts.get("met") is not None:
        sel = sel & (_to_np(arrays, "met") < cuts["met"])
    for jet in (0, 1):
        if cuts.get("mreg") is not None:
            sel = sel & (mreg[jet] > cuts["mreg"][0]) & (mreg[jet] < cuts["mreg"][1])
        if cuts.get("msd") is not None:
            sel = sel & (msd[jet] > cuts["msd"][0]) & (msd[jet] < cuts["msd"][1])
        if cuts.get("msd_mreg_ratio") is not None:
            shifted = msd[jet] + cuts["msd_mreg_shift"]
            sel = (
                sel
                & (shifted > cuts["msd_mreg_ratio"][0] * mreg[jet])
                & (shifted < cuts["msd_mreg_ratio"][1] * mreg[jet])
            )

    if not np.any(sel):
        return None

    swap = swap[sel]
    order = np.where(swap[:, None], [[1, 0]], [[0, 1]])
    out = {}
    for col, (vals0, vals1) in jets.items():
        vals0, vals1 = vals0[sel], vals1[sel]
        out[col] = np.stack([np.where(swap, vals1, vals0), np.where(swap, vals0, vals1)], axis=1)

    if is_data:
        out["weight"] = np.ones((np.sum(sel), 1))
        return out

    arrays = arrays[sel]

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
    # per-event xsec * gen weight without lumi/PU/trigSF, for filtering low-stat high-weight events
    out["xsecGenWeight"] = (_to_np(arrays, "xsecWeight") * _to_np(arrays, "genWeight"))[:, None]

    for col, branch in gen.items():
        if col == "dr_vdaus":
            dr = np.take_along_axis(_jet_pair(arrays, branch), order, axis=1)
            out["bbFatJetVQQMatch"] = dr < VQQ_MATCH_DR
        else:
            out[col] = _to_np(arrays, branch)[:, None]

    return out


def xrootd_url(path) -> str:
    return f"root://eosuser.cern.ch/{path}" if str(path).startswith("/eos/user/") else str(path)


def _jet_branches(jet_columns: list[str] | None = None) -> dict[str, tuple[str, str]]:
    if jet_columns is None:
        return JET_BRANCHES
    keep = {*jet_columns, *SELECTION_JET_COLUMNS}
    return {col: branches for col, branches in JET_BRANCHES.items() if col in keep}


def read_branches(sample: str, jet_columns: list[str] | None = None) -> list[str]:
    branches = [b for pair in _jet_branches(jet_columns).values() for b in pair]
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
    jet_columns: list[str] | None = None,
):
    """
    Reads the NanoTrees of a sample one chunk at a time, for samples too large to hold in memory (e.g. QCD).
    Arguments as in load_nanotrees, plus file_pattern: glob of the sample files relative to nanotrees_dir,
    overriding SAMPLE_FILES (e.g. to read mc/parts of a production whose merged files are incomplete),
    and jet_columns: bbFatJet columns to read and return besides the ones the selection needs (None: all).

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

    branches = read_branches(sample, jet_columns)
    if cuts.get("met") is not None:
        branches = [*branches, "met"]

    print(f"Loading {sample} from {len(files)} NanoTrees file(s)")
    for f in files:
        for arrays in _iterate_file(xrootd_url(f), branches, step_size):
            yield len(arrays), _select_chunk(arrays, cuts, is_data, gen, mc_lumi, apply_trig_sf, jet_columns)


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
    file_patterns: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    events_dict = {}

    for sample in samples:
        is_data = sample == "data"

        chunks = []
        n_read = 0
        for n_chunk, chunk in iterate_nanotrees(
            nanotrees_dir,
            sample,
            selection,
            mc_lumi,
            apply_trig_sf,
            step_size,
            file_pattern=(file_patterns or {}).get(sample),
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
