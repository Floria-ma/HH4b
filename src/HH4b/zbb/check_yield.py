from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from HH4b.hh_vars import LUMI

SAMPLE_FILES = {
    "2023": ("zqq_2023_full/Zto2Q-4Jets_HT-*", "*/nano_skim_*.root", "*/outfiles/*.pkl"),
    "2024": ("skimmer/02Nov2026_2024_v15_scouting_zbb/2024/Zto2Q-4Jets_*", "root/*.root", "pickles/*.pkl"),
}

branches = {
    "bbFatJetPt": ("bbFatJetPt0", "bbFatJetPt1"),
    "bbFatJetEta": ("bbFatJetEta0", "bbFatJetEta1"),
    "bbFatJetPhi": ("bbFatJetPhi0", "bbFatJetPhi1"),
    "bbFatJetMass": ("bbFatJetMass0", "bbFatJetMass1"),
    "bbFatJetMsd": ("bbFatJetMsd0", "bbFatJetMsd1"),
    "bbFatJetScoutParTmassCorrectedX2p": ("bbFatJetScoutParTmassCorrectedX2p0", "bbFatJetScoutParTmassCorrectedX2p1"),
    "bbFatJetScoutParTTXbb": ("bbFatJetScoutParTTXbb0", "bbFatJetScoutParTTXbb1"),  # Xbb / (Xbb + QCD)
    "bbFatJetScoutParTmassCorrResonance": ("bbFatJetScoutParTmassCorrResonance0", "bbFatJetScoutParTmassCorrResonance1"),
}
# per-event branches (GenZBB is one flag per event, not per jet)
event_branches = ["weight", "ht", "GenZBB"]

# branch: (x label, bins); jets are ordered by TXbb, so jet 0 is the one the TXbb >= 0.3 cut is applied to
PLOTS = {
    "bbFatJetPt0": (r"Leading-$T_{Xbb}$ jet $p_T$ (GeV)", np.linspace(300, 1200, 37)),
    "bbFatJetPt1": (r"Second jet $p_T$ (GeV)", np.linspace(200, 1000, 33)),
    "bbFatJetEta0": (r"Leading-$T_{Xbb}$ jet $\eta$", np.linspace(-2.5, 2.5, 26)),
    "bbFatJetPhi0": (r"Leading-$T_{Xbb}$ jet $\phi$", np.linspace(-np.pi, np.pi, 26)),
    "bbFatJetMass0": (r"Leading-$T_{Xbb}$ jet mass (GeV)", np.linspace(0, 250, 51)),
    "bbFatJetMsd0": (r"Leading-$T_{Xbb}$ jet $m_\mathrm{SD}$ (GeV)", np.linspace(0, 250, 51)),
    "bbFatJetScoutParTmassCorrectedX2p0": (r"Leading-$T_{Xbb}$ jet $m_\mathrm{X2p}$ (GeV)", np.linspace(0, 250, 51)),
    "bbFatJetScoutParTmassCorrResonance0": (r"Leading-$T_{Xbb}$ jet $m_\mathrm{res}$ (GeV)", np.linspace(0, 250, 51)),
    "bbFatJetScoutParTTXbb0": (r"Leading-$T_{Xbb}$ jet $T_{Xbb}$", np.linspace(0.3, 1.0, 36)),
    "ht": (r"$H_T$ (GeV)", np.linspace(600, 2500, 39)),
}
# per-year stack components, bottom to top: (label, fill colour, GenZBB value)
COMPONENTS = [
    (r"Z$\to$qq (non-bb)", "blue", False),
    (r"Z$\to$bb", "orange", True),
]
COLORS = {"2023": "#2a78d6", "2024": "#eb6834"}
STYLES = {"2023": "-", "2024": "--"}  # so the years are not told apart by colour only

parser = argparse.ArgumentParser("plot yields")
parser.add_argument("--basedir", help="Input directory", default="/eos/user/z/zima/bbbb/")
parser.add_argument("--outdir", help="default: <basedir>/plots_zqq_2023_vs_2024", default=None)
parser.add_argument("--shape", action="store_true", help="normalise each year to unit area instead of the 2024 lumi")
args = parser.parse_args()

# scale other years to 2024 lumi
def lumi_scale(year):
    if year not in LUMI:
        raise ValueError(f"Unknown year: {year}")
    return LUMI["2024"] / LUMI[year]

def np_nominal(pkl: Path) -> float:
    with pkl.open("rb") as f:
        out = pickle.load(f)
    # {year: {dataset: {"totals": ..., "cutflow": ...}}}
    return next(iter(next(iter(out.values())).values()))["totals"]["np_nominal"]


def load(year: str) -> dict[str, np.ndarray]:
    bins_glob, root_glob, pkl_glob = SAMPLE_FILES[year]
    read = [*event_branches, *[b for pair in branches.values() for b in pair]]
    arrays = []
    for d in sorted(Path(args.basedir).glob(bins_glob)):
        pkls = sorted(d.glob(pkl_glob))
        # condor jobs with 0 selected events write a ROOT file without an Events tree
        roots = [r for r in sorted(d.glob(root_glob)) if "Events" in uproot.open(r)]
        if not roots:
            print(f"{year} {d.name}: no events selected ({len(pkls)} pickles)")
            continue
        np_sum = sum(np_nominal(p) for p in pkls)
        a = uproot.concatenate([f"{r}:Events" for r in roots], read, library="np")
        a["finalWeight"] = a["weight"] / np_sum * lumi_scale(year)
        print(f"{year} {d.name}: {len(pkls)} jobs, {len(a['weight'])} events, yield {a['finalWeight'].sum():.4g}")
        arrays.append(a)
    events = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    events["GenZBB"] = events["GenZBB"].astype(bool)
    return events


plt.style.use(hep.style.CMS)
outdir = Path(args.outdir) if args.outdir else Path(args.basedir) / "plots_zqq_2023_vs_2024"
outdir.mkdir(parents=True, exist_ok=True)

events = {year: load(year) for year in SAMPLE_FILES}
yields = {year: ev["finalWeight"].sum() for year, ev in events.items()}
print(f"\nZ->qq at {LUMI['2024'] / 1e3:.1f} fb^-1: "
      + ", ".join(f"{year} {y:.4g}" for year, y in yields.items())
      + f", 2024/2023 = {yields['2024'] / yields['2023']:.3f}")

def save(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight")
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png")


suffix = "_shape" if args.shape else ""
ylabel = "Normalised to unit area" if args.shape else "Events / bin"
for var, (xlabel, bins) in PLOTS.items():
    # 2023 vs 2024 step histograms with ratio panel
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07}, sharex=True
    )
    hists = {}
    for year, ev in events.items():
        w = ev["finalWeight"]
        h, _ = np.histogram(ev[var], bins=bins, weights=w)
        err = np.sqrt(np.histogram(ev[var], bins=bins, weights=w**2)[0])
        norm = h.sum() if args.shape else 1.0
        hists[year] = (h / norm, err / norm)
        hep.histplot(
            h / norm, bins, yerr=err / norm, histtype="step", color=COLORS[year], ls=STYLES[year], lw=2,
            ax=ax, label=f"{year} ({yields[year]:.3g} events)",
        )
    (a, ea), (b, eb) = hists["2023"], hists["2024"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b / a
        er = r * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)
    rax.errorbar(0.5 * (bins[1:] + bins[:-1]), r, yerr=er, fmt="o", ms=4, color="black")
    rax.axhline(1, color="gray", ls=":")
    rax.set_ylim(0, 2)
    rax.set_ylabel("2024 / 2023")
    rax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    hep.cms.label(ax=ax, data=False, year="2024", lumi=LUMI["2024"] / 1e3, fontsize=16)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)  # headroom for the legend
    ax.legend(title=r"Z$\to$qq, 2023 scaled to 2024 lumi")
    save(fig, outdir / f"{var}{suffix}")

    # one stacked Zbb / non-Zbb plot per year
    for year, ev in events.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        w = ev["finalWeight"]
        norm = np.histogram(ev[var], bins=bins, weights=w)[0].sum() if args.shape else 1.0
        stack, labels, colors = [], [], []
        for label, color, is_bb in COMPONENTS:
            sel = ev["GenZBB"] == is_bb
            stack.append(np.histogram(ev[var][sel], bins=bins, weights=w[sel])[0] / norm)
            labels.append(f"{label} ({w[sel].sum():.3g} events)")
            colors.append(color)
        hep.histplot(stack, bins, stack=True, histtype="fill", color=colors, label=labels, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        hep.cms.label(ax=ax, data=False, year="2024", lumi=LUMI["2024"] / 1e3, fontsize=16)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
        ax.legend(title=f"{year}" + ("" if year == "2024" else ", scaled to 2024 lumi"))
        save(fig, outdir / f"{var}_stack_{year}{suffix}")
