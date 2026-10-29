# Pre-fit plots of the pass regions with the estimated QCD of qcd_estimation.py stacked with the MC, against data.
# Reads templates_{year}_{pT bin}.root of scouting_templates.py after qcd_estimation.py wrote qcd_{region} into them:
#    python plot_with_qcd.py --basedir /eos/user/z/zima/scouting/templates/nanotrees_tagger_Cali_2024_had_vh

from __future__ import annotations

import argparse
import re
from pathlib import Path

import hist
import numpy as np
import uproot
from hist import Hist

from HH4b import plotting

MC_KEYS = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar", "singletop"]  # bkg_keys of scouting_templates.py
# QCD is the MC QCD, while qcd is the estimated
QCD_KEY = "qcd"
DATA_KEY = "data"
BG_ORDER = [QCD_KEY, "Wto2Q", "Zto2Q_QQ", "Zto2Q_CC", "ttbar", "singletop", "Zto2Q_BB"]  # bottom to top of the stack
BLIND_WINDOW = (100, 150)  # BLIND_WINDOW of scouting_templates.py, data hidden there
MASS_LABEL = r"$m_\mathrm{res}$ (GeV)"
# sig_keys of scouting_templates.py, plotted unstacked (step) and scaled by SIG_SCALE
SIG_KEYS = ["hbb"]
SIG_SCALE = 50
# samples with their yield in the legend
ZBB_KEY = "Zto2Q_BB"
YIELD_LABELS = {key: plotting.label_by_sample.get(key, key) for key in [*SIG_KEYS, ZBB_KEY]}


def region_hist(f, region: str, mc_keys: list[str]) -> Hist:
    """Hist with a Sample axis (data, MC, qcd) of one region, the format of the template pickles"""
    data = f[f"{DATA_KEY}_{region}"]
    edges = data.axes[0].edges()
    h = Hist(
        hist.axis.StrCategory([DATA_KEY, *mc_keys, QCD_KEY], name="Sample"),
        hist.axis.Variable(edges, name="mass", label=MASS_LABEL),
        storage="weight",
    )
    view = h.view(flow=False)
    for i, sample in enumerate(h.axes[0]):
        th1 = f[f"{sample}_{region}"]
        view.value[i] = th1.values()
        view.variance[i] = th1.variances()

    # the pass data is zeroed in the blinded window: hide it instead of plotting zeros
    centers = h.axes[1].centers
    blind = (centers > BLIND_WINDOW[0]) & (centers < BLIND_WINDOW[1])
    view.value[0, blind] = np.nan
    view.variance[0, blind] = np.nan
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, help="output directory of scouting_templates.py")
    parser.add_argument("--year", default="2024All")
    parser.add_argument("--outdir", default=None, help="default: <basedir>/plots_withQCD")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(args.basedir) / "plots_withQCD"
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.basedir).glob(f"templates_{args.year}_pT*.root"))
    if not files:
        raise FileNotFoundError(f"No templates_{args.year}_pT*.root in {args.basedir}")

    for fname in files:
        f = uproot.open(fname)
        names = {key.split(";")[0] for key in f.keys()}
        # nominal pass regions with an estimated QCD
        regions = sorted(
            m.group(1)
            for name in names
            if (m := re.fullmatch(r"data_(pass_TXbb.*_pT\d+to\d+)", name)) and f"{QCD_KEY}_{m.group(1)}" in names
        )
        if not regions:
            print(f"No pass regions with {QCD_KEY}_<region> in {fname}, run qcd_estimation.py first")
            continue

        for region in regions:
            # templates of an older scouting_templates.py run can miss a sample (e.g. singletop)
            mc_keys = [key for key in MC_KEYS if f"{key}_{region}" in names]
            sig_keys = [key for key in SIG_KEYS if f"{key}_{region}" in names]
            if missing := set(MC_KEYS + SIG_KEYS) - set(mc_keys + sig_keys):
                print(f"Warning: no {sorted(missing)} templates for {region} in {fname}, plotting without them")
            hists = region_hist(f, region, mc_keys + sig_keys)
            # signal (before the SIG_SCALE scaling) and Zbb yields in the legend (ratioHistPlot takes the labels
            # from label_by_sample)
            for key in [k for k in YIELD_LABELS if k in sig_keys or k in mc_keys]:
                n = hists[key, :].sum().value
                plotting.label_by_sample[key] = f"{YIELD_LABELS[key]} ({n:.1f} events)"
                print(f"  {region} {key}: {n:.1f} events")
            plotting.ratioHistPlot(
                hists=hists,
                year=args.year,
                sig_keys=sig_keys,
                sig_scale_dict={key: SIG_SCALE for key in sig_keys},
                bg_keys=[*mc_keys, QCD_KEY],
                bg_order=[key for key in BG_ORDER if key in [*mc_keys, QCD_KEY]],
                title=f"{region} Region Pre-Fit Shapes",
                name=str(outdir / f"{region}_withQCD.pdf"),
                show=False,
                energy="13.6",
                ratio_ylims = [0.5, 1.5]
            )
            print(f"Saved {outdir / f'{region}_withQCD.pdf'}")


if __name__ == "__main__":
    main()
