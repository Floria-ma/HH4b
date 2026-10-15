# Copies a NanoTrees file keeping only the branches nanotrees_loader reads (all events kept):
# python skim_nanotrees.py --sample QCD --input <in.root> --output <out.root>

from __future__ import annotations

import argparse

import ROOT

from HH4b.zbb import nanotrees_loader

# kept on top of nanotrees_loader.read_branches, e.g. to redo the Hbb | Hcc > 0.3 preselection of other productions
EXTRA_BRANCHES = ["ak8_ParT_HccVsQCD", "ak8_2_ParT_HccVsQCD"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="key of nanotrees_loader.SAMPLE_FILES, sets the branches")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    branches = nanotrees_loader.read_branches(args.sample) + EXTRA_BRANCHES
    if args.sample != "data":
        branches.append("nPSWeight")  # counter of PSWeight and PSWeightNorm

    df = ROOT.RDataFrame("Events", nanotrees_loader.xrootd_url(args.input))
    df.Snapshot("Events", args.output, list(dict.fromkeys(branches)))


if __name__ == "__main__":
    main()
