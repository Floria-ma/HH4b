#one file:
#    python skim_nanotrees.py --sample QCD --input <in.root> --output <out.root>
#all files of a sample on condor, one job per file:
#    python skim_nanotrees.py --sample QCD --submit --nanotrees-dir <dir> --file-pattern "mc/parts/QCD*_tree.root" \
#        --outdir <dir>/mc/skim --jobdir <afs dir for the condor files>

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from HH4b.zbb import nanotrees_loader

# kept on top of nanotrees_loader.read_branches, e.g. to redo the Hbb | Hcc > 0.3 preselection of other productions
EXTRA_BRANCHES = ["ak8_ParT_HccVsQCD", "ak8_2_ParT_HccVsQCD"]

LCG_VIEW = "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh"

def skim_branches(sample: str, extra_branches: list[str] = ()) -> list[str]:
    branches = nanotrees_loader.read_branches(sample) + EXTRA_BRANCHES + list(extra_branches)
    if sample != "data":
        branches.append("nPSWeight")  # counter of PSWeight and PSWeightNorm
    return list(dict.fromkeys(branches))


def skim(
    sample: str,
    in_file: str,
    out_file: str,
    nthreads: int = 1,
    max_entries: int | None = None,
    extra_branches: list[str] = (),
):
    import ROOT

    if nthreads > 1 and max_entries is None:  # RDataFrame.Range does not work with implicit MT
        ROOT.EnableImplicitMT(nthreads)

    f_in = ROOT.TFile.Open(nanotrees_loader.xrootd_url(in_file))
    n_expected = f_in.Get("Events").GetEntries()
    f_in.Close()

    df = ROOT.RDataFrame("Events", nanotrees_loader.xrootd_url(in_file))
    if max_entries is not None:
        df = df.Range(max_entries)
        n_expected = min(max_entries, n_expected)

    opts = ROOT.RDF.RSnapshotOptions()
    opts.fCompressionAlgorithm = ROOT.RCompressionSetting.EAlgorithm.kLZ4  # as the NanoTrees, fast to read
    opts.fCompressionLevel = 4

    branches = skim_branches(sample, extra_branches)
    out = df.Snapshot("Events", out_file, branches, opts)
    n_out = out.Count().GetValue()
    if n_out != n_expected:
        raise RuntimeError(f"{out_file} has {n_out} events, expected {n_expected}")
    print(f"Wrote {n_out} events with {len(branches)} branches to {out_file}")


def submit(
    sample: str,
    nanotrees_dir: str,
    file_pattern: str,
    outdir: str,
    jobdir: str,
    nthreads: int,
    extra_branches: list[str] = (),
):
    files = sorted(Path(nanotrees_dir).glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files in {nanotrees_dir}/{file_pattern}")

    jobdir = Path(jobdir)
    (jobdir / "logs").mkdir(parents=True, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    src_dir = Path(__file__).resolve().parents[2]
    extra_arg = f' --extra-branches "{",".join(extra_branches)}"' if extra_branches else ""
    script = jobdir / "skim.sh"
    script.write_text(
        f"""#!/bin/bash
set -e
source {LCG_VIEW}
export PYTHONPATH={src_dir}:$PYTHONPATH
python {Path(__file__).resolve()} --sample {sample} --input "$1" --output skim.root --nthreads {nthreads}{extra_arg}
xrdcp -f skim.root "{nanotrees_loader.xrootd_url(outdir)}/$2"
rm skim.root
"""
    )
    script.chmod(0o755)

    (jobdir / "files.txt").write_text("".join(f"{f} {f.name}\n" for f in files))
    sub = jobdir / "skim.sub"
    sub.write_text(
        f"""executable = {script}
arguments = $(infile) $(outname)
output = {jobdir}/logs/$(outname).out
error = {jobdir}/logs/$(outname).err
log = {jobdir}/logs/skim.log
request_cpus = {nthreads}
request_memory = {2000 * nthreads}
request_disk = 30GB
+JobFlavour = "workday"
queue infile, outname from {jobdir}/files.txt
"""
    )
    print(f"Submitting {len(files)} jobs writing to {outdir}")
    subprocess.run(["condor_submit", str(sub)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True, help="key of nanotrees_loader.SAMPLE_FILES, sets the branches")
    parser.add_argument("--input", help="NanoTrees file to skim")
    parser.add_argument("--output", help="skimmed output file")
    parser.add_argument("--max-entries", type=int, default=None, help="only skim the first entries, for tests")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--submit", action="store_true", help="skim every file of --file-pattern on condor")
    parser.add_argument("--nanotrees-dir")
    parser.add_argument("--file-pattern", help="glob relative to --nanotrees-dir")
    parser.add_argument("--outdir", help="EOS directory for the skimmed files (condor)")
    parser.add_argument("--jobdir", help="AFS directory for the condor files and logs")
    parser.add_argument(
        "--extra-branches",
        default="",
        help="comma separated branches to keep on top of the ones nanotrees_loader reads",
    )
    args = parser.parse_args()

    extra = [b for b in args.extra_branches.split(",") if b]

    if args.submit:
        submit(
            args.sample, args.nanotrees_dir, args.file_pattern, args.outdir, args.jobdir, args.nthreads, extra
        )
    else:
        skim(args.sample, args.input, args.output, args.nthreads, args.max_entries, extra)


if __name__ == "__main__":
    main()
