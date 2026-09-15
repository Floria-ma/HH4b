"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
import warnings
import gc
from math import ceil
from pathlib import Path
from string import Template
import subprocess
import time

from HH4b import run_utils

from concurrent.futures import ThreadPoolExecutor

def condor_submit(jdl):
    subprocess.run(["condor_submit", jdl], check=True)

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
    "ucsd": "root://redirector.t2.ucsd.edu:1095//",
    "cern": "root://eosuser.cern.ch//",
}


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        f.write(templ.substitute(templ_args))


def process_subsample(args, sample, subsample, tot_files, local_dir, outdir, t2_prefixes, proxy, tag):
    """Process a single subsample with batching and subdirectory organization"""
    print(f"\nProcessing {subsample} with {tot_files} files")
    
    subsample_base_dir = local_dir / subsample
    subsample_base_dir.mkdir(parents=True, exist_ok=True)
    
    sample_dir = outdir / args.year / subsample
    njobs = ceil(tot_files / args.files_per_job)
    
    print(f"  Total jobs needed: {njobs} ({args.files_per_job} files per job)")
    
    # Organize jobs into subdirectories to avoid too many files in one directory
    # This actually became a problem
    jobs_per_subdir = 1000 
    
    nsubmit_local = 0
    total_subdirs = ceil(njobs / jobs_per_subdir)
    
    for subdir_num in range(total_subdirs):
        start_job = subdir_num * jobs_per_subdir
        end_job = min((subdir_num + 1) * jobs_per_subdir, njobs)
        
        # Create subdirectory for this batch of jobs
        subdir_name = f"{start_job:04d}-{end_job-1:04d}"
        subsample_dir = subsample_base_dir / subdir_name
        subsample_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing jobs {start_job} to {end_job-1} in directory {subdir_name}...")
        
        batch_submissions = []
        
        for j in range(start_job, end_job):
            if args.test and j == 2:
                break
            
            prefix = f"{args.year}_{subsample}"
            localcondor = subsample_dir / f"{prefix}_{j}.jdl"
            jdl_args = {"dir": subsample_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
            write_template(jdl_templ, localcondor, jdl_args)
            
            localsh = subsample_dir / f"{prefix}_{j}.sh"
            sh_args = {
                "branch": args.git_branch,
                "gituser": args.git_user,
                "script": args.script,
                "year": args.year,
                "starti": j * args.files_per_job,
                "endi": (j + 1) * args.files_per_job,
                "sample": sample,
                "subsample": subsample,
                "processor": args.processor,
                "maxchunks": args.maxchunks,
                "chunksize": args.chunksize,
                "t2_prefixes": " ".join(t2_prefixes),
                "outdir": sample_dir,
                "jobnum": j,
                "nano_version": args.nano_version,
                "save_root": ("--save-root" if args.save_root else "--no-save-root"),
                "txbb": args.txbb,
                "save_systematics": (
                    "--save-systematics" if args.save_systematics else "--no-save-systematics"
                ),
                "region": f"--region {args.region}" if "skimmer" in args.processor else "",
                "use_scouting": (
                    "--use-scouting" if args.use_scouting else "--no-use-scouting"
                ),
            }
            write_template(sh_templ, localsh, sh_args)
            os.system(f"chmod u+x {localsh}")
            
            if Path(f"{localcondor}.log").exists():
                Path(f"{localcondor}.log").unlink()
            
            batch_submissions.append(str(localcondor))
            nsubmit_local += 1
        
        # Submit this batch if requested
        if args.submit and batch_submissions:
            print(f"Submitting {len(batch_submissions)} jobs from directory {subdir_name}...")
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                executor.map(condor_submit, batch_submissions)
            
            print(f"Submitted batch {subdir_num + 1}/{total_subdirs}")
        
        batch_submissions.clear()
        gc.collect()
        
        time.sleep(0.1)
    
    return nsubmit_local


def main(args):
    run_utils.check_branch(args.git_branch, args.git_user, args.allow_diff_local_repo)
    username = os.environ["USER"]

    if args.site == "lpc" or args.site == "cern":
        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    elif args.site == "ucsd": 
        if username == "rkansal":
            proxy = "/home/users/rkansal/x509up_u31735"
        elif username == "dprimosc":
            proxy = "/tmp/x509up_u150012"  # "/home/users/dprimosc/x509up_u150012"
        elif username == "woodson":
            proxy = "/home/users/woodson/x509up_u31135"
    else:
        raise ValueError(f"Invalid site {args.site}")

    if args.site not in args.save_sites:
        print(args.save_sites)
        warnings.warn(
            f"Your local site {args.site} is not in save sites {args.save_sites}!", stacklevel=1
        )

    t2_prefixes = [t2_redirectors[site] for site in args.save_sites]

    tag = f"{args.tag}_{args.nano_version}_{args.region}"

    # make eos dir
    if args.site == "cern":
        pdir = Path(f"eos/user/{username[0]}/{username}/bbbb/{args.processor}/")
    else: 
        pdir = Path(f"store/user/{username}/bbbb/{args.processor}/")
        
    outdir = pdir / tag

    # make local directory
    local_dir = Path(f"condor/{args.processor}/{tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("Condor work dir: ", local_dir)

    print(args.subsamples)
    fileset = run_utils.get_fileset(
        args.processor,
        args.year,
        args.nano_version,
        args.samples,
        args.subsamples,
        get_num_files=True,
    )

    print(f"fileset: {fileset}")

    # Global template file paths
    global jdl_templ, sh_templ
    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    
    for sample in fileset:
        for subsample, tot_files in fileset[sample].items():
            nsubmit += process_subsample(
                args, sample, subsample, tot_files, local_dir, outdir, 
                t2_prefixes, proxy, tag
            )
            
            gc.collect() # Saw some hickups so trying this
            
            time.sleep(0.5)
    
    print(f"\nTotal {nsubmit} jobs {'submitted' if args.submit else 'prepared'}")


def parse_args(parser):
    parser.add_argument("--script", default="src/run.py", help="script to run", type=str)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--site",
        default="cern",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd", "cern"],
    )
    parser.add_argument(
        "--save-sites",
        default=["cern"],
        help="tier 2s in which we want to save the files",
        type=str,
        nargs="+",
        choices=["lpc", "ucsd", "cern"],
    )
    run_utils.add_bool_arg(
        parser,
        "test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
    )
    parser.add_argument("--files-per-job", default=10, help="# files per condor job", type=int)
    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument("--git-user", default="Floria-ma", help="which user's repo to use", type=str)
    run_utils.add_bool_arg(
        parser,
        "allow-diff-local-repo",
        default=False,
        help="Allow the local repo to be different from the specified remote repo (not recommended!)."
        "If false, submit script will exit if the latest commits locally and on Github are different.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_utils.parse_common_args(parser)
    parse_args(parser)
    args = parser.parse_args()
    main(args)