#!/bin/bash
# Runs scouting_templates.py on condor with the hh4b micromamba env (submit with: condor_submit templates.sub)
set -e

ZBB_DIR=/afs/cern.ch/user/z/zima/HH4b/src/HH4b/zbb
PYTHON=/eos/home-z/zima/micromamba/envs/hh4b/bin/python

export MPLBACKEND=Agg  # no display on the batch nodes
cd "$ZBB_DIR"
"$PYTHON" -u scouting_templates.py
