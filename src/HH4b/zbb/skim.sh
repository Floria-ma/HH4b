#!/bin/bash
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh
export PYTHONPATH=/afs/cern.ch/user/z/zima/HH4b/src:$PYTHONPATH
python /afs/cern.ch/user/z/zima/HH4b/src/HH4b/zbb/skim_nanotrees.py --sample data --input "$1" --output skim.root --nthreads 4
xrdcp -f skim.root "root://eosuser.cern.ch//eos/user/z/zima/HadronicVH/tagger_Cali_2024_had/data/skim/HJ/$2"
rm skim.root
