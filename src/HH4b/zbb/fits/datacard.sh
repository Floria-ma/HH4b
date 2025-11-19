#!/bin/bash


ZBB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#templates_dir=${ZBB_DIR}/templates_zbb
#templates_dir = "/eos/user/z/zima/zbb_templates/Test_v15_scouting_zbb/"
templates_dir="/eos/user/z/zima/zbb_templates/Nov07_v15_scouting_zbb"

for year in 2023 2023BPix; do
    cards_dir="${ZBB_DIR}/fits/cards/${year}"
    python3 "${ZBB_DIR}/fits/CreateDatacard.py" \
        --templates-dir "${templates_dir}" \
        --cards-dir "${cards_dir}" \
        --year "$year" \
        --nTF 1
done
