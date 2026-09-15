#!/bin/bash

# "14Feb2026_VJets_TT_Only_v15_scouting_zbb"
# ZBB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# templates_dir=${ZBB_DIR}/templates_zbb

ZBB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
templates_dir=/eos/user/z/zima/scouting/templates/14Feb2026_VJets_TT_Only_v15_scouting_zbb #12Feb2026_Scouting_Fixed_v15_scouting_zbb #11Feb2026_StandardOfflineCuts_v15_scouting_zbb    #07Feb2026_PTl300TXbb0p3_PTsl300_HT600_v15_scouting_zbb #07Feb2026_PTl450TXbb0p3_PTsl200_HT1000_v15_scouting_zbb

for year in 2023All; do # 2022All 2023All
    cards_dir="${ZBB_DIR}/fits/cards/${year}"
    python3 "${ZBB_DIR}/fits/CreateDatacard.py" \
        --templates-dir "${templates_dir}" \
        --cards-dir "${cards_dir}" \
        --year "$year" \
        --nTF 1
done
