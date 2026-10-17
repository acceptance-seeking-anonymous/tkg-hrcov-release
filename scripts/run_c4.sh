#!/bin/bash
# scripts/run_c4.sh -- second TiRGN tkgl-polecat base seed (43) (registration/c4_protocol.json): training (~1.7 h), pipeline,
# v7 fits, valid-selected controls, official test (c4/c4_chain.sh), then c4_agg.py.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir c4; bash $SRC/c4/c4_chain.sh
python3 $SRC/c4/c4_agg.py --s43 $WORK_DIR/tracks/hardening/c4/blend/official_test_polecat_tirgn2.npz --out c4_results.json
