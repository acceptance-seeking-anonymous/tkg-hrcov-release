#!/bin/bash
# scripts/run_pilot_smallpedia.sh -- pilot 2 on tkgl-smallpedia (RE-GCN, LogCL, TiRGN): verify + static edges, prep, base grid
# (epoch budgets 3..60, valid selection), TiRGN train + dump, full rows, lists, features, 8 arms x 3 seeds, list controls,
# official evaluation; aggregation.  Per-base chain about 50 min (results/reports/pilot_pipeline_notes.md section 7.4).
#   bash scripts/run_pilot_smallpedia.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
P=$WORK_DIR/tracks/pilot_smallpedia; mkdir -p $P/dumps $P/work $P/stage; rundir pilot_smallpedia
python3 -u $SRC/pilot_smallpedia/sp_verify.py 2>&1 | tee verify.log
python3 -u $SRC/pilot_smallpedia/sp_prep.py --workers 20 2>&1 | tee prep.log
for M in regcn logcl; do bash $SRC/pilot_smallpedia/sp_base_grid.sh $M $GPU 3,6,12,30,60; done
env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/pilot_smallpedia/sp_tirgn.py --epochs 30 2>&1 | tee tirgn.log
for B in regcn logcl; do env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/pilot_smallpedia/sp_fullrows.py --base $B 2>&1 | tee fullrows_$B.log; done
for B in regcn logcl tirgn; do bash $SRC/pilot_smallpedia/sp_chain.sh $B $GPU 10 0 2>&1 | tee chain_$B.log; done
echo "aggregation: python3 $SRC/pilot_smallpedia/sp_agg.py --root aggroot --rule $R/registration/pilots/smallpedia_decision_rule.json --out smallpedia_pilot_verdict.json"
echo "direct-static-edge control: bash $SRC/pilot_smallpedia/sp_chain_nd.sh <base> $GPU 10; python3 $SRC/pilot_smallpedia/sp_agg_nd.py"
