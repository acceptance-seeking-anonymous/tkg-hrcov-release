#!/bin/bash
# scripts/run_pilot_icews.sh <A|B> -- pilot 3 on tkgl-icews.  Stage A (registration/stageA_decision_rule.json): relation
# normalisation, prep, verify, window-trained RE-GCN / LogCL, lists, lookup + official base, features, re-rankers, official systems
# (about 35 min per base after the bases).  Stage B (registration/stageB_rule.json): full train, every test query
# (bases 3-14 h each on one RTX PRO 6000; chains 5-7 h; results/reports/pilot_pipeline_notes.md section 9).
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
S=${1:-B}; rundir pilot_icews_stage$S
if [ "$S" = "A" ]; then
  python3 -u $SRC/pilot_icews/pi_relnorm.py; python3 -u $SRC/pilot_icews/pi_prep.py --workers 20; python3 -u $SRC/pilot_icews/pi_verify.py
  for B in regcn logcl; do env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/pilot_icews/pi_base.py --model $B; bash $SRC/pilot_icews/pi_chain.sh $B $GPU 20; done
  echo "aggregation: python3 $SRC/pilot_icews/pi_agg.py --root aggroot --rule $R/registration/stageA_decision_rule.json --out icews_stageA_verdict.json"
else
  python3 -u $SRC/pilot_icews/pi_prepB.py
  for B in regcn logcl; do bash $SRC/pilot_icews/pi_chainB2.sh $B $GPU 20; done
  python3 -u $SRC/pilot_icews/pi_aggB.py --rule $R/registration/stageB_rule.json --out icews_stageB_verdict.json --workers 16
fi
