#!/bin/bash
# scripts/run_forward_holdout.sh <build|stage1|stage2> -- GDELT25X forward holdout (registration/fh_decision_rule.json).
#   build    data/build_gdelt25x.py (downloads ~26k GDELT 2.0 export files, ~35 min with 16 threads) + SHA-256 check
#   stage1   prepare and freeze (TiRGN base, lists, features, re-rankers, controls on validation); no test score (~6 h)
#   stage2   the single registered test pass, then aggregation (fh_agg.py)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
FH=$WORK_DIR/tracks/phase2_forward_holdout; DD=$FH/data/GDELT25X; mkdir -p $FH/data; rundir forward_holdout
case "$1" in
  build) python3 $R/data/build_gdelt25x.py --out $DD --raw $WORK_DIR/tracks/datasets/raw/gdelt --skip-list $R/data/gdelt25x_missing_export_files.txt
         python3 $R/data/build_gdelt25x.py --check --out $DD ;;
  stage1) bash $SRC/forward_holdout/fh_stage1_chain.sh ;;
  stage2) bash $SRC/forward_holdout/fh_stage2_test.sh ;;
  *) echo "usage: $0 build|stage1|stage2"; exit 2 ;;
esac
