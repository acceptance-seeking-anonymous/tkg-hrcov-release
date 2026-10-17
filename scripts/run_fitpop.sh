#!/bin/bash
# scripts/run_fitpop.sh <item> -- post-verdict fit-population audit of the v7.8 ladder and HRcov (sensitivity analysis; it does
# not replace the registered v7.8 verdict).
#   refit       fit-population counts, common-intersection refit of SA, +C, +S, +T, HRcov and official scoring (fitpop/fitpop_chain.sh)
#   diag        size-matched random-subsample refits (trainer harness ranks) and test gold-pool flags (fitpop/fitpop_diag_chain.sh)
#   aggregate   stage the per-query arrays (analysis/stage_v78_inputs.sh) and rebuild fitpop_audit.json / .md (fitpop/fitpop_run.py)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir fitpop_$1
case "$1" in
  refit) bash $SRC/fitpop/fitpop_chain.sh ;;
  diag) bash $SRC/fitpop/fitpop_diag_chain.sh ;;
  aggregate) bash $R/analysis/stage_v78_inputs.sh "${2:-}" && FITPOP_OUT=$PWD python3 $SRC/fitpop/fitpop_run.py ;;
  *) echo "usage: $0 refit | diag | aggregate [npz dir]"; exit 2 ;;
esac
