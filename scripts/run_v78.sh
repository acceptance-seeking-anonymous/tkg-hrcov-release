#!/bin/bash
# scripts/run_v78.sh <item> [dataset] -- v7.8 standalone ladder and external learned-history comparison
# (registration/v78_framing_rule.json + v78_rule_amendment_1.json; protocol notes v78_external_protocol_note*.json).
#   ladder <ds>     history / lists / lookup / pools / train / official for SA, +C, +S, +T, widths, REL_LIN (v78/v78_chain.sh)
#   external        EdgeReMIND and CountTRuCoLa runs and per-query score dumps on both datasets (v78_external/*_job.sh;
#                   needs bases/fetch_external_methods.sh --venvs)
#   ext <ds>        SA_ERM and the external methods alone and blended with the frozen bases (v78/v78_ext_chain.sh)
#   adapted         post-verdict, registered-optional: label-adapted CountTRuCoLa (v78_external/v78_ct_adapted_job.sh, then
#                   v78/v78_adapted_chain.sh)
#   aggregate       stage the per-query arrays (analysis/stage_v78_inputs.sh) and rebuild v78_adapted_check.json / .md, with
#                   a bit-level check of every registered v7.8 contrast (v78/v78_adapted_run.py)
# <ds> = smallpedia | polecat.  GPU index from GPU; CPU stages use up to 16 workers.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir v78_$1${2:+_$2}
case "$1" in
  ladder) bash $SRC/v78/v78_chain.sh "${2:?dataset}" 0 ;;
  external) for ds in tkgl-smallpedia tkgl-polecat; do bash $SRC/v78_external/v78_er_job.sh $ds 1337 16 gate; bash $SRC/v78_external/v78_ct_job.sh $ds 16; done ;;
  ext) bash $SRC/v78/v78_ext_chain.sh "${2:?dataset}" 0 ;;
  adapted) echo "per dataset: bash $SRC/v78_external/v78_ct_adapted_job.sh <tkgl-ds> 16 <cut ts id> [harness valid rows]; the cut ids are in results/v78/runs/sel_ext_adapted_*.json (field cut)"
           bash $SRC/v78/v78_adapted_chain.sh ;;
  aggregate) bash $R/analysis/stage_v78_inputs.sh "${2:-}" && V78_OUT=$PWD python3 $SRC/v78/v78_adapted_run.py ;;
  *) echo "usage: $0 ladder <ds> | external | ext <ds> | adapted | aggregate [npz dir]"; exit 2 ;;
esac
