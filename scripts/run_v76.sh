#!/bin/bash
# scripts/run_v76.sh <item> -- reviewer-requested analyses (registration/v76_rule.json + v76_rule_amendment_1.json).
#   p1        M2b LambdaMART on the four bases, M4 cross-fitted RecB selection, M3a inputs (v76/v76_p1.sh)
#   p2 | p2b  official scoring of the M2b / M4 systems (v76/v76_p2.sh, v76_p2b.sh)
#   m3a       validation-trained TiRGN on tkgl-smallpedia (v76/sp_tirgn_valfit.py via m3a_fetch.sh)
#   m6_tirgn  TiRGN tkgl-smallpedia seeds 43 and 44 (v76/m6_tirgn.sh; ~20 min training each)
#   m6_regcn  TGB-configuration RE-GCN seeds 43 and 44 (v76/m6_regcn.sh "43 44"; RTX 4090)
#   scan      finiteness scan and RE-GCN recheck (v76/v76_scan.sh, v76_regcn_recheck.sh)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir v76_$1
case "$1" in
  p1) bash $SRC/v76/v76_p1.sh ;;
  p2) bash $SRC/v76/v76_p2.sh ;;
  p2b) bash $SRC/v76/v76_p2b.sh ;;
  m3a) bash $SRC/v76/m3a_fetch.sh ;;
  m6_tirgn) bash $SRC/v76/m6_tirgn.sh ;;
  m6_regcn) cp -n $STAGE_DIR/g3_tgb_bits.tgz . ; bash $SRC/v76/m6_regcn.sh "43 44" ;;
  scan) bash $SRC/v76/v76_scan.sh; bash $SRC/v76/v76_regcn_recheck.sh ;;
  *) echo "usage: $0 p1|p2|p2b|m3a|m6_tirgn|m6_regcn|scan"; exit 2 ;;
esac
echo "aggregation: v76/v76_agg.py run_all(CFG) over the official outputs (see the module docstring)"
