#!/bin/bash
# scripts/run_v7.sh <item> -- pre-specified later analyses of registration/tois_v7_rule.json.
#   extract   A items: re-analysis of stored per-query outputs (v7/v7_extract.sh; CPU)
#   b         B1-B3: ranker form, matched pool, common-pool base ablation (v7/v7_b.sh all; GPU)
#   b4        B4: G1 on tkgl-icews Stage B (v7/v7_b4.sh, v7_b4c3.sh)
#   c2        C2: pattern design variants (v7/v7_c2.sh, v7_c2b.sh)
#   official  official scoring of all v7 systems (v7/v7_official.sh, v7_off02.sh)
#   scan      finiteness scan of v7 outputs (v7/v7_scan.sh)
#   report    consolidated report from the per-item JSONs (v7/v7_build_report.py)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir v7_$1
case "$1" in
  extract) bash $SRC/v7/v7_extract.sh ;;
  b) bash $SRC/v7/v7_b.sh all ;;
  b4) bash $SRC/v7/v7_b4.sh; bash $SRC/v7/v7_b4c3.sh ;;
  c2) bash $SRC/v7/v7_c2.sh; bash $SRC/v7/v7_c2b.sh ;;
  official) bash $SRC/v7/v7_official.sh; bash $SRC/v7/v7_off02.sh ;;
  scan) bash $SRC/v7/v7_scan.sh ;;
  report) python3 -c "import sys; sys.path.insert(0, '$SRC/v7'); import v7_build_report as B; B.build({})" ;;
  *) echo "usage: $0 extract|b|b4|c2|official|scan|report"; exit 2 ;;
esac
