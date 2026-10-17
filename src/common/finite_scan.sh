#!/bin/bash
# finite scan (read-only, CPU) of the stored arrays behind Manuscript A: main table, G2, Stage B, forward holdout, v7 (+ A10/A11 ablations).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; T=$WORK_DIR/tracks; H=$T/hardening
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
CPU="env OMP_NUM_THREADS=1 HD_HOST=local"
$CPU python3 -u $SRC/common/finite_scan.py --roots smoke=$H/presub_hcpat2 --out ./finite_scan_smoke.json --workers 4 > smoke.log 2>&1; tm smoke $?
[ -f finite_scan_smoke.json ] || { echo "smoke failed" >> chain.log; exit 4; }
$CPU python3 -u $SRC/common/finite_scan.py --workers 10 --out ./finite_scan_serverA.json --roots \
  "main_table_polecat=$T/pilot_polecat,$T/pc_tirgn,$H/tirgn_polecat,$H/presub_hcpat,$H/presub_hcpat2,$H/rb_polecat" \
  "main_table_controls_recb_blends=$H/work" \
  "main_table_smallpedia=$T/pilot_smallpedia,$H/rb_smallpedia" \
  "G2_ICEWS18_GDELT=$H/presub_g2" \
  "stageB_tkgl_icews=$T/pilot_icews/work_stageB,$T/pilot_icews/dumps" \
  "forward_holdout_GDELT25X=$T/phase2_forward_holdout" \
  "v7_items=$H/v7" \
  "ablations_A10_A11=$H/ablations_A" > scan.log 2>&1; tm scan $?
echo ALLDONE >> chain.log
