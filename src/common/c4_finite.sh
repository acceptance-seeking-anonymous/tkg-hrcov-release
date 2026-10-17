#!/bin/bash
# C4 validity check (same discipline as the G8 check; read-only, CPU): count NaN / +inf / -inf in every stored array of the
# seed-43 TiRGN run (checkpoint, dumps), the pc pipeline (lists, lookups, features), the v7 fits and the official outputs; then
# locate any -inf in *_logp tables (padding vs real candidate).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; H=$WORK_DIR/tracks/hardening; C=$H/c4
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
CPU="env OMP_NUM_THREADS=12"
$CPU python3 -u $SRC/common/finite_scan.py --workers 8 --out ./finite_scan_c4.json --roots \
  "c4_tirgn_seed43=$C/tirgn" "c4_pc_pipeline=$C/pc/work" "c4_v7_fits=$C/train" "c4_official_outputs=$C/blend" > scan_c4.log 2>&1; tm scan_c4 $?
python3 - > neginf_files.txt <<'PY'
import json
d = json.load(open("finite_scan_c4.json"))
for r in d.get("files", d.get("per_file", [])):
    if r.get("neginf", 0) > 0: print(r["path"])
PY
if [ -s neginf_files.txt ]; then $CPU python3 -u $SRC/common/neginf_masks.py > masks.log 2>&1; tm masks $?; fi
echo ALLDONE >> chain.log
