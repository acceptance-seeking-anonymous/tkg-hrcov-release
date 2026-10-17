#!/bin/bash
# v76: finiteness scan of every new output of items M2b / M3a / M4 on serverA (read-only), after phase 2b; -inf location check.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; H=$WORK_DIR/tracks/hardening; V=$H/v76
# (orchestration wait of the original multi-job run removed)
R=""
for d in $V/m2b/bt_* $V/m2b/lm2_*; do R="$R m2b_$(basename $d)=$d"; done
R="$R m4_recb_xf=$V/recb_xf m4_rbf=$V/rbf m4_g1f_polecat=$V/m4/g1f_polecat m4_g1f_smallpedia=$V/m4/g1f_smallpedia m4_g1t_polecat=$V/m4/g1t_polecat m4_g1t_smallpedia=$V/m4/g1t_smallpedia"
R="$R official_v76=$V/official m3a_bt_valbase=$V/m3a/bt_valbase m3a_pipeline=$V/m3a/sp/work"
mkdir -p ctl && cp $V/m4/official_test_*.npz $V/m3a/official_test_*.npz ctl/ 2>/dev/null
env OMP_NUM_THREADS=1 \
  python3 -u $SRC/common/finite_scan.py --workers 12 --out ./finite_scan_v76_serverA.json --roots $R m4_m3a_official=./ctl > scan.log 2>&1
echo rc=$? >> scan.log; rm -rf ctl
python3 - > neginf_files.txt <<'PY'
import json
d = json.load(open("finite_scan_v76_serverA.json"))
for r in d.get("files", []):
    if sum(m.get("neginf", 0) for m in r.get("members", {}).values()) > 0: print(r["path"])
PY
if [ -s neginf_files.txt ]; then python3 -u $SRC/common/neginf_masks.py > masks.log 2>&1; fi
echo done >> scan.log
