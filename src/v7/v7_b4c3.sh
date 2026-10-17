#!/bin/bash
# v7 B4/C3 aggregation on serverA (CPU): smoke (G1 slot filled by LogCL HRCOV150, B=20) now; full run once the B4 official G1 scoring has finished.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; S=$WORK_DIR/tracks/pilot_icews/work_stageB; V=$WORK_DIR/tracks/hardening/v7
B4J=$WORK_DIR/runs
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
free -g >> chain.log
CPU="env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8"
$CPU python3 -u $SRC/v7/v7_b4c3_agg.py --stageB $S --g1 $S/blend/official_test_icews_logcl_sys.npz --g1_prefix HRCOV150 --B 20 --out ./smoke_b4c3.json > smoke.log 2>&1; tm smoke $?
for i in $(seq 1 600); do grep -q "stage-time\] official rc=" $B4J/chain.log 2>/dev/null && break; sleep 60; done
grep "official rc=" $B4J/chain.log >> chain.log
G1=$V/b4/official_test_icews_recb_g1.npz; ls -la $G1 >> chain.log 2>&1
[ -f $G1 ] || { echo "no G1 official file" >> chain.log; exit 5; }
$CPU python3 -u $SRC/v7/v7_b4c3_agg.py --stageB $S --g1 $G1 --out ./v7_b4c3_tkgl_icews.json > agg.log 2>&1; tm agg $?
cp $V/b4/blend_test_icews_recb_g1.json $J/ 2>/dev/null
echo ALLDONE >> chain.log
