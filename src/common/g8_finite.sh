#!/bin/bash
# G8 validity check (no protocol change): after the G8 chain (job) finishes, count non-finite values in every
# CognTKE score array (valid + test dumps), the pc features / lookups, the official outputs and the training state, and
# recompute every re-ranker score (HR, HRCOV, HRCOV_PAT; valid + test) from the saved models. Read-only, CPU.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; T=$WORK_DIR/tracks; H=$T/hardening; G=$H/g8_cogntke; P=$G/pc
CJ=$WORK_DIR/runs
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
echo "waiting for the G8 chain since $(date -u)" > chain.log
for i in $(seq 1 720); do grep -q ALLDONE $CJ/chain.log 2>/dev/null && break; [ -f $CJ/_status.json ] && grep -q '"state":"\(failed\|succeeded\|cancelled\)' $CJ/_status.json && break; sleep 120; done
grep "stage-time" $CJ/chain.log >> chain.log 2>/dev/null; grep -q ALLDONE $CJ/chain.log || { echo "G8 chain not complete" >> chain.log; tail -n 20 $CJ/chain.log >> chain.log; exit 5; }
CPU="env RS_THREADS=12 OMP_NUM_THREADS=12 HD_HOST=local"
$CPU python3 -u $SRC/common/finite_scan.py --workers 8 --out ./finite_scan_g8.json --roots \
  "g8_cogntke_dumps=$G/dumps" "g8_pc_pipeline=$P/work" "g8_official_outputs=$G/blend" "g8_training_state=$G/run_B" > scan_g8.log 2>&1; tm scan_g8 $?
TD=$P/work/train_cogn; HP=$P/work/hcpat_cogn
$CPU python3 -u $SRC/common/rescore_check.py --label polecat_cogn --feat $P/work/feat_cogn \
  --arm HR=$TD/train_cogn.json:$TD/models --arm HRCOV=$TD/train_cogn.json:$TD/models --arm HRCOV_PAT=$HP/train_hcpat.json:$HP/models \
  --out ./rescore_polecat_cogn.json > rs_g8.log 2>&1; tm rescore_g8 $?
echo ALLDONE >> chain.log
