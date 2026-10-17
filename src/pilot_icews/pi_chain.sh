#!/bin/bash
# pilot_icews / pi_chain.sh -- per-base Stage-A driver (run from a job dir on host serverA).
# usage: bash pi_chain.sh <base> <gpu> <cpu_workers> [stages]
#   stages (comma list, default all): lists,lookup,features,train,official,collect
# Every stage writes under $P/work and is skipped when its completion marker exists (resume after outage).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; GPU=$2; NW=$3; STAGES=${4:-lists,lookup,features,train,official,collect}
P=$WORK_DIR/tracks/pilot_icews
W=$P/work
JD=$PWD
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
GPUC="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
stamp() { echo "[stage-time] $1 $(date +%s)" | tee -a chain_$B.log; }
stamp start
if has lists && [ ! -f $W/lists_$B/lists_summary_$B.json ]; then
  $CPU python3 -u $SRC/pilot_icews/pi_lists.py --base $B --workers $NW 2>&1 | grep --line-buffered -v Warning | tee -a chain_$B.log; stamp lists; fi
if has lookup && [ ! -f $W/lookup_$B/lookup_$B.json ]; then
  eval $GPUC bash -c "'python3 -u $SRC/pilot_icews/pi_replay.py --base $B --mode lookup'" 2>&1 | tee -a chain_$B.log; stamp lookup; fi
if has features && [ ! -f $W/feat_summary_$B.json ]; then
  $CPU python3 -u $SRC/pilot_icews/pi_features.py --base $B 2>&1 | grep --line-buffered -v Warning | tee -a chain_$B.log; stamp features; fi
if has train && [ ! -f $W/train_$B/promlist_HR_s0.npy ]; then
  eval $GPUC python3 -u $SRC/pilot_icews/pi_train.py --base $B 2>&1 | tee -a chain_$B.log; stamp train; fi
if has official && [ ! -f $W/official_$B/systems_$B.json ]; then
  SYS=$(python3 -c "import json;d=json.load(open('$W/train_$B/train_$B.json'));print(','.join(f.replace('promlist_','').replace('.npy','')+'=$W/train_$B/'+f for f in d['promlists']))")
  eval $GPUC bash -c "'python3 -u $SRC/pilot_icews/pi_replay.py --base $B --mode systems --systems $SYS'" 2>&1 | tee -a chain_$B.log; stamp official; fi
if has collect; then
  mkdir -p bundle_$B
  cp $W/feat_${B}_orig/test_q.npz bundle_$B/test_q_orig.npz; cp $W/feat_${B}_rn/test_q.npz bundle_$B/test_q_rn.npz
  cp official_*_ranks.npz bundle_$B/ 2>/dev/null || true
  cp $W/train_$B/ranks_test.npz $W/train_$B/ranks_select.npz $W/train_$B/train_$B.json bundle_$B/
  cp $W/lookup_$B/lookup_$B.json $W/lookup_$B/test.npz bundle_$B/ 2>/dev/null || true
  mv bundle_$B/test.npz bundle_$B/lookup_test.npz 2>/dev/null || true
  cp $W/official_$B/systems_$B.json bundle_$B/ 2>/dev/null || true
  cp $W/official_$B/test.npz bundle_$B/official_systems_test.npz 2>/dev/null || true
  cp $W/lists_$B/lists_summary_$B.json $W/feat_summary_$B.json $P/dumps/base_${B}_stageA.json bundle_$B/ 2>/dev/null || true
  cp $P/dumps/ckpt_${B}_stageA/grid_log.json bundle_$B/grid_log_$B.json 2>/dev/null || true
  tar czf bundle_$B.tgz bundle_$B; ls -lh bundle_$B; stamp collect
fi
