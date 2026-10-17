#!/bin/bash
# pilot_polecat / pc_chain.sh <base> <gpu> <workers> [smoke_n] [stages]
# Runs lists (CPU) -> lookup (GPU, full rows + official TGB eval of the base) -> features (CPU)
#      -> train (GPU) || refs (CPU) -> aug oracle (GPU) -> collect bundle, for one base.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; G=$2; WK=$3; SM=${4:-0}; STAGES=${5:-lists,lookup,features,train,refs,aug,collect}
H=$WORK_DIR; P=$H/tracks/pilot_polecat
# (code staging step of the original run removed: code is read from $SRC)
SMA=""; [ "$SM" != "0" ] && SMA="--smoke $SM"
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
GPU="env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $CPU"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
T0=$(date +%s); tm() { echo "[stage-time] $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times_$B.txt; }
if has lists; then $CPU python3 -u $SRC/pilot_polecat/pc_lists.py --base $B --workers $WK $SMA 2>&1 | tee lists_$B.log; tm lists; fi
if has lookup; then $GPU bash -c "python3 -u $SRC/pilot_polecat/pc_lookup.py --base $B $SMA" 2>&1 | tee lookup_$B.log; tm lookup; fi
if has features; then $CPU python3 -u $SRC/pilot_polecat/pc_features.py --base $B $SMA 2>&1 | tee feat_$B.log; tm features; fi
if has train || has refs; then
  PIDS=""
  if has train; then ( $GPU python3 -u $SRC/pilot_polecat/pc_train.py --base $B $SMA 2>&1 | tee train_$B.log ) & PIDS="$PIDS $!"; fi
  if has refs && [ "$SM" == "0" ]; then ( $CPU python3 -u $SRC/pilot_polecat/pc_refs.py --base $B 2>&1 | tee refs_$B.log ) & PIDS="$PIDS $!"; fi
  for p in $PIDS; do wait $p; done
  tm train_refs
fi
if has aug && [ "$SM" == "0" ]; then $GPU python3 -u $SRC/pilot_polecat/pc_aug.py --base $B 2>&1 | tee aug_$B.log; tm aug; fi
if has collect && [ "$SM" == "0" ]; then
  W=$P/work; D=bundle_$B; mkdir -p $D
  cp $W/train_$B/ranks_test.npz $W/train_$B/ranks_select.npz $W/train_$B/train_$B.json $D/
  cp $W/refs_$B/ranks_test.npz $D/refs_ranks_test.npz; cp $W/refs_$B/refs_$B.json $D/
  cp $W/feat_$B/test_q.npz $D/; cp feat_summary_$B.json lists_summary_$B.json lookup_$B.json $D/ 2>/dev/null || true
  cp $W/aug_$B/aug_oracle_$B.json $D/ 2>/dev/null || true
  cp $P/dumps/${B}_POLECAT_dump_log.json $D/dump_log_$B.json 2>/dev/null || true
  tar czf bundle_$B.tgz $D; ls -la bundle_$B.tgz; tm collect
fi
ls -la
