#!/bin/bash
# pilot_polecat / pc_ts_chain.sh <base> <workers> [K] [stages]
# Time-safe bridge rerun: identical pipeline under $P/work_ts, except the stage-2 bridge TRAINING cohort is built from
# snapshot-before-t indices (pc_lists.py --timesafe K).  GPU stages run on physical card 0 under a flock so the two
# bases never share the card at the same time (card 0 is shared with other users; card 1 is reserved).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; WK=$2; K=${3:-32}; STAGES=${4:-lists,lookup,features,train,diag,systems,agg}
H=$WORK_DIR; P=$H/tracks/pilot_polecat; W=$P/work_ts
mkdir -p $W $P/code_ts; cp pc_*.py pc_*.sh $P/code_ts/ 2>/dev/null || true
for f in valid_sample_idx.npy te_POLECAT.npz strata_valid.npz strata_test.npz tgb_ns prep_summary.json recb_test_rank.npy recb_valid_rank.npy; do
  [ -e $W/$f ] || ln -s $P/work/$f $W/$f
done
CPU="env PC_WORK=$W OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 $CPU"
LOCK=$P/gpu0_ts.lock
has() { [[ ",$STAGES," == *",$1,"* ]]; }
T0=$(date +%s); tm() { echo "[stage-time] $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times_ts_$B.txt; }
gpufree() { nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1 | tee -a gpu_ts_$B.txt; }
if has lists; then $CPU python3 -u $P/code_ts/pc_lists.py --base $B --workers $WK --timesafe $K 2>&1 | tee lists_ts_$B.log; tm lists; fi
if has lookup; then flock $LOCK bash -c "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1; $GPU bash -c 'python3 -u $P/code_ts/pc_lookup.py --base $B --no_official'" 2>&1 | tee lookup_ts_$B.log; tm lookup; fi
if has features; then $CPU python3 -u $P/code_ts/pc_features.py --base $B 2>&1 | tee feat_ts_$B.log; tm features; fi
if has train; then flock $LOCK $GPU python3 -u $P/code_ts/pc_train.py --base $B --arms HRCOV,HRCOVFEAT 2>&1 | tee train_ts_$B.log; tm train; fi
if has diag; then flock $LOCK $GPU python3 -u $P/code_ts/pc_train_diag.py --base $B --arms HRCOV_BR,HRCOV_PAT --save_lists HRCOV_BR 2>&1 | tee diag_ts_$B.log; tm diag; fi
if has systems; then flock $LOCK $GPU bash -c "python3 -u $P/code_ts/pc_lookup.py --base $B --mode systems --systems HRCOV_s0=$W/train_$B/promlist_HRCOV_s0.npy,HRCOV_BR_s0=$W/train_$B/promlist_HRCOV_BR_s0.npy" 2>&1 | tee official_ts_$B.log; tm systems; fi
if has agg; then $CPU python3 -u $P/code_ts/pc_ts_agg.py --base $B --systems systems_$B.json 2>&1 | tee agg_ts_$B.log; tm agg; fi
cp $W/lists_$B/lists_summary.json lists_summary_ts_$B.json 2>/dev/null || true
cp $W/train_$B/train_$B.json train_ts_$B.json 2>/dev/null || true
cp $W/train_$B/train_diag_$B.json train_diag_ts_$B.json 2>/dev/null || true
ls -la
