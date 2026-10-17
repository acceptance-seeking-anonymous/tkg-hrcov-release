#!/bin/bash
# fusion_followups step 3 / sp_chain_dated.sh <workers> [bases] [modes]
# dated static-edge re-run of HRCOV_ST (mode dated) and HRCOV_ST_NODIRECT (mode dated_nd) on tkgl-smallpedia, host serverA.
# ste refit (dated) -> per base, per mode: lists (CPU; pilot-2 bridge rankers, C_cov asserted identical) -> features (CPU)
# -> train HRCOV_ST (3 seeds; GPU 0 only if idle, else CPU) -> collect ranks.  GPU 1 is never used.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
WK=$1; BASES=${2:-regcn,logcl,tirgn}; MODES=${3:-dated,dated_nd}
H=$WORK_DIR; SPT=$H/tracks/pilot_smallpedia; P=$H/tracks/fusion_followups/sp; W=$P/work
T0=$(date +%s); tm() { echo "[stage-time] $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times.txt; }
gpu0_ok() { nvidia-smi -i ${GPU:-0} --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{ if ($1<2000 && $2==0) print 1; else print 0}'; }
BASEDR="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PC_P=$P"
if [ ! -f $W/ste_SMALLPEDIA_dt.npz ]; then
  $BASEDR SP_STATIC_MODE=dated python3 -u $SRC/pilot_smallpedia/sp_ste_dated.py --workers $WK > ste_dated.log 2>&1; tm ste_dated
fi
for B in ${BASES//,/ }; do
  for M in ${MODES//,/ }; do
    case $M in dated) S=_dt;; dated_nd) S=_dtnd;; esac
    CPU="env SP_STATIC_MODE=$M $BASEDR"
    $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers $WK > lists_$B$S.log 2>&1; tm lists_$B$S
    $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B > feat_$B$S.log 2>&1; tm feat_$B$S
    G=$(gpu0_ok); echo "gpu0_ok=$G before train $B $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')" | tee -a gpu_checks.txt
    if [ "$G" = "1" ]; then GF="CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8"; else GF="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8"; fi
    $CPU $GF python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B --arms HRCOV_ST > train_$B$S.log 2>&1; tm train_$B$S
    mkdir -p out; cp $W/train_$B$S/ranks_test.npz out/ranks_test_$B$S.npz; cp $W/train_$B$S/ranks_select.npz out/ranks_select_$B$S.npz
    cp $W/train_$B$S/train_$B.json out/train_$B$S.json; cp lists_summary_$B$S.json feat_summary_$B$S.json out/ 2>/dev/null
    rm -f $W/train_$B$S/promlist_*.npy $W/feat_$B$S/*_X.npy $W/lists_$B$S/*_X?c.npy $W/lists_$B$S/*_X?s.npy   # large reproducible tensors
  done
  cp $SPT/work/train_$B/ranks_test.npz out/p2_ranks_test_$B.npz; cp $SPT/work/train_${B}_nd/ranks_test.npz out/p2_ranks_test_${B}_nd.npz
  cp $SPT/work/train_${B}_nds/ranks_test.npz out/p2_ranks_test_${B}_nds.npz 2>/dev/null
  cp $SPT/work/feat_$B/test_q.npz out/p2_test_q_$B.npz
done
cp ste_dated_summary.json out/ 2>/dev/null; cp $W/ste_dated_summary.json out/ 2>/dev/null
ls -la out
