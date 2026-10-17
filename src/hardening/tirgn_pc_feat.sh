#!/bin/bash
# hardening / tirgn_pc_feat.sh -- TiRGN polecat: features + HR/HRCOV (3 seeds) on serverA CPU (16 threads) from the GPU
# lookup computed on serverB (job; the serverA CPU lookup of job was too slow and was cancelled).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
M=$WORK_DIR
P=$M/tracks/pc_tirgn
DD=$M/tracks/datasets/harness_data/tkgl-polecat
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
mkdir -p $P/work/lookup_tirgn
cp $P/work/lookup_tirgn_serverB/valid.npz $P/work/lookup_tirgn_serverB/test.npz $P/work/lookup_tirgn/; cp $P/work/lookup_tirgn_serverB/lookup_tirgn.json $W/
md5sum $P/work/lookup_tirgn/*.npz >> chain.log
rm -rf $P/work/feat_tirgn $P/work/train_tirgn
# cross-GPU lookup: dump on serverA RTX PRO 6000, lookup on serverB RTX 4090 -> exact-rank agreement 0.980 (valid) / 0.978 (test);
# all disagreements are deep near-tie blocks (median gold rank ~121k of 150,931), gold log p equal to 1.4e-5, MRR equal to 1e-8,
# top-1 identical; the pilot threshold 0.99 was set for same-GPU runs. Lowered to 0.97 for this base only (hardening/diag/rank_agree.py).
export PC_RANK_AGREE_MIN=0.97   # rank-agreement threshold for this base (the original run edited pc_features.py in place)
grep -n "rank_agree > 0.9" $SRC/pilot_polecat/pc_features.py >> chain.log
ENVS="env PC_HARNESS=$M PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
cpu1() { env OMP_NUM_THREADS=1 $ENVS "$@" ; }
cpu16() { env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $ENVS "$@" ; }
cpu1 python3 -u $SRC/pilot_polecat/pc_features.py --base tirgn > feat_tirgn.log 2>&1; tm features $?
cpu16 python3 -u $SRC/pilot_polecat/pc_train.py --base tirgn --arms HR,HRCOV > train_tirgn.log 2>&1; tm train $?
TD=$P/work/train_tirgn
cp $TD/train_tirgn.json $W/ 2>/dev/null; cp $TD/ranks_test.npz $W/ranks_test_pc_tirgn.npz 2>/dev/null
cp $P/work/lists_tirgn/lists_summary.json $W/lists_summary_tirgn.json 2>/dev/null; cp feat_summary_tirgn.json $W/ 2>/dev/null
echo ALLDONE >> chain.log
