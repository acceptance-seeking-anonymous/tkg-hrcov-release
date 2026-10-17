#!/bin/bash
# hardening / tirgn_lookup.sh -- GPU copy of the TiRGN polecat list lookup (hd_lookup.py) on the serverB RTX 4090, run in
# parallel to the slower CPU lookup of job on serverA; results copied back to serverA as lookup_tirgn_serverB/ (not overwriting).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
S=$WORK_DIR   # one machine: the tree of the second server is the same $WORK_DIR
SCP="cp"
R_A=$WORK_DIR/tracks/pc_tirgn/work
WK=$H/pc_tirgn_lk/work
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
mkdir -p $WK/lists_tirgn
# (code staging step of the original run removed: code is read from $SRC)
$SCP $R_A/valid_sample_idx.npy $WK/
for f in valid_Lpat valid_Lbr test_Lpat test_Lbr; do $SCP $R_A/lists_tirgn/$f.npy $WK/lists_tirgn/; done; tm fetch_lists $?
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
$GPU python3 -u $SRC/hardening/hd_lookup.py --base tirgn --source tirgn --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --work $WK --tree $H/stage/TiRGN > lookup_tirgn.log 2>&1; tm lookup $?
bash -c "mkdir -p $R_A/lookup_tirgn_serverB"
$SCP $WK/lookup_tirgn/valid.npz $WK/lookup_tirgn/test.npz lookup_tirgn.json $R_A/lookup_tirgn_serverB/; tm send_back $?
echo ALLDONE >> chain.log
