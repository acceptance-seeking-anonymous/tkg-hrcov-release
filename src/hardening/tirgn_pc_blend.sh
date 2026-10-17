#!/bin/bash
# hardening / tirgn_pc_blend.sh -- BASE(+)RecB / RecB-POOL controls for the TiRGN polecat base on the serverB RTX 4090
# (serverA GPU 0 occupied by other users; the serverA CPU job runs lookup -> features -> HR/HRCOV training).
# 1. fetch the TiRGN checkpoint from serverA, 2. valid grid (168 configs, pilot 200k valid sample) + valid-only selection,
# 3. wait for the HR / HRCOV promoted lists of seeds 0-2 on serverA, fetch them, 4. one official-evaluator test pass.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
S=$WORK_DIR   # one machine: the tree of the second server is the same $WORK_DIR
SCP="cp"
SSH="bash -c"
R_A=$WORK_DIR
TD_A=$R_A/tracks/pc_tirgn/work/train_tirgn
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
mkdir -p $H/tirgn_polecat $H/code $H/work/blend $H/pc_tirgn_lists
# (code staging step of the original run removed: code is read from $SRC)
$SCP $R_A/tracks/hardening/tirgn_polecat/tirgn_POLECAT_seed42.pt $R_A/tracks/hardening/tirgn_polecat/tirgn_POLECAT_epoch_log.json $H/tirgn_polecat/; tm fetch_ckpt $?
md5sum $H/tirgn_polecat/tirgn_POLECAT_seed42.pt >> chain.log; $SSH "md5sum $R_A/tracks/hardening/tirgn_polecat/tirgn_POLECAT_seed42.pt" >> chain.log
CK=$H/tirgn_polecat/tirgn_POLECAT_seed42.pt; TREE=$H/stage/TiRGN
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
$GPU python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage valid --source tirgn --tirgn_tree $TREE --ckpt $CK --recb_dir $H/work/recb --out $H/work/blend > blend_valid_pc_tirgn.log 2>&1; tm blend_valid $?
$GPU python3 $SRC/hardening/hd_select.py --ds polecat --bases tirgn --dir $H/work/blend --out $H/work/blend/sel_polecat_tirgn.json > select_tirgn.log 2>&1; tm select $?
cp $H/work/blend/blend_valid_polecat_tirgn.json $H/work/blend/sel_polecat_tirgn.json $W/ 2>/dev/null
# wait for the promoted lists (written by pc_train.py on serverA at the end of training)
n=0
until $SSH "test -f $TD_A/train_tirgn.json && ls $TD_A/promlist_HRCOV_s2.npy" > /dev/null 2>&1; do sleep 120; n=$((n+1)); [ $n -ge 360 ] && { echo "promlists never appeared" >> chain.log; exit 5; }; done
sleep 60
for a in HR HRCOV; do for s in 0 1 2; do $SCP $TD_A/promlist_${a}_s$s.npy $H/pc_tirgn_lists/; done; done
$SCP $TD_A/train_tirgn.json $TD_A/ranks_test.npz $W/ 2>/dev/null; mv $W/ranks_test.npz $W/ranks_test_pc_tirgn.npz 2>/dev/null
tm fetch_promlists $?
( cd $H/pc_tirgn_lists && md5sum *.npy ) >> chain.log; $SSH "cd $TD_A && md5sum promlist_HR*_s*.npy" >> chain.log
L=$H/pc_tirgn_lists
SYS=HR_s0=$L/promlist_HR_s0.npy,HR_s1=$L/promlist_HR_s1.npy,HR_s2=$L/promlist_HR_s2.npy,HRCOV_s0=$L/promlist_HRCOV_s0.npy,HRCOV_s1=$L/promlist_HRCOV_s1.npy,HRCOV_s2=$L/promlist_HRCOV_s2.npy
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage test --source tirgn --tirgn_tree $TREE --ckpt $CK --recb_dir $H/work/recb --sel $H/work/blend/sel_polecat_tirgn.json --systems $SYS --out $H/work/blend" > blend_test_pc_tirgn.log 2>&1; tm blend_test $?
cp $H/work/blend/blend_test_polecat_tirgn.json $H/work/blend/official_test_polecat_tirgn.npz $W/ 2>/dev/null
echo ALLDONE >> chain.log
