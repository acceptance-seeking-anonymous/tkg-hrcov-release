#!/bin/bash
# hardening / pc_tirgn_chain.sh -- pilot-1 HR / HRCOV pipeline on the TiRGN polecat base (serverB), then blend + official test.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
S=$WORK_DIR   # one machine: the tree of the second server is the same $WORK_DIR
SCP="cp"
P=$M/tracks/pc_tirgn
mkdir -p $P/work $P/code $P/dumps $P/stage
for f in te_POLECAT.npz strata_valid.npz strata_test.npz prep_summary.json recb_valid_rank.npy recb_test_rank.npy; do [ -f $P/work/$f ] || $SCP $S/tracks/pilot_polecat/work/$f $P/work/; done
cp $M/tracks/pilot_polecat/work/valid_sample_idx.npy $P/work/; [ -d $P/work/tgb_ns ] || cp -r $M/tracks/pilot_polecat/work/tgb_ns $P/work/
cp $M/tracks/pilot_polecat/stage/*.npy $P/stage/
# (code staging step of the original run removed: code is read from $SRC)
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
grep -n tirgn $SRC/pilot_polecat/pc_common.py > pc_common_patch.txt
ln -sf $H/tirgn_polecat/tirgn_scores_POLECAT_traintest.npz $P/dumps/tirgn_scores_POLECAT_traintest.npz
ln -sf $H/tirgn_polecat/tirgn_POLECAT_seed42.pt $P/dumps/tirgn_POLECAT_seed42.pt
# (code staging step of the original run removed: code is read from $SRC)
ENVS="env PC_HARNESS=$M PC_P=$P PC_DD=$M/tracks/datasets/harness_data/tkgl-polecat PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M"
CPU="env OMP_NUM_THREADS=1 $ENVS"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $ENVS"
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s" >> chain.log; }
$CPU python3 -u $SRC/pilot_polecat/pc_lists.py --base tirgn --workers 16 > lists_tirgn.log 2>&1; tm lists $?
$GPU python3 -u $SRC/hardening/hd_lookup.py --base tirgn --source tirgn --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --work $P/work --tree $H/stage/TiRGN > lookup_tirgn.log 2>&1; tm lookup $?
$CPU python3 -u $SRC/pilot_polecat/pc_features.py --base tirgn > feat_tirgn.log 2>&1; tm features $?
$GPU python3 -u $SRC/pilot_polecat/pc_train.py --base tirgn --arms HR,HRCOV > train_tirgn.log 2>&1; tm train $?
$GPU python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage valid --source tirgn --tirgn_tree $H/stage/TiRGN --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --out $H/work/blend > blend_valid_pc_tirgn.log 2>&1; tm blend_valid $?
$GPU python3 $SRC/hardening/hd_select.py --ds polecat --bases tirgn --dir $H/work/blend --out $H/work/blend/sel_polecat_tirgn.json > select_tirgn.log 2>&1
TD=$P/work/train_tirgn
SYS=HR_s0=$TD/promlist_HR_s0.npy,HR_s1=$TD/promlist_HR_s1.npy,HR_s2=$TD/promlist_HR_s2.npy,HRCOV_s0=$TD/promlist_HRCOV_s0.npy,HRCOV_s1=$TD/promlist_HRCOV_s1.npy,HRCOV_s2=$TD/promlist_HRCOV_s2.npy
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage test --source tirgn --tirgn_tree $H/stage/TiRGN --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --sel $H/work/blend/sel_polecat_tirgn.json --systems $SYS --out $H/work/blend" > blend_test_pc_tirgn.log 2>&1; tm blend_test $?
cp $TD/train_tirgn.json $TD/ranks_test.npz $W/ 2>/dev/null; mv $W/ranks_test.npz $W/ranks_test_pc_tirgn.npz 2>/dev/null
cp $H/work/blend/*polecat_tirgn* $H/work/blend/sel_polecat_tirgn.json $W/ 2>/dev/null
cp lists_summary_tirgn.json feat_summary_tirgn.json lookup_tirgn.json $W/ 2>/dev/null
