#!/bin/bash
# hardening / tirgn_pc.sh -- TiRGN on tkgl-polecat (authors' config, history L=10) on serverA GPU 0 ONLY
# (96 GB card; GPU 1 reserved for students), valid-best epoch, then the pilot-1 HR / HRCOV pipeline on that base and the
# BASE(+)RecB / RecB-POOL controls with the official evaluator (same steps as pc_tirgn_chain.sh, paths of the serverA originals).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
OIH=$WORK_DIR
H=$OIH/tracks/hardening
M=$OIH
P=$M/tracks/pc_tirgn
DD=$M/tracks/datasets/harness_data/tkgl-polecat
TREE=$M/tracks/pilot_smallpedia/stage/TiRGN
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0})
echo "GPU0 used MiB at dispatch: $used" > chain.log
if [ "$used" -gt 2000 ]; then echo "GPU0 busy -> abort" >> chain.log; exit 3; fi
mkdir -p $H/code $H/tirgn_polecat $P/work $P/code $P/dumps $P/stage $H/work/blend
# (code staging step of the original run removed: code is read from $SRC)
md5sum $DD/test.txt $DD/train.txt >> chain.log
ENVS="env PC_HARNESS=$M PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
gpu() { env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS "$@" ; }
cpu() { env OMP_NUM_THREADS=1 $ENVS "$@" ; }
# ---- 1. TiRGN training (12 epochs max, patience 3, <= 7 h), valid-best checkpoint, harness-schema top-200 dump
gpu python3 -u $SRC/hardening/hd_tirgn.py --data_dir $DD --n_ent 150931 --n_rel 16 --tree $TREE --ds POLECAT --L 10 --epochs 12 --patience 3 \
    --max_train_hours 7 --dec_chunk 4096 --out $H/tirgn_polecat > tirgn_polecat.log 2>&1; tm tirgn_train_dump $?
cp $H/tirgn_polecat/*.json $W/ 2>/dev/null
[ -f $H/tirgn_polecat/tirgn_scores_POLECAT_traintest.npz ] || { echo "no TiRGN dump -> stop" >> chain.log; exit 4; }
# ---- 2. pilot-1 pipeline on the TiRGN base
for f in te_POLECAT.npz strata_valid.npz strata_test.npz prep_summary.json recb_valid_rank.npy recb_test_rank.npy valid_sample_idx.npy; do cp $M/tracks/pilot_polecat/work/$f $P/work/; done
[ -d $P/work/tgb_ns ] || cp -r $M/tracks/pilot_polecat/work/tgb_ns $P/work/
cp $M/tracks/pilot_polecat/stage/*.npy $P/stage/
# (code staging step of the original run removed: code is read from $SRC)
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
grep -n tirgn $SRC/pilot_polecat/pc_common.py > pc_common_patch.txt
ln -sf $H/tirgn_polecat/tirgn_scores_POLECAT_traintest.npz $P/dumps/tirgn_scores_POLECAT_traintest.npz
ln -sf $H/tirgn_polecat/tirgn_POLECAT_seed42.pt $P/dumps/tirgn_POLECAT_seed42.pt
cpu python3 -u $SRC/pilot_polecat/pc_lists.py --base tirgn --workers 16 > lists_tirgn.log 2>&1; tm lists $?
gpu python3 -u $SRC/hardening/hd_lookup.py --base tirgn --source tirgn --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --work $P/work --tree $TREE > lookup_tirgn.log 2>&1; tm lookup $?
cpu python3 -u $SRC/pilot_polecat/pc_features.py --base tirgn > feat_tirgn.log 2>&1; tm features $?
gpu python3 -u $SRC/pilot_polecat/pc_train.py --base tirgn --arms HR,HRCOV > train_tirgn.log 2>&1; tm train $?
# ---- 3. controls: valid grid -> valid-only selection -> one test pass (official evaluator)
gpu python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage valid --source tirgn --tirgn_tree $TREE --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --out $H/work/blend > blend_valid_pc_tirgn.log 2>&1; tm blend_valid $?
gpu python3 $SRC/hardening/hd_select.py --ds polecat --bases tirgn --dir $H/work/blend --out $H/work/blend/sel_polecat_tirgn.json > select_tirgn.log 2>&1
TD=$P/work/train_tirgn
SYS=HR_s0=$TD/promlist_HR_s0.npy,HR_s1=$TD/promlist_HR_s1.npy,HR_s2=$TD/promlist_HR_s2.npy,HRCOV_s0=$TD/promlist_HRCOV_s0.npy,HRCOV_s1=$TD/promlist_HRCOV_s1.npy,HRCOV_s2=$TD/promlist_HRCOV_s2.npy
gpu bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn --stage test --source tirgn --tirgn_tree $TREE --ckpt $P/dumps/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --sel $H/work/blend/sel_polecat_tirgn.json --systems $SYS --out $H/work/blend" > blend_test_pc_tirgn.log 2>&1; tm blend_test $?
cp $TD/train_tirgn.json $W/ 2>/dev/null; cp $TD/ranks_test.npz $W/ranks_test_pc_tirgn.npz 2>/dev/null
cp $H/work/blend/blend_*_polecat_tirgn.json $H/work/blend/official_test_polecat_tirgn.npz $H/work/blend/sel_polecat_tirgn.json $W/ 2>/dev/null
cp lists_summary_tirgn.json feat_summary_tirgn.json lookup_tirgn.json $W/ 2>/dev/null
echo ALLDONE >> chain.log
