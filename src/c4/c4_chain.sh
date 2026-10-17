#!/bin/bash
# v7 C4 (c4_protocol.json): second TiRGN base seed (43) on tkgl-polecat, same recipe as seed 42, on serverA GPU 0
# ONLY after the G8 chain has released it (GO marker written after the G8 aggregation). Then the pilot-1 pc pipeline on base
# 'tirgn2', v7 fits (MLP64_PAT = HRCOV_PAT, EV_PT = EV_NF_PT), valid-selected controls and one official test pass.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; OIH=$WORK_DIR; T=$OIH/tracks; H=$T/hardening; V=$H/v7; C4=$H/c4; P=$C4/pc
DD=$T/datasets/harness_data/tkgl-polecat; TREE=$T/pilot_smallpedia/stage/TiRGN; SEED=43
CK=$C4/tirgn/tirgn_POLECAT_seed$SEED.pt
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
fail() { echo "BLOCKED at $1" >> chain.log; exit 7; }
mkdir -p $C4/tirgn $C4/blend $C4/train $P/work $P/code $P/dumps $P/stage
echo "waiting for $C4/GO since $(date -u)" > chain.log
for i in $(seq 1 360); do [ -f $C4/GO ] && break; sleep 300; done
[ -f $C4/GO ] || fail "no GO marker after 30 h"
for i in $(seq 1 60); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}); [ "$U" -lt 4000 ] && break; sleep 60; done
echo "GO at $(date -u); GPU0 used MiB $U" >> chain.log; [ "$U" -lt 4000 ] || fail "GPU0 busy ($U MiB)"
ENVS="env PC_HARNESS=$OIH PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1 HD_HAR=$OIH HD_HOST=local"
VOL=""
gpu() { env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS $VOL "$@" ; }
cpu() { env OMP_NUM_THREADS=1 $ENVS $VOL "$@" ; }
# 1. TiRGN seed 43 (recipe of the seed-42 run; only --seed and --out differ)
gpu python3 -u $SRC/hardening/hd_tirgn.py --data_dir $DD --n_ent 150931 --n_rel 16 --tree $TREE --ds POLECAT --L 10 --epochs 12 --patience 3 \
    --max_train_hours 7 --dec_chunk 4096 --seed $SEED --out $C4/tirgn > tirgn_train.log 2>&1; tm tirgn_train_dump $?
cp $C4/tirgn/*.json $J/ 2>/dev/null
[ -f $C4/tirgn/tirgn_scores_POLECAT_traintest.npz ] && [ -f $CK ] || fail "TiRGN training/dump"
# 2. pc pipeline on base 'tirgn2'
for f in te_POLECAT.npz strata_valid.npz strata_test.npz prep_summary.json recb_valid_rank.npy recb_test_rank.npy valid_sample_idx.npy; do cp $T/pc_tirgn/work/$f $P/work/; done
[ -d $P/work/tgb_ns ] || cp -r $T/pc_tirgn/work/tgb_ns $P/work/
cp $T/pc_tirgn/stage/*.npy $P/stage/; cp $T/pc_tirgn/code/pc_*.py $P/code/
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
grep -n '"tirgn2"' $SRC/pilot_polecat/pc_common.py > pc_common_patch.txt; [ $(wc -l < pc_common_patch.txt) -eq 2 ] || fail "pc_common patch"
ln -sf $C4/tirgn/tirgn_scores_POLECAT_traintest.npz $P/dumps/tirgn2_scores_POLECAT_traintest.npz; ln -sf $CK $P/dumps/tirgn2_POLECAT_seed43.pt
cpu python3 -u $SRC/pilot_polecat/pc_lists.py --base tirgn2 --workers 16 > lists.log 2>&1; tm lists $?; [ -d $P/work/lists_tirgn2 ] || fail lists
gpu python3 -u $SRC/hardening/hd_lookup.py --base tirgn2 --source tirgn --ckpt $P/dumps/tirgn2_POLECAT_seed43.pt --work $P/work --tree $TREE > lookup.log 2>&1; tm lookup $?
cpu python3 -u $SRC/pilot_polecat/pc_features.py --base tirgn2 > feat.log 2>&1; tm features $?; [ -d $P/work/feat_tirgn2 ] || fail features
# 3. v7 fits: HRCOV_PAT (MLP64_PAT) and EV_NF_PT (EV_PT), seeds 0-2; smoke first
gpu python3 -u $SRC/v7/v7_btrain.py --layout pc --feat $P/work/feat_tirgn2 --lists $P/work/lists_tirgn2 --out $J/smoke_train --label polecat_tirgn2 --arms MLP64_PAT,EV_PT --smoke 3000 > smoke_train.log 2>&1; tm smoke_train $?
[ -f $J/smoke_train/train_v7.json ] || fail "fit smoke"
gpu python3 -u $SRC/v7/v7_btrain.py --layout pc --feat $P/work/feat_tirgn2 --lists $P/work/lists_tirgn2 --out $C4/train --label polecat_tirgn2 --arms MLP64_PAT,EV_PT > train.log 2>&1; tm train $?
cp $C4/train/train_v7.json $J/train_v7_polecat_tirgn2.json 2>/dev/null; cp $C4/train/ranks_test.npz $J/ranks_test_tirgn2.npz 2>/dev/null
ls $C4/train/promlist_MLP64_PAT_fused_s2.npy $C4/train/promlist_EV_PT_alone_s2.npy > /dev/null || fail "fits"
# 4. controls on valid (grid, valid-only selection), then one official test pass
gpu python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn2 --stage valid --source tirgn --tirgn_tree $TREE --ckpt $CK --recb_dir $H/work/recb --out $C4/blend > blend_valid.log 2>&1; tm blend_valid $?
gpu python3 $SRC/hardening/hd_select.py --ds polecat --bases tirgn2 --dir $C4/blend --out $C4/blend/sel_polecat_tirgn2.json > select.log 2>&1; tm select $?
[ -f $C4/blend/sel_polecat_tirgn2.json ] || fail select
SYS=""; for s in 0 1 2; do SYS="$SYS,MLP64_PAT_fused_s$s=$C4/train/promlist_MLP64_PAT_fused_s$s.npy,EV_PT_alone_s$s=$C4/train/promlist_EV_PT_alone_s$s.npy"; done; SYS=${SYS#,}
gpu bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn2 --stage test --source tirgn --tirgn_tree $TREE --ckpt $CK --recb_dir $H/work/recb --sel $C4/blend/sel_polecat_tirgn2.json --systems $SYS --bs 256 --out $C4/blend" > blend_test.log 2>&1; tm blend_test $?
cp $C4/blend/sel_polecat_tirgn2.json $C4/blend/blend_*_polecat_tirgn2.json $J/ 2>/dev/null; cp lists_summary_tirgn2.json feat_summary_tirgn2.json lookup_tirgn2.json $J/ 2>/dev/null
[ -f $C4/blend/official_test_polecat_tirgn2.npz ] || fail "official test"
echo ALLDONE >> chain.log
