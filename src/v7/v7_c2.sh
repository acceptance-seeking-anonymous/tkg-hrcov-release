#!/bin/bash
# v7 C2 pattern-design variants on serverA: lists (CPU) -> exact lookup (GPU 1) -> features (CPU) -> fits (GPU 1) -> official (GPU 1).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; C=$V/c2; mkdir -p $V/code $C
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 12000 ]; }
DK="env PYTHONDONTWRITEBYTECODE=1"
CPU="env OMP_NUM_THREADS=1 $DK"
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 V7_THREADS=4 HD_HOST=local $DK"
PCENV="PC_P=$T/pc_tirgn PYTHONPATH=$SRC/pilot_polecat:$PYTHONPATH"; SPENV="PC_P=$T/pilot_smallpedia PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"
# 0. smoke of the lists
$CPU python3 -u $SRC/v7/v7_c2_lists.py --ds polecat --out $J/smoke_pc --workers 4 --smoke 400 > smoke_lists_pc.log 2>&1; tm smoke_lists_pc $?
$CPU python3 -u $SRC/v7/v7_c2_lists.py --ds smallpedia --out $J/smoke_sp --workers 4 --smoke 400 > smoke_lists_sp.log 2>&1; tm smoke_lists_sp $?
tail -3 smoke_lists_pc.log smoke_lists_sp.log >> chain.log
[ -f $J/smoke_pc/c2_lists_summary.json ] && [ -f $J/smoke_sp/c2_lists_summary.json ] || { echo "lists smoke failed" >> chain.log; exit 4; }
# 1. lists
$CPU python3 -u $SRC/v7/v7_c2_lists.py --ds polecat --out $C/polecat_tirgn --workers 14 > lists_pc.log 2>&1 & P1=$!
$CPU python3 -u $SRC/v7/v7_c2_lists.py --ds smallpedia --out $C/smallpedia_tirgn --workers 6 > lists_sp.log 2>&1 & P2=$!
wait $P1; tm lists_pc $?; wait $P2; tm lists_sp $?
cp $C/*/c2_lists_summary.json $J/ 2>/dev/null; for d in polecat_tirgn smallpedia_tirgn; do cp $C/$d/c2_lists_summary.json $J/c2_lists_summary_$d.json 2>/dev/null; done
for d in polecat_tirgn smallpedia_tirgn; do for sp in valid test; do
  $CPU python3 -c "import numpy as np; b=np.load('$C/$d/${sp}_blocks.npy'); np.save('$C/$d/${sp}_req.npy', b.reshape(b.shape[0], -1))"; done; done
# 2. exact lookup (wait until the B official-scoring polecat containers have finished: GPU-1 budget)
# (orchestration wait of the original multi-job run removed)
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
PCA="--ds polecat --source tirgn --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --feat $T/pc_tirgn/work/feat_tirgn --valid_rows $T/pc_tirgn/work/valid_sample_idx.npy"
SPA="--ds smallpedia --source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32 --feat $T/pilot_smallpedia/work/feat_tirgn"
( for sp in valid test; do $GPU python3 -u $SRC/v7/v7_lookup.py $PCA --split $sp --req $C/polecat_tirgn/${sp}_req.npy --out $C/polecat_tirgn/lk > lk_pc_$sp.log 2>&1; tm lk_pc_$sp $?; done ) &
for sp in valid test; do $GPU python3 -u $SRC/v7/v7_lookup.py $SPA --split $sp --req $C/smallpedia_tirgn/${sp}_req.npy --out $C/smallpedia_tirgn/lk > lk_sp_$sp.log 2>&1; tm lk_sp_$sp $?; done
wait
# 3. features
$CPU $PCENV python3 -u $SRC/v7/v7_c2feat.py --layout pc --common pc_common --feat $T/pc_tirgn/work/feat_tirgn --c2 $C/polecat_tirgn --out $C/pools_polecat_tirgn > feat_pc.log 2>&1 & P1=$!
$CPU $SPENV python3 -u $SRC/v7/v7_c2feat.py --layout sp --common sp_common --feat $T/pilot_smallpedia/work/feat_tirgn --c2 $C/smallpedia_tirgn --out $C/pools_smallpedia_tirgn > feat_sp.log 2>&1 & P2=$!
wait $P1; tm feat_pc $?; wait $P2; tm feat_sp $?
cp $C/pools_*/c2_pools_summary.json $J/ 2>/dev/null; for d in polecat_tirgn smallpedia_tirgn; do cp $C/pools_$d/c2_pools_summary.json $J/c2_pools_summary_$d.json 2>/dev/null; done
# 4. fits
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
ARMS=C2_UTIL12,C2_CONF12,C2_UTIL1,C2_UTIL2
$GPU python3 -u $SRC/v7/v7_btrain.py --layout pc --feat $T/pc_tirgn/work/feat_tirgn --lists $T/pc_tirgn/work/lists_tirgn --c2 $C/pools_polecat_tirgn --out $V/b/train_polecat_tirgn_c2 --label polecat_tirgn_c2 --arms $ARMS > train_pc.log 2>&1 & P1=$!
$GPU python3 -u $SRC/v7/v7_btrain.py --layout sp --feat $T/pilot_smallpedia/work/feat_tirgn --c2 $C/pools_smallpedia_tirgn --out $V/b/train_smallpedia_tirgn_c2 --label smallpedia_tirgn_c2 --arms $ARMS > train_sp.log 2>&1 & P2=$!
wait $P1; tm train_pc $?; wait $P2; tm train_sp $?
for d in polecat_tirgn smallpedia_tirgn; do cp $V/b/train_${d}_c2/train_v7.json $J/train_${d}_c2_train_v7.json; cp $V/b/train_${d}_c2/ranks_test.npz $J/train_${d}_c2_ranks_test.npz; done
# 5. official
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
for d in polecat_tirgn smallpedia_tirgn; do
  SYS=$(ls $V/b/train_${d}_c2/promlist_*_fused_s*.npy | awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}')
  if [ $d = polecat_tirgn ]; then A="--ds polecat --base v7_c2 --stage test --source tirgn --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --bs 256"
  else A="--ds smallpedia --base v7_c2 --stage test --source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32 --recb_dir $H/work/recb --bs 256"; fi
  ( $GPU bash -c "python3 -u $SRC/hardening/hd_blend.py $A --no_controls --systems $SYS --out $V/official" > off_c2_$d.log 2>&1; tm off_c2_$d $? ) &
  sleep 20
done
wait
cp $V/official/blend_test_*_v7_c2.json $J/ 2>/dev/null
echo ALLDONE >> chain.log
