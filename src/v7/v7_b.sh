#!/bin/bash
# v7 B-items on serverA: RB pool features (CPU) then re-ranker arms (GPU 1 only, guard), exact fused ranks + promoted lists.
# Usage: bash v7_b.sh <stage: rbfeat|smoke|train|all>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
ST=${1:-all}
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; mkdir -p $V/code $V/b
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 24000 ]; }
CPU="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1"
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 V7_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
PCENV="PC_P=$T/pc_tirgn PYTHONPATH=$SRC/pilot_polecat:$PYTHONPATH"; SPENV="PC_P=$T/pilot_smallpedia PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"
PCF="--layout pc --feat $T/pc_tirgn/work/feat_tirgn --lists $T/pc_tirgn/work/lists_tirgn"; SPF="--layout sp --feat $T/pilot_smallpedia/work/feat_tirgn"
if [ $ST = rbfeat ] || [ $ST = all ]; then
  $CPU $PCENV python3 -u $SRC/v7/v7_rbfeat.py $PCF --common pc_common --scan $V/scan/polecat_tirgn --out $V/b/rb_polecat_tirgn > rbfeat_pc.log 2>&1 &
  P1=$!
  $CPU $SPENV python3 -u $SRC/v7/v7_rbfeat.py $SPF --common sp_common --scan $V/scan/smallpedia_tirgn --out $V/b/rb_smallpedia_tirgn > rbfeat_sp.log 2>&1 &
  P2=$!
  wait $P1; tm rbfeat_pc $?; wait $P2; tm rbfeat_sp $?
  cp $V/b/rb_*/*_rb_summary.json $J/ 2>/dev/null
fi
if [ $ST = smoke ] || [ $ST = all ]; then
  guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
  $GPU python3 -u $SRC/v7/v7_btrain.py $PCF --rb $V/b/rb_polecat_tirgn --out $J/smoke_pc --label polecat_tirgn --arms MLP64_PAT,QBLEND,RB_gen,MULTIPOS --seeds 0 --smoke 3000 > smoke_btrain_pc.log 2>&1; tm smoke_pc $?
  $GPU python3 -u $SRC/v7/v7_btrain.py $SPF --rb $V/b/rb_smallpedia_tirgn --out $J/smoke_sp --label smallpedia_tirgn --arms MLP64_PAT,EV_RB --seeds 0 --smoke 3000 > smoke_btrain_sp.log 2>&1; tm smoke_sp $?
  grep -q "ALL DONE" smoke_btrain_pc.log && grep -q "ALL DONE" smoke_btrain_sp.log || { echo "smoke failed" >> chain.log; tail -5 smoke_btrain_*.log >> chain.log; exit 4; }
fi
if [ $ST = train ] || [ $ST = all ]; then
  guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
  run() { $GPU python3 -u $SRC/v7/v7_btrain.py $1 --rb $V/b/rb_$2 --out $V/b/train_$2_$3 --label $2_$3 --arms $4 > train_$2_$3.log 2>&1; tm train_$2_$3 $?; }
  run "$PCF" polecat_tirgn g1 MLP64_PAT,LIN_PAT,MLP16_PAT,QBLEND &
  run "$PCF" polecat_tirgn g2 PT_common,RB_common,RB_gen,EV_PT,EV_RB &
  run "$PCF" polecat_tirgn g3 NB_PAT,noDyadLR,UNMASKED,MULTIPOS &
  run "$SPF" smallpedia_tirgn g1 MLP64_PAT,LIN_PAT,MLP16_PAT,QBLEND,PT_common,RB_common &
  run "$SPF" smallpedia_tirgn g2 RB_gen,EV_PT,EV_RB,NB_PAT,noDyadLR &
  wait
  for d in $V/b/train_*; do b=$(basename $d); cp $d/train_v7.json $J/${b}_train_v7.json 2>/dev/null; cp $d/ranks_test.npz $J/${b}_ranks_test.npz 2>/dev/null; cp $d/ranks_select.npz $J/${b}_ranks_select.npz 2>/dev/null; done
fi
echo ALLDONE >> chain.log
