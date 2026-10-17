#!/bin/bash
# v78 external comparators + SA_ERM on serverA (v78_framing_rule + v78_rule_amendment_1).  GPU 0 only; CPU <= 16.
# Usage: bash v78_ext_chain.sh <ds> <smoke N (0 = full)> <stages: erm_hist,erm_lists,erm_pools,erm_train,erm_official,ext_valid,ext_test>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
DS=$1; SM=${2:-0}; STAGES=${3:-erm_hist,erm_lists,erm_pools,erm_train,erm_official,ext_valid,ext_test}
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v78; E=$T/v78_external
SFX=$([ "$SM" -gt 0 ] && echo "_smoke" || echo ""); W=$V/$DS$SFX; CD=$SRC/v78; mkdir -p $W   # code is read from the release tree
# (code staging of the original run removed: code is read from $SRC; hd_* modules come from $SRC/hardening via PYTHONPATH)
T0=$(date +%s); tm() { echo "[stage-time] $DS$SFX $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
st() { [[ ",$STAGES," == *",$1,"* ]]; }
MNT=""
RUNG="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $MNT"
TGB=""
SMA=$([ "$SM" -gt 0 ] && echo "--smoke $SM" || echo "")
ERM=$E/runs/edgeremind/tkgl-$DS/seed1337/dump; CTD=$E/runs/counttrucola/tkgl-$DS/dump
if [ $DS = polecat ]; then BASES="tirgn cogn"; else BASES="tirgn"; fi
if st erm_hist; then $RUNG python3 -u $CD/v78_hist.py --ds $DS --out $W/hist_erm --erm $ERM $SMA > erm_hist_$DS$SFX.log 2>&1; tm erm_hist $?; fi
if st erm_lists; then $CPU python3 -u $CD/v78_lists.py --ds $DS --hist $W/hist_erm --out $W/lists_erm --workers 14 --erm $ERM $SMA > erm_lists_$DS$SFX.log 2>&1; tm erm_lists $?; fi
if st erm_pools; then $RUNG python3 -u $CD/v78_pools.py --ds $DS --hist $W/hist_erm --lists $W/lists_erm --lookup "" --erm $ERM --out $W/pools_erm $SMA > erm_pools_$DS$SFX.log 2>&1; tm erm_pools $?; fi
if st erm_train; then $RUNG python3 -u $CD/v78_train.py --ds $DS --pools $W/pools_erm --arms SA_ERM --out $W/train_erm $SMA > erm_train_$DS$SFX.log 2>&1; tm erm_train $?
  cp $W/train_erm/train_v78.json $J/train_v78_erm_$DS$SFX.json 2>/dev/null; fi
if st erm_official; then R=$W/train_erm; SYS=""; for s in 0 1 2; do SYS="$SYS,SA_ERM_s$s=$R/promlist_SA_ERM_s$s.npy"; done; SYS=${SYS#,}
  $RUNG bash -c "$TGB python3 -u $CD/v78_official.py --ds $DS --label v78_erm$SFX --erm $ERM --systems $SYS --out $W $SMA" > erm_off_$DS$SFX.log 2>&1; tm erm_official $?
  cp $W/official_test_${DS}_v78_erm$SFX.npz $W/official_test_${DS}_v78_erm$SFX.json $J/ 2>/dev/null; fi
for b in $BASES; do
  if [ $DS = smallpedia ]; then SRC="--fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
  elif [ $b = tirgn ]; then SRC="--ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --tirgn_tree $T/pilot_smallpedia/stage/TiRGN"
  else SRC="--store $H/g8_cogntke/dumps"; fi
  MS="edgeremind=$ERM,counttrucola=$CTD"
  if st ext_valid; then $RUNG python3 -u $CD/v78_ext.py --ds $DS --base $b --stage valid --hist $W/hist --methods $MS --out $W $SRC $SMA > ext_valid_${DS}_$b$SFX.log 2>&1; tm ext_valid_$b $?
    cp $W/sel_ext_${DS}_$b.json $J/sel_ext_${DS}_$b$SFX.json 2>/dev/null; fi
  if st ext_test; then $RUNG bash -c "$TGB python3 -u $CD/v78_ext.py --ds $DS --base $b --stage test --hist $W/hist --methods $MS --sel $W/sel_ext_${DS}_$b.json --out $W $SRC $SMA" > ext_test_${DS}_$b$SFX.log 2>&1; tm ext_test_$b $?
    cp $W/official_test_${DS}_v78_ext_$b.npz $W/official_test_${DS}_v78_ext_$b.json $J/ 2>/dev/null; fi
done
cp $W/hist_erm/hist_summary.json $J/hist_erm_summary_$DS$SFX.json 2>/dev/null; cp $W/lists_erm/lists_summary.json $J/lists_erm_summary_$DS$SFX.json 2>/dev/null
cp $W/pools_erm/pools_summary.json $J/pools_erm_summary_$DS$SFX.json 2>/dev/null
echo "CHAIN_DONE $DS$SFX" >> chain.log
