#!/bin/bash
# v78 standalone ladder chain on serverA (v78_framing_rule + v78_rule_amendment_1).  GPU 0 only; CPU <= 16.
# Usage: bash v78_chain.sh <ds: polecat|smallpedia> <smoke N (0 = full)> <stages: hist,lists,lookup,pools,train,official>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
DS=$1; SM=${2:-0}; STAGES=${3:-hist,lists,lookup,pools,train,official}
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v78
SFX=$([ "$SM" -gt 0 ] && echo "_smoke" || echo ""); W=$V/$DS$SFX; CD=$SRC/v78; mkdir -p $W   # code is read from the release tree
# (code staging of the original run removed: code is read from $SRC; hd_* modules come from $SRC/hardening via PYTHONPATH)
T0=$(date +%s); tm() { echo "[stage-time] $DS$SFX $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
st() { [[ ",$STAGES," == *",$1,"* ]]; }
MNT=""
RUNG="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $MNT"
TGB=""
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); echo "GPU0 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 20000 ]; }
SMA=$([ "$SM" -gt 0 ] && echo "--smoke $SM" || echo "")
if [ $DS = polecat ]; then BASES="tirgn cogn"; else BASES="tirgn"; fi
if st hist; then guard || { echo "GPU0 busy" >> chain.log; exit 3; }
  $RUNG python3 -u $CD/v78_hist.py --ds $DS --out $W/hist $SMA > hist_$DS$SFX.log 2>&1; tm hist $?; fi
if st lists; then
  $CPU python3 -u $CD/v78_lists.py --ds $DS --hist $W/hist --out $W/lists --workers 14 $SMA > lists_$DS$SFX.log 2>&1; tm lists $?; fi
if st lookup; then
  for b in $BASES; do for sp in valid test; do
    if [ $DS = smallpedia ]; then SRC="--fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
    elif [ $b = tirgn ]; then SRC="--ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --tirgn_tree $T/pilot_smallpedia/stage/TiRGN"
    else SRC="--store $H/g8_cogntke/dumps"; fi
    $RUNG python3 -u $CD/v78_lookup.py --ds $DS --base $b --lists $W/lists --hist $W/hist --split $sp --out $W/lk_$b $SRC > lookup_${DS}_${b}_$sp$SFX.log 2>&1; tm lookup_${b}_$sp $?
  done; done; fi
if st pools; then
  LKA=$(for b in $BASES; do printf "%s=%s," $b $W/lk_$b; done); LKA=${LKA%,}
  $RUNG python3 -u $CD/v78_pools.py --ds $DS --hist $W/hist --lists $W/lists --lookup $LKA --out $W/pools $SMA > pools_$DS$SFX.log 2>&1; tm pools $?; fi
if st train; then
  ARMS="SA_W100,SA_GEN,SA_W50,SA_W150"; for b in $BASES; do ARMS="$ARMS,LAD_C_$b,LAD_S_$b,LAD_T_$b,HR_W50_$b,HR_W150_$b,REL_LIN_$b,REL_LIN_global_$b"; done
  $RUNG python3 -u $CD/v78_train.py --ds $DS --pools $W/pools --arms $ARMS --out $W/train $SMA > train_$DS$SFX.log 2>&1; tm train $?
  cp $W/train/train_v78.json $J/train_v78_$DS$SFX.json 2>/dev/null; fi
if st official; then
  R=$W/train; SYS=""; for a in SA_W100 SA_GEN SA_W50 SA_W150 $(for b in $BASES; do echo LAD_C_$b LAD_S_$b; done); do for s in 0 1 2; do [ -f $R/promlist_${a}_s$s.npy ] && SYS="$SYS,${a}_s$s=$R/promlist_${a}_s$s.npy"; done; done; SYS=${SYS#,}
  $RUNG bash -c "$TGB python3 -u $CD/v78_official.py --ds $DS --label v78_hist$SFX --systems $SYS --out $W $SMA" > off_hist_$DS$SFX.log 2>&1; tm official_hist $?
  cp $W/official_test_${DS}_v78_hist$SFX.npz $W/official_test_${DS}_v78_hist$SFX.json $J/ 2>/dev/null
  for b in $BASES; do
    SYS=""; for a in LAD_T_$b HR_W50_$b HR_W150_$b REL_LIN_$b REL_LIN_global_$b; do for s in 0 1 2; do [ -f $R/promlist_${a}_s$s.npy ] && SYS="$SYS,${a}_s$s=$R/promlist_${a}_s$s.npy"; done; done; SYS=${SYS#,}
    if [ $DS = smallpedia ]; then SRC="--source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
    elif [ $b = tirgn ]; then SRC="--source tirgn --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt"
    else SRC="--source cogn --ckpt $H/g8_cogntke/dumps"; fi
    $RUNG bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds $DS --base v78_${b}$SFX --stage test $SRC --recb_dir $H/work/recb --no_controls --systems $SYS --bs 256 --out $W $SMA" > off_base_${DS}_$b$SFX.log 2>&1; tm official_base_$b $?
    cp $W/official_test_${DS}_v78_${b}$SFX.npz $W/blend_test_${DS}_v78_${b}$SFX.json $J/ 2>/dev/null
  done; fi
cp $W/hist/hist_summary.json $J/hist_summary_$DS$SFX.json 2>/dev/null; cp $W/lists/lists_summary.json $J/lists_summary_$DS$SFX.json 2>/dev/null
cp $W/pools/pools_summary.json $J/pools_summary_$DS$SFX.json 2>/dev/null
echo "CHAIN_DONE $DS$SFX" >> chain.log
