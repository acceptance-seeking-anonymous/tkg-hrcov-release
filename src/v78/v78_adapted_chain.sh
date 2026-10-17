#!/bin/bash
# v78 post-verdict check: EXT_adapted (CountTRuCoLa, label-adapted) alone + blend with the frozen bases.  GPU 0 only.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v78; E=$T/v78_external
CD=$SRC/v78   # code is read from the release tree
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
RUNG="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
TGB=""
for cfg in "smallpedia tirgn" "polecat tirgn" "polecat cogn"; do set -- $cfg; DS=$1; b=$2; W=$V/$DS
  if [ $DS = smallpedia ]; then SRC="--fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
  elif [ $b = tirgn ]; then SRC="--ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --tirgn_tree $T/pilot_smallpedia/stage/TiRGN"
  else SRC="--store $H/g8_cogntke/dumps"; fi
  MS="counttrucola_adapted=$E/dumps/counttrucola_adapted/tkgl-$DS"
  $RUNG python3 -u $CD/v78_ext.py --ds $DS --base $b --stage valid --tag _adapted --hist $W/hist --methods $MS --out $W $SRC > adapted_valid_${DS}_$b.log 2>&1; tm adapted_valid_${DS}_$b $?
  $RUNG bash -c "$TGB python3 -u $CD/v78_ext.py --ds $DS --base $b --stage test --tag _adapted --hist $W/hist --methods $MS --sel $W/sel_ext_adapted_${DS}_$b.json --out $W $SRC" > adapted_test_${DS}_$b.log 2>&1; tm adapted_test_${DS}_$b $?
  cp $W/official_test_${DS}_v78_ext_adapted_$b.npz $W/official_test_${DS}_v78_ext_adapted_$b.json $W/sel_ext_adapted_${DS}_$b.json $J/ 2>/dev/null
done
echo CHAIN_DONE >> chain.log
