#!/bin/bash
# presub / g1_run.sh -- G1 base-free learned recurrency on serverB (RTX 4090 + CPU).
#   bash g1_run.sh <smoke_n>        smoke_n > 0: first smoke_n valid / test rows, 2 re-ranker epochs (code check only)
#   bash g1_run.sh 0                full run
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
SMOKE=${1:-2000}
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
G=$H/presub_g1; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
SFX=$( [ "$SMOKE" -gt 0 ] && echo "_smoke" || echo "" )
TRAIN_SMOKE=$( [ "$SMOKE" -gt 0 ] && echo "--smoke 1" || echo "" )
BLEND_SMOKE=$( [ "$SMOKE" -gt 0 ] && echo "--smoke $SMOKE" || echo "" )
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
for DS in polecat smallpedia; do
  if [ $DS = polecat ]; then COMMON=pc_common; PP=$M/tracks/pilot_polecat; DD=$M/tracks/datasets/harness_data/tkgl-polecat
  else COMMON=sp_common; PP=$M/tracks/pilot_smallpedia; DD=$M/tracks/datasets/harness_data/tkgl-smallpedia; fi
  ENVS="env PC_HARNESS=$M PC_P=$PP PC_DD=$DD HD_HAR=$M PYTHONDONTWRITEBYTECODE=1 HD_HOST=local PYTHONPATH=$SRC/$(basename $PP):$PYTHONPATH"
  GPU="env OMP_NUM_THREADS=8 G1_THREADS=8 $ENVS"
  F=$G/feat_$DS$SFX; R=$G/train_$DS$SFX
  $GPU python3 -u $SRC/presub/g1_features.py --ds $DS --common $COMMON --recb_dir $H/work/recb --out $F --smoke $SMOKE > feat_$DS$SFX.log 2>&1; tm features_$DS $?
  cp $F/g1_features_summary.json $W/g1_features_summary_$DS$SFX.json 2>/dev/null
  $GPU python3 -u $SRC/presub/g1_train.py --feat $F --out $R $TRAIN_SMOKE > train_$DS$SFX.log 2>&1; tm train_$DS $?
  cp $R/train_g1.json $W/train_g1_$DS$SFX.json 2>/dev/null
  SYS=""; for arm in G1_K50 G1_K150; do for s in 0 1 2; do [ -f $R/promlist_${arm}_s$s.npy ] && SYS="$SYS,${arm}_s$s=$R/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
  $GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds $DS --base recb_g1$SFX --stage test --source recb --no_controls --recb_dir $H/work/recb --systems $SYS $BLEND_SMOKE --out $G" > official_$DS$SFX.log 2>&1; tm official_$DS $?
  cp $G/official_test_${DS}_recb_g1$SFX.npz $G/blend_test_${DS}_recb_g1$SFX.json $W/ 2>/dev/null
done
echo ALLDONE >> chain.log
