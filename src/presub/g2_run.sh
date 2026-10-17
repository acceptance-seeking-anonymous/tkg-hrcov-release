#!/bin/bash
# presub / g2_run.sh -- G2: HRCOV_PAT + recurrency controls on ICEWS18 / GDELT (RE-GCN split), CPU on serverA.
#   bash g2_run.sh <smoke_n> [datasets] [bases]
#     smoke_n > 0 : code check -- first smoke_n valid / test rows, RecB on the first 50 queries per relation, 2 re-ranker
#                   epochs; outputs under presub_g2_smoke (numbers not used)
#     smoke_n = 0 : full run
# Base rows: tracks/retriever/fullscores (float32 log-softmax, valid + test).  CPU only (no --gpus): <= $WORKERS workers.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
SMOKE=${1:-1000}
DSL=${2:-"ICEWS18 GDELT"}
BASES=${3:-"tirgn regcn logcl"}
WORKERS=${WORKERS:-8}
W=$PWD
H=$WORK_DIR
HD=$H/tracks/hardening
SFX=$( [ "$SMOKE" -gt 0 ] && echo "_smoke" || echo "" )
G=$HD/presub_g2$SFX; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > gpu_at_start.txt 2>&1
uptime > load_at_start.txt
PSM=""; RSM=""; TSM=""
[ "$SMOKE" -gt 0 ] && PSM="--smoke $SMOKE" && RSM="--smoke 50" && TSM="--smoke 1"
for DS in $DSL; do
  if [ $DS = ICEWS18 ]; then DD=$EXT_DIR/logcl_redun/data/ICEWS18; STEP=24; LDS=icews18
  else DD=$EXT_DIR/gdelt_armAD/data/GDELT; STEP=15; LDS=gdelt; fi
  P=$G/$DS; RB=$G/recb; mkdir -p $P/work $RB
  ENVS="env G2_DS=$DS PC_P=$P G2_VALID_ROWS=$P/work/valid_sample_idx.npy HD_DIR=$HD PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
  CPU="env OMP_NUM_THREADS=1 G2_THREADS=$WORKERS $ENVS"
  CPUT="env OMP_NUM_THREADS=$WORKERS G2_THREADS=$WORKERS $ENVS"
  $CPU python3 -u $SRC/presub/g2_prep.py --workers $WORKERS $PSM > prep_$DS$SFX.log 2>&1; tm prep_$DS $?
  cp prep_summary.json prep_summary_$DS$SFX.json 2>/dev/null
  $CPU python3 -u $SRC/hardening/hd_recb.py --data_dir $DD --ts_step $STEP --stage valid --workers $WORKERS --P 16 --out $RB $RSM > recb_valid_$DS$SFX.log 2>&1; tm recb_valid_$DS $?
  $CPU python3 -u $SRC/hardening/hd_recb.py --data_dir $DD --ts_step $STEP --stage test --workers $WORKERS --P 16 --out $RB $RSM > recb_test_$DS$SFX.log 2>&1; tm recb_test_$DS $?
  cp $RB/recb_selection_$DS.json $RB/recb_test_meta_$DS.json $W/ 2>/dev/null
  SELB=""
  for B in $BASES; do
    FS=$H/tracks/retriever/fullscores
    FR=$FS/${B}_${DS}_valid_scores_f32.bin,$FS/${B}_${DS}_test_scores_f32.bin
    $CPU python3 -u $SRC/presub/g2_lists.py --base $B --workers $WORKERS $PSM > lists_${DS}_$B$SFX.log 2>&1; tm lists_${DS}_$B $?
    $CPU python3 -u $SRC/presub/g2_features.py --base $B $PSM > feat_${DS}_$B$SFX.log 2>&1; tm features_${DS}_$B $?
    cp feat_summary_$B$SFX.json feat_summary_${DS}_$B$SFX.json 2>/dev/null; cp lists_summary_$B$SFX.json lists_summary_${DS}_$B$SFX.json 2>/dev/null
    $CPUT python3 -u $SRC/presub/g2_train.py --feat $P/work/feat_$B$SFX --out $P/work/train_$B$SFX $TSM > train_${DS}_$B$SFX.log 2>&1; tm train_${DS}_$B $?
    cp $P/work/train_$B$SFX/train_g2.json train_g2_${DS}_$B$SFX.json 2>/dev/null
    $CPUT python3 -u $SRC/hardening/hd_blend.py --ds $LDS --base $B --stage valid --source fullrows --fullrows $FR --recb_dir $RB --threads $WORKERS $PSM --out $G > blend_valid_${DS}_$B$SFX.log 2>&1; tm blend_valid_${DS}_$B $?
    SELB="$SELB,$B"
  done
  SELB=${SELB#,}
  $CPU python3 $SRC/hardening/hd_select.py --ds $LDS --bases $SELB --dir $G --out $G/sel_$LDS.json > select_$DS$SFX.log 2>&1; tm select_$DS $?
  for B in $BASES; do
    FS=$H/tracks/retriever/fullscores
    FR=$FS/${B}_${DS}_valid_scores_f32.bin,$FS/${B}_${DS}_test_scores_f32.bin
    TD=$P/work/train_$B$SFX; SYS=""
    for arm in HR HRCOV_PAT; do for s in 0 1 2; do [ -f $TD/promlist_${arm}_s$s.npy ] && SYS="$SYS,${arm}_s$s=$TD/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
    $CPUT python3 -u $SRC/hardening/hd_blend.py --ds $LDS --base $B --stage test --source fullrows --fullrows $FR --recb_dir $RB --sel $G/sel_$LDS.json --systems $SYS --threads $WORKERS $PSM --out $G > blend_test_${DS}_$B$SFX.log 2>&1; tm blend_test_${DS}_$B $?
  done
  cp $G/blend_valid_${LDS}_*.json $G/blend_test_${LDS}_*.json $G/official_test_${LDS}_*.npz $G/sel_$LDS.json $W/ 2>/dev/null
  cp $P/work/strata_test.npz $W/strata_test_$DS$SFX.npz 2>/dev/null; cp $RB/recb_test_ranks_$DS.npz $W/ 2>/dev/null
done
# ---- G2 decision (g2_agg.py; smoke: first smoke_n test rows)
AGG="env PYTHONDONTWRITEBYTECODE=1"
$AGG python3 $SRC/presub/g2_agg_cfg.py --G $G --dsl "$DSL" --bases "$BASES" --sfx "$SFX" --out g2_agg_cfg$SFX.json > agg_cfg$SFX.log 2>&1
ASM=""; [ "$SMOKE" -gt 0 ] && ASM="--smoke $SMOKE"
$AGG python3 -u $SRC/presub/g2_agg.py --cfg g2_agg_cfg$SFX.json --rule a_presub_rule.json --out g2_verdict$SFX.json $ASM > agg$SFX.log 2>&1; tm agg $?
echo ALLDONE >> chain.log
