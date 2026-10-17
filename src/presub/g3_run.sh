#!/bin/bash
# presub / g3_run.sh -- G3: TGB 2.0 RE-GCN config on tkgl-smallpedia (target 0.594), harness-order replay dump,
# official base check, then the unchanged smallpedia HR / HRCOV / HRCOV_ST pipeline + controls on base label regcn_tgb.
#   bash g3_run.sh 1     smoke: 2 epochs x 8 snapshots, first valid / test snapshot, official check on those rows,
#                            pipeline path check only (code check; numbers not used)
#   bash g3_run.sh 0     full run
# Host: serverB (RTX 4090 24 GB, 32 threads).  Inputs: g3_tgb_regcn.py g3_dump.py hd_blend.py
# g3_tgb_bits.tgz (TGB repo sources, no data) tkgl-smallpedia.zip (raw TGB zip).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
SMOKE=${1:-1}
RESUME=${2:-0}      # 1: keep the trained TGB checkpoint in dumps/ and skip stage 1 (re-run dump onward)
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
G=$H/presub_g3; mkdir -p $G/code
SFX=$( [ "$SMOKE" -gt 0 ] && echo "_smoke" || echo "" )
# (code staging step of the original run removed: code is read from $SRC)
[ -f $G/tkgl-smallpedia.zip ] || cp $W/tkgl-smallpedia.zip $G/
RUN=$G/tgb_repo$SFX
# fresh repo copy + dump dir (earlier container runs leave root-owned files, so remove them from a container)
if [ "$RESUME" -eq 0 ]; then
  rm -rf $RUN $G/dumps$SFX
  mkdir -p $RUN; tar xzf $W/g3_tgb_bits.tgz -C $RUN
fi
D=$G/dumps$SFX; mkdir -p $D
SHIM=$M/tracks/pilot_smallpedia/stage/TiRGN
DD=$M/tracks/datasets/harness_data/tkgl-smallpedia
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > gpu_at_start.txt
ENVS="env PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS"
# 1) TGB's own script (train with valid-best checkpoint, then the script's own official test)
[ "$RESUME" -eq 0 ] && $GPU bash -c "python3 -u $SRC/presub/g3_tgb_regcn.py --repo $RUN --shim_dir $SHIM --zip $G/tkgl-smallpedia.zip --out $D --smoke $SMOKE" > tgb_regcn$SFX.log 2>&1; tm tgb_regcn $?
cp $D/g3_tgb_run.json $W/g3_tgb_run$SFX.json 2>/dev/null; cp $D/tgb_regcn_results.json $W/tgb_regcn_results$SFX.json 2>/dev/null
# 2) replay dump in harness order (top-200 dump + full float32 rows)
$GPU python3 -u $SRC/presub/g3_dump.py --repo $RUN --shim_dir $SHIM --ckpt $D/tgb_regcn_ckpt.pt --dd $DD --har $M --out $D --smoke $SMOKE > dump$SFX.log 2>&1; tm dump $?
cp $D/g3_dump_log.json $W/g3_dump_log$SFX.json 2>/dev/null
# 3) official base MRR from the dumped rows (py-tgb evaluator, TGB negatives); must equal the script's test MRR to 1e-3
NSM=""; [ "$SMOKE" -gt 0 ] && NSM="--smoke $(python3 -c "import json;print(json.load(open('$D/g3_dump_log.json'))['first_test_snapshot_rows'])" 2>/dev/null || echo 2000)"
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb_check$SFX --stage test --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --no_controls $NSM --out $G" > official_base$SFX.log 2>&1; tm official_base $?
cp $G/blend_test_smallpedia_regcn_tgb_check$SFX.json $W/ 2>/dev/null
# 4) smallpedia pipeline on base label regcn_tgb (sp_* code unchanged; new track dir, base-independent inputs from sp_new)
P=$M/tracks/sp_g3$SFX; mkdir -p $P/work $P/stage $P/code
SN=$M/tracks/sp_new
for f in prep_summary.json valid_sample_idx.npy te_SMALLPEDIA.npz ste_SMALLPEDIA.npz strata_valid.npz strata_test.npz recb_valid_rank.npy recb_test_rank.npy reach_sample.npz; do [ -f $P/work/$f ] || cp $SN/work/$f $P/work/; done
[ -d $P/work/tgb_ns ] || cp -r $SN/work/tgb_ns $P/work/
cp $SN/stage/* $P/stage/ 2>/dev/null; cp $SN/code/sp_*.py $P/code/
[ -L $P/dumps ] || ln -s $D $P/dumps
PENV="env PC_HARNESS=$M PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1"
CPU="env OMP_NUM_THREADS=1 $PENV"
GPUP="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PENV $ENVS"
$CPU python3 -c "
import os, sys; sys.path.insert(0, '$P/code'); import sp_common as C
need = [C.DUMPS['regcn'], C.FULLROWS['regcn']['valid'], C.FULLROWS['regcn']['test'], C.STATIC] + [os.path.join(C.WORK, f) for f in ('prep_summary.json','valid_sample_idx.npy','te_SMALLPEDIA.npz','strata_valid.npz','strata_test.npz','recb_valid_rank.npy','recb_test_rank.npy','tgb_ns')]
print('PATHCHECK', {p: os.path.exists(p) for p in need}); print('PATHCHECK_ALL', all(os.path.exists(p) for p in need))
grep = open('$SRC/pilot_smallpedia/sp_train.py').read().count('if arm in SAVE_LISTS:'); print('SAVE_LISTS_PATCH', grep)
" > pathcheck$SFX.log 2>&1; tm pathcheck $?
if [ "$SMOKE" -eq 0 ]; then
  B=regcn; TB=$(date +%s)
  $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers 12 > lists_g3.log 2>&1; tm lists $?
  $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B > feat_g3.log 2>&1; tm features $?
  $GPUP python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B --arms HR,HRCOV,HRCOV_ST > train_g3.log 2>&1; tm train $?
  $GPUP python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb --stage valid --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --out $G > blend_valid_g3.log 2>&1; tm blend_valid $?
  $GPUP python3 $SRC/hardening/hd_select.py --ds smallpedia --bases regcn_tgb --dir $G --out $G/sel_smallpedia_tgb.json > select_g3.log 2>&1; tm select $?
  TD=$P/work/train_$B; SYS=""
  for arm in HR HRCOV HRCOV_ST; do for s in 0 1 2; do SYS="$SYS,${arm}_s$s=$TD/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
  $GPUP bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb --stage test --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --sel $G/sel_smallpedia_tgb.json --systems $SYS --out $G" > blend_test_g3.log 2>&1; tm blend_test $?
  cp $TD/train_$B.json $W/train_g3.json 2>/dev/null; cp $TD/ranks_test.npz $W/ranks_test_g3.npz 2>/dev/null; cp $P/work/feat_$B/test_q.npz $W/test_q_g3.npz 2>/dev/null
  cp lists_summary_*.json feat_summary_*.json $W/ 2>/dev/null
  cp $G/blend_valid_smallpedia_regcn_tgb.json $G/blend_test_smallpedia_regcn_tgb.json $G/official_test_smallpedia_regcn_tgb.npz $G/sel_smallpedia_tgb.json $W/ 2>/dev/null
fi
echo ALLDONE >> chain.log
