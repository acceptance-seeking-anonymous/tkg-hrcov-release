#!/bin/bash
# v76 M6 (lowest priority; v76_rule.json caab7495): RE-GCN TGB configuration on tkgl-smallpedia, new base seeds, on serverB (RTX 4090).
# Per seed = the G3 recipe (g3_tgb_regcn.py, TGB regcn.py unchanged, --seed S) -> g3_dump.py full rows -> sp pipeline
# (sp_lists, sp_features: HRCOV_PAT pool) -> hcpat_train.py (stored recipe) -> BLEND / POOL valid grid + hd_select (full valid)
# -> one official test pass (base, BLEND, POOL, HRCOV_PAT_s0-2).  Usage: bash m6_regcn.sh "43 44"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
SEEDS=${1:-"43 44"}
W=$PWD; H=$WORK_DIR/tracks/hardening; M=$WORK_DIR; G=$H/presub_g3; V=$H/v76/m6; mkdir -p $V/code
# (code staging step of the original run removed: code is read from $SRC)
SHIM=$M/tracks/pilot_smallpedia/stage/TiRGN; DD=$M/tracks/datasets/harness_data/tkgl-smallpedia; SN=$M/tracks/sp_new
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
ENVS="env PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local"
MNT=""
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS $MNT"
for S in $SEEDS; do
  RUN=$V/tgb_repo_s$S; D=$V/dumps_regcn_s$S; P=$M/tracks/sp_g3_s$S; TS=$V/hcpat_regcn_tgb_s$S
  [ -f $D/regcn_test_logp.f32 ] && [ -f $D/g3_dump_log.json ] && echo "seed $S dumps exist, skip training" >> chain.log
  if [ ! -f $D/g3_dump_log.json ]; then
    rm -rf $RUN; mkdir -p $RUN $D; tar xzf $W/g3_tgb_bits.tgz -C $RUN
    $GPU bash -c "python3 -u $SRC/presub/g3_tgb_regcn.py --repo $RUN --shim_dir $SHIM --zip $G/tkgl-smallpedia.zip --out $D --smoke 0 --seed $S" > tgb_regcn_s$S.log 2>&1; tm tgb_regcn_s$S $?
    cp $D/g3_tgb_run.json $W/g3_tgb_run_s$S.json 2>/dev/null
    $GPU python3 -u $SRC/presub/g3_dump.py --repo $RUN --shim_dir $SHIM --ckpt $D/tgb_regcn_ckpt.pt --dd $DD --har $M --out $D --smoke 0 > dump_s$S.log 2>&1; tm dump_s$S $?
    cp $D/g3_dump_log.json $W/g3_dump_log_s$S.json 2>/dev/null
  fi
  mkdir -p $P/work $P/stage $P/code
  for f in prep_summary.json valid_sample_idx.npy te_SMALLPEDIA.npz ste_SMALLPEDIA.npz strata_valid.npz strata_test.npz recb_valid_rank.npy recb_test_rank.npy reach_sample.npz; do [ -f $P/work/$f ] || cp $SN/work/$f $P/work/; done
  [ -d $P/work/tgb_ns ] || cp -r $SN/work/tgb_ns $P/work/
  cp $SN/stage/* $P/stage/ 2>/dev/null; cp $SN/code/sp_*.py $P/code/
  [ -L $P/dumps ] || ln -s $D $P/dumps
  PENV="env PC_HARNESS=$M PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1"
  CPU="env OMP_NUM_THREADS=1 $PENV $MNT"
  GPUP="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PENV $ENVS $MNT"
  if [ ! -f $P/work/feat_regcn/test_A_X.npy ]; then
    $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base regcn --workers 12 > lists_s$S.log 2>&1; tm lists_s$S $?
    $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base regcn > feat_s$S.log 2>&1; tm features_s$S $?
  fi
  $GPUP python3 -u $SRC/hardening/hcpat_train.py --layout sp --feat $P/work/feat_regcn --out $TS --label smallpedia_regcn_tgb_s$S > hcpat_s$S.log 2>&1; tm hcpat_s$S $?
  cp $TS/train_hcpat.json $W/train_hcpat_regcn_tgb_s$S.json 2>/dev/null
  $GPUP python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb_s$S --stage valid --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --out $V > blend_valid_s$S.log 2>&1; tm blend_valid_s$S $?
  $GPUP python3 $SRC/hardening/hd_select.py --ds smallpedia --bases regcn_tgb_s$S --dir $V --out $V/sel_smallpedia_regcn_tgb_s$S.json > select_s$S.log 2>&1; tm select_s$S $?
  SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
  $GPUP bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb_s$S --stage test --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --sel $V/sel_smallpedia_regcn_tgb_s$S.json --systems $SYS --out $V" > blend_test_s$S.log 2>&1; tm blend_test_s$S $?
  cp $V/blend_valid_smallpedia_regcn_tgb_s$S.json $V/blend_test_smallpedia_regcn_tgb_s$S.json $V/official_test_smallpedia_regcn_tgb_s$S.npz $V/sel_smallpedia_regcn_tgb_s$S.json $W/ 2>/dev/null
  cp $TS/ranks_test.npz $W/ranks_test_hcpat_regcn_tgb_s$S.npz 2>/dev/null
  echo "seed $S done" >> chain.log
done
echo ALLDONE >> chain.log
