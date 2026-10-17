#!/bin/bash
# phase2_forward_holdout / fh_stage1_chain.sh -- STAGE 1: prepare and freeze on GDELT25X; NO test-split score is computed.
#   A (CPU, starts at once): valid sample + early-train TE tables + VALID strata; RecB per-relation selection on valid;
#                            RecB blend CSR for valid only
#   B (GPU, after fh_gpu_wait): TiRGN on the train window (hd_tirgn.py, GDELT G2 hyper-parameters his 7 / angle 10,
#                            --no_dump), then VALID full log-prob rows + valid dump (fh_tirgn_rows.py)
#   C (CPU, after A and B):  valid pattern-prior lists, valid features (25), BASE(+)RecB / RecB-POOL grids on valid,
#                            valid selection, re-ranker fit + freeze (HR, HRCOV_PAT, seeds 0-2)
#   D:                       sha256 freeze manifest of every frozen file (fh_freeze.py)
# Usage (job cwd = harvest dir):  bash fh_stage1_chain.sh
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD
H=$WORK_DIR
FH=$H/tracks/phase2_forward_holdout
P=$FH/GDELT25X; W=$P/work; C=$FH/code; L=$FH/logs; RUN=$P/run_stage1
mkdir -p $W/recb $W/controls $W/base_tirgn $L $RUN
WORKERS=${WORKERS:-8}
export FH_GPU_CAP_GB=${FH_GPU_CAP_GB:-30}
source $C/fh_gpu_wait.sh
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u +%FT%TZ)" >> $L/stage1_chain.log; }
echo "[stage1] start $(date -u +%FT%TZ) host $(hostname) workers $WORKERS" >> $L/stage1_chain.log
uptime >> $L/stage1_chain.log
DD=$FH/data/GDELT25X
ENVS="env PC_HARNESS=$H FH_DIR=$FH PC_P=$P FH_DD=$DD G2_VALID_ROWS=$W/valid_sample_idx.npy PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
MNT=""
CPU="env OMP_NUM_THREADS=1 G2_THREADS=$WORKERS $ENVS $MNT"
CPUT="env OMP_NUM_THREADS=$WORKERS G2_THREADS=$WORKERS $ENVS $MNT"

# ---------------- A: CPU preparation (no base needed)
(
  $CPU python3 -u $C/fh_prep.py --workers $WORKERS --stage 1 > $L/prep_stage1.log 2>&1; tm prep_stage1 $?
  $CPU python3 -u $C/fh_recb.py --data_dir $DD --ts_step 1 --stage valid --workers $WORKERS --P 16 --out $W/recb > $L/recb_valid.log 2>&1; tm recb_valid $?
  $CPU python3 -u $C/fh_recb.py --data_dir $DD --ts_step 1 --stage test --only_valid --workers $WORKERS --P 16 --out $W/recb > $L/recb_validcsr.log 2>&1; tm recb_validcsr $?
) &
PA=$!

# ---------------- B: GPU base (waits for an allowed card)
(
  fh_gpu_wait
  ENVG=$(echo "$ENVS" | sed "s|HD_HOST=local")
  GPUR="env CUDA_VISIBLE_DEVICES=$FH_GPU FH_GPU_CAP_GB=$FH_GPU_CAP_GB OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVG $MNT"
  echo "[gpu] tirgn_train start $(date -u +%FT%TZ) gpu $FH_GPU" >> $L/stage1_chain.log; G0=$(date +%s)
  $GPUR python3 -u $C/fh_tirgn_train.py --data_dir $DD --n_ent 1173 --n_rel 218 --tree $FH/stage/TiRGN --ds GDELT25X \
      --L 7 --hr 0.3 --lr 0.001 --epochs 30 --patience 5 --max_train_hours 10 --seed 42 --dec_chunk 8192 --no_dump \
      --out $W/base_tirgn > $L/tirgn_train.log 2>&1; tm tirgn_train $?
  $GPUR python3 -u $C/fh_tirgn_rows.py --ckpt $W/base_tirgn/tirgn_GDELT25X_seed42.pt --split valid --tree $FH/stage/TiRGN \
      --L 7 --hr 0.3 --dec_chunk 4096 > $L/tirgn_rows_valid.log 2>&1; tm tirgn_rows_valid $?
  echo "[gpu] stage1 GPU wall seconds $(( $(date +%s) - G0 )) on gpu $FH_GPU" >> $L/stage1_chain.log
) &
PB=$!
wait $PA; wait $PB

# ---------------- C: CPU, needs the valid base rows
if [ -f $W/fullrows/tirgn_GDELT25X_valid_scores_f32.bin ] && [ -f $W/recb/recb_csr_valid_GDELT25X.npz ]; then
  FR=$W/fullrows/tirgn_GDELT25X_valid_scores_f32.bin,$W/fullrows/tirgn_GDELT25X_test_scores_f32.bin
  $CPU python3 -u $C/fh_lists.py --base tirgn --workers $WORKERS --splits valid > $L/lists_valid.log 2>&1; tm lists_valid $?
  $CPU python3 -u $C/fh_features.py --base tirgn --splits valid > $L/features_valid.log 2>&1; tm features_valid $?
  $CPUT python3 -u $C/fh_blend.py --ds gdelt25x --base tirgn --stage valid --source fullrows --fullrows $FR --recb_dir $W/recb \
      --threads $WORKERS --out $W/controls > $L/blend_valid.log 2>&1; tm blend_valid $?
  $CPU python3 $C/hd_select.py --ds gdelt25x --bases tirgn --dir $W/controls --out $W/controls/sel_gdelt25x.json > $L/select.log 2>&1; tm select $?
  $CPUT python3 -u $C/fh_train.py --feat $W/feat_tirgn --out $W/train_tirgn > $L/train_rr.log 2>&1; tm train_rr $?
else
  echo "[stage1] C skipped: valid base rows or valid RecB CSR missing" >> $L/stage1_chain.log
fi

# ---------------- D: freeze manifest (host python3, stdlib only)
python3 $C/fh_freeze.py --fh $FH --out $FH/freeze/freeze_manifest_remote.json > $L/freeze.log 2>&1; tm freeze $?

# ---------------- harvest copies (small files only)
mkdir -p $J/h
cp $L/*.log $J/h/ 2>/dev/null
for f in $W/prep_summary_stage1.json $W/recb/recb_selection_GDELT25X.json $W/recb/recb_valid_meta_GDELT25X.json $W/recb/recb_validcsr_meta_GDELT25X.json \
         $W/base_tirgn/tirgn_GDELT25X_epoch_log.json $W/base_tirgn/rows_valid_log.json $W/lists_tirgn/lists_summary_valid.json \
         $W/feat_tirgn/feat_summary_valid.json $W/controls/blend_valid_gdelt25x_tirgn.json $W/controls/sel_gdelt25x.json \
         $W/train_tirgn/train_fh.json $FH/freeze/freeze_manifest_remote.json; do
  [ -f $f ] && cp $f $J/h/
done
echo "[stage1] end $(date -u +%FT%TZ)" >> $L/stage1_chain.log
cp $L/stage1_chain.log $J/h/
