#!/bin/bash
# phase2_forward_holdout / fh_stage2_test.sh -- STAGE 2 (the ONE test pass of fh_decision_rule.json).  Run only after the
# go (the status pilot's own Stage-2 test must be complete).  Nothing is fitted or selected here.
#   0. verify every Stage-1 frozen file against freeze_manifest (abort, exit 3, on any mismatch)
#   1. test strata (fh_prep --stage 2); RecB test ranks + valid/test blend CSR into recb_stage2/ (frozen selection copied
#      in; the recomputed valid CSR must equal the frozen one byte-for-byte)
#   2. GPU: TiRGN test full rows + test dump with the frozen checkpoint (fh_tirgn_rows --split test)
#   3. test lists + features; frozen re-rankers applied (fh_apply: select-rank reproduction check first)
#   4. every arm scored once (fh_blend --stage test, harness exact filtered ranks); decision (fh_agg -> fh_verdict.json)
# Usage (job cwd = harvest dir):  bash fh_stage2_test.sh
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD
H=$WORK_DIR
FH=$H/tracks/phase2_forward_holdout
P=$FH/GDELT25X; W=$P/work; C=$FH/code; L=$FH/logs; RUN=$P/run_stage2; DD=$FH/data/GDELT25X
mkdir -p $RUN $W/recb_stage2 $W/test_out $J/h
WORKERS=${WORKERS:-8}
export FH_GPU_CAP_GB=${FH_GPU_CAP_GB:-30}
source $C/fh_gpu_wait.sh
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u +%FT%TZ)" >> $L/stage2_test.log; }
echo "[stage2] start $(date -u +%FT%TZ)" >> $L/stage2_test.log
python3 $C/fh_freeze.py --fh $FH --verify $FH/freeze/freeze_manifest_remote.json > $L/stage2_verify.json 2>&1
VRC=$?; tm verify_freeze $VRC
if [ $VRC -ne 0 ]; then echo "[stage2] ABORT: frozen files changed or missing (see stage2_verify.json)" >> $L/stage2_test.log; cp $L/stage2_* $J/h/; exit 3; fi
ENVS="env PC_HARNESS=$H FH_DIR=$FH PC_P=$P FH_DD=$DD G2_VALID_ROWS=$W/valid_sample_idx.npy PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
MNT=""
CPU="env OMP_NUM_THREADS=1 G2_THREADS=$WORKERS $ENVS $MNT"
CPUT="env OMP_NUM_THREADS=$WORKERS G2_THREADS=$WORKERS $ENVS $MNT"
# ---- 1. CPU (test strata, RecB test)
(
  $CPU python3 -u $C/fh_prep.py --workers $WORKERS --stage 2 > $L/prep_stage2.log 2>&1; tm prep_stage2 $?
  cp $W/recb/recb_selection_GDELT25X.json $W/recb_stage2/
  $CPU python3 -u $C/fh_recb.py --data_dir $DD --ts_step 1 --stage test --workers $WORKERS --P 16 --out $W/recb_stage2 > $L/recb_test.log 2>&1; tm recb_test $?
  $CPU python3 -c "import numpy as np; a=np.load('$W/recb/recb_csr_valid_GDELT25X.npz'); b=np.load('$W/recb_stage2/recb_csr_valid_GDELT25X.npz'); \
ok=all(np.array_equal(a[k], b[k]) for k in a.files); print('valid CSR arrays identical to frozen:', ok)" >> $L/stage2_test.log 2>&1
) &
PA=$!
# ---- 2. GPU (TiRGN test rows, frozen checkpoint)
(
  fh_gpu_wait
  GPUR="env CUDA_VISIBLE_DEVICES=$FH_GPU FH_GPU_CAP_GB=$FH_GPU_CAP_GB OMP_NUM_THREADS=8 $ENVS $MNT"
  G0=$(date +%s)
  $GPUR python3 -u $C/fh_tirgn_rows.py --ckpt $W/base_tirgn/tirgn_GDELT25X_seed42.pt --split test --tree $FH/stage/TiRGN \
      --L 7 --hr 0.3 --dec_chunk 4096 > $L/tirgn_rows_test.log 2>&1; tm tirgn_rows_test $?
  echo "[gpu] stage2 GPU wall seconds $(( $(date +%s) - G0 )) on gpu $FH_GPU" >> $L/stage2_test.log
) &
PB=$!
wait $PA; wait $PB
# ---- 3. test lists / features / frozen re-rankers
$CPU python3 -u $C/fh_lists.py --base tirgn --workers $WORKERS --splits test > $L/lists_test.log 2>&1; tm lists_test $?
$CPU python3 -u $C/fh_features.py --base tirgn --splits test > $L/features_test.log 2>&1; tm features_test $?
$CPUT python3 -u $C/fh_apply.py --feat $W/feat_tirgn --train $W/train_tirgn > $L/apply_test.log 2>&1; tm apply_test $?
# ---- 4. score every arm once + decision
FR=$W/fullrows/tirgn_GDELT25X_valid_scores_f32.bin,$W/fullrows/tirgn_GDELT25X_test_scores_f32.bin
TD=$W/train_tirgn; SYS=""
for arm in HR HRCOV_PAT; do for s in 0 1 2; do SYS="$SYS,${arm}_s$s=$TD/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
$CPUT python3 -u $C/fh_blend.py --ds gdelt25x --base tirgn --stage test --source fullrows --fullrows $FR --recb_dir $W/recb_stage2 \
    --sel $W/controls/sel_gdelt25x.json --systems $SYS --threads $WORKERS --out $W/test_out > $L/blend_test.log 2>&1; tm blend_test $?
RSHA=$(sha256sum $FH/freeze/fh_decision_rule.json | cut -d' ' -f1)
cat > $W/test_out/fh_agg_cfg.json <<EOF
{"dd": "$DD", "strata": "$W/strata_test.npz", "official": "$W/test_out/official_test_gdelt25x_tirgn.npz",
 "blend_test_json": "$W/test_out/blend_test_gdelt25x_tirgn.json", "recb_ranks": "$W/recb_stage2/recb_test_ranks_GDELT25X.npz",
 "apply_json": "$W/train_tirgn/apply_fh.json", "feat_summary_test": "$W/feat_tirgn/feat_summary_test.json",
 "hash_check_passed": true, "rule_sha256": "$RSHA", "deviations": []}
EOF
$CPU python3 -u $C/fh_agg.py --cfg $W/test_out/fh_agg_cfg.json --rule $FH/freeze/fh_decision_rule.json --out $W/test_out/fh_verdict.json > $L/agg.log 2>&1; tm agg $?
for f in $W/test_out/fh_verdict.json $W/test_out/fh_agg_cfg.json $W/test_out/blend_test_gdelt25x_tirgn.json $W/train_tirgn/apply_fh.json \
         $W/feat_tirgn/feat_summary_test.json $W/lists_tirgn/lists_summary_test.json $W/prep_summary_stage2.json $W/base_tirgn/rows_test_log.json \
         $W/recb_stage2/recb_test_meta_GDELT25X.json; do [ -f $f ] && cp $f $J/h/; done
cp $L/*.log $L/stage2_verify.json $J/h/ 2>/dev/null
echo "[stage2] end $(date -u +%FT%TZ)" >> $L/stage2_test.log; cp $L/stage2_test.log $J/h/
