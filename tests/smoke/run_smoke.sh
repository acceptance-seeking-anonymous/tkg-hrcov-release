#!/bin/bash
# tests/smoke/run_smoke.sh [N] -- end-to-end smoke test of the HRCOV_PAT pipeline on TGB 2.0 tkgl-polecat:
# data download + conversion -> TGB id bridge -> RE-GCN base (1 epoch, full data) + harness-order score dump -> prep (full
# splits, RecB-lite skipped) -> candidate lists for the first N validation-sample and test queries -> exact base lookup + official TGB evaluation of the base -> features
# -> HRCOV_PAT re-ranker (3 seeds, 2 epochs) -> official TGB evaluation of HRCOV_PAT -> checks.
# Numbers are NOT comparable with the paper (1-epoch base, N queries); the test checks that every stage runs and that the
# official evaluator agrees with the harness ranks.  The tkgl-polecat RE-GCN base needs a GPU with more than 24 GB (it ran on
# a 96 GB NVIDIA RTX PRO 6000; on a 24 GB RTX 4090 the first epoch runs out of memory).  About 12 min, 6 GB disk.
#   source configs/env.sh && bash tests/smoke/run_smoke.sh 3000
set -eo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$R/configs/env.sh"
N=${1:-3000}; WK=${SMOKE_WORKERS:-8}
P=$WORK_DIR/tracks/pilot_polecat; DD=$WORK_DIR/tracks/datasets/harness_data/tkgl-polecat
mkdir -p "$P/dumps" "$P/work" "$P/stage"
OUT=${SMOKE_OUT:-$PWD/smoke_out}; mkdir -p "$OUT"; cd "$OUT"
T0=$(date +%s); tm() { echo "[stage-time] $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times.txt; }
GPUE="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
python3 -u "$R/data/download_tgb.py" --names tkgl-polecat 2>&1 | tee download.log; tm data
python3 -u "$SRC/pilot_polecat/pc_tgb_verify.py" 2>&1 | tee verify.log; tm tgb_verify
if [ ! -f "$P/dumps/regcn_scores_POLECAT_traintest.npz" ]; then
  $GPUE python3 -u "$SRC/harness/base_dump.py" --model regcn --data_dir "$DD" --dataset POLECAT --out_dir "$P/dumps" --epochs 1 --seed 42 --threads 8 2>&1 | tee base.log
fi; tm base
# prep on the full splits (base-independent; ~1-4 min): pc_prep --smoke subsamples test rows differently from pc_lists --smoke,
# so the strata of a smoke prep would not line up with the smoke lists
python3 -u "$SRC/pilot_polecat/pc_prep.py" --workers $WK --skip_recb 2>&1 | tee prep.log; tm prep
bash "$SRC/pilot_polecat/pc_chain.sh" regcn "$GPU" $WK $N lists,lookup,features 2>&1 | tee chain.log; tm lists_lookup_features
$GPUE python3 -u "$SRC/hardening/hcpat_train.py" --layout pc --feat "$P/work/feat_regcn_smoke" --lists "$P/work/lists_regcn_smoke" \
  --out "$P/work/hcpat_regcn_smoke" --label polecat_regcn --smoke $N 2>&1 | tee hcpat.log; tm hcpat
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$P/work/hcpat_regcn_smoke/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPUE python3 -u "$SRC/pilot_polecat/pc_lookup.py" --base regcn --mode systems --systems "$SYS" --smoke $N 2>&1 | tee systems.log; tm official_systems
python3 "$R/tests/smoke/check_smoke.py" --out smoke_result.json; tm check
