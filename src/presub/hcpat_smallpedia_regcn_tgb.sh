#!/bin/bash
# presub / hcpat_smallpedia_regcn_tgb.sh -- definitional re-run extension: HRCOV_PAT on the G3 base regcn_tgb (TGB 2.0 RE-GCN config on
# tkgl-smallpedia), which G3 made a decision base; train + official scoring on the serverB RTX 4090.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD; H=$WORK_DIR/tracks/hardening; M=$WORK_DIR; G=$H/presub_hcpat; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > gpu_at_start.txt
ENVS="env PC_HARNESS=$M PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local G2_THREADS=8"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS"
TS=$G/train_smallpedia_regcn_tgb; D=$H/presub_g3/dumps
$GPU python3 -u $SRC/hardening/hcpat_train.py --layout sp --feat $M/tracks/sp_g3/work/feat_regcn --out $TS --label smallpedia_regcn_tgb > train_smallpedia_regcn_tgb.log 2>&1; tm train $?
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tgb_hcpat --stage test --source fullrows --fullrows $D/regcn_valid_logp.f32,$D/regcn_test_logp.f32 --recb_dir $H/work/recb --no_controls --systems $SYS --out $G" > official_sp_regcn_tgb.log 2>&1; tm official $?
cp $G/official_test_smallpedia_regcn_tgb_hcpat.npz $G/blend_test_smallpedia_regcn_tgb_hcpat.json $W/ 2>/dev/null; cp $TS/train_hcpat.json $W/train_hcpat_smallpedia_regcn_tgb.json 2>/dev/null
echo ALLDONE >> chain.log
