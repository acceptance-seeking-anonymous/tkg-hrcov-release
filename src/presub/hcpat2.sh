#!/bin/bash
# G2 round, part 2 (investigators' go-ahead for G2): HRCOV_PAT (bridge-free; definition of hcpat_definitional / G2) on the two remaining decision
# bases -- polecat LogCL (pilot-1 checkpoint) and smallpedia TiRGN (pilot-2, valid-best epoch 20) -- trained with hcpat_train.py
# (seeds 0-2) and scored by the official py-tgb evaluator; GPU 1 of serverA only, <= 4 CPUs.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; G=$H/presub_hcpat2; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > gpu_at_start.txt
GPU1="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 G2_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 HD_HAR=$O HD_HOST=local"
# ---- polecat LogCL
TS=$G/train_polecat_logcl
$GPU1 python3 -u $SRC/hardening/hcpat_train.py --layout pc --feat $T/pilot_polecat/work/feat_logcl --lists $T/pilot_polecat/work/lists_logcl --out $TS --label polecat_logcl > train_polecat_logcl.log 2>&1; tm train_polecat_logcl $?
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPU1 bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base logcl_hcpat --stage test --source replay --ckpt $T/pilot_polecat/dumps/logcl_POLECAT_seed42.pt --L 10 --recb_dir $H/work/recb --no_controls --systems $SYS --out $G" > official_polecat_logcl.log 2>&1; tm official_polecat_logcl $?
# ---- smallpedia TiRGN
TS2=$G/train_smallpedia_tirgn
$GPU1 python3 -u $SRC/hardening/hcpat_train.py --layout sp --feat $T/pilot_smallpedia/work/feat_tirgn --out $TS2 --label smallpedia_tirgn > train_smallpedia_tirgn.log 2>&1; tm train_smallpedia_tirgn $?
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS2/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPU1 bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_hcpat --stage test --source fullrows --fullrows none,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32 --recb_dir $H/work/recb --no_controls --systems $SYS --out $G" > official_smallpedia_tirgn.log 2>&1; tm official_smallpedia_tirgn $?
cp $G/official_test_polecat_logcl_hcpat.npz $G/official_test_smallpedia_tirgn_hcpat.npz $G/blend_test_*_hcpat.json $J/ 2>/dev/null
cp $TS/train_hcpat.json $J/train_hcpat_polecat_logcl.json; cp $TS2/train_hcpat.json $J/train_hcpat_smallpedia_tirgn.json
echo ALLDONE >> chain.log
