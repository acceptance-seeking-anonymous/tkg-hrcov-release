#!/bin/bash
# v7 B4: G1 (base-free learned recurrency) on tkgl-icews Stage B, serverA GPU 1 only (guard), CPU for features' evidence part.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; P=$T/pilot_icews; B=$V/b4; mkdir -p $B/code
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 16000 ]; }
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 G1_THREADS=8 PI_STAGE=stageB PI_GPU_MEM_GB=20 PYTHONPATH=$SRC/pilot_icews:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
$GPU python3 -u $SRC/v7/v7_g1_features_icews.py --ds icews --common pi_common --recb_dir $P/work_stageB/recb --out $J/feat_smoke --W 150 --bs 256 --threads 8 --smoke 2000 > feat_smoke.log 2>&1; tm feat_smoke $?
grep -q "test" feat_smoke.log && [ -f $J/feat_smoke/test_q.npz ] || { echo "feature smoke failed" >> chain.log; tail -20 feat_smoke.log >> chain.log; exit 4; }
cp $J/feat_smoke/g1_features_summary.json $J/g1_features_summary_smoke.json 2>/dev/null
$GPU python3 -u $SRC/v7/v7_g1_features_icews.py --ds icews --common pi_common --recb_dir $P/work_stageB/recb --out $B/feat_icews --W 150 --bs 256 --threads 8 > feat.log 2>&1; tm feat $?
cp $B/feat_icews/g1_features_summary.json $J/ 2>/dev/null
guard || { echo "GPU1 busy before train -> abort" >> chain.log; exit 3; }
$GPU python3 -u $SRC/v7/v7_g1_train.py --feat $B/feat_icews --out $B/train_icews --arms G1_K150 > train.log 2>&1; tm train $?
cp $B/train_icews/train_g1.json $J/train_g1_icews.json 2>/dev/null; cp $B/train_icews/ranks_test.npz $J/ranks_test_g1_icews.npz 2>/dev/null
SYS=""; for s in 0 1 2; do [ -f $B/train_icews/promlist_G1_K150_s$s.npy ] && SYS="$SYS,G1_K150_s$s=$B/train_icews/promlist_G1_K150_s$s.npy"; done; SYS=${SYS#,}
guard || { echo "GPU1 busy before official -> abort" >> chain.log; exit 3; }
$GPU bash -c "python3 -u $SRC/v7/v7_pi_blend.py --ds icews --base recb_g1 --stage test --source recb --no_controls --recb_dir $P/work_stageB/recb --systems $SYS --bs 256 --out $B" > official.log 2>&1; tm official $?
cp $B/blend_test_icews_recb_g1.json $J/ 2>/dev/null
echo ALLDONE >> chain.log
