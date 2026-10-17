#!/bin/bash
# G8 option B full training on one GPU (README config, 20 epochs, prefetch). Resumable: re-run continues from
# $G/run_B/ckpt_last.pt. Usage: bash g8_train.sh <mode B|A> <epochs>   (after g8_setup.sh)
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -u
MODE=${1:-B}; EP=${2:-20}
J=$PWD
G=$WORK_DIR/tracks/hardening/g8_cogntke
RD=$G/run_$MODE
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0})
if [ "$U" -gt 2000 ]; then echo "GPU busy ($U MiB) -> abort"; exit 3; fi
RUN="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$G/pylib_ts:$G/pylib:${PYTHONPATH:-} G8_HOST=local"
cd $G/CognTKE   # CognTKE code reads ./data/POLECAT from its repository root
$RUN python3 -u $SRC/g8_cogntke/g8_train.py --run_dir $RD --mode $MODE --epochs $EP >> $G/train_$MODE.log 2>&1
echo "train exit $?"
tail -30 $G/train_$MODE.log
cp $RD/epoch_log.json $J/ 2>/dev/null; cp $G/train_$MODE.log $J/ 2>/dev/null
