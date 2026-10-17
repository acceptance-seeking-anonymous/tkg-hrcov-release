#!/bin/bash
# hardening / pc_epoch_curve.sh -- diagnostic for the polecat pilot-1 bases (RE-GCN, LogCL; history 10, lr 1e-3, time-ordered,
# the pilot-1 configuration): per-epoch FULL-valid filtered MRR for epochs 1..12 (patience 12 = full curve), no dump.
# Answers whether the pilot-1 fixed 12-epoch checkpoints are the valid-best epoch of that configuration.
# serverA GPU 0 ONLY (the 24 GB 4090 OOMs at history 10 on polecat), queued after the TiRGN polecat job.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
JD=$WORK_DIR/runs
D=$M/tracks/datasets/harness_data/tkgl-polecat
# (code staging step of the original run removed: code is read from $SRC)
while grep -q '"state":"running"' $JD/_status.json 2>/dev/null; do sleep 120; done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU0 used MiB at start: $used" >> curve.log
if [ "$used" -gt 2000 ]; then echo "GPU0 busy -> abort" >> curve.log; exit 3; fi
mkdir -p $H/pc_curve
RUN="env CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HAR=$M HD_HOST=local"
for B in regcn logcl; do
  echo "start $B $(date)" >> curve.log
  $RUN python3 -u $SRC/hardening/hd_base_train.py --model $B --data_dir $D --dataset POLECAT_curve --out_dir $H/pc_curve --epochs 12 --patience 12 \
     --grid "10,1e-3,0" --no_dump > curve_$B.log 2>&1
  echo "EXIT $B $? $(date)" >> curve.log
  cp $H/pc_curve/${B}_POLECAT_curve_grid_log.json $W/ 2>/dev/null
done
