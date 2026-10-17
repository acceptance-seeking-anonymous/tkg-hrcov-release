#!/bin/bash
# pilot_smallpedia / sp_base_grid.sh <model> <gpu> <epochs-list>
# Trains + dumps a frozen base with the UNMODIFIED harness base_dump.py for each epoch budget in the list
# (valid-based selection of the epoch budget happens later from the dump logs; test is never used).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
M=$1; G=$2; EL=${3:-12,30,60}
H=$WORK_DIR; P=$H/tracks/pilot_smallpedia; DD=$H/tracks/datasets/harness_data/tkgl-smallpedia
mkdir -p $P/dumps $P/code $P/work $P/stage
for E in ${EL//,/ }; do
  mkdir -p $P/dumps/e$E
  T0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
     \
     python3 -u $SRC/harness/base_dump.py --model $M --data_dir $DD --dataset SMALLPEDIA --out_dir $P/dumps/e$E --epochs $E --threads 8 2>&1 | tee base_${M}_e$E.log
  cp ${M}_SMALLPEDIA_dump_log.json dump_log_${M}_e$E.json
  echo "[stage-time] ${M} e$E $(( $(date +%s) - T0 ))s" | tee -a stage_times_${M}.txt
done
ls -la $P/dumps/*
