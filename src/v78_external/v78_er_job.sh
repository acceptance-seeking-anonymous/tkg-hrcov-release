#!/bin/bash
# EdgeReMIND on one dataset: (1) unmodified authors' example script (gate), (2) er_run.py faithful re-train + per-query dumps.
# usage: bash v78_er_job.sh <dataset> <seed> <workers> [gate|nogate] [extra er_run args...]
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
# one BLAS/OpenMP thread per process: the default (48 per worker x 16 workers) drove host load to ~500 on the first attempt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W=$PWD; DSN=$1; SEED=$2; NW=$3; GATE=${4:-gate}; shift 4; EXTRA="$@"
B=$WORK_DIR/tracks/v78_external; E=$B/env/er/bin/python
# (code staging of the original job removed: the scripts are read from $SRC/v78_external)
RD=$B/runs/edgeremind/$DSN/seed$SEED; mkdir -p $RD/gate $RD/dump
echo "host $(hostname) start $(date -u +%FT%TZ) nproc $(nproc)"; free -g | head -2; uptime
if [ "$GATE" = "gate" ]; then
  cd $RD/gate
  $E $SRC/v78_external/memmon.py --out mem_gate.json -- $E $B/repos/EdgeReMIND/examples/linkproppred/tkgl/edgeremind.py \
      --dataset $DSN --seed $SEED --num-workers $NW --log-file-path gate_metrics.log > gate.log 2>&1
  echo "gate rc=$?"; tr '\r' '\n' < gate.log | grep -v "it/s\]$\|s/it\]$" | tail -60 > gate_tail.txt
  cp mem_gate.json gate_tail.txt gate_metrics.log $W/ 2>/dev/null
fi
cd $RD
$E $SRC/v78_external/memmon.py --out $RD/mem_dump.json -- $E $SRC/v78_external/er_run.py --dataset $DSN --seed $SEED --num-workers $NW --out $RD/dump $EXTRA > $RD/dump.log 2>&1
echo "dump rc=$?"
tr '\r' '\n' < $RD/dump.log | grep -v "it/s\]$\|s/it\]$" | tail -80 > $W/dump_tail.txt
cp $RD/mem_dump.json $RD/dump/meta.json $RD/dump/val_curve.json $W/ 2>/dev/null
ls -la $RD/dump $RD/dump/* | head -60
echo "end $(date -u +%FT%TZ)"
