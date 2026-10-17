#!/bin/bash
# G6 latency/memory profiling on serverA GPU 1 only (GPU 0 = G8 training, untouched).
# Usage: bash g6_run.sh [datasets]   -> per dataset: smoke (2 test ts, 1 repeat) then, if rc=0, the full registered run (10k sample, 1 warm-up + 3 repeats)
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
DSL=${1:-"polecat smallpedia"}
J=$PWD; O=$WORK_DIR; G=$O/tracks/hardening/g6_latency; mkdir -p $G/code $G/logs
# (code staging step of the original run removed: code is read from $SRC)
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 used MiB at start: $U" > chain.log
if [ "$U" -gt 2000 ]; then echo "GPU1 busy -> abort" >> chain.log; exit 3; fi
nvidia-smi > nvidia_smi_start.txt; uptime > load_at_start.txt; lscpu > lscpu.txt; free -g > free_start.txt
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
RUN="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1"
for DS in $DSL; do
  $RUN python3 -u $SRC/latency_g6/g6_profile.py --ds $DS --out $G --smoke_ts 2 --repeats 1 > g6_${DS}_smoke.log 2>&1; RC=$?; tm ${DS}_smoke $RC
  if [ $RC -ne 0 ]; then tail -5 g6_${DS}_smoke.log >> chain.log; continue; fi
  echo "$DS $(uptime)" >> load_per_ds.txt
  nvidia-smi -i ${GPU:-0},1 --query-gpu=index,timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > smi_${DS}.csv & SPID=$!
  $RUN python3 -u $SRC/latency_g6/g6_profile.py --ds $DS --out $G --repeats 3 > g6_${DS}.log 2>&1; tm ${DS}_full $?
  kill $SPID; tail -2 g6_${DS}.log >> chain.log
done
cp $G/g6_*.json $J/ 2>/dev/null; cp $J/*.log $J/*.csv $J/*.txt $G/logs/ 2>/dev/null
echo ALLDONE >> chain.log
