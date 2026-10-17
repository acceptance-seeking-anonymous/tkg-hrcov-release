#!/bin/bash
# scripts/run_latency_g6.sh -- per-query latency and memory of the plug-in (registration/g6_protocol.json): smoke, then 10k-query
# sample, 1 warm-up + 3 repeats per dataset (latency_g6/g6_run.sh); table with g6_build.py.  About 1 h on one GPU.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir latency_g6; bash $SRC/latency_g6/g6_run.sh ${1:-}; python3 $SRC/latency_g6/g6_build.py
