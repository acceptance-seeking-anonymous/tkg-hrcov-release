#!/bin/bash
# presub / hcpat_polecat_tirgn.sh -- definitional re-run: HRCOV_PAT re-ranker (3 seeds) on the hardening TiRGN polecat HRCOV features, serverA CPU (8 threads).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD; R=$WORK_DIR; H=$R/tracks/hardening; G=$H/presub_hcpat; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
P=$R/tracks/pc_tirgn/work
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
uptime > load_at_start.txt
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 G2_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local \
  \
  python3 -u $SRC/hardening/hcpat_train.py --layout pc --feat $P/feat_tirgn --lists $P/lists_tirgn --out $G/train_polecat_tirgn --label polecat_tirgn > train_polecat_tirgn.log 2>&1; tm train_polecat_tirgn $?
echo ALLDONE >> chain.log
