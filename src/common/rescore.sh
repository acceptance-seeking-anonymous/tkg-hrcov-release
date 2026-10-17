#!/bin/bash
# re-ranker score recomputation (CPU, read-only) for the default-base main-table rows: polecat TiRGN HR / HRCOV / HRCOV_PAT,
# smallpedia TiRGN HRCOV_PAT.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; T=$WORK_DIR/tracks; H=$T/hardening
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
CPU="env RS_THREADS=12 OMP_NUM_THREADS=12"
TD=$T/pc_tirgn/work/train_tirgn; HP=$H/presub_hcpat/train_polecat_tirgn
$CPU python3 -u $SRC/common/rescore_check.py --label polecat_tirgn --feat $T/pc_tirgn/work/feat_tirgn \
  --arm HR=$TD/train_tirgn.json:$TD/models --arm HRCOV=$TD/train_tirgn.json:$TD/models --arm HRCOV_PAT=$HP/train_hcpat.json:$HP/models \
  --out ./rescore_polecat_tirgn.json > rs_pc.log 2>&1; tm rs_pc $?
HS=$H/presub_hcpat2/train_smallpedia_tirgn
$CPU python3 -u $SRC/common/rescore_check.py --label smallpedia_tirgn --feat $T/pilot_smallpedia/work/feat_tirgn --pre A_ \
  --arm HRCOV_PAT=$HS/train_hcpat.json:$HS/models --out ./rescore_smallpedia_tirgn.json > rs_sp.log 2>&1; tm rs_sp $?
echo ALLDONE >> chain.log
