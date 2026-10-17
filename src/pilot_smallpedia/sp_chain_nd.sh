#!/bin/bash
# pilot_smallpedia / sp_chain_nd.sh <base> <gpu> <workers> [modes]   -- direct-static-edge leakage control (addendum)
# for each mode in {nodirect, nodirect_strict}: lists (CPU; pilot-2 bridge rankers, C_cov asserted identical) -> features (CPU)
# -> train HRCOV_ST only (GPU, 3 seeds) -> official py-tgb eval of seed 0 (CPU) -> collect bundle_<base>_<sfx>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; G=$2; WK=$3; MODES=${4:-nodirect,nodirect_strict}
H=$WORK_DIR; P=$H/tracks/pilot_smallpedia; W=$P/work
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $B $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times_$B.txt; }
for M in ${MODES//,/ }; do
  case $M in nodirect) S=_nd; NM=HRCOV_ST_NODIRECT;; nodirect_strict) S=_nds; NM=HRCOV_ST_NODIRECT_STRICT;; *) echo bad mode $M; exit 2;; esac
  CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SP_STATIC_MODE=$M"
  GPU="env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $CPU"
  $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers $WK 2>&1 | tee lists_$B$S.log; tm lists$S
  $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B 2>&1 | tee feat_$B$S.log; tm features$S
  $GPU python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B --arms HRCOV_ST 2>&1 | tee train_$B$S.log; tm train$S
  TD=$W/train_$B$S
  $CPU bash -c "python3 -u $SRC/pilot_smallpedia/sp_official.py --base $B --workers $WK --systems ${NM}_s0=$TD/promlist_HRCOV_ST_s0.npy" 2>&1 | tail -6 | tee official_$B$S.log; tm official$S
  D=bundle_$B$S; mkdir -p $D
  cp $TD/ranks_test.npz $TD/ranks_select.npz $TD/train_$B.json $D/
  cp $W/feat_$B$S/test_q.npz $D/; cp feat_summary_$B$S.json lists_summary_$B$S.json official_systems${S}_$B.json official_systems${S}_${B}_ranks.npz $D/ 2>/dev/null || true
  tar czf $D.tgz $D; ls -la $D.tgz; tm collect$S
done
ls -la
