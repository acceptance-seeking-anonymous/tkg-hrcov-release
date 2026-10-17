#!/bin/bash
# pilot_smallpedia / sp_chain.sh <base> <gpu> <workers> [smoke_n] [stages]
# fullrows (GPU; RE-GCN/LogCL only) -> lists (CPU) -> features (CPU) -> train (GPU) -> listctl (CPU) -> official (CPU) -> collect
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; G=$2; WK=$3; SM=${4:-0}; STAGES=${5:-fullrows,lists,features,train,listctl,official,collect}
H=$WORK_DIR; P=$H/tracks/pilot_smallpedia
# (code staging step of the original run removed: code is read from $SRC)
SMA=""; [ "$SM" != "0" ] && SMA="--smoke $SM"
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
GPU="env CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $CPU"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
T0=$(date +%s); tm() { echo "[stage-time] $1 $(( $(date +%s) - T0 ))s" | tee -a stage_times_$B.txt; }
if has fullrows && [ "$B" != "tirgn" ]; then $GPU python3 -u $SRC/pilot_smallpedia/sp_fullrows.py --base $B 2>&1 | tee fullrows_$B.log; tm fullrows; fi
if has lists; then $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers $WK $SMA 2>&1 | tee lists_$B.log; tm lists; fi
if has features; then $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B $SMA 2>&1 | tee feat_$B.log; tm features; fi
if has train; then $GPU python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B $SMA 2>&1 | tee train_$B.log; tm train; fi
if has listctl && [ "$SM" == "0" ]; then $CPU python3 -u $SRC/pilot_smallpedia/sp_listctl.py --base $B 2>&1 | tail -3 | tee listctl_$B.log; tm listctl; fi
if has official && [ "$SM" == "0" ]; then
  TD=$P/work/train_$B
  $CPU bash -c "python3 -u $SRC/pilot_smallpedia/sp_official.py --base $B --workers $WK; python3 -u $SRC/pilot_smallpedia/sp_official.py --base $B --workers $WK --systems HR_s0=$TD/promlist_HR_s0.npy,HRCOV_s0=$TD/promlist_HRCOV_s0.npy,HRCOV_ST_s0=$TD/promlist_HRCOV_ST_s0.npy,HR_ST_s0=$TD/promlist_HR_ST_s0.npy" 2>&1 | tail -6 | tee official_$B.log
  tm official
fi
if has collect && [ "$SM" == "0" ]; then
  W=$P/work; D=bundle_$B; mkdir -p $D
  cp $W/train_$B/ranks_test.npz $W/train_$B/ranks_select.npz $W/train_$B/train_$B.json $D/
  cp $W/feat_$B/test_q.npz $D/; cp feat_summary_$B.json lists_summary_$B.json listctl_$B.json official_*_$B.json official_*_${B}_ranks.npz $D/ 2>/dev/null || true
  cp fullrows_$B.json $D/ 2>/dev/null || true
  cp $P/dumps/${B}_SMALLPEDIA_dump_log.json $D/dump_log_$B.json 2>/dev/null || true
  tar czf bundle_$B.tgz $D; ls -la bundle_$B.tgz; tm collect
fi
ls -la
