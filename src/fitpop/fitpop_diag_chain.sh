#!/bin/bash
# fitpop diagnostics: size-matched refits (training only, harness ranks) + test gold-location flags.  GPU 0 only.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v78; FP=$V/fitpop; CD=$SRC/fitpop   # code is read from the release tree
# (code staging of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
RUNG="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
CPU="env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $MNT"
guard() { for i in $(seq 1 12); do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
            echo "GPU0 free MiB $F util $U% before $1 at $(date -u)" >> chain.log; [ "$F" -gt 20000 ] && return 0; sleep 300; done; return 1; }
$CPU python3 -u $CD/fitpop_diag.py --mode prep --v78 $V --out /workspace > prep.log 2>&1; tm prep $?
for cfg in "S1 smallpedia tirgn" "S2 polecat tirgn" "S3 polecat cogn"; do set -- $cfg; c=$1; DS=$2; b=$3; W=$V/$DS
  for run in "sa SA_W100" "pc LAD_C_$b,LAD_S_$b,LAD_T_$b" "pt HRCOV_RE_$b"; do set -- $run; tag=$1; ARMS=$2
    guard sm_${c}_$tag || { echo "GPU0 busy, stop" >> chain.log; exit 3; }
    $RUNG python3 -u $CD/v78_train_fitpop.py --ds $DS --pools $W/pools --arms $ARMS --fit_rows $FP/sizematch_${c}_$tag.npy --out $FP/${c}_sm_$tag > train_sm_${c}_$tag.log 2>&1; tm train_sm_${c}_$tag $?
    cp $FP/${c}_sm_$tag/train_v78.json $J/train_sm_${c}_$tag.json
  done
done
$CPU python3 -u $CD/fitpop_diag.py --mode collect --v78 $V --out /workspace > collect.log 2>&1; tm collect $?
