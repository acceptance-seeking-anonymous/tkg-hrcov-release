#!/bin/bash
# fitpop audit: common-intersection refit of SA, +C, +S, +T, HRcov (HRCOV_RE) per configuration + HRCOV_RE own-population
# reproduction; official scoring as in v78_chain.sh.  GPU 0 only (never GPU 1); one GPU process at a time.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v78; FP=$V/fitpop
CD=$SRC/fitpop   # code is read from the release tree
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
RUNG="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
TGB=""
guard() { for i in $(seq 1 12); do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
            echo "GPU0 free MiB $F util $U% before $1 at $(date -u)" >> chain.log; [ "$F" -gt 20000 ] && return 0; sleep 300; done; return 1; }
for cfg in "S1 smallpedia tirgn" "S2 polecat tirgn" "S3 polecat cogn"; do set -- $cfg; c=$1; DS=$2; b=$3; W=$V/$DS; OUT=$FP/${c}_int; OUTR=$FP/${c}_own
  guard train_int_$c || { echo "GPU0 busy, stop" >> chain.log; exit 3; }
  $RUNG python3 -u $CD/v78_train_fitpop.py --ds $DS --pools $W/pools --arms SA_W100,LAD_C_$b,LAD_S_$b,LAD_T_$b,HRCOV_RE_$b --fit_rows $FP/fitint_$c.npy --out $OUT > train_int_$c.log 2>&1; tm train_int_$c $?
  guard train_own_$c || { echo "GPU0 busy, stop" >> chain.log; exit 3; }
  $RUNG python3 -u $CD/v78_train_fitpop.py --ds $DS --pools $W/pools --arms HRCOV_RE_$b --out $OUTR > train_own_$c.log 2>&1; tm train_own_$c $?
  cp $OUT/train_v78.json $J/train_int_$c.json; cp $OUTR/train_v78.json $J/train_own_$c.json
  SYS=""; for a in SA_W100 LAD_C_$b LAD_S_$b; do for s in 0 1 2; do SYS="$SYS,${a}_s$s=$OUT/promlist_${a}_s$s.npy"; done; done; SYS=${SYS#,}
  guard off_hist_$c || { echo "GPU0 busy, stop" >> chain.log; exit 3; }
  $RUNG bash -c "$TGB python3 -u $SRC/v78/v78_official.py --ds $DS --label fitpop_${c}_hist --systems $SYS --out $FP" > off_hist_$c.log 2>&1; tm off_hist_$c $?
  SYS=""; for s in 0 1 2; do SYS="$SYS,LAD_T_${b}_s$s=$OUT/promlist_LAD_T_${b}_s$s.npy,HRCOV_RE_${b}_s$s=$OUT/promlist_HRCOV_RE_${b}_s$s.npy,HRCOVOWN_${b}_s$s=$OUTR/promlist_HRCOV_RE_${b}_s$s.npy"; done; SYS=${SYS#,}
  if [ $DS = smallpedia ]; then SRC="--source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
  elif [ $b = tirgn ]; then SRC="--source tirgn --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt"
  else SRC="--source cogn --ckpt $H/g8_cogntke/dumps"; fi
  guard off_base_$c || { echo "GPU0 busy, stop" >> chain.log; exit 3; }
  $RUNG bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds $DS --base fitpop_${c}_$b --stage test $SRC --recb_dir $H/work/recb --no_controls --systems $SYS --bs 256 --out $FP" > off_base_$c.log 2>&1; tm off_base_$c $?
  cp $FP/official_test_${DS}_fitpop_${c}_hist.npz $FP/official_test_${DS}_fitpop_${c}_$b.npz $J/ 2>/dev/null
done
echo CHAIN_DONE >> chain.log
