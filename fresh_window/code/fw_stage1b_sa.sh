#!/bin/bash
# phase2_fresh_window / fw_stage1b_sa.sh -- STAGE 1 part S only (rerun: the first S attempt failed at the cut-ts parse, nothing was computed) on GDELTFW (validation only; no test row is scored).
#   C (CPU):  HRcov valid pattern lists + 25 features (fh_lists / fh_features), BLEND grid on valid + selection (fh_blend /
#             hd_select), HRcov re-ranker fit + freeze (fh_train --arms HRCOV_PAT, seeds 0-2)
#   S:        RecB_xf (v76_recb_xf_select on the selection part + valid CSR), SA history order / conf_hist lists / pools on
#             valid (GPU 0 / CPU), SA_W100 fit + freeze (fw_sa_train --stage fit)
set +e
J=$PWD
H=/home/anon/cs_scratch/open_ideas_harness
FW=$H/tracks/phase2_fresh_window
P=$FW/GDELTFW; W=$P/work; C=$FW/code; CS=$FW/code_sa; L=$FW/logs; RUN=$P/run_stage1
mkdir -p $W/controls $W/recb_xf $W/sa/hist $W/sa/lists $W/sa/pools $W/sa/train $L $RUN $CS
cp -f $J/fw_stage1b_sa.sh $C/
for f in fw_ext.py fw_pc_shim.py fw_sa_official.py fw_sa_train.py hd_blend.py hd_recb.py v76_recb_xf_select.py v78_common.py v78_hist.py v78_lists.py v78_official.py v78_pools.py v78_train.py; do cp -f $J/$f $CS/$f; done
WORKERS=${WORKERS:-8}
export FH_GPU_CAP_GB=${FH_GPU_CAP_GB:-30}
source $C/fw_gpu_wait.sh
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u +%FT%TZ)" >> $L/stage1_chain.log; }
echo "[stage1b-sa] start $(date -u +%FT%TZ) host $(hostname) workers $WORKERS" >> $L/stage1_chain.log; uptime >> $L/stage1_chain.log
DD=$FW/data/GDELTFW
ENVS="-e PC_HARNESS=$H -e FH_DIR=$FW -e PC_P=$P -e FH_DD=$DD -e G2_VALID_ROWS=$W/valid_sample_idx.npy -e PYTHONDONTWRITEBYTECODE=1 -e V78_RECB_XF=$W/recb_xf -e FW_CODE=$C -e HD_HOST=serverA_(CPU)"
MNT="-v /home/anon/cs_scratch:/home/anon/cs_scratch -w $RUN"
CPU="docker run --rm --cpus $WORKERS -e OMP_NUM_THREADS=1 -e G2_THREADS=$WORKERS $ENVS $MNT tkg-gpu:latest"
CPUT="docker run --rm --cpus $WORKERS -e OMP_NUM_THREADS=$WORKERS -e G2_THREADS=$WORKERS -e V7_THREADS=$WORKERS $ENVS $MNT tkg-gpu:latest"
for f in $W/fullrows/tirgn_GDELTFW_valid_scores_f32.bin $W/recb/recb_csr_valid_GDELTFW.npz $W/recb/recb_valid_ranks_clean_GDELTFW.npy; do
  [ -f $f ] || { echo "[stage1b] missing $f" >> $L/stage1_chain.log; exit 3; }
done
# ---------------- S: RecB_xf + SA
(
  CUT=$($CPU python3 -c "
import numpy as np
q = np.loadtxt('$DD/valid.txt', dtype=np.int64, usecols=(3,)); u = np.unique(q); print(int(u[int(0.7 * len(u))]))" 2>/dev/null | tail -1)
  echo "[stage1b] validation cut ts = $CUT" >> $L/stage1_chain.log
  case "$CUT" in ""|*[!0-9]*) echo "[stage1b-sa] bad cut [$CUT]" >> $L/stage1_chain.log; exit 4;; esac
  $CPU python3 -u $CS/v76_recb_xf_select.py --data_dir $DD --name GDELTFW --cut $CUT --orig_dir $W/recb --out $W/recb_xf > $L/recb_xf_select.log 2>&1; tm recb_xf_select $?
  $CPU python3 -u $C/fh_recb.py --data_dir $DD --name GDELTFW --ts_step 1 --stage test --only_valid --workers $WORKERS --P 16 --out $W/recb_xf > $L/recb_xf_validcsr.log 2>&1; tm recb_xf_validcsr $?
  fh_gpu_wait
  ENVG=$(echo "$ENVS" | sed "s|HD_HOST=serverA_(CPU)|HD_HOST=serverA_(GPU${FH_GPU}_RTX_PRO_6000)|")
  GPUR="docker run --rm --gpus device=$FH_GPU -e CUDA_VISIBLE_DEVICES=0 --cpus $WORKERS -e OMP_NUM_THREADS=$WORKERS -e V7_THREADS=$WORKERS $ENVG $MNT tkg-gpu:latest"
  $GPUR python3 -u $CS/v78_hist.py --ds gdeltfw --out $W/sa/hist --splits valid > $L/sa_hist_valid.log 2>&1; tm sa_hist_valid $?
  $CPU python3 -u $CS/v78_lists.py --ds gdeltfw --hist $W/sa/hist --out $W/sa/lists --workers $WORKERS --splits valid > $L/sa_lists_valid.log 2>&1; tm sa_lists_valid $?
  $GPUR python3 -u $CS/v78_pools.py --ds gdeltfw --hist $W/sa/hist --lists $W/sa/lists --lookup none=none --out $W/sa/pools --splits valid > $L/sa_pools_valid.log 2>&1; tm sa_pools_valid $?
  $GPUR python3 -u $CS/fw_sa_train.py --stage fit --ds gdeltfw --pools $W/sa/pools --arms SA_W100 --out $W/sa/train > $L/sa_fit.log 2>&1; tm sa_fit $?
) &
PS=$!
wait $PS
mkdir -p $J/h
cp $L/recb_xf_*.log $L/sa_*.log $J/h/ 2>/dev/null
for f in $W/recb_xf/recb_selection_GDELTFW.json $W/recb_xf/recb_validcsr_meta_GDELTFW.json \
         $W/sa/hist/hist_summary_valid.json $W/sa/lists/lists_summary.json $W/sa/pools/pools_summary_valid.json $W/sa/train/train_fw_sa.json; do
  [ -f $f ] && cp $f $J/h/
done
echo "[stage1b-sa] end $(date -u +%FT%TZ)" >> $L/stage1_chain.log
cp $L/stage1_chain.log $J/h/
