#!/bin/bash
# pilot_icews / pi_chainB2.sh -- per-base STAGE B driver, official scoring split in two passes (run from a job dir on host serverA).
# usage: bash pi_chainB.sh <base> <gpu> <cpu_workers> [stages]
#   stages: lists,lookup,features,train,blendvalid,select,official_blend,official_sys,collect
#   official_blend = base + BASE(+)RecB + RecB-POOL (needs only the base + selection; run early, in parallel)
#   official_sys   = base + HRCOV_B* seeds (needs the re-ranker promoted lists), files tagged _sys
# Every stage writes under $P/work_stageB and is skipped when its completion marker exists (resume after outage).
# env GPU_CAP=<GB> caps every GPU stage process (co-scheduling on a shared card).
# Stage B env: PI_STAGE=stageB PI_TEST_FRAC=1.0 PI_BGRID=150,200,300 PI_LPLEN=400 (W_MAX = 350).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
B=$1; GPU=$2; NW=$3; STAGES=${4:-lists,lookup,features,train,blendvalid,select,official_blend,official_sys,collect}
P=$WORK_DIR/tracks/pilot_icews
W=$P/work_stageB; RB=$W/recb; BL=$W/blend
JD=$PWD
ENVB="env PI_STAGE=stageB PI_TEST_FRAC=1.0 PI_BGRID=150,200,300 PI_LPLEN=400 PI_WIN_T0=0 PI_GRID=1,2,3,4,6,8,10,12 PI_GRID_EXT="
CPU="env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $ENVB"
GPUC="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local PI_GPU_MEM_GB=${GPU_CAP:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVB"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
stamp() { echo "[stage-time] $1 $(date +%s) disk_free_GB=$(df -BG --output=avail $EXT_DIR | tail -1)" | tee -a chainB_$B.log; }
waitfor() { while [ ! -f "$1" ]; do sleep 120; done; }
# GPU stages start only when the card has >= GPU_NEED MiB free (default 26000), so co-tenants are never squeezed
waitgpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)" -lt "${GPU_NEED:-26000}" ]; do sleep 120; done; }
mkdir -p $BL
stamp start
if has lists && [ ! -f $W/lists_$B/lists_summary_$B.json ]; then
  $CPU python3 -u $SRC/pilot_icews/pi_lists.py --base $B --workers $NW 2>&1 | grep --line-buffered -v Warning | tee -a chainB_$B.log; stamp lists; fi
if has lookup && [ ! -f $W/lookup_$B/lookup_$B.json ]; then
  waitgpu; stamp gpu_free_ok
  eval $GPUC bash -c "'python3 -u $SRC/pilot_icews/pi_replay.py --base $B --mode lookup --no_official'" 2>&1 | tee -a chainB_$B.log; stamp lookup; fi
if has features && [ ! -f $W/feat_summary_$B.json ]; then
  $CPU python3 -u $SRC/pilot_icews/pi_features.py --base $B 2>&1 | grep --line-buffered -v Warning | tee -a chainB_$B.log; stamp features; fi
# pattern lists are folded into the feature tensors; drop the big per-query list arrays (regenerable with pi_lists.py).
# Files are root-owned (written in the container) -> delete from a container.
if [ -f $W/feat_summary_$B.json ] && ls $W/lists_$B/*_Lpat.npy >/dev/null 2>&1; then
  $CPU bash -c "rm -f $W/lists_$B/*_Lpat.npy $W/lists_$B/*_Ccov.npy $W/lists_$B/*_Xcov.npy"; stamp lists_pruned; fi
if has train && [ ! -f $W/train_$B/train_$B.json ]; then
  waitgpu; stamp gpu_free_ok
  eval $GPUC python3 -u $SRC/pilot_icews/pi_train.py --base $B 2>&1 | tee -a chainB_$B.log; stamp train; fi
if has blendvalid && [ ! -f $BL/blend_valid_icews_$B.json ]; then
  waitfor $RB/recb_csr_valid_tkgl-icews.npz; stamp recb_csr_valid_present
  waitgpu; stamp gpu_free_ok
  eval $GPUC python3 -u $SRC/pilot_icews/pi_blend.py --ds icews --base $B --stage valid --source replay --ckpt $P/dumps/${B}_stageB_best.pt --L 10 --recb_dir $RB --out $BL 2>&1 | tee -a chainB_$B.log; stamp blendvalid; fi
if has select && [ ! -f $BL/sel_icews_$B.json ]; then
  $CPU python3 -u $SRC/hardening/hd_select.py --ds icews --bases $B --dir $BL --out $BL/sel_icews_$B.json 2>&1 | tee -a chainB_$B.log; stamp select; fi
if has official_blend && [ ! -f $BL/blend_test_icews_$B.json ]; then
  waitfor $RB/recb_csr_test_tkgl-icews.npz; waitfor $BL/sel_icews_$B.json; stamp official_blend_inputs_present
  waitgpu; stamp gpu_free_ok
  eval $GPUC bash -c "'python3 -u $SRC/pilot_icews/pi_blend.py --ds icews --base $B --stage test --source replay --ckpt $P/dumps/${B}_stageB_best.pt --L 10 --recb_dir $RB --sel $BL/sel_icews_$B.json --out $BL'" 2>&1 | tee -a chainB_$B.log; stamp official_blend; fi
if has official_sys && [ ! -f $BL/blend_test_icews_${B}_sys.json ]; then
  waitfor $W/train_$B/train_$B.json; stamp official_sys_inputs_present
  SYS=$(python3 -c "import json;d=json.load(open('$W/train_$B/train_$B.json'));print(','.join(f.replace('promlist_','').replace('.npy','')+'=$W/train_$B/'+f for f in d['promlists']))")
  echo "systems: $SYS" | tee -a chainB_$B.log
  waitgpu; stamp gpu_free_ok
  eval $GPUC bash -c "'python3 -u $SRC/pilot_icews/pi_blend.py --ds icews --base $B --stage test --source replay --ckpt $P/dumps/${B}_stageB_best.pt --L 10 --base_only --systems $SYS --out_tag _sys --out $BL'" 2>&1 | tee -a chainB_$B.log; stamp official_sys; fi
if has collect; then
  mkdir -p bundle_$B
  cp $W/train_$B/train_$B.json $W/feat_summary_$B.json $W/lists_$B/lists_summary_$B.json $P/dumps/base_${B}_stageB.json bundle_$B/ 2>/dev/null || true
  cp $BL/blend_valid_icews_$B.json $BL/sel_icews_$B.json $BL/blend_test_icews_$B.json $BL/blend_test_icews_${B}_sys.json bundle_$B/ 2>/dev/null || true
  cp $P/dumps/ckpt_${B}_stageB/grid_log.json bundle_$B/grid_log_$B.json 2>/dev/null || true
  ls -lh bundle_$B; stamp collect
fi
