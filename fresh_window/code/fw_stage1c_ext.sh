#!/bin/bash
# phase2_fresh_window / fw_stage1c_ext.sh -- STAGE 1: EXT_alone / EXT_blend (52 per config) of the three CountTRuCoLa
# configurations on the validation SELECTION part (fw_ext.py --stage valid), GPU 0.  Validation numbers only.
set +e
J=$PWD
H=/home/anon/cs_scratch/open_ideas_harness; FW=$H/tracks/phase2_fresh_window
P=$FW/GDELTFW; W=$P/work; C=$FW/code; CS=$FW/code_sa; L=$FW/logs; RUN=$P/run_stage1; DD=$FW/data/GDELTFW
mkdir -p $W/ext $J/h; cp -f $J/fw_ext.py $CS/fw_ext.py; cp -f $J/fw_stage1c_ext.sh $C/
source $C/fw_gpu_wait.sh
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u +%FT%TZ)" >> $L/stage1_chain.log; }
ENVS="-e PC_HARNESS=$H -e FH_DIR=$FW -e PC_P=$P -e FH_DD=$DD -e PYTHONDONTWRITEBYTECODE=1 -e HD_HOST=serverA_(GPU0)"
MNT="-v /home/anon/cs_scratch:/home/anon/cs_scratch -w $RUN"
fh_gpu_wait
GPUR="docker run --rm --gpus device=$FH_GPU -e CUDA_VISIBLE_DEVICES=0 --cpus 8 -e OMP_NUM_THREADS=8 $ENVS $MNT tkg-gpu:latest"
FR=$W/fullrows/tirgn_GDELTFW_valid_scores_f32.bin,$W/fullrows/tirgn_GDELTFW_test_scores_f32.bin
M=""; for c in C0 C1 C2; do [ -f $FW/ct/$c/dump_val/top_ids.npy ] && M="$M,$c=$FW/ct/$c"; done; M=${M#,}
echo "[stage1c] EXT valid methods $M" >> $L/stage1_chain.log
$GPUR python3 -u $CS/fw_ext.py --ds gdeltfw --stage valid --hist $W/sa/hist --methods $M --fullrows $FR --out $W/ext > $L/ext_valid.log 2>&1; tm ext_valid $?
cp $W/ext/sel_ext_gdeltfw.json $L/ext_valid.log $J/h/ 2>/dev/null
for c in C0 C1 C2; do mkdir -p $J/h/$c; cp $FW/ct/$c/run.txt $FW/ct/$c/dump_val/meta.json $FW/ct/$c/mem_*.json $J/h/$c/ 2>/dev/null; done
