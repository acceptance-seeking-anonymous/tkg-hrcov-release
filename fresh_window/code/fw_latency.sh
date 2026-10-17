#!/bin/bash
# phase2_fresh_window / fw_latency.sh -- v79 latency protocol on serverA (validation slice; descriptive).  Run when no other job
# of this track is active.  1) fw_latency.py in the tkg-gpu container on GPU 0 (8 CPUs); 2) fw_ct_latency.py in the
# CountTRuCoLa venv for the selected configuration (frozen Stage-1 rules), threads pinned.
set +e
J=$PWD
H=/home/anon/cs_scratch/open_ideas_harness; FW=$H/tracks/phase2_fresh_window
P=$FW/GDELTFW; W=$P/work; C=$FW/code; CS=$FW/code_sa; CC=$FW/code_ct; L=$FW/logs; RUN=$P/run_latency; DD=$FW/data/GDELTFW
mkdir -p $W/latency $RUN $J/h; cp -f $J/fw_latency.py $J/fw_latency.sh $C/; cp -f $J/fw_ct_latency.py $CC/
mkdir -p $W/ext; cp -f $J/ext_comparator.json $W/ext/ext_comparator.json
source $C/fw_gpu_wait.sh
echo "[latency] start $(date -u +%FT%TZ)" >> $L/latency.log; uptime >> $L/latency.log; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> $L/latency.log
ps -eo pid,pcpu,etime,args --sort=-pcpu | head -15 >> $L/latency.log
COMP=$W/ext/ext_comparator.json; CFGC=$(python3 -c "import json; print(json.load(open('$COMP'))['config'])"); CTPAR=$(python3 -c "import json; print(json.load(open('$COMP'))['ct_params'])")
ENVS="-e PC_HARNESS=$H -e FH_DIR=$FW -e PC_P=$P -e FH_DD=$DD -e PYTHONDONTWRITEBYTECODE=1 -e V78_RECB_XF=$W/recb_xf -e FW_CODE=$C -e HD_HOST=serverA_(GPU0_RTX_PRO_6000,8cpu)"
MNT="-v /home/anon/cs_scratch:/home/anon/cs_scratch -w $RUN"
fh_gpu_wait
GPUR="docker run --rm --gpus device=$FH_GPU -e CUDA_VISIBLE_DEVICES=0 --cpus 8 -e OMP_NUM_THREADS=8 -e G2_THREADS=8 $ENVS $MNT tkg-gpu:latest"
$GPUR python3 -u $C/fw_latency.py --n 2000 --reps 5 --workers 8 --out $W/latency/latency_systems.json --comparator $COMP --ct_dump $FW/ct/$CFGC/dump_val > $L/latency_systems.log 2>&1
echo "[latency] systems rc=$? $(date -u +%FT%TZ)" >> $L/latency.log
B=$H/tracks/v78_external; PY=$B/env/ct/bin/python; RD=$FW/ct/$CFGC
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -rf $RD/repo_latency; cp -a $RD/repo $RD/repo_latency; cp -f $CC/fw_ct_latency.py $RD/repo_latency/rule_based/
cd $RD/repo_latency
PYTHONPATH=$RD/repo_latency/rule_based:$RD/repo_latency $PY rule_based/fw_ct_latency.py --fw_dd $DD --fw_root $RD/root --n 2000 --reps 5 --nc 10 \
   --out $W/latency/latency_ct.json --params DATASET_NAME=tkgl-gdeltfw NUM_CPUS=10 $CTPAR < /dev/null > $L/latency_ct.log 2>&1
echo "[latency] ct rc=$? $(date -u +%FT%TZ)" >> $L/latency.log
cp $W/latency/*.json $L/latency*.log $J/h/ 2>/dev/null
