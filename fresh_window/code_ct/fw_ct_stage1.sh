#!/bin/bash
# phase2_fresh_window / fw_ct_stage1.sh -- STAGE 1 of the external comparator on GDELTFW: the released CountTRuCoLa
# (counttrucola_submission @ 5d9aedf, code unchanged; wrapper fw_ct_main.py) for the three released configurations C0/C1/C2
# of the implementation note, run concurrently: examples + rule learning on TRAIN, application to VALIDATION only
# (EVAL_TESTSET_FLAG=False), then the validation rankings are scored by fw_ct_score.py.  No test row is touched.
# usage: bash fw_ct_stage1.sh <num_cpus_per_config>
set +e
J=$PWD; NC=${1:-10}
B=/home/anon/cs_scratch/open_ideas_harness/tracks/v78_external; PY=$B/env/ct/bin/python; CTR=$B/repos/counttrucola_submission
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; DD=$FW/data/GDELTFW; CC=$FW/code_ct
mkdir -p $CC; cp -f $J/fw_ct_main.py $J/fw_ct_score.py $J/fw_ct_stage1.sh $CC/; cp -f $B/code/memmon.py $CC/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COMMON="AGGREGATION_DECAY=0.8 NUM_TOP_RULES=10 RULE_TYPE_C=True F_UNSEEN_NEGATIVES=10 Z_RULES_FACTOR=0.1"
declare -A CFG
CFG[C0]="LEARN_WINDOW_SIZE=10 DATAPOINT_THRESHOLD_MULTI=0 RULE_UNSEEN_NEGATIVES=30 $COMMON"
CFG[C1]="LEARN_WINDOW_SIZE=50 DATAPOINT_THRESHOLD_MULTI=0 RULE_UNSEEN_NEGATIVES=30 $COMMON"
CFG[C2]="LEARN_WINDOW_SIZE=50 DATAPOINT_THRESHOLD_MULTI=50 RULE_UNSEEN_NEGATIVES=100 $COMMON"
echo "host $(hostname) start $(date -u +%FT%TZ) ncpu/config $NC"; free -g | head -2; uptime
( cd $CTR && git rev-parse HEAD && git status --short | head ) > $J/ct_repo_state.txt
run_cfg() {
  c=$1; RD=$FW/ct/$c; mkdir -p $RD/root $RD/dump_val
  rsync -a --delete --exclude files --exclude .git --exclude '__pycache__' --exclude '*.pkl' --exclude '*.csv' --exclude '*.zip' $CTR/ $RD/repo/
  cp -f $CC/fw_ct_main.py $RD/repo/rule_based/
  ( cd $RD/repo && diff -rq $CTR/rule_based rule_based -x __pycache__ ) > $RD/code_diff_vs_release.txt 2>&1
  cd $RD/repo; touch $RD/.start_marker
  echo "[$c] start $(date -u +%FT%TZ) params: ${CFG[$c]}" > $RD/run.txt
  PYTHONPATH=$RD/repo/rule_based:$RD/repo $PY $CC/memmon.py --out $RD/mem_ct_stage1.json -- \
     $PY rule_based/fw_ct_main.py --fw_dd $DD --fw_root $RD/root --params DATASET_NAME=tkgl-gdeltfw NUM_CPUS=$NC \
     EVAL_TESTSET_FLAG=False EVAL_VALSET_FLAG=True ${CFG[$c]} < /dev/null > $RD/ct_stage1.log 2>&1
  echo "[$c] main rc=$? $(date -u +%FT%TZ)" >> $RD/run.txt
  tr '\r' '\n' < $RD/ct_stage1.log | grep -v "it/s\]$\|s/it\]$" > $RD/ct_stage1_clean.log
  RV=$(find $RD/repo/files/rankings -type f -newer $RD/.start_marker -name "*rankings_val*" | head -1)
  RT=$(find $RD/repo/files/rankings -type f -newer $RD/.start_marker -name "*rankings_test*" | head -1)
  echo "[$c] rankings val=$RV test=${RT:-none}" >> $RD/run.txt
  ls -la $RD/repo/files/rules >> $RD/run.txt 2>&1
  PYTHONPATH=$RD/repo $PY $CC/memmon.py --out $RD/mem_score_val.json -- $PY $CC/fw_ct_score.py --dd $DD --rankings "$RV" --split val \
     --out $RD/dump_val --workers $NC > $RD/score_val.log 2>&1
  echo "[$c] score rc=$? $(date -u +%FT%TZ)" >> $RD/run.txt
  sha256sum $RD/repo/files/rules/* "$RV" >> $RD/run.txt 2>/dev/null
  mkdir -p $J/h/$c; cp $RD/run.txt $RD/code_diff_vs_release.txt $RD/mem_*.json $RD/dump_val/meta.json $RD/score_val.log $J/h/$c/ 2>/dev/null
  grep -n "options: \|\[fw\]\|Error\|Traceback\|large\|C_X_COUNT\|time to\|took\|WARNING" $RD/ct_stage1_clean.log | head -60 > $J/h/$c/ct_key_lines.txt
  tail -40 $RD/ct_stage1_clean.log > $J/h/$c/ct_tail.txt
}
run_cfg C0 & P0=$!; sleep 60
run_cfg C1 & P1=$!; sleep 60
run_cfg C2 & P2=$!
wait $P0 $P1 $P2
cp $J/ct_repo_state.txt $J/h/ 2>/dev/null
echo "end $(date -u +%FT%TZ)"
