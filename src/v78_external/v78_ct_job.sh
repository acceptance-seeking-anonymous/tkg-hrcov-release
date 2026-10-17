#!/bin/bash
# CountTRuCoLa on one dataset: faithful rule_based/main.py with the released config (DATASET_OVERRIDES), then official scoring + dumps.
# usage: bash v78_ct_job.sh <dataset> <num_cpus>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W=$PWD; DSN=$1; NC=$2
B=$WORK_DIR/tracks/v78_external; E=$B/env/er/bin/python; C=$B/env/ct/bin/python
# (code staging of the original job removed: the scripts are read from $SRC/v78_external)
RD=$B/runs/counttrucola/$DSN; mkdir -p $RD
CTR=$B/repos/counttrucola_submission
echo "host $(hostname) start $(date -u +%FT%TZ)"; free -g | head -2; uptime
touch $RD/.start_marker
cd $CTR
$E $SRC/v78_external/memmon.py --out $RD/mem_ct.json -- $C rule_based/main.py --params DATASET_NAME=$DSN NUM_CPUS=$NC < /dev/null > $RD/ct.log 2>&1
echo "ct main rc=$?"
tr '\r' '\n' < $RD/ct.log | grep -v "it/s\]$\|s/it\]$" > $RD/ct_clean.log
grep -n "options: \|mean mrr\|mean hits\|eval mode\|Error\|Traceback\|large\|C_X_COUNT\|total time\|eval time" $RD/ct_clean.log | head -80 > $W/ct_key_lines.txt
tail -60 $RD/ct_clean.log > $W/ct_tail.txt
cp $CTR/files/results/$DSN-results.csv $RD/ 2>/dev/null; cp $RD/$DSN-results.csv $W/ 2>/dev/null
RT=$(find $CTR/files/rankings -type f -newer $RD/.start_marker -name "*-rankings_test_*" | grep -- "$DSN" | head -1)
RV=$(find $CTR/files/rankings -type f -newer $RD/.start_marker -name "*-rankings_val_*" | grep -- "$DSN" | head -1)
echo "rankings test=$RT val=$RV"; ls -la $(dirname "$RT")
$E $SRC/v78_external/memmon.py --out $RD/mem_score.json -- $E $SRC/v78_external/ct_score.py --dataset $DSN --out $RD/dump --rankings-val "$RV" --rankings-test "$RT" \
    --workers $NC --own-results $RD/$DSN-results.csv > $RD/score.log 2>&1
echo "score rc=$?"; tail -30 $RD/score.log > $W/score_tail.txt
cp $RD/mem_ct.json $RD/mem_score.json $RD/dump/meta.json $W/ 2>/dev/null
echo "end $(date -u +%FT%TZ)"
