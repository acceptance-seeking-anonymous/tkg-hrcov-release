#!/bin/bash
# EXT_adapted CountTRuCoLa (registered-optional, after the v78 verdict): released code/config on a repo COPY, validation fit
# window (ts_id < cut) moved into the learning data by ct_adapted_main.py, then official scoring + dumps.
# usage: bash v78_ct_adapted_job.sh <dataset> <num_cpus> <cut_ts_id> [harness_valid_rows.npy]
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W=$PWD; DSN=$1; NC=$2; CUT=$3; HROWS=$4
T=$WORK_DIR/tracks; B=$T/v78_external; E=$B/env/er/bin/python; C=$B/env/ct/bin/python
DST=$B/repos/counttrucola_adapted
# (code staging of the original job removed: the scripts are read from $SRC/v78_external)
[ -d $DST ] || cp -R $B/repos/counttrucola_submission $DST   # the copy of the released repository (release: created here if missing)
cp -f $SRC/v78_external/ct_adapted_main.py $DST/rule_based/ct_adapted_main.py; rm -f $DST/ct_adapted_main.py
RD=$B/runs/counttrucola_adapted/$DSN; mkdir -p $RD; touch $RD/.start_marker
echo "host $(hostname) start $(date -u +%FT%TZ)"; uptime; free -g | head -2
cd $DST
V78_HARNESS_DD=$T/datasets/harness_data/$DSN V78_CUT_TS_ID=$CUT V78_SPLIT_LOG=$RD/split_moved.json \
  $E $SRC/v78_external/memmon.py --out $RD/mem_ct.json -- $C rule_based/ct_adapted_main.py --params DATASET_NAME=$DSN NUM_CPUS=$NC < /dev/null > $RD/ct.log 2>&1
echo "ct adapted main rc=$?"
tr '\r' '\n' < $RD/ct.log | grep -v "it/s\]$\|s/it\]$" > $RD/ct_clean.log
grep -n "v78 EXT_adapted\|'NUM_CPUS'\|mean mrr\|eval mode\|Error\|Traceback" $RD/ct_clean.log | cut -c1-600 | head -40 > $W/ct_key_lines.txt
tail -40 $RD/ct_clean.log > $W/ct_tail.txt
cp $DST/files/results/$DSN-results.csv $RD/ 2>/dev/null; cp $RD/$DSN-results.csv $RD/split_moved.json $W/ 2>/dev/null
RT=$(ls $DST/files/rankings/$DSN/*-rankings_test_*.txt); RV=$(ls $DST/files/rankings/$DSN/*-rankings_val_*.txt)
echo "rankings test=$RT val=$RV"
EXTRA=""; [ -n "$HROWS" ] && EXTRA="--harness-valid-rows $HROWS"
$E $SRC/v78_external/memmon.py --out $RD/mem_score.json -- $E $SRC/v78_external/ct_score_adapted.py --dataset $DSN --out $RD/dump --rankings-val "$RV" --rankings-test "$RT" \
    --workers $NC --own-results $RD/$DSN-results.csv --cut-ts-id $CUT $EXTRA > $RD/score.log 2>&1
echo "score rc=$?"; tail -8 $RD/score.log | cut -c1-3000 > $W/score_tail.txt
mkdir -p $B/dumps/counttrucola_adapted; ln -sfn $RD/dump $B/dumps/counttrucola_adapted/$DSN
cp $RD/mem_ct.json $RD/mem_score.json $RD/dump/meta.json $W/ 2>/dev/null
echo "end $(date -u +%FT%TZ)"
