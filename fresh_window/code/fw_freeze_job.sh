#!/bin/bash
# phase2_fresh_window / fw_freeze_job.sh -- end of Stage 1: install the rule documents + Stage-2 code, write the sha256 manifest.
set +e
J=$PWD; H=/home/anon/cs_scratch/open_ideas_harness; FW=$H/tracks/phase2_fresh_window; C=$FW/code; W=$FW/GDELTFW/work
mkdir -p $FW/freeze $W/ext $J/h
for f in v79_fresh_window_rule.json v79_fresh_window_rule_amendment_1.json fresh_window_implementation_note.json fresh_window_implementation_note_addendum_1.json fresh_window_uninspected_check.json; do cp -f $J/$f $FW/freeze/$f; done
cp -f $J/ext_comparator.json $W/ext/ext_comparator.json
cp -f $J/fw_stage2.sh $J/fw_freeze.py $J/fw_agg.py $J/v7_common.py $J/fw_freeze_job.sh $C/
cmp -s $J/ext_comparator.json $W/ext/ext_comparator.json && echo "comparator installed"
python3 $C/fw_freeze.py --fh $FW --out $FW/freeze/freeze_manifest_remote.json > $J/h/freeze_out.txt 2>&1; echo "freeze rc=$?" >> $J/h/freeze_out.txt
cp $FW/freeze/freeze_manifest_remote.json $J/h/
cd $FW && tar czf $J/h/fw_code_frozen.tgz --exclude='__pycache__' code code_sa code_ct freeze/*.json 2>>$J/h/freeze_out.txt
ls -la $J/h >> $J/h/freeze_out.txt
