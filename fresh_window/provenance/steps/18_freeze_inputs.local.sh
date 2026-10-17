#!/bin/bash
# Verbatim local notebook cell(s), session-archive index 620, 622 (local bash kernel, workspace root as cwd).
# Stage the registration JSONs into fw/freeze_in/ and write code/fw_freeze_job.sh (cell 622: tar option order fix).
# Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
# are the job output directories copied back from serverA. Only change: user name restored.
# ===== cell 620
mkdir -p fw/freeze_in && cp "/Users/anon/.jobs/orgs/<id>/artifacts/<id>/13197939-8bbd-4f32-88c2-9197a2badc8b/ve10ab60d_v79_fresh_window_rule.json" fw/freeze_in/v79_fresh_window_rule.json && cp "/Users/anon/.jobs/orgs/<id>/artifacts/<id>/a4c9bafc-09fe-4dc9-b6b3-aedeaab12d59/v6debf06f_v79_fresh_window_rule_amendment_1.json" fw/freeze_in/v79_fresh_window_rule_amendment_1.json && cp fw/out/fresh_window_implementation_note.json fw/out/fresh_window_implementation_note_addendum_1.json fw/out/fresh_window_uninspected_check.json fw/out/ext_comparator.json fw/freeze_in/ && sha256sum fw/freeze_in/* && cat > fw/code_fw/fw_freeze_job.sh <<'EOF'
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
cd $FW && tar czf $J/h/fw_code_frozen.tgz code code_sa code_ct freeze/*.json --exclude='__pycache__' 2>>$J/h/freeze_out.txt
ls -la $J/h >> $J/h/freeze_out.txt
EOF
bash -n fw/code_fw/fw_freeze_job.sh && echo ok
# ===== cell 622
cd fw/code_fw && sed -i '' "s|tar czf \$J/h/fw_code_frozen.tgz code code_sa code_ct freeze/\*.json --exclude='__pycache__'|tar czf \$J/h/fw_code_frozen.tgz --exclude='__pycache__' code code_sa code_ct freeze/*.json|" fw_freeze_job.sh && grep -n "tar czf" fw_freeze_job.sh && bash -n fw_freeze_job.sh && echo ok
