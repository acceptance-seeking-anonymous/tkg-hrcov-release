#!/bin/bash
# Extracted verbatim from the orchestration cell at session-archive index 496 and 498 (sent to serverA serverA as one
# shell command). Only change: the archive's display redaction of the user name was restored to /home/anon.
# Purpose: smoke test that the CountTRuCoLa RuleDataset builds from GDELTFW through the fw_ct_main.py adapter (no rule
# learning, no scoring). Run between the failed first CT Stage-1 attempt and the rerun.
# Second change: cell 498 shipped fw_ct_main.py inline as base64 ('echo <b64> | base64 -d'); here it is copied from
# code_ct/fw_ct_main.py. Checked on serverA after the run: the smoke-test copy /tmp/fwct_smoke/fw_ct_main.py (written
# 2026-10-16 14:57:12Z), $FW/code_ct/fw_ct_main.py, $FW/ct/C0/repo/rule_based/fw_ct_main.py and code_ct/fw_ct_main.py in
# this package all have md5 1aa4023c35e36cff2c07bd94c954dc89.
# --- part 1 (cell 496): write the smoke script
set -e
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; B=/home/anon/cs_scratch/open_ideas_harness/tracks/v78_external
mkdir -p /tmp/fwct_smoke && cd $FW/ct/C0/repo
cat > /tmp/fwct_smoke/t.py <<'EOF'
import sys, time
sys.argv = ["x", "--fw_dd", "/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window/data/GDELTFW", "--fw_root", "/tmp/fwct_smoke/root"]
sys.path.insert(0, "rule_based")
t0 = time.time()
import fw_ct_main as W
import rule_dataset
d = rule_dataset.RuleDataset(name="tkgl-gdeltfw")
print("ok", d.num_rels, d.num_rels_half, len(d.all_quads), len(d.train_data), len(d.val_data), len(d.test_data), d.large_data_flag, len(d.nodes_id_to_string), len(d.rels_id_to_string), round(time.time()-t0,1))
print(list(d.rels_id_to_string.items())[:2], list(d.nodes_id_to_string.items())[:2], d.inverse_rel_dict[0], d.inverse_rel_dict[215])
EOF

# --- part 2 (cell 498): install the adapter and run it
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; B=/home/anon/cs_scratch/open_ideas_harness/tracks/v78_external
cd $FW/ct/C0/repo && cp "${PKG:?set PKG to the unpacked package root}/code_ct/fw_ct_main.py" /tmp/fwct_smoke/fw_ct_main.py && cp /tmp/fwct_smoke/fw_ct_main.py rule_based/fw_ct_main.py && OMP_NUM_THREADS=1 PYTHONPATH=$FW/ct/C0/repo/rule_based:$FW/ct/C0/repo timeout 600 $B/env/ct/bin/python /tmp/fwct_smoke/t.py 2>&1 | grep -v "^>>>" | tail -8
