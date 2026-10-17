"""Verbatim orchestration cell, session-archive index 496, 498 (call_command on serverA).
Adapter smoke test (same commands as code_ct/fw_ct_dataset_smoke.sh).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 496
print(jct.state())
r = c_serverA.call_command("""set -e
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
""", intent="Write CT adapter smoke script to /tmp")
print(r.exit_code, r.stderr[-500:])
# ===== cell 498
import base64
src = open("fw/code_ct/fw_ct_main.py").read()
b64 = base64.b64encode(src.encode()).decode()
r = c_serverA.call_command(f"""FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; B=/home/anon/cs_scratch/open_ideas_harness/tracks/v78_external
cd $FW/ct/C0/repo && echo {b64} | base64 -d > /tmp/fwct_smoke/fw_ct_main.py && cp /tmp/fwct_smoke/fw_ct_main.py rule_based/fw_ct_main.py && OMP_NUM_THREADS=1 PYTHONPATH=$FW/ct/C0/repo/rule_based:$FW/ct/C0/repo timeout 600 $B/env/ct/bin/python /tmp/fwct_smoke/t.py 2>&1 | grep -v "^>>>" | tail -8""", intent="Smoke test: build CountTRuCoLa RuleDataset from GDELTFW (no learning, no scoring)")
print(r.exit_code, r.stdout[-3000:], r.stderr[-1000:])
