"""Verbatim orchestration cell, session-archive index 614 (remote job submission).
Latency protocol (job c46772b5): fw_latency.py (base/BLEND/HRcov/SA) + fw_ct_latency.py (CT comparator).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 614
jlat = c_serverA.submit_job(command="bash fw_latency.sh",
    intent="v79 latency protocol (validation slice, descriptive): base/BLEND/HRcov/SA/CT-comparator stage timings on GPU 0, batch 1 and batched; CT rule application in its venv",
    inputs=[{"src": "fw/code_fw/fw_latency.py", "dst": "fw_latency.py"}, {"src": "fw/code_fw/fw_latency.sh", "dst": "fw_latency.sh"},
            {"src": "fw/code_ct/fw_ct_latency.py", "dst": "fw_ct_latency.py"}, {"src": "fw/out/ext_comparator.json", "dst": "ext_comparator.json"}],
    outputs=["h/*"], run_timeout_s=6*3600)
print(jlat)
