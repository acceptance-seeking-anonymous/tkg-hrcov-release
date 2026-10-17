"""Verbatim orchestration cell, session-archive index 554 (remote job submission).
Stage 1B-SA rerun (job 9ccd0e7d): RecB_xf selection + valid CSR, SA history/lists/pools, SA_W100 fit.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 554
inputs = [{"src": "fw/code_fw/fw_stage1b_sa.sh", "dst": "fw_stage1b_sa.sh"}] + [{"src": f"fw/code_sa/{f}", "dst": f} for f in sa_files if f.endswith(".py")]
j1s = c_serverA.submit_job(command="bash fw_stage1b_sa.sh",
    intent="v79 fresh window Stage 1B-SA rerun (validation only): RecB_xf selection on selection part + valid CSR, SA history order/lists/pools, SA_W100 fit on GPU 0",
    inputs=inputs, outputs=["h/*"], run_timeout_s=12*3600)
print(j1s)
