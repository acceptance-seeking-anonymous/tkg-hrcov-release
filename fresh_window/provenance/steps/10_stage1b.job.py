"""Verbatim orchestration cell, session-archive index 524, 528 (remote job submission).
Stage 1B (job dc39717c): HRcov lists/features/fit, BLEND grid + selection, RecB_xf, SA (SA part failed at the
selection-cut parse). Cell 524 was rejected client-side before submission (input dst must be a bare filename); 528 is the submission.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 524
import os
sa_files = sorted(os.listdir("fw/code_sa"))
print(sa_files)
inputs = [{"src": "fw/code_fw/fh_blend.py", "dst": "fh_blend.py"}, {"src": "fw/code_fw/fw_stage1b.sh", "dst": "fw_stage1b.sh"}] + \
         [{"src": f"fw/code_sa/{f}", "dst": f"code_sa/{f}"} for f in sa_files if f.endswith(".py")]
j1b = c_serverA.submit_job(command="bash fw_stage1b.sh",
    intent="v79 fresh window Stage 1B (validation only): HRcov lists/features/fit, BLEND grid+selection, RecB_xf selection+valid CSR, SA history/lists/pools/fit on GPU 0",
    inputs=inputs, outputs=["h/*"], run_timeout_s=16*3600)
print(j1b)
# ===== cell 528
inputs = [{"src": "fw/code_fw/fh_blend.py", "dst": "fh_blend.py"}, {"src": "fw/code_fw/fw_stage1b.sh", "dst": "fw_stage1b.sh"}] + \
         [{"src": f"fw/code_sa/{f}", "dst": f} for f in sa_files if f.endswith(".py")]
j1b = c_serverA.submit_job(command="bash fw_stage1b.sh",
    intent="v79 fresh window Stage 1B (validation only): HRcov lists/features/fit, BLEND grid+selection, RecB_xf selection+valid CSR, SA history/lists/pools/fit on GPU 0",
    inputs=inputs, outputs=["h/*"], run_timeout_s=16*3600)
print(j1b)
