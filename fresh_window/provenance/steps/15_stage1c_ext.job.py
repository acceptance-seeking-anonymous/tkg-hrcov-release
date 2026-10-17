"""Verbatim orchestration cell, session-archive index 604 (remote job submission).
Stage 1C (job 953e4300): CT C0 alone and 52 TiRGN blends on the validation selection part.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 604
j1c = c_serverA.submit_job(command="bash fw_stage1c_ext.sh",
    intent="v79 fresh window Stage 1C (validation selection part only): CountTRuCoLa C0 alone and 52 TiRGN blends, GPU 0",
    inputs=[{"src": "fw/code_sa/fw_ext.py", "dst": "fw_ext.py"}, {"src": "fw/code_fw/fw_stage1c_ext.sh", "dst": "fw_stage1c_ext.sh"}],
    outputs=["h/*"], run_timeout_s=4*3600)
print(j1c)
