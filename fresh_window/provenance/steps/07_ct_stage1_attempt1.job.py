"""Verbatim orchestration cell, session-archive index 474 (remote job submission).
CountTRuCoLa Stage 1, first attempt (job 7a51145d): failed at dataset construction in the adapter
(id-file handling), before any rule was learned; the adapter was fixed (code_ct/fw_ct_main.py is the fixed version).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 474
jct = c_serverA.submit_job(command="bash fw_ct_stage1.sh 10",
    intent="v79 fresh window Stage 1: released CountTRuCoLa (3 released configs, 10 CPUs each) learned on GDELTFW train and applied to validation only; validation rankings scored",
    inputs=[{"src": "fw/code_ct/fw_ct_main.py", "dst": "fw_ct_main.py"}, {"src": "fw/code_ct/fw_ct_score.py", "dst": "fw_ct_score.py"},
            {"src": "fw/code_ct/fw_ct_stage1.sh", "dst": "fw_ct_stage1.sh"}],
    outputs=["h/*", "ct_repo_state.txt"], run_timeout_s=30*3600)
print(jct)
