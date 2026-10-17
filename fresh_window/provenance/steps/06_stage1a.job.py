"""Verbatim orchestration cell, session-archive index 460 (remote job submission).
Stage 1A (job c2ac489f): GDELTFW prep, RecB validation grid, TiRGN training on GPU 0, validation rows.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 460
FWC = "/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window/code"
cmd = f"cp fw_gpu_wait.sh fw_stage1a.sh {FWC}/ && bash {FWC}/fw_stage1a.sh"
j1a = c_serverA.submit_job(command=cmd, intent="v79 fresh window Stage 1A: GDELTFW prep, RecB validation grid, TiRGN training on GPU 0 and validation rows (no test scoring)",
                      inputs=[{"src": "fw/code/fw_gpu_wait.sh", "dst": "fw_gpu_wait.sh"}, {"src": "fw/code/fw_stage1a.sh", "dst": "fw_stage1a.sh"}],
                      outputs=["h/*"], run_timeout_s=14*3600)
print(j1a)
