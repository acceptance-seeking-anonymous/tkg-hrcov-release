"""Verbatim orchestration cell, session-archive index 634 (remote job submission).
Stage 2 (job 7349980c): verify the freeze manifest, then the single test pass per system.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 634
j2 = c_serverA.submit_job(command="bash /home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window/code/fw_stage2.sh",
    intent="v79 fresh window STAGE 2: verify freeze, then the single test pass per system (TiRGN, BLEND, HRcov, SA, CountTRuCoLa comparator) on GPU 0 / CPU",
    inputs=[], outputs=["h/*"], run_timeout_s=12*3600)
print(j2)
