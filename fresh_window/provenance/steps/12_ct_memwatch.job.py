"""Verbatim orchestration cell, session-archive index 588 (remote job submission).
Host-memory watchdog for the three concurrent CT runs (job eea61388).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 588
r = c_serverA.call_command("""FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; for c in C0 C1 C2; do pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$c/root" | head -1; done""", intent="Read-only check that the watchdog matches the three CT main processes")
print(r.stdout)
jmw = c_serverA.submit_job(command="bash fw_ct_memwatch.sh",
    intent="Memory watchdog for the three concurrent CountTRuCoLa Stage-1 runs (stops C2, then C1, only if host MemAvailable < 60 / 40 GB)",
    inputs=[{"src": "fw/code_ct/fw_ct_memwatch.sh", "dst": "fw_ct_memwatch.sh"}], outputs=[], run_timeout_s=30*3600)
print(jmw)
