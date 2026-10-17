"""Verbatim orchestration cell, session-archive index 624 (remote job submission).
End of Stage 1 (job 117823f0): install rule documents + Stage-2 code, write freeze_manifest_remote.json, tar the frozen code.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 624
fi = ["v79_fresh_window_rule.json", "v79_fresh_window_rule_amendment_1.json", "fresh_window_implementation_note.json", "fresh_window_implementation_note_addendum_1.json", "fresh_window_uninspected_check.json", "ext_comparator.json"]
inputs = [{"src": f"fw/freeze_in/{f}", "dst": f} for f in fi] + [{"src": f"fw/code_fw/{f}", "dst": f} for f in ("fw_stage2.sh", "fw_freeze.py", "fw_agg.py", "v7_common.py", "fw_freeze_job.sh")]
jfz = c_serverA.submit_job(command="bash fw_freeze_job.sh", intent="v79 fresh window end of Stage 1: install rule docs + Stage-2 code, write SHA-256 freeze manifest (no test computation)",
                      inputs=inputs, outputs=["h/*"], run_timeout_s=3*3600)
print(jfz)
