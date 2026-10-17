"""Verbatim orchestration cell, session-archive index 299 (remote job submission).
Read-only scan of serverA/serverB/serverC for any GDELT-derived file dated >= 2025-10-01 (jobs 24868d2a, 7281237c, 6f3c1306).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 299
c_serverC = host.compute.create("serverC")
cmd = r'''set +e
if command -v docker >/dev/null 2>&1 && docker image inspect tkg-gpu:latest >/dev/null 2>&1; then
  docker run --rm -v /home/anon:/home/anon:ro -v $PWD:/workspace -w /workspace tkg-gpu:latest python3 fw_uninspected_scan.py HOST /home/anon > scan.json 2> scan.err
  echo "mode=docker-root-readonly" > scan_mode.txt
else
  python3 fw_uninspected_scan.py HOST /home/anon > scan.json 2> scan.err
  echo "mode=host-user" > scan_mode.txt
fi
ls -la; wc -c scan.json
'''
scan_jobs = {}
for nm, c in (("serverA", c_serverA), ("serverB", c_serverB), ("serverC", c_serverC)):
    scan_jobs[nm] = c.submit_job(command=cmd.replace("HOST", f"ssh:server.{nm}"), intent=f"v79 uninspected-window audit on .{nm}: read-only scan for data dated >= 2025-10-01",
                                 inputs=[{"src": "fw/code/fw_uninspected_scan.py", "dst": "fw_uninspected_scan.py"}],
                                 outputs=["scan.json", "scan.err", "scan_mode.txt"], run_timeout_s=3*3600)
    print(nm, scan_jobs[nm].job_id if hasattr(scan_jobs[nm], "job_id") else scan_jobs[nm])
