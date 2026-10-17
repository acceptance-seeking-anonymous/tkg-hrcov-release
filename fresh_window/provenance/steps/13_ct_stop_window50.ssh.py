"""Verbatim orchestration cell, session-archive index 594 (call_command on serverA).
Manual stop of CT C1/C2 at 16:08Z (same commands as code_ct/fw_ct_stop_window50.sh).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 594
r = c_serverA.call_command("""FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; LOG=$FW/logs/ct_memwatch.log
killtree() { local p=$1; for c in $(pgrep -P $p); do killtree $c; done; kill -TERM $p 2>/dev/null; }
for c in C1 C2; do p=$(pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$c/root" | head -1); echo "$(date -u +%FT%TZ) MANUAL stop $c pid $p (projected memory/time infeasible; no score exists)" >> $LOG; [ -n "$p" ] && killtree $p; done
sleep 5; for c in C0 C1 C2; do echo $c $(pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$c/root" | head -1); done; awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo""", intent="Stop my CountTRuCoLa C1/C2 runs (resource-infeasible; projected >250 GB each); C0 keeps running")
print(r.stdout, r.stderr[-300:])
