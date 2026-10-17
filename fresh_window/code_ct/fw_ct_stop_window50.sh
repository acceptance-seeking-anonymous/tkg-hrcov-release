#!/bin/bash
# Extracted verbatim from the orchestration cell at session-archive index 594 (sent to serverA serverA as one
# shell command). Only change: the archive's display redaction of the user name was restored to /home/anon.
# Purpose: manual stop of the CountTRuCoLa C1/C2 (LEARN_WINDOW_SIZE=50) Stage-1 runs at 2026-10-16T16:08Z
# (projected memory/time infeasible on the shared host; no C1/C2 rule set, ranking or score exists). C0 kept running.
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; LOG=$FW/logs/ct_memwatch.log
killtree() { local p=$1; for c in $(pgrep -P $p); do killtree $c; done; kill -TERM $p 2>/dev/null; }
for c in C1 C2; do p=$(pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$c/root" | head -1); echo "$(date -u +%FT%TZ) MANUAL stop $c pid $p (projected memory/time infeasible; no score exists)" >> $LOG; [ -n "$p" ] && killtree $p; done
sleep 5; for c in C0 C1 C2; do echo $c $(pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$c/root" | head -1); done; awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo
