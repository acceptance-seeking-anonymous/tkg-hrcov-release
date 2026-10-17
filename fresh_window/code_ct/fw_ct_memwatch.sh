#!/bin/bash
# phase2_fresh_window / fw_ct_memwatch.sh -- protects the shared host during the concurrent CountTRuCoLa Stage-1 runs.
# Every 30 s: log MemAvailable and the RSS of each config's main process.  If MemAvailable < 60 GB, stop configuration C2
# (whole process tree); if it is still < 40 GB, stop C1.  C0 is never stopped.  Decisions use memory only (no scores exist).
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window; LOG=$FW/logs/ct_memwatch.log
pid_of() { pgrep -f "^[^ ]*bin/python rule_based/fw_ct_main.py --fw_dd .* --fw_root $FW/ct/$1/root" | head -1; }
killtree() { local p=$1; for c in $(pgrep -P $p); do killtree $c; done; kill -TERM $p 2>/dev/null; }
stopped=""
while true; do
  P0=$(pid_of C0); P1=$(pid_of C1); P2=$(pid_of C2)
  [ -z "$P0$P1$P2" ] && { echo "$(date -u +%FT%TZ) no CT main process left; exit" >> $LOG; exit 0; }
  AV=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  r() { [ -n "$1" ] && ps -o rss= -p $1 | awk '{print int($1/1048576)}' || echo -; }
  echo "$(date -u +%FT%TZ) avail_GB=$AV C0=$(r $P0)GB C1=$(r $P1)GB C2=$(r $P2)GB stopped=[$stopped]" >> $LOG
  if [ "$AV" -lt 60 ] && [ -n "$P2" ]; then echo "$(date -u +%FT%TZ) MemAvailable $AV GB < 60: stopping C2 (pid $P2)" >> $LOG; killtree $P2; stopped="$stopped C2"; sleep 20; continue; fi
  if [ "$AV" -lt 40 ] && [ -n "$P1" ]; then echo "$(date -u +%FT%TZ) MemAvailable $AV GB < 40: stopping C1 (pid $P1)" >> $LOG; killtree $P1; stopped="$stopped C1"; sleep 20; continue; fi
  sleep 30
done
