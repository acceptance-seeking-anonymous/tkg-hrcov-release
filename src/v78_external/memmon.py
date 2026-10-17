#!/usr/bin/env python3
"""memmon.py -- run a command, sample the process tree's PSS / RSS and host MemAvailable every 10 s, record peaks and wall time.
Kills the tree if host MemAvailable drops below --min-avail-gb (protects co-tenants).  Usage: memmon.py --out mem.json [--min-avail-gb 80] -- cmd ..."""
import sys, os, time, json, subprocess, argparse, signal
import psutil
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--min-avail-gb", type=float, default=80.0)
ap.add_argument("--interval", type=float, default=10.0)
a, rest = ap.parse_known_args()
cmd = rest[1:] if rest and rest[0] == "--" else rest
t0 = time.time(); p = subprocess.Popen(cmd); P = psutil.Process(p.pid)
peak_pss = peak_rss = 0.0; min_avail = 1e18; killed = False; samples = 0; peak_nproc = 0
def tree():
    try: return [P] + P.children(recursive=True)
    except psutil.Error: return []
while p.poll() is None:
    pss = rss = 0.0; procs = tree()
    for q in procs:
        try:
            mi = q.memory_full_info(); pss += mi.pss; rss += mi.rss
        except psutil.Error: pass
    av = psutil.virtual_memory().available
    peak_pss = max(peak_pss, pss); peak_rss = max(peak_rss, rss); min_avail = min(min_avail, av); peak_nproc = max(peak_nproc, len(procs)); samples += 1
    if av < a.min_avail_gb * 2**30 and not killed:
        print(f"[memmon] host MemAvailable {av/2**30:.1f} GB < {a.min_avail_gb} GB -> terminating run to protect co-tenants", flush=True)
        for q in reversed(procs):
            try: q.kill()
            except psutil.Error: pass
        killed = True
    if samples % 30 == 0:
        json.dump(dict(running=True, elapsed_s=round(time.time()-t0,1), peak_pss_gb=peak_pss/2**30, peak_rss_sum_gb=peak_rss/2**30,
                       min_host_avail_gb=min_avail/2**30), open(a.out, "w"))
    time.sleep(a.interval)
rc = p.wait()
json.dump(dict(cmd=cmd, returncode=rc, wall_s=round(time.time()-t0, 1), peak_pss_gb=round(peak_pss/2**30, 2), peak_rss_sum_gb=round(peak_rss/2**30, 2),
               min_host_avail_gb=round(min_avail/2**30, 1), peak_nproc=peak_nproc, killed_for_memory=killed, samples=samples,
               note="peak_pss_gb = max over samples of summed PSS of the process tree (shared pages counted once); rss_sum double-counts fork-shared pages"),
          open(a.out, "w"), indent=1)
sys.exit(rc)
