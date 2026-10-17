#!/usr/bin/env python3
"""cost of the hardening RecB (hd_recb.walk_pairs) on tkgl-icews: per query, # facts (s, r', *, k) with k < t."""
import numpy as np, json, time, sys
DD = sys.argv[1]
t0 = time.time()
Q = {sp: np.loadtxt(f"{DD}/{sp}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)) for sp in ("train", "valid", "test")}
N, R = [int(x) for x in open(f"{DD}/stat.txt").read().split()[:2]]
allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
s = np.concatenate([allq[:, 0], allq[:, 2]]); r = np.concatenate([allq[:, 1], allq[:, 1] + R]); t = np.concatenate([allq[:, 3], allq[:, 3]])
TT = int(t.max()) + 2
key = (r * N + s) * TT + t; key.sort()
out = {}
for sp in ("valid", "test"):
    q = Q[sp]
    qs = np.concatenate([q[:, 0], q[:, 2]]); qr = np.concatenate([q[:, 1], q[:, 1] + R]); qt = np.concatenate([q[:, 3], q[:, 3]])
    base = (qr * N + qs) * TT
    n = np.searchsorted(key, base + qt, "left") - np.searchsorted(key, base, "left")
    # distinct objects among them (pairs after unique) is <= n; report n
    pr = np.percentile(n, [50, 90, 99, 99.9]).tolist()
    out[sp] = dict(n_queries=int(len(n)), total_edges=int(n.sum()), mean=float(n.mean()), max=int(n.max()), pct_50_90_99_999=pr,
                   frac_zero=float((n == 0).mean()))
    # window-100 variant
    n100 = np.searchsorted(key, base + qt, "left") - np.searchsorted(key, base + np.maximum(qt - 100, 0), "left")
    out[sp]["w100_total_edges"] = int(n100.sum())
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
