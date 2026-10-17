#!/usr/bin/env python3
"""pilot_icews / pi_ghcheck.py -- exact-equivalence + speed check of pi_common.fast_global_hist vs harness logcl.GlobalHist
on real tkgl-icews train snapshots (both query directions)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from pi_common import DS, Splits, fast_global_hist
from logcl import GlobalHist
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
S = Splits(DS); N, R = S.n_ent, S.R
q = S.q["train"]; ts = q[:, 3]; uts = np.unique(ts)
g0, g1 = GlobalHist(N), fast_global_hist()(N)
res = dict(device=dev, checked_snapshots=[], max_abs_diff=0.0, n_rows=0)
t_old = t_new = 0.0
for k, u in enumerate(uts[:6200]):
    tr = q[ts == u][:, :3]
    if k >= 6000 and k % 20 == 0:
        s = np.concatenate([tr[:, 0], tr[:, 2]]); r = np.concatenate([tr[:, 1], tr[:, 1] + R])
        a0 = time.time(); v0 = g0.freq_vec(s, r, dev); torch.cuda.synchronize() if dev == "cuda" else None; t_old += time.time() - a0
        a1 = time.time(); v1 = g1.freq_vec(s, r, dev); torch.cuda.synchronize() if dev == "cuda" else None; t_new += time.time() - a1
        assert v0.dtype == v1.dtype and v0.shape == v1.shape
        res["max_abs_diff"] = max(res["max_abs_diff"], float((v0 - v1).abs().max())); res["n_rows"] += len(s)
        res["checked_snapshots"].append(int(u)); assert torch.equal(v0, v1), u
    g0.update(tr, R); g1.update(tr, R)
assert g0.d == g1.d
res.update(seconds_old=round(t_old, 2), seconds_new=round(t_new, 2), speedup=round(t_old / max(t_new, 1e-9), 1), identical=True)
print(json.dumps(res))
