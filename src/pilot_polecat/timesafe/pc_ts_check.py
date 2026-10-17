#!/usr/bin/env python3
"""sanity check: the snapshot index changes the bridge candidate set of a cohort query (vs the static all-train index)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import DS, DUMPS, WORK, Splits, BaseDump
import pc_lists as L
from bridge_features import BridgeIndex
S = Splits(DS); N, R = S.n_ent, S.R
D = BaseDump(DUMPS["regcn"], S)
z = np.load(f"{WORK}/lists_regcn_smoke/cohort_timesafe.npz")
reached = z["reached"]; cset = z["cset"]; samp = z["samp"]; t = z["t"]
keep = np.flatnonzero(reached)[:1000]
print("median cset first1000 reached", float(np.median(cset[keep])), "all reached", float(np.median(cset[reached])), "n reached", int(reached.sum()))
trq = S.q["train"]; ts_tr = trq[:, 3]
BIf = BridgeIndex(trq, N, R)
L.G.update(D=D, N=N, R=R)
rows = samp[keep[:5]]
for i in rows:
    s_, r_, o_, tt = (int(x) for x in D.sro[i])
    L.G["BI"] = BIf; zf = L.bridge_one(s_, r_, o_, D.topk_ids[i, :200], D.topk_scores[i, :200])
    BIk = BridgeIndex(trq[ts_tr < tt], N, R); L.G["BI"] = BIk
    zk = L.bridge_one(s_, r_, o_, D.topk_ids[i, :200], D.topk_scores[i, :200])
    print(int(i), tt, "static |C|", len(zf[0]["cands"]) if zf else None, "reached", zf[0]["reached"] if zf else None,
          "| snapshot |C|", len(zk[0]["cands"]) if zk else None, "reached", zk[0]["reached"] if zk else None)
