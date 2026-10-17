#!/usr/bin/env python3
"""hardening / hd_lookup.py -- pc_lookup.py --mode lookup for any hd_blend.py row source (used for TiRGN on polecat).

Replays the base with ground-truth online history and writes, per query of the pilot valid sample / every test query,
from the FULL log-prob row (same keys as pc_lookup.py, so pc_features.py / pc_refs.py run unchanged):
  lp_req [n,500] exact log-prob of the requested list members (Lpat ++ Lbr), rk_req raw rank (1 + #higher),
  ck [n,6] #non-filter entities (gold excluded) with log-prob > gold + delta (D_GRID), gold_lp, rank_filt (exact), tie_n.
  python3 hd_lookup.py --base tirgn --source tirgn --ckpt C --work W   (PC_* env as pc_common)
"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_blend as HB

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--source", default="tirgn"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--work", required=True); ap.add_argument("--L", type=int, default=10); ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--tree", default=f"{HB.HD}/stage/TiRGN")
a = ap.parse_args()
t0 = time.time()
D_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]; LP_LEN, LB_LEN = 300, 200
cfg = HB.DSCFG["polecat"]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
Fi = HB.Filt(Q, N, R)
sro_full = {sp: HB.split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
rows_req = {"valid": np.load(f"{a.work}/valid_sample_idx.npy"), "test": np.arange(len(sro_full["test"]))}
req = {}
for sp in ("valid", "test"):
    ld = f"{a.work}/lists_{a.base}"
    req[sp] = np.concatenate([np.load(f"{ld}/{sp}_Lpat.npy"), np.load(f"{ld}/{sp}_Lbr.npy")], 1)
    assert len(req[sp]) == len(rows_req[sp])
slot = {}; need = {}
for sp in ("valid", "test"):
    s_ = -np.ones(len(sro_full[sp]), np.int64); s_[rows_req[sp]] = np.arange(len(rows_req[sp])); slot[sp] = s_
O = {sp: dict(lp_req=np.zeros((len(rows_req[sp]), LP_LEN + LB_LEN), np.float32), rk_req=np.zeros((len(rows_req[sp]), LP_LEN + LB_LEN), np.int32),
              ck=np.zeros((len(rows_req[sp]), len(D_GRID)), np.int32), gold_lp=np.zeros(len(rows_req[sp]), np.float32),
              rank_filt=np.zeros(len(rows_req[sp]), np.int32), tie_n=np.zeros(len(rows_req[sp]), np.int32)) for sp in ("valid", "test")}
if a.source == "cogn":            # G8: CognTKE exact rows from the g8_dump.py sparse store (--ckpt = store dir)
    import g8_rows
    it = g8_rows.source(cfg, a.ckpt, ("valid", "test"), dev)
elif a.source == "tirgn":
    import hd_tirgn_replay as TRP
    it = TRP.source(cfg, a.ckpt, a.L, ("valid", "test"), dev, a.tree)
else:
    it = HB.replay_source(cfg, a.base, a.ckpt, a.L, ("valid", "test"), dev, print)
Dg = torch.tensor(D_GRID, device=dev, dtype=torch.float32)
pos = {"valid": 0, "test": 0}
with torch.no_grad():
    for sp, ts, sro, fn, trip in it:
        if sro is None: continue
        p0 = pos[sp]; nq = len(sro); pos[sp] += nq
        idx = np.arange(p0, p0 + nq); loc = np.flatnonzero(slot[sp][idx] >= 0)
        for b0 in range(0, len(loc), a.bs):
            lb = loc[b0:b0 + a.bs]; sl = slot[sp][idx[lb]]
            lp = fn(sro[lb, 0], sro[lb, 1])
            gold = torch.as_tensor(sro[lb, 2], device=dev)
            ar = torch.arange(len(lb), device=dev)
            glp = lp[ar, gold]
            fr, fc = Fi.pairs(sro[lb])
            lf = lp.clone()
            if len(fr): lf[torch.as_tensor(fr, device=dev), torch.as_tensor(fc, device=dev)] = float("-inf")
            Ox = O[sp]
            Ox["gold_lp"][sl] = glp.cpu().numpy(); Ox["rank_filt"][sl] = ((lf > glp.unsqueeze(1)).sum(1) + 1).cpu().numpy()
            Ox["tie_n"][sl] = ((lf == glp.unsqueeze(1)).sum(1) - 1).cpu().numpy()
            Ox["ck"][sl] = (lf.unsqueeze(1) > (glp.unsqueeze(1) + Dg.unsqueeze(0)).unsqueeze(2)).sum(2).cpu().numpy()
            rq = torch.as_tensor(req[sp][sl], device=dev, dtype=torch.long)
            lpr = lp.gather(1, rq.clamp(min=0))
            srt, _ = torch.sort(lp, dim=1)
            cle = torch.searchsorted(srt, lpr.contiguous(), right=True)
            Ox["lp_req"][sl] = torch.where(rq >= 0, lpr, torch.full_like(lpr, -1e30)).cpu().numpy()
            Ox["rk_req"][sl] = torch.where(rq >= 0, N - cle + 1, torch.zeros_like(cle)).cpu().numpy()
            del lp, lf, srt, cle, lpr
res = dict(base=a.base, source=a.source, host=os.environ.get("HD_HOST", ""), elapsed_s=round(time.time() - t0, 1))
for sp in ("valid", "test"):
    d = f"{a.work}/lookup_{a.base}"; os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/{sp}.npz", rows=rows_req[sp], **O[sp])
    rf = O[sp]["rank_filt"].astype(np.float64)
    res[f"{sp}_harness"] = dict(MRR=float(np.mean(1 / rf)), H10=float(np.mean(rf <= 10)), n=int(len(rf)))
json.dump(res, open(f"lookup_{a.base}.json", "w"), indent=1)
print("ALL DONE", json.dumps(res), flush=True)
