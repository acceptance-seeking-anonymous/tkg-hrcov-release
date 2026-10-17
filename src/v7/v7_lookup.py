#!/usr/bin/env python3
"""v7 / v7_lookup.py -- exact base log-prob / raw base rank / babove / filt for requested candidate ids (n x K, -1 pad) in
feature row order, from the frozen base's exact score rows (TiRGN replay / float32 full rows).  No evaluation.
  python3 v7_lookup.py --ds polecat --source tirgn --ckpt C --tirgn_tree TT --split test --req R.npy --feat F --out O [--valid_rows V]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_blend as HB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--source", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--fullrows", default=""); ap.add_argument("--tirgn_tree", default=""); ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--split", required=True); ap.add_argument("--req", required=True); ap.add_argument("--feat", required=True)
    ap.add_argument("--valid_rows", default=""); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); sro_full = HB.split_queries(Q[a.split], R, a.split)
    sro_f = np.load(f"{a.feat}/{a.split}_q.npz")["sro"].astype(np.int64)
    if a.split == "test" and len(sro_f) == len(sro_full): rows_req = np.arange(len(sro_full))
    elif a.valid_rows: rows_req = np.load(a.valid_rows).astype(np.int64)
    else: rows_req = np.arange(len(sro_full))
    assert (sro_full[rows_req] == sro_f).all(), "feature rows != split order"
    REQ = np.load(a.req, mmap_mode="r"); n, K = REQ.shape; assert n == len(rows_req)
    req = np.zeros(len(sro_full), bool); req[rows_req] = True; slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(n)
    os.makedirs(a.out, exist_ok=True)
    mm = lambda nm, dt: np.lib.format.open_memmap(f"{a.out}/{a.split}_lk_{nm}.npy", mode="w+", dtype=dt, shape=(n, K))
    LPo = mm("logp", np.float32); RKo = mm("brank", np.float32); BAo = mm("babove", np.bool_); FIo = mm("filt", np.bool_)
    if a.source == "tirgn":
        import hd_tirgn_replay as TRP
        it = ((sp, ts, sro, fn, trip) for sp, ts, sro, fn, trip in TRP.source(cfg, a.ckpt, a.L, (a.split,), dev, a.tirgn_tree))
    else:
        vp, tp = a.fullrows.split(","); it = ((sp, ts, sro, fn, trip) for sp, ts, sro, fn, trip, _ in HB.fullrow_source(cfg, {"valid": vp, "test": tp}, (a.split,), dev))
        MMr = np.memmap({"valid": vp, "test": tp}[a.split], dtype=np.float32, mode="r").reshape(-1, N)
    pos = 0; done = 0
    with torch.no_grad():
        for sp, ts, sro, fn, trip in it:
            if sro is None or sp != a.split: continue
            p0 = pos; nq = len(sro); pos += nq; idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[idx_all])
            for b0 in range(0, len(loc), a.bs):
                lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[rows]; B = len(lb)
                lp = (torch.as_tensor(np.asarray(MMr[rows]), device=dev) if a.source == "fullrows" else fn(sro[lb, 0], sro[lb, 1])).float()
                gold = torch.as_tensor(sro[lb, 2], device=dev); ar = torch.arange(B, device=dev)
                fr_, fc_ = Fi.pairs(sro[lb]); filt = torch.zeros((B, N), dtype=torch.bool, device=dev)
                if len(fr_): filt[torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev)] = True
                filt[ar, gold] = False
                ids = torch.as_tensor(np.asarray(REQ[sl]).astype(np.int64), device=dev); v = ids >= 0; idc = ids.clamp(min=0)
                lc = torch.gather(lp, 1, idc); srt = torch.sort(lp, dim=1).values
                brk = (N - torch.searchsorted(srt, lc.contiguous(), right=True)).float() + 1
                g = lp[ar, gold]
                LPo[sl] = torch.where(v, lc, torch.full_like(lc, float("-inf"))).cpu().numpy()
                RKo[sl] = torch.where(v, brk, torch.full_like(brk, float(N))).cpu().numpy()
                BAo[sl] = (v & (lc > g.unsqueeze(1))).cpu().numpy(); FIo[sl] = (v & torch.gather(filt, 1, idc) & (ids != gold[:, None])).cpu().numpy()
                done += B
    assert done == n, (done, n)
    for z in (LPo, RKo, BAo, FIo): z.flush()
    json.dump(dict(split=a.split, n=int(n), K=int(K), elapsed_s=round(time.time() - t0, 1)), open(f"{a.out}/{a.split}_lk_summary.json", "w"))
    log("DONE", n)


if __name__ == "__main__":
    main()
