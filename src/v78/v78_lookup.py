#!/usr/bin/env python3
"""v78 / v78_lookup.py -- exact base log p / raw base rank / babove / filt for the base-dependent v7.8 pool columns that are
not in the stored HRcov feature dir: [PC block 50 | HU25 block 25 | HU75 block 75 | base top-75 columns 51-75 (25)] = 175
columns per query (v7_lookup.py logic; sources tirgn replay, fullrows, cogn sparse store).
  python3 v78_lookup.py --ds polecat --base tirgn --lists L --split test --out O [--ckpt C --tirgn_tree TT | --fullrows V,T | --store D]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import hd_blend as HB

LAYOUT = [("PC", 50), ("HU25", 25), ("HU75", 75), ("T75tail", 25)]


def build_req(lists, base, split):
    fb = np.load(f"{lists}/{split}_firstblocks.npz")
    parts = [np.load(f"{lists}/{split}_PC_{base}_blk.npy"), np.load(f"{lists}/{split}_HU25_{base}_blk.npy"),
             np.load(f"{lists}/{split}_HU75_{base}_blk.npy"), fb[f"T75_{base}"][:, 50:75]]
    return np.concatenate([p.astype(np.int64) for p in parts], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--base", required=True); ap.add_argument("--lists", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True); ap.add_argument("--hist", required=True)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--tirgn_tree", default=""); ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--fullrows", default=""); ap.add_argument("--store", default=""); ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); sro_full = HB.split_queries(Q[a.split], R, a.split)
    Hh = np.load(f"{a.hist}/{a.split}_hist.npz")
    REQ = build_req(a.lists, a.base, a.split); n, K = REQ.shape
    rows_req = Hh["rows"].astype(np.int64)[:n]
    assert len(rows_req) == n
    req = np.zeros(len(sro_full), bool); req[rows_req] = True; slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(n)
    os.makedirs(a.out, exist_ok=True)
    mm = lambda nm, dt: np.lib.format.open_memmap(f"{a.out}/{a.split}_lk_{nm}.npy", mode="w+", dtype=dt, shape=(n, K))
    LPo = mm("logp", np.float32); RKo = mm("brank", np.float32); BAo = mm("babove", np.bool_); FIo = mm("filt", np.bool_)
    if a.ckpt:
        import hd_tirgn_replay as TRP
        it = ((sp, ts, sro, fn, trip) for sp, ts, sro, fn, trip in TRP.source(cfg, a.ckpt, a.L, (a.split,), dev, a.tirgn_tree))
    elif a.store:
        import g8_rows
        it = g8_rows.source(cfg, a.store, (a.split,), dev)
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
                lp = (torch.as_tensor(np.asarray(MMr[rows]), device=dev) if a.fullrows else fn(sro[lb, 0], sro[lb, 1])).float()
                gold = torch.as_tensor(sro[lb, 2], device=dev); ar = torch.arange(B, device=dev)
                fr_, fc_ = Fi.pairs(sro[lb]); filt = torch.zeros((B, N), dtype=torch.bool, device=dev)
                if len(fr_): filt[torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev)] = True
                filt[ar, gold] = False
                ids = torch.as_tensor(REQ[sl], device=dev); v = ids >= 0; idc = ids.clamp(min=0)
                lc = torch.gather(lp, 1, idc); srt = torch.sort(lp, dim=1).values
                brk = (N - torch.searchsorted(srt, lc.contiguous(), right=True)).float() + 1
                g = lp[ar, gold]
                LPo[sl] = torch.where(v, lc, torch.full_like(lc, float("-inf"))).cpu().numpy()
                RKo[sl] = torch.where(v, brk, torch.full_like(brk, float(N))).cpu().numpy()
                BAo[sl] = (v & (lc > g.unsqueeze(1))).cpu().numpy(); FIo[sl] = (v & torch.gather(filt, 1, idc) & (ids != gold[:, None])).cpu().numpy()
                done += B
            if (done // 50000) != ((done - len(loc)) // 50000): log(a.split, done, "/", n)
    assert done == n, (done, n)
    for z in (LPo, RKo, BAo, FIo): z.flush()
    np.save(f"{a.out}/{a.split}_lk_req.npy", REQ.astype(np.int32))
    C.jdump(dict(split=a.split, base=a.base, n=int(n), K=int(K), layout=LAYOUT, elapsed_s=round(time.time() - t0, 1)), f"{a.out}/{a.split}_lk_summary.json")
    log("DONE", n)


if __name__ == "__main__":
    main()
