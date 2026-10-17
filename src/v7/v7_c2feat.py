#!/usr/bin/env python3
"""v7 / v7_c2feat.py -- C2 variant pools in the v7 'RB' layout (25 columns: 0-18 base + evidence, 19 is_extra,
20-24 the variant's pattern features), one directory per variant, from v7_c2_lists.py blocks + v7_lookup.py exact base values.
Columns 0-18 of the base top-50 part are copied from the stored PT feature tensor; block columns use pc_features.py formulas.
  PC_P=... python3 v7_c2feat.py --layout pc --common pc_common --feat F --c2 C2DIR --out O
"""
import os, sys, json, time, argparse, importlib
import numpy as np
VARS = ["UTIL12", "CONF12", "UTIL1", "UTIL2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True); ap.add_argument("--common", required=True); ap.add_argument("--feat", required=True)
    ap.add_argument("--c2", required=True); ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)
    PC = importlib.import_module(a.common); sys.path.insert(0, f"{PC.TRACKS}/residual/code")
    from rs_features import Hist, DyadReader
    S = PC.Splits(PC.DS); Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad")
    summ = {}
    for split in ("valid", "test"):
        pre = f"{a.feat}/{split}_" + ("A_" if a.layout == "sp" else "")
        Q = dict(np.load(f"{a.feat}/{split}_q.npz"))
        X = np.load(pre + "X.npy", mmap_mode="r"); LP = np.load(pre + "logp.npy", mmap_mode="r"); CA = np.load(pre + "cand.npy", mmap_mode="r")
        FI = np.load(pre + "filt.npy", mmap_mode="r"); BA = np.load(pre + "babove.npy", mmap_mode="r")
        BL = np.load(f"{a.c2}/{split}_blocks.npy", mmap_mode="r"); PX = np.load(f"{a.c2}/{split}_patX.npy", mmap_mode="r")
        lk = {k: np.load(f"{a.c2}/lk/{split}_lk_{k}.npy", mmap_mode="r") for k in ("logp", "brank", "babove", "filt")}
        n = BL.shape[0] if not a.smoke else min(BL.shape[0], a.smoke); sro = Q["sro"].astype(np.int64)[:n]
        gp_pt = (Q["gold_pos_A"] if a.layout == "sp" else Q["gold_pos"]).astype(np.int64)[:n]
        for v, nm in enumerate(VARS):
            od = f"{a.out}/{nm}"; os.makedirs(od, exist_ok=True)
            mm = lambda f, shp, dt: np.lib.format.open_memmap(f"{od}/{split}_{f}.npy", mode="w+", dtype=dt, shape=shp)
            Xo = mm("X", (n, 100, 25), np.float16); LPo = mm("logp", (n, 100), np.float32); CAo = mm("cand", (n, 100), np.int32)
            FIo = mm("filt", (n, 100), np.bool_); BAo = mm("babove", (n, 100), np.bool_)
            for lo in range(0, n, a.chunk):
                hi = min(n, lo + a.chunk); nb = hi - lo; s, r, t = sro[lo:hi, 0], sro[lo:hi, 1], sro[lo:hi, 3]
                Xt = np.asarray(X[lo:hi, :50, :19]).astype(np.float32)
                c = np.asarray(BL[lo:hi, v]).astype(np.int64); vv = c >= 0; cs = slice(50 * v, 50 * v + 50)
                lp = np.asarray(lk["logp"][lo:hi, cs]).astype(np.float64); bk = np.asarray(lk["brank"][lo:hi, cs]).astype(np.float64)
                top1 = np.asarray(LP[lo:hi, 0]).astype(np.float64)
                Xb = np.zeros((nb, 50, 19), np.float32)
                Xb[:, :, 0] = np.maximum(np.where(vv, lp, -30.0), -30.0); Xb[:, :, 1] = Xb[:, :, 0] - top1[:, None]
                Xb[:, :, 2] = np.log(np.maximum(np.where(vv, bk, 1.0), 1.0)); Xb[:, :, 4:8] = Xt[:, :1, 4:8]
                hf = Hh.feats(s, r, t, np.where(vv, c, -1)); A_, _ = BM.score(s, r, t, np.where(vv, c, -1), hf["dyad_hi"])
                Xb[:, :, 8] = np.log1p(hf["ntrip"]); Xb[:, :, 9] = np.log1p(hf["ndyad"]); Xb[:, :, 10] = hf["rec_trip"]
                Xb[:, :, 11] = hf["rec_dyad"]; Xb[:, :, 12] = np.log1p(hf["nobjrel"]); Xb[:, :, 13] = np.log1p(hf["nent"])
                Xb[:, :, 14] = hf["rec_objrel"]; Xb[:, :, 15] = np.log1p(hf["ntrip30"]); Xb[:, :, 16] = A_
                Xb[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xb[:, :, 18] = np.log1p(hf["nsubj"])[:, None]; Xb[~vv] = 0.0
                Xf = np.zeros((nb, 100, 25), np.float32); Xf[:, :50, :19] = Xt; Xf[:, 50:, :19] = Xb; Xf[:, 50:, 19] = vv
                Xf[:, :, 20:25] = np.asarray(PX[lo:hi, v]).astype(np.float32); Xf[:, 50:][~vv] = 0.0
                Xo[lo:hi] = Xf.astype(np.float16)
                LPo[lo:hi, :50] = np.asarray(LP[lo:hi, :50]); LPo[lo:hi, 50:] = np.where(vv, lp, -np.inf).astype(np.float32)
                CAo[lo:hi, :50] = np.asarray(CA[lo:hi, :50]); CAo[lo:hi, 50:] = c.astype(np.int32)
                FIo[lo:hi, :50] = np.asarray(FI[lo:hi, :50]); FIo[lo:hi, 50:] = np.asarray(lk["filt"][lo:hi, cs]) & vv
                BAo[lo:hi, :50] = np.asarray(BA[lo:hi, :50]); BAo[lo:hi, 50:] = np.asarray(lk["babove"][lo:hi, cs]) & vv
            for z in (Xo, LPo, CAo, FIo, BAo): z.flush()
            o = sro[:, 2]; ca = np.asarray(CAo); isg = (ca == o[:, None]) & (ca >= 0); gp = np.where(isg.any(1), isg.argmax(1), -1)
            assert ((gp_pt >= 0) & (gp_pt < 50) <= (gp >= 0) & (gp < 50)).all()
            np.savez(f"{od}/{split}_q.npz", rows=Q["rows"][:n], sro=sro, r_base=Q["r_base"][:n], gold_pos=gp.astype(np.int16))
            summ.setdefault(nm, {})[split] = dict(recall=float((gp >= 0).mean()), mean_size=float(50 + (np.asarray(BL[:n, v]) >= 0).sum(1).mean()))
            log(split, nm, json.dumps(summ[nm][split]))
    json.dump(summ, open(f"{a.out}/c2_pools_summary.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
