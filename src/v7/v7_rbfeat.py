#!/usr/bin/env python3
"""v7 / v7_rbfeat.py -- B2 RecB-expanded pool RB = stored base top-50 ++ first <= 50 RecB-trained (clean) candidates not in
the top-50 (from v7_scan.py), with the SAME feature code as pc_features.py / sp_features.py for the new block:
  layout (25 columns): 0-7 base, 8-18 evidence (rs_features Hist / DyadReader, facts with ts < t), 19 is_extra (j >= 50),
  20-24 RecB generator features (log q, log q - log q1, log RecB rank, has_psi, in RecB top-50).
Columns 0-18 of the top-50 part are copied from the stored PT feature tensor (identical candidates).
Also writes the RecB features of the stored PT pool (for QBLEND / generator-specific PT checks): {split}_PT_recbF.npy.
  PC_P=... python3 v7_rbfeat.py --layout pc --common pc_common --feat F --lists L --scan S --out O
"""
import os, sys, json, time, argparse, importlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True); ap.add_argument("--common", required=True)
    ap.add_argument("--feat", required=True); ap.add_argument("--lists", default=""); ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, default=20000); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)
    PC = importlib.import_module(a.common)
    sys.path.insert(0, f"{PC.TRACKS}/residual/code")
    from rs_features import Hist, DyadReader
    S = PC.Splits(PC.DS); Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad", BM.n_transitions)
    os.makedirs(a.out, exist_ok=True)
    for split in ("valid", "test"):
        pre = f"{a.feat}/{split}_" + ("A_" if a.layout == "sp" else "")
        Q = dict(np.load(f"{a.feat}/{split}_q.npz"))
        X = np.load(pre + "X.npy", mmap_mode="r"); LP = np.load(pre + "logp.npy", mmap_mode="r"); CA = np.load(pre + "cand.npy", mmap_mode="r")
        FI = np.load(pre + "filt.npy", mmap_mode="r"); BA = np.load(pre + "babove.npy", mmap_mode="r")
        Z = np.load(f"{a.scan}/{split}_scan.npz")
        rbc = np.load(f"{a.scan}/{split}_rb_cand.npy", mmap_mode="r"); rbl = np.load(f"{a.scan}/{split}_rb_logp.npy", mmap_mode="r")
        rbk = np.load(f"{a.scan}/{split}_rb_brank.npy", mmap_mode="r"); rba = np.load(f"{a.scan}/{split}_rb_babove.npy", mmap_mode="r")
        rbf = np.load(f"{a.scan}/{split}_rb_filt.npy", mmap_mode="r")
        FT = np.load(f"{a.scan}/{split}_recbF_T50.npy", mmap_mode="r"); FP = np.load(f"{a.scan}/{split}_recbF_PTblock.npy", mmap_mode="r")
        FR = np.load(f"{a.scan}/{split}_recbF_RBblock.npy", mmap_mode="r")
        n = X.shape[0] if not a.smoke else a.smoke
        assert len(Z["rows"]) == X.shape[0], (len(Z["rows"]), X.shape)
        sro = Q["sro"].astype(np.int64)
        assert (np.asarray(CA[:min(n, 5000), :50]) >= 0).all()
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{a.out}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        Xo = mm("X", (n, 100, 25), np.float16); LPo = mm("logp", (n, 100), np.float32); CAo = mm("cand", (n, 100), np.int32)
        FIo = mm("filt", (n, 100), np.bool_); BAo = mm("babove", (n, 100), np.bool_); PTr = mm("PT_recbF", (n, 100, 5), np.float16)
        gp_pt = (Q["gold_pos_A"] if a.layout == "sp" else Q["gold_pos"]).astype(np.int64)
        gpos_rb = Z["gpos_RB"].astype(np.int64)
        gp = np.where((gp_pt >= 0) & (gp_pt < 50), gp_pt, np.where(gpos_rb >= 0, 50 + gpos_rb, -1))
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk); nb = hi - lo
            s, r, t = sro[lo:hi, 0], sro[lo:hi, 1], sro[lo:hi, 3]
            Xt = np.asarray(X[lo:hi, :50, :19]).astype(np.float32)
            c = np.asarray(rbc[lo:hi]).astype(np.int64); v = c >= 0
            lp = np.asarray(rbl[lo:hi]).astype(np.float64); bk = np.asarray(rbk[lo:hi]).astype(np.float64)
            Xb = np.zeros((nb, 50, 19), np.float32)
            top1 = np.asarray(LP[lo:hi, 0]).astype(np.float64)
            Xb[:, :, 0] = np.maximum(np.where(v, lp, -30.0), -30.0); Xb[:, :, 1] = Xb[:, :, 0] - top1[:, None]
            Xb[:, :, 2] = np.log(np.maximum(np.where(v, bk, 1.0), 1.0)); Xb[:, :, 3] = 0.0
            Xb[:, :, 4:8] = Xt[:, :1, 4:8]                                   # query-level base features
            hf = Hh.feats(s, r, t, np.where(v, c, -1)); A_, _ = BM.score(s, r, t, np.where(v, c, -1), hf["dyad_hi"])
            Xb[:, :, 8] = np.log1p(hf["ntrip"]); Xb[:, :, 9] = np.log1p(hf["ndyad"]); Xb[:, :, 10] = hf["rec_trip"]
            Xb[:, :, 11] = hf["rec_dyad"]; Xb[:, :, 12] = np.log1p(hf["nobjrel"]); Xb[:, :, 13] = np.log1p(hf["nent"])
            Xb[:, :, 14] = hf["rec_objrel"]; Xb[:, :, 15] = np.log1p(hf["ntrip30"]); Xb[:, :, 16] = A_
            Xb[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xb[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
            Xb[~v] = 0.0
            Xf = np.zeros((nb, 100, 25), np.float32)
            Xf[:, :50, :19] = Xt; Xf[:, 50:, :19] = Xb; Xf[:, 50:, 19] = v
            Xf[:, :50, 20:25] = np.asarray(FT[lo:hi]).astype(np.float32); Xf[:, 50:, 20:25] = np.asarray(FR[lo:hi]).astype(np.float32)
            Xf[:, 50:][~v] = 0.0
            Xo[lo:hi] = Xf.astype(np.float16)
            LPo[lo:hi, :50] = np.asarray(LP[lo:hi, :50]); LPo[lo:hi, 50:] = np.where(v, lp, -np.inf).astype(np.float32)
            CAo[lo:hi, :50] = np.asarray(CA[lo:hi, :50]); CAo[lo:hi, 50:] = c.astype(np.int32)
            FIo[lo:hi, :50] = np.asarray(FI[lo:hi, :50]); FIo[lo:hi, 50:] = np.asarray(rbf[lo:hi]) & v
            BAo[lo:hi, :50] = np.asarray(BA[lo:hi, :50]); BAo[lo:hi, 50:] = np.asarray(rba[lo:hi]) & v
            PTr[lo:hi, :50] = np.asarray(FT[lo:hi]); PTr[lo:hi, 50:] = np.asarray(FP[lo:hi])
            log(split, hi, "/", n)
        for z in (Xo, LPo, CAo, FIo, BAo, PTr): z.flush()
        o = sro[:n, 2]; ca = np.asarray(CAo); isg = (ca == o[:, None]) & (ca >= 0)
        gchk = np.where(isg.any(1), isg.argmax(1), -1)
        assert (gchk == gp[:n]).all(), ("gold position mismatch", int((gchk != gp[:n]).sum()))
        qo = dict(rows=Q["rows"][:n], sro=sro[:n], r_base=Q["r_base"][:n], gold_pos=gp[:n].astype(np.int16), n_rb=Z["n_rb"][:n])
        np.savez(f"{a.out}/{split}_q.npz", **qo)
        json.dump(dict(split=split, n=int(n), recall_RB=float((gp[:n] >= 0).mean()), recall_T50=float(((gp_pt[:n] >= 0) & (gp_pt[:n] < 50)).mean()),
                       mean_n_rb=float(Z["n_rb"][:n].mean()), elapsed_s=round(time.time() - t0, 1)), open(f"{a.out}/{split}_rb_summary.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
