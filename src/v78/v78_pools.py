#!/usr/bin/env python3
"""v78 / v78_pools.py -- assemble every v7.8 pool (v78_rule_amendment_1) in the v7 'RB' column layout (25 columns):
  0-7   generator slots: base pools = HRcov base features (stored for the base top-50, recomputed from exact lookups
        otherwise, as v7_c2feat.py); SA pools = RecB_xf generator features (G1 formulas on q, used only by SA_GEN)
  8-18  evidence (rs_features Hist / DyadReader, HRcov code)        19  is_extra (block column)
  20-24 pattern features of the pool's list (v78_lists.py)
Per pool and split: X, logp (base log p | log q for SA), cand, filt, bah (above gold in history order), bab (above gold in
base order; base pools), q.npz (rows, sro, r_hist, r_base, gold_pos, K).  One GPU pass recomputes the strict history order
(asserted equal to v78_hist.py) to place every column relative to the gold.
  python3 v78_pools.py --ds polecat --hist HD --lists LD --lookup b=DIR,... --out OUT [--smoke N]
"""
import os, sys, json, time, argparse, importlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import hd_blend as HB

LKOFF = {"PC": (0, 50), "HU25": (50, 75), "HU75": (75, 150), "T75tail": (150, 175)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(C.DS)); ap.add_argument("--hist", required=True); ap.add_argument("--lists", required=True)
    ap.add_argument("--lookup", required=True, help="base=dir,..."); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--erm", default="")
    a = ap.parse_args(); t0 = time.time(); log = C.log_fn(t0)
    cfg = C.DS[a.ds]; hcfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["PC_P"] = cfg["P"]; sys.path.insert(0, cfg["code"])
    PCm = importlib.import_module(cfg["common"]); sys.path.insert(0, f"{PCm.TRACKS}/residual/code")
    from rs_features import Hist, DyadReader
    Sp_ = PCm.Splits(PCm.DS); Hh = Hist(Sp_); BM = DyadReader(Sp_, Hh); log("hist + dyad")
    LK = dict(kv.split("=") for kv in a.lookup.split(",")) if a.lookup else {}
    bases = [] if a.erm else list(cfg["bases"])
    ERM = {sp: C.erm_load(a.erm, sp) for sp in ("valid", "test")} if a.erm else None
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R)
    freq = C.train_freq(cfg["dd"], N); gorder, _ = C.freq_order(freq); gorder_t = torch.as_tensor(gorder, device=dev)
    sro_full = {sp: HB.split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
    csr = {sp: HB.RecbCSR(f"{C.RECB_XF}/recb_csr_{sp}_{cfg['name']}.npz") for sp in ("valid", "test")}
    D = {}
    POOLS = [(f"SA{K}", K, None) for K in C.SA_K] + [(f"PC_{b}", 50, b) for b in bases] + [(f"HU{K}_{b}", K, b) for b in bases for K in (25, 75)]
    if a.erm: POOLS = [("SA50", 50, None)]
    for sp in ("valid", "test"):
        Hd = dict(np.load(f"{a.hist}/{sp}_hist.npz")); n = len(Hd["rows"]) if not a.smoke else min(a.smoke, len(Hd["rows"]))
        fb = dict(np.load(f"{a.lists}/{sp}_firstblocks.npz")); assert len(fb["htop75"]) == n, "lists / hist n"
        fidx = Hd["fidx"][:n].astype(np.int64)
        d = dict(n=n, fidx=fidx, rows=Hd["rows"][:n].astype(np.int64), sro=Hd["sro"][:n].astype(np.int64), htop=Hd["htop"][:n], hq=Hd["hq"][:n],
                 r_hist_ref=Hd["r_hist"][:n], fb=fb, blk={nm: np.load(f"{a.lists}/{sp}_{nm}_blk.npy", mmap_mode="r") for nm, _, _ in POOLS},
                 patX={nm: np.load(f"{a.lists}/{sp}_{nm}_patX.npy", mmap_mode="r") for nm, _, _ in POOLS}, base={})
        for b in bases:
            bc = cfg["bases"][b]; pre = C.feat_pre(bc, cfg["layout"], sp); Qf = dict(np.load(f"{bc['feat']}/{sp}_q.npz"))
            assert (Qf["sro"][fidx].astype(np.int64) == d["sro"]).all()
            lkreq = np.load(f"{LK[b]}/{sp}_lk_req.npy", mmap_mode="r")
            d["base"][b] = dict(X=np.load(pre + "X.npy", mmap_mode="r"), LP=np.load(pre + "logp.npy", mmap_mode="r"),
                                BA=np.load(pre + "babove.npy", mmap_mode="r"), CA=np.load(pre + "cand.npy", mmap_mode="r"), r_base=Qf["r_base"][fidx],
                                lk={k: np.load(f"{LK[b]}/{sp}_lk_{k}.npy", mmap_mode="r") for k in ("logp", "brank", "babove", "filt")}, lkreq=lkreq)
        D[sp] = d
    os.makedirs(a.out, exist_ok=True)
    OUT = {}
    for sp in ("valid", "test"):
        n = D[sp]["n"]; OUT[sp] = {}
        for nm, K, b in POOLS:
            od = f"{a.out}/{nm}"; os.makedirs(od, exist_ok=True); W = 2 * K
            mm = lambda f, shp, dt: np.lib.format.open_memmap(f"{od}/{sp}_{f}.npy", mode="w+", dtype=dt, shape=shp)
            OUT[sp][nm] = dict(X=mm("X", (n, W, 25), np.float16), LP=mm("logp", (n, W), np.float32), CA=mm("cand", (n, W), np.int32),
                               FI=mm("filt", (n, W), np.bool_), BAH=mm("bah", (n, W), np.bool_), BAB=(mm("bab", (n, W), np.bool_) if b else None),
                               gp=np.full(n, -1, np.int64), r_hist=np.zeros(n, np.int64))
    req = {sp: np.zeros(len(sro_full[sp]), bool) for sp in ("valid", "test")}; slot = {sp: -np.ones(len(sro_full[sp]), np.int64) for sp in ("valid", "test")}
    for sp in ("valid", "test"): req[sp][D[sp]["rows"]] = True; slot[sp][D[sp]["rows"]] = np.arange(D[sp]["n"])
    xi = HB.XiCounts(N, R, dev); pos_ = {"valid": 0, "test": 0}; done = {"valid": 0, "test": 0}
    chk = dict(htop_equal=0, n=0, evid_maxabs_vs_stored={}, base_feat_maxabs_vs_stored={})
    with torch.no_grad():
        for sp, ts, sro, _, trip, _ in HB.recb_source(hcfg, ("valid", "test")):
            if sro is not None:
                p0 = pos_[sp]; nq = len(sro); pos_[sp] += nq
                idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[sp][idx_all]); d = D[sp]
                for b0 in range(0, len(loc), a.bs):
                    lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[sp][rows]; B = len(lb)
                    s_q, r_q, o_q, t_q = sro[lb, 0], sro[lb, 1], sro[lb, 2], sro[lb, 3]
                    S = HB.recb_row(csr[sp], rows, r_q, xi.xi(torch.as_tensor(r_q, device=dev)), dev, "tr")
                    pos, ent = C.order_positions(S, gorder_t, C.erm_dense(ERM[sp], rows, N, dev) if ERM else None)
                    eqh = (ent[:, :75].int().cpu().numpy() == d["htop"][sl, :75]).all(1); chk["htop_equal"] += int(eqh.sum()); chk["n"] += B
                    ar = torch.arange(B, device=dev); gold = torch.as_tensor(o_q, device=dev); pg = pos[ar, gold]
                    fr_, fc_ = Fi.pairs(sro[lb]); FM = torch.zeros((B, N), dtype=torch.bool, device=dev)
                    if len(fr_): FM[torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev)] = True
                    nbf = (FM & (pos < pg.unsqueeze(1))).sum(1); r_hist = (1 + pg - nbf).cpu().numpy()
                    assert (r_hist == d["r_hist_ref"][sl]).all(), "r_hist differs from v78_hist"
                    tot = S.double().sum(1, keepdim=True)
                    hq = d["hq"][sl].astype(np.float64)
                    q1 = hq[:, 0]; p50 = hq[:, :50] / np.maximum(hq[:, :50].sum(1, keepdims=True), 1e-300)
                    ent50 = -(p50 * np.log(np.maximum(p50, 1e-300))).sum(1)
                    marg = np.clip(np.log(np.maximum(hq[:, 0], 1e-300)) - np.log(np.maximum(hq[:, 1], 1e-300)), -30, 30)
                    tailm = np.log(np.maximum(1.0 - hq[:, :200].sum(1), 1e-12))
                    # union of candidate columns for the evidence features
                    cols = {}; U = []; off = 0
                    def addu(key, arr):
                        nonlocal off
                        cols[key] = (off, off + arr.shape[1]); U.append(arr.astype(np.int64)); off += arr.shape[1]
                    addu("htop75", d["htop"][sl, :75])
                    for nm, K, b in POOLS: addu(nm, np.asarray(d["blk"][nm][sl]))
                    for b in bases: addu(f"T75_{b}", d["fb"][f"T75_{b}"][sl])
                    Uc = np.concatenate(U, 1)
                    hf = Hh.feats(s_q, r_q, t_q, Uc); A_, _ = BM.score(s_q, r_q, t_q, Uc, hf["dyad_hi"])
                    EV = np.zeros(Uc.shape + (11,), np.float32)
                    EV[..., 0] = np.log1p(hf["ntrip"]); EV[..., 1] = np.log1p(hf["ndyad"]); EV[..., 2] = hf["rec_trip"]; EV[..., 3] = hf["rec_dyad"]
                    EV[..., 4] = np.log1p(hf["nobjrel"]); EV[..., 5] = np.log1p(hf["nent"]); EV[..., 6] = hf["rec_objrel"]
                    EV[..., 7] = np.log1p(hf["ntrip30"]); EV[..., 8] = A_; EV[..., 9] = np.log1p(hf["nquery"])[:, None]; EV[..., 10] = np.log1p(hf["nsubj"])[:, None]
                    Ut = torch.as_tensor(Uc, device=dev); Uv = Ut >= 0; Uc0 = Ut.clamp(min=0)
                    Upos = torch.gather(pos, 1, Uc0); Ubah = (Uv & (Upos < pg.unsqueeze(1))).cpu().numpy()
                    Ufi = (Uv & torch.gather(FM, 1, Uc0) & (Ut != gold.unsqueeze(1))).cpu().numpy()
                    Uq = (torch.gather(S, 1, Uc0).double() / tot.clamp(min=1e-300)).cpu().numpy(); Upos = Upos.cpu().numpy()
                    def U_(key, arr): lo, hi = cols[key]; return arr[:, lo:hi]
                    for nm, K, b in POOLS:
                        W = 2 * K; o_ = OUT[sp][nm]
                        if nm.startswith("SA"): fk = ("htop75", 0, K)
                        else: fk = (f"T75_{b}", 0, K)
                        first = U_(fk[0], Uc)[:, :K]; blk = U_(nm, Uc)
                        CA = np.concatenate([first, blk], 1); v = CA >= 0
                        cat = lambda arr: np.concatenate([U_(fk[0], arr)[:, :K], U_(nm, arr)], 1)
                        X = np.zeros((B, W, 25), np.float32)
                        X[:, :, 8:19] = cat(EV)
                        X[:, K:, 19] = (blk >= 0)
                        X[:, :, 20:25] = np.asarray(d["patX"][nm][sl]).astype(np.float32)
                        if b is None:                       # RecB_xf generator slots (SA_GEN only)
                            qq = cat(Uq); lq = np.where(qq > 0, np.log(np.maximum(qq, 1e-300)), -30.0)
                            X[:, :, 0] = np.maximum(lq, -30.0); X[:, :, 1] = X[:, :, 0] - np.maximum(np.log(np.maximum(q1, 1e-300)), -30.0)[:, None]
                            hp = cat(Upos); X[:, :, 2] = np.log(hp + 1.0); X[:, :, 3] = hp < 50
                            X[:, :, 4] = q1[:, None]; X[:, :, 5] = ent50[:, None]; X[:, :, 6] = marg[:, None]; X[:, :, 7] = tailm[:, None]
                            LP = np.where(v, X[:, :, 0], -np.inf)
                            BAB = None
                        else:
                            bb = d["base"][b]; lk = bb["lk"]
                            fx = d["fidx"][sl]
                            LPf = np.asarray(bb["LP"][fx, :50]).astype(np.float64); BAf = np.asarray(bb["BA"][fx, :50]); Xf = np.asarray(bb["X"][fx, :50, :8]).astype(np.float32)
                            top1 = LPf[:, 0]
                            if nm.startswith("PC"): seg = [("PC",)]
                            elif K == 25: seg = [("HU25",)]
                            else: seg = [("T75tail",), ("HU75",)]
                            lp_x = np.concatenate([np.asarray(lk["logp"][sl, LKOFF[s_[0]][0]:LKOFF[s_[0]][1]]) for s_ in seg], 1).astype(np.float64)
                            rk_x = np.concatenate([np.asarray(lk["brank"][sl, LKOFF[s_[0]][0]:LKOFF[s_[0]][1]]) for s_ in seg], 1).astype(np.float64)
                            ba_x = np.concatenate([np.asarray(lk["babove"][sl, LKOFF[s_[0]][0]:LKOFF[s_[0]][1]]) for s_ in seg], 1)
                            rq_x = np.concatenate([np.asarray(bb["lkreq"][sl, LKOFF[s_[0]][0]:LKOFF[s_[0]][1]]) for s_ in seg], 1).astype(np.int64)
                            k0 = min(K, 50)                  # columns taken from the stored HRcov feature dir
                            CAx = CA[:, k0:]; assert (rq_x == CAx).all(), f"lookup ids != pool ids ({nm})"
                            LP = np.concatenate([LPf[:, :k0], lp_x], 1); LP = np.where(v, LP, -np.inf)
                            BAB = np.concatenate([BAf[:, :k0], ba_x], 1) & v
                            X[:, :k0, :8] = Xf[:, :k0]
                            vx = CAx >= 0
                            X[:, k0:, 0] = np.maximum(np.where(vx, lp_x, -30.0), -30.0); X[:, k0:, 1] = X[:, k0:, 0] - top1[:, None]
                            X[:, k0:, 2] = np.log(np.maximum(np.where(vx, rk_x, 1.0), 1.0)); X[:, k0:, 3] = 0.0
                            X[:, k0:, 4:8] = Xf[:, :1, 4:8]
                            assert (np.asarray(bb["CA"][fx, :k0]).astype(np.int64) == CA[:, :k0]).all(), "first block ids"
                            if done[sp] == 0 and nm.startswith("PC"):
                                chk["evid_maxabs_vs_stored"][f"{sp}_{b}"] = float(np.abs(X[:, :50, 8:19] - np.asarray(bb["X"][fx, :50, 8:19]).astype(np.float32)).max())
                        X[~v] = 0.0
                        o_["X"][sl] = X.astype(np.float16); o_["LP"][sl] = LP.astype(np.float32); o_["CA"][sl] = CA.astype(np.int32)
                        o_["FI"][sl] = cat(Ufi) & v; o_["BAH"][sl] = cat(Ubah) & v
                        if BAB is not None: o_["BAB"][sl] = BAB
                        isg = (CA == o_q[:, None]) & v; o_["gp"][sl] = np.where(isg.any(1), isg.argmax(1), -1); o_["r_hist"][sl] = r_hist
                    done[sp] += B
                    del S, pos, ent, FM
                if len(loc) and (done[sp] // 20000) != ((done[sp] - len(loc)) // 20000): log(sp, done[sp], "/", D[sp]["n"])
            xi.add(trip)
    summ = dict(ds=a.ds, pools=[p[0] for p in POOLS], checks=chk, smoke=a.smoke)
    for sp in ("valid", "test"):
        assert done[sp] == D[sp]["n"], (sp, done[sp], D[sp]["n"])
        for nm, K, b in POOLS:
            o_ = OUT[sp][nm]
            for k in ("X", "LP", "CA", "FI", "BAH", "BAB"):
                if o_[k] is not None: o_[k].flush()
            extra = dict(r_base=D[sp]["base"][b]["r_base"]) if b else {}
            np.savez(f"{a.out}/{nm}/{sp}_q.npz", rows=D[sp]["rows"], sro=D[sp]["sro"], r_hist=o_["r_hist"], gold_pos=o_["gp"], K=K, **extra)
            summ.setdefault(nm, {})[sp] = dict(recall=float((o_["gp"] >= 0).mean()), mean_size=float((np.asarray(o_["CA"]) >= 0).sum(1).mean())) if sp == "valid" else dict(n=int(D[sp]["n"]))
    summ["checks"]["htop_equal_frac"] = chk["htop_equal"] / max(chk["n"], 1)
    summ["elapsed_s"] = round(time.time() - t0, 1)
    C.jdump(summ, f"{a.out}/pools_summary.json"); C.jdump(summ, f"pools_summary_{a.ds}{'_smoke' if a.smoke else ''}.json")
    log("valid pools", json.dumps({k: v for k, v in summ.items() if k not in ("pools",)})[:3000])
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
