#!/usr/bin/env python3
"""v78 / v78_lists.py -- pattern blocks of every v7.8 pool (v78_rule_amendment_1), one pattern instantiation per query.
Pattern set: HRcov recipe (1-2-hop entity-anonymised path patterns, M1 = 64, M2 = 32, online history < t, te_hier of the
HRcov TE tables).  Two candidate lists (length 300):
  conf_hist : max te_hier desc, support (sum max(te_hier, 0)) desc, RecB_xf score (float64, CPU) desc, train freq desc, id asc
  util_<b>  : HRcov utility u = te_hier - log1p(mass_base (e-1)) per base b (v7_c2_lists UTIL12: best u desc, support desc,
              base log p desc, id asc)
Pools (first block F, block = first <= K list members not in F):
  SA<K>      F = history top-K (v78_hist.py), list conf_hist                       K in 25, 50, 75
  PC_<b>     F = base top-50 (stored HRcov first block), list conf_hist            (+C / +S / +T)
  HU<K>_<b>  F = base top-K (stored top-50 then dump order), list util_<b>         K in 25, 50 (= stored block check), 75
Per pool and column: 5 pattern features (best util of the pool's base [PAT_FILL for SA], best te, log1p #patterns, reach,
log1p(min(position in the pool's list, 300) + 1)).
Outputs <out>/<split>_<pool>_{blk,patX}.npy, <split>_firstblocks.npz, lists_summary.json.
  python3 v78_lists.py --ds polecat --hist HISTDIR --out OUT --workers 14 [--smoke N]
"""
import os, sys, json, time, argparse, importlib
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True, choices=list(C.DS)); ap.add_argument("--hist", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=14); ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--splits", default="valid,test"); ap.add_argument("--erm", default="")
a = ap.parse_args()
CFG = C.DS[a.ds]
os.environ["PC_P"] = CFG["P"]
sys.path.insert(0, CFG["code"]); sys.path.insert(0, f"{C.T}/retriever/src"); sys.path.insert(0, f"{C.T}/residual/code")
PCm = importlib.import_module(CFG["common"])
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
KC, BP, LP_LEN, KREF, PAT_FILL, E1 = 50, 50, C.LP_LEN, 200, C.PAT_FILL, C.E1
BASES = [] if a.erm else list(CFG["bases"])
G = {}


def pools():
    P = [(f"SA{K}", K, None, "conf") for K in C.SA_K]
    for b in BASES:
        P.append((f"PC_{b}", 50, b, "conf"))
        P += [(f"HU{K}_{b}", K, b, "util") for K in C.HU_K]
    return P


POOLS = pools() if not a.erm else [("SA50", 50, None, "conf")]


def recb_cpu(i, cand, r, t):
    """RecB_xf score (float64) of candidate ids for query i: 1000 a psi + 1000 (1 - a) xi (facts with ts < t)."""
    row = G["crow"][i]; lo, hi = G["ptr"][row], G["ptr"][row + 1]
    psi = np.zeros(len(cand))
    if hi > lo:
        cc = G["ccand"][lo:hi].astype(np.int64); pv = G["cpsi"][lo:hi].astype(np.float64); o = np.argsort(cc)
        cc = cc[o]; pv = pv[o]; j = np.searchsorted(cc, cand); j = np.minimum(j, len(cc) - 1); m = cc[j] == cand
        psi[m] = pv[j[m]]
    N, TT = G["N"], G["TT"]
    key = (r * N + cand) * TT
    cnt = np.searchsorted(G["AK"], key + t, "left") - np.searchsorted(G["AK"], key, "left")
    tot = np.searchsorted(G["TK"], r * TT + t, "left") - np.searchsorted(G["TK"], r * TT, "left")
    al = float(G["alpha"][r])
    return 1000.0 * al * psi + 1000.0 * (1.0 - al) * cnt / max(float(tot), 1.0)


def block_feats(Fb, order_list, rank_full, cand, best_u, bte, npat, K, use_util):
    """first block Fb (K ids) ++ <= K list members not in Fb -> (block ids [K], X [2K, 5])."""
    seen = set(Fb.tolist()); Cl = list(Fb.tolist())
    for e in order_list[:LP_LEN]:
        if len(Cl) >= 2 * K: break
        e = int(e)
        if e not in seen: Cl.append(e); seen.add(e)
    blk = np.full(K, -1, np.int32); nb = len(Cl) - K; blk[:nb] = Cl[K:]
    X = np.zeros((2 * K, 5), np.float32); X[:, 0] = PAT_FILL; X[:, 4] = np.log1p(LP_LEN + 1)
    if cand is not None and len(cand):
        Cv = np.array(Cl, np.int64); j = np.searchsorted(cand, Cv); j = np.minimum(j, len(cand) - 1); m = cand[j] == Cv
        mm = np.zeros(2 * K, bool); mm[:len(Cl)] = m; jj = np.zeros(2 * K, np.int64); jj[:len(Cl)] = j
        if use_util: X[mm, 0] = best_u[jj[mm]]
        X[mm, 1] = bte[jj[mm]]; X[mm, 2] = np.log1p(npat[jj[mm]]); X[mm, 3] = 1.0
        X[mm, 4] = np.log1p(np.minimum(rank_full[jj[mm]], LP_LEN) + 1)
    return blk, X, nb


def one(i):
    s, r, o, t = (int(x) for x in G["sro"][i]); N = G["N"]; R = G["R"]
    res = {}
    d = instantiate(G["EI"], s, t, M1, M2)
    diag = dict(has_pat=d is not None, n_union=0, gpos_conf=-1)
    cand = None; bte = npat = None; util_b = {}
    if d is not None:
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = G["te"].features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        cand, inv = np.unique(cc, return_inverse=True)
        bte = np.full(len(cand), -np.inf); np.maximum.at(bte, inv, te_h[pp])
        bte0 = np.zeros(len(cand)); np.maximum.at(bte0, inv, te_h[pp])          # HRcov feature convention (max with 0 init)
        supp_c = np.bincount(inv, weights=np.maximum(te_h[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        if G.get("erm_ids") is not None:          # SA_ERM: EdgeReMIND score replaces RecB_xf in the tie-break (floor = min listed)
            ei = G["erm_ids"][i].astype(np.int64); es = G["erm_sc"][i].astype(np.float64); oo = np.argsort(ei); ei = ei[oo]; es = es[oo]
            jx = np.minimum(np.searchsorted(ei, cand), len(ei) - 1); Sx = np.where(ei[jx] == cand, es[jx], es.min())
        else:
            Sx = recb_cpu(i, cand, r, t)
        oc = np.lexsort((cand, -G["freq"][cand], -Sx, -supp_c, -bte)); conf_list = cand[oc]
        rank_c = np.empty(len(cand), np.int64); rank_c[oc] = np.arange(len(cand))
        diag["n_union"] = int(len(cand))
        gi = np.searchsorted(cand, o)
        if gi < len(cand) and cand[gi] == o: diag["gpos_conf"] = int(rank_c[gi])
        for b in BASES:
            bb = G["base"][b]
            if bb["mass"] == "top200":
                drow = bb["drow"][i]; tki = bb["D_ids"][drow, :KREF].astype(np.int64); tks = bb["D_sc"][drow, :KREF].astype(np.float64)
                tail = max(1.0 - np.exp(tks).sum(), 1e-12); lp_tail = np.log(tail / max(N - KREF, 1))
                pb = G["pos_buf"]; pb[tki] = np.arange(KREF); pk = pb[cc]; lpc = np.where(pk >= 0, tks[np.maximum(pk, 0)], lp_tail); pb[tki] = -1
            else:
                lprow = np.asarray(bb["MM"][bb["srow"][i]], np.float64); lpc = lprow[cc]
            mass = np.bincount(pp, weights=np.exp(lpc), minlength=P_)
            util = te_h - np.log1p(mass * E1)
            bu = np.full(len(cand), -np.inf); np.maximum.at(bu, inv, util[pp])
            supp_u = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
            lpcand = np.zeros(len(cand)); lpcand[inv] = lpc
            ou = np.lexsort((-lpcand, -supp_u, -bu)); rank_u = np.empty(len(cand), np.int64); rank_u[ou] = np.arange(len(cand))
            util_b[b] = (cand[ou], rank_u, bu)
        bte = bte0
    for nm, K, b, kind in POOLS:
        if nm.startswith("SA"): Fb = G["htop"][i, :K].astype(np.int64)
        else: Fb = G["base"][b]["T75"][i, :K].astype(np.int64)
        if cand is None:
            blk, X, nb = block_feats(Fb, np.zeros(0, np.int64), None, None, None, None, None, K, False)
        elif kind == "conf":
            bu = util_b[b][2] if b is not None else None
            blk, X, nb = block_feats(Fb, conf_list, rank_c, cand, bu, bte, npat, K, b is not None)
        else:
            lst, rk, bu = util_b[b]
            blk, X, nb = block_feats(Fb, lst, rk, cand, bu, bte, npat, K, True)
        res[nm] = (blk, X.astype(np.float16), nb)
    return i, res, diag


def chunk(ix):
    return [one(int(i)) for i in ix]


def main():
    t0 = time.time(); log = C.log_fn(t0)
    S = PCm.Splits(PCm.DS); N, R = S.n_ent, S.R; assert (N, R) == (CFG["N"], CFG["R"])
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, N, R, 1); te = TETables(CFG["te"]); log("event index + TE")
    h, rr, tt, ts = allq.T.astype(np.int64)
    TT = int(ts.max()) + 1
    s_all = np.concatenate([h, tt]); r_all = np.concatenate([rr, rr + R]); o_all = np.concatenate([tt, h]); t_all = np.concatenate([ts, ts])
    AK = np.sort((r_all * N + o_all) * TT + t_all); TK = np.sort(r_all * TT + t_all)
    freq = C.train_freq(CFG["dd"], N)
    G.update(EI=EI, te=te, N=N, R=R, TT=TT, AK=AK, TK=TK, freq=freq, pos_buf=np.full(N, -1, np.int64), base={})
    for b, bc in [(b_, CFG["bases"][b_]) for b_ in BASES]:
        G["base"][b] = dict(mass=bc["mass"])
        if bc["mass"] == "top200":
            z = np.load(bc["dump"]); G["base"][b].update(D_ids=z["topk_ids"], D_sc=z["topk_scores"], D_sro=z["sro"])
            log("dump", b, z["topk_ids"].shape)
    os.makedirs(a.out, exist_ok=True); summ = dict(ds=a.ds, pools=[p[0] for p in POOLS], smoke=a.smoke)
    for split in a.splits.split(","):
        Hh = np.load(f"{a.hist}/{split}_hist.npz")
        n = len(Hh["rows"]) if not a.smoke else min(a.smoke, len(Hh["rows"]))
        sro = Hh["sro"].astype(np.int64)
        csr = np.load(f"{C.RECB_XF}/recb_csr_{split}_{CFG['name']}.npz")
        fidx = Hh["fidx"].astype(np.int64)[:n]
        G.update(sro=sro, crow=Hh["rows"].astype(np.int64), htop=Hh["htop"][:, :75].astype(np.int64), ptr=csr["ptr"], ccand=csr["cand"],
                 cpsi=csr["psi_tr"], alpha=np.asarray(csr["alpha_tr"], np.float64))
        assert (csr["sro"][G["crow"]] == sro).all(), "CSR rows"
        if a.erm:
            E = C.erm_load(a.erm, split); assert (np.asarray(E["keys"][G["crow"][:n], :3]) == sro[:n, :3]).all(), "ERM rows"
            G["erm_ids"] = np.asarray(E["top_ids"][G["crow"][:n]]); G["erm_sc"] = np.asarray(E["top_scores"][G["crow"][:n]])
        chk = {}
        for b, bc in [(b_, CFG["bases"][b_]) for b_ in BASES]:
            Q = dict(np.load(f"{bc['feat']}/{split}_q.npz")); assert (Q["sro"][fidx].astype(np.int64) == sro[:n]).all(), f"feat rows {b}"
            CA = np.load(C.feat_pre(bc, CFG["layout"], split) + "cand.npy", mmap_mode="r")
            T50 = np.asarray(CA[fidx, :50]).astype(np.int64)
            gb = G["base"][b]
            if bc["mass"] == "top200":
                drow = Q["rows"][fidx].astype(np.int64); assert (gb["D_sro"][drow[:2000]].astype(np.int64) == sro[:2000]).all(), "dump rows"
                gb["drow"] = drow; ext = gb["D_ids"][drow, :100].astype(np.int64)
                chk[b] = float((ext[:, :50] == T50).all(1).mean())
            else:
                MM = np.memmap(bc["rows"][split], dtype=np.float32, mode="r").reshape(-1, N); gb["MM"] = MM; gb["srow"] = Q["rows"][fidx].astype(np.int64)
                ext = np.zeros((n, 100), np.int64)
                for lo in range(0, n, 4096):
                    blk = np.asarray(MM[gb["srow"][lo:min(n, lo + 4096)]]); ip = np.argpartition(-blk, 100, axis=1)[:, :100]
                    vv = np.take_along_axis(blk, ip, 1); od = np.lexsort((ip, -vv), axis=-1); ext[lo:lo + len(blk)] = np.take_along_axis(ip, od, 1)
                chk[b] = float((ext[:n, :50] == T50[:n]).all(1).mean())
            T75 = np.zeros((n, 75), np.int64); T75[:, :50] = T50
            for i in range(n):                       # stored first 50, then the base order without duplicates
                sset = set(T50[i].tolist()); k = 50
                for e in ext[i]:
                    if k >= 75: break
                    if int(e) not in sset: T75[i, k] = e; k += 1
            gb["T75"] = T75
        log(split, "first-50 = base order (fraction)", chk)
        mm = lambda f, shp, dt: np.lib.format.open_memmap(f"{a.out}/{split}_{f}.npy", mode="w+", dtype=dt, shape=shp)
        OUT = {}
        for nm, K, b, kind in POOLS:
            OUT[nm] = (mm(f"{nm}_blk", (n, K), np.int32), mm(f"{nm}_patX", (n, 2 * K, 5), np.float16), np.zeros(n, np.int16))
        diag = dict(has_pat=np.zeros(n, bool), n_union=np.zeros(n, np.int64), gpos_conf=np.full(n, -1, np.int64))
        with mp.get_context("fork").Pool(a.workers) as pool:
            for part in pool.imap_unordered(chunk, np.array_split(np.arange(n), max(1, n // 2000)), chunksize=1):
                for i, res, dg in part:
                    for nm, (blk, X, nb) in res.items():
                        OUT[nm][0][i] = blk; OUT[nm][1][i] = X; OUT[nm][2][i] = nb
                    for k_, v_ in dg.items(): diag[k_][i] = v_
        for nm in OUT: OUT[nm][0].flush(); OUT[nm][1].flush()
        fb = dict(htop75=G["htop"][:n].astype(np.int32), **{f"T75_{b}": G["base"][b]["T75"][:n].astype(np.int32) for b in BASES},
                  **{f"nblk_{nm}": OUT[nm][2] for nm in OUT}, **diag)
        np.savez(f"{a.out}/{split}_firstblocks.npz", **fb)
        # check: HU50 block == stored HRcov pattern block
        st_chk = {}
        for b, bc in [(b_, CFG["bases"][b_]) for b_ in BASES]:
            pre = C.feat_pre(bc, CFG["layout"], split); CA = np.load(pre + "cand.npy", mmap_mode="r")
            if CFG["layout"] == "sp": npb = np.load(f"{bc['feat']}/{split}_q.npz")["n_pat_block"].astype(np.int64)[fidx]
            else: npb = np.load(f"{bc['lists']}/{split}_q.npz")["n_pat_block"].astype(np.int64)[fidx]
            stb = np.asarray(CA[fidx, 50:100]).astype(np.int64); jj = np.arange(50)[None, :]; stb = np.where(jj < npb[:, None], stb, -1)
            mine = np.asarray(OUT[f"HU50_{b}"][0]).astype(np.int64)
            Xs = np.asarray(np.load(pre + "X.npy", mmap_mode="r")[fidx[:20000], :100, 19:24]).astype(np.float32)
            Xm = np.asarray(OUT[f"HU50_{b}"][1][:min(n, 20000)]).astype(np.float32)
            vm = np.concatenate([np.ones((Xs.shape[0], 50), bool), stb[:Xs.shape[0]] >= 0], 1)
            st_chk[b] = dict(block_equal=float((mine == stb).all(1).mean()), patX_maxabs_first20k=float(np.abs(Xs - Xm)[vm].max()),
                             patX_frac_close=float((np.abs(Xs - Xm)[vm] < 2e-2).mean()))
        summ[split] = dict(n=int(n), stored_block_check=st_chk, first50_base_order=chk, frac_has_pat=float(diag["has_pat"].mean()),
                           mean_union=float(diag["n_union"].mean()),
                           mean_block={nm: float(OUT[nm][2].mean()) for nm in OUT}, seconds=round(time.time() - t0, 1))
        log(split, json.dumps(summ[split]))
    C.jdump(summ, f"{a.out}/lists_summary.json"); C.jdump(summ, f"lists_summary_{a.ds}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
