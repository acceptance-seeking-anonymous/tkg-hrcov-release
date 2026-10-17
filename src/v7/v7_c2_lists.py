#!/usr/bin/env python3
"""v7 / v7_c2_lists.py -- C2 pattern-design variants of the HRCOV_PAT pattern-prior block (pc_lists / sp_lists recipe,
pattern part only; tois_v7_rule.json C2).  One pattern instantiation per query (1-2-hop entity-anonymised path patterns,
online history < t, M1 = 64, M2 = 32), then four orderings / pattern sets at the same block budget (<= 50 candidates not in the
stored base top-50, dedupe in list order):
  UTIL12 : proposed  -- utility u = te_hier - log1p(mass_base (e - 1)), 1- and 2-hop patterns (re-derived; checked against the stored PT block)
  CONF12 : confidence only -- order by te_hier (no base-mass penalty), same pattern set
  UTIL1  : utility, 1-hop patterns only          UTIL2 : utility, 2-hop patterns only
Pattern features per candidate column (same definitions for every variant, computed on the variant's pattern set):
  max utility, max te_hier, log1p(#patterns), reach flag, log1p(min(position in the variant's list, 300) + 1).
Base mass: polecat = dump top-200 exact + uniform tail (pilot-1); smallpedia = exact full rows (pilot-2).
  python3 v7_c2_lists.py --ds polecat --out O --workers 16
"""
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True, choices=["polecat", "smallpedia"]); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args()
O = os.path.expandvars("$WORK_DIR"); T = f"{O}/tracks"; H = f"{T}/hardening"
CFG = {"polecat": dict(code=f"{T}/pc_tirgn/code", P=f"{T}/pc_tirgn", common="pc_common", te=f"{T}/pc_tirgn/work/te_POLECAT.npz", mass="top200",
                       feat=f"{T}/pc_tirgn/work/feat_tirgn", lists=f"{T}/pc_tirgn/work/lists_tirgn", layout="pc"),
       "smallpedia": dict(code=f"{T}/pilot_smallpedia/code", P=f"{T}/pilot_smallpedia", common="sp_common", te=f"{T}/pilot_smallpedia/work/te_SMALLPEDIA.npz",
                          mass="exact", feat=f"{T}/pilot_smallpedia/work/feat_tirgn", lists=None, layout="sp",
                          rows={"valid": f"{T}/pilot_smallpedia/dumps/tirgn_valid_logp.f32", "test": f"{T}/pilot_smallpedia/dumps/tirgn_test_logp.f32"})}[a.ds]
os.environ["PC_P"] = CFG["P"]
sys.path.insert(0, CFG["code"]); sys.path.insert(0, f"{T}/retriever/src"); sys.path.insert(0, f"{T}/residual/code")
C = __import__(CFG["common"])
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
KC, BP, LP_LEN, KREF, PAT_FILL, E1 = 50, 50, 300, 200, -6.0, np.e - 1.0
VARS = ["UTIL12", "CONF12", "UTIL1", "UTIL2"]
G = {}


def one(i):
    s, r, o, t = (int(x) for x in G["sro"][i]); N = G["N"]; R = G["R"]
    top50 = G["T50"][i].astype(np.int64); seen0 = set(top50.tolist())
    blocks = np.full((len(VARS), BP), -1, np.int32); X = np.zeros((len(VARS), KC + BP, 5), np.float32)
    X[:, :, 0] = PAT_FILL; X[:, :, 4] = np.log1p(LP_LEN + 1)
    nblk = np.zeros(len(VARS), np.int16)
    d = instantiate(G["EI"], s, t, M1, M2)
    if d is None:
        return i, blocks, X, nblk
    pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
    if G["mass"] == "top200":
        tki = G["D"].topk_ids[G["drow"][i], :KREF].astype(np.int64); tks = G["D"].topk_scores[G["drow"][i], :KREF].astype(np.float64)
        tail = max(1.0 - np.exp(tks).sum(), 1e-12); lp_tail = np.log(tail / max(N - KREF, 1))
        pb = G["pos_buf"]; pb[tki] = np.arange(KREF); pk = pb[cc]; lpc = np.where(pk >= 0, tks[np.maximum(pk, 0)], lp_tail); pb[tki] = -1
    else:
        lprow = np.asarray(G["MM"][G["srow"][i]], np.float64); lpc = lprow[cc]
    mass = np.bincount(pp, weights=np.exp(lpc), minlength=P_)
    hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
    te_h = G["te"].features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
    util = te_h - np.log1p(mass * E1)
    for v, nm in enumerate(VARS):
        pm = np.ones(P_, bool) if nm in ("UTIL12", "CONF12") else (hop == (1 if nm == "UTIL1" else 2))
        sel = pm[pp]
        if not sel.any(): continue
        pps = pp[sel]; ccs = cc[sel]; lpcs = lpc[sel]
        score = te_h if nm == "CONF12" else util
        cand, inv = np.unique(ccs, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, score[pps])
        bu = np.full(len(cand), -np.inf); np.maximum.at(bu, inv, util[pps])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pps])
        supp = np.bincount(inv, weights=np.maximum(score[pps], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        lpcand = np.zeros(len(cand)); lpcand[inv] = lpcs
        order = np.lexsort((-lpcand, -supp, -best)); cand_o = cand[order]
        rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        Cl = list(top50.tolist()); seen = set(seen0)
        for e in cand_o[:LP_LEN]:
            if len(Cl) >= KC + BP: break
            if int(e) not in seen: Cl.append(int(e)); seen.add(int(e))
        nb = len(Cl) - KC; nblk[v] = nb; blocks[v, :nb] = Cl[KC:]
        Cv = np.array(Cl, np.int64); j = np.searchsorted(cand, Cv); j = np.minimum(j, len(cand) - 1); m = cand[j] == Cv
        X[v, :len(Cl)][m, 0] = bu[j[m]]; X[v, :len(Cl)][m, 1] = bte[j[m]]; X[v, :len(Cl)][m, 2] = np.log1p(npat[j[m]])
        X[v, :len(Cl)][m, 3] = 1.0; X[v, :len(Cl)][m, 4] = np.log1p(np.minimum(rank_o[j[m]], LP_LEN) + 1)
    return i, blocks, X, nblk


def chunk(ix):
    return [one(int(i)) for i in ix]


def main():
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)
    S = C.Splits(C.DS); N, R = S.n_ent, S.R
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, N, R, 1); te = TETables(CFG["te"]); log("event index + TE")
    G.update(EI=EI, te=te, N=N, R=R, mass=CFG["mass"], pos_buf=np.full(N, -1, np.int64))
    if CFG["mass"] == "top200":
        D = C.BaseDump(C.DUMPS["tirgn"], S); G["D"] = D; log("dump", D.N, D.K)
    os.makedirs(a.out, exist_ok=True); summ = {}
    for split in ("valid", "test"):
        pre = f"{CFG['feat']}/{split}_" + ("A_" if CFG["layout"] == "sp" else "")
        Q = dict(np.load(f"{CFG['feat']}/{split}_q.npz")); CA = np.load(pre + "cand.npy", mmap_mode="r")
        n = len(Q["rows"]) if not a.smoke else a.smoke
        G["sro"] = Q["sro"].astype(np.int64); G["T50"] = np.asarray(CA[:, :50]).astype(np.int64)
        if CFG["mass"] == "top200":
            G["drow"] = Q["rows"].astype(np.int64)
            assert (G["D"].sro[G["drow"][:1000]].astype(np.int64) == G["sro"][:1000]).all()
        else:
            G["MM"] = np.memmap(CFG["rows"][split], dtype=np.float32, mode="r").reshape(-1, N); G["srow"] = Q["rows"].astype(np.int64)
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(chunk, np.array_split(np.arange(n), a.workers * 16), chunksize=1)
        BL = np.full((n, len(VARS), BP), -1, np.int32); XP = np.zeros((n, len(VARS), KC + BP, 5), np.float16); NB = np.zeros((n, len(VARS)), np.int16)
        for p in parts:
            for i, b, x, nb in p:
                BL[i] = b; XP[i] = x.astype(np.float16); NB[i] = nb
        np.save(f"{a.out}/{split}_blocks.npy", BL); np.save(f"{a.out}/{split}_patX.npy", XP); np.save(f"{a.out}/{split}_nblock.npy", NB)
        # check: re-derived UTIL12 block == stored PT block (bridge-free HRCOV_PAT pool)
        npb = (Q["n_pat_block"] if CFG["layout"] == "sp" else np.load(f"{CFG['lists']}/{split}_q.npz")["n_pat_block"][:len(Q["rows"])]).astype(np.int64)[:n]
        st = np.asarray(CA[:n, 50:100]).astype(np.int64); jj = np.arange(50)[None, :]; st = np.where(jj < npb[:, None], st, -1)
        eq = (BL[:, 0, :] == st).all(1)
        summ[split] = dict(n=int(n), util12_equals_stored_PT_block=float(eq.mean()), mean_block={nm: float(NB[:, v].mean()) for v, nm in enumerate(VARS)},
                           frac_empty_block={nm: float((NB[:, v] == 0).mean()) for v, nm in enumerate(VARS)}, seconds=round(time.time() - t0, 1))
        log(split, json.dumps(summ[split]))
    json.dump(summ, open(f"{a.out}/c2_lists_summary.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
