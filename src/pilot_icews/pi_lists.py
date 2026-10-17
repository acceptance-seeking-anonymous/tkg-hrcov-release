#!/usr/bin/env python3
"""pilot_icews / pi_lists.py -- per-base history-aware pool-widening lists (CPU), both relation schemes.

For the valid sample and the test sample of one frozen base, and for scheme in {orig, rn}:
  * 6.1 pattern prior (pilot-1 pc_lists.query_lists, bridge part removed): instantiate the subject's 1-2-hop
    entity-anonymised path patterns (M1=64, M2=32, FULL online history < t), score each pattern by the predicted residual
    utility u = te_hier - log1p(mass_base(C_p) (e-1)) (base mass from the dump top-200, tail-uniform elsewhere), rank
    candidates by best utility -> L_pat (top 300).
    scheme rn: the event index and the query relation use relation-normalised ids (pi_relnorm relmap); TE tables shared
    (relmap is the identity on every relation that occurs in train).
  * C_cov = base top-50 ++ first 150 L_pat members outside the top-50 (columns 50..199; HRCOV_B uses the first B);
    per candidate the 6 coverage features COV_FEATS.
Outputs under work/lists_<base>/ : {split}_{scheme}_{Lpat,Ccov,Xcov}.npy + {split}_{scheme}_q.npz (gold diagnostics).
  python3 pi_lists.py --base regcn --workers 22
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import (DUMPS, DS, WORK, P, KC, W_MAX, LP_LEN, KREF, COV_FEATS, Splits, SplitsView, relmap2, jdump, log_fn)
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
import multiprocessing as mp

G = {}
E1 = np.e - 1.0
PAT_FILL = -6.0


def query_lists(i):
    EI = G["EI"]; te = G["te"]; N = G["N"]; R = G["R"]
    s, r, o, t = (int(x) for x in G["sro"][i])
    rq = int(G["rmap"][r])
    tki = G["tki"][i].astype(np.int64); tks = G["tks"][i].astype(np.float64)
    p_top = np.exp(tks); tail = max(1.0 - p_top.sum(), 1e-12); lp_tail = np.log(tail / max(N - KREF, 1))
    top50 = tki[:KC]
    Lpat = np.full(LP_LEN, -1, np.int32); Ccov = np.full(W_MAX, -1, np.int32); Xcov = np.zeros((W_MAX, len(COV_FEATS)), np.float16)
    q = dict(gpos_pat=-1, g_in_union=False, n_union=0, has_pat=False, n_pat_block=0)
    pat = {}
    d = instantiate(EI, s, t, M1, M2)
    if d is not None:
        q["has_pat"] = True
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        pos_in_top = G["pos_buf"]; pos_in_top[tki] = np.arange(KREF)
        pk = pos_in_top[cc]
        lpc = np.where(pk >= 0, tks[np.maximum(pk, 0)], lp_tail)
        pos_in_top[tki] = -1
        mass = np.bincount(pp, weights=np.exp(lpc), minlength=P_)
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = te.features(rq, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        util = te_h - np.log1p(mass * E1)
        cand, inv = np.unique(cc, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, util[pp])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pp])
        supp = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        lpcand = np.zeros(len(cand)); lpcand[inv] = lpc
        order = np.lexsort((-lpcand, -supp, -best))
        cand_o = cand[order]
        rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        Lp = cand_o[:LP_LEN]; Lpat[:len(Lp)] = Lp
        q["n_union"] = len(cand)
        gi = np.searchsorted(cand, o)
        if gi < len(cand) and cand[gi] == o:
            q["g_in_union"] = True
            if rank_o[gi] < LP_LEN:
                q["gpos_pat"] = int(rank_o[gi])
        pat = dict(cand=cand, best=best, bte=bte, npat=npat, rank=rank_o)
    C = list(top50.tolist()); seen = set(C)
    for e in Lpat:
        if len(C) >= W_MAX or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    C = np.array(C, np.int64); Ccov[:len(C)] = C
    X = np.zeros((len(C), len(COV_FEATS)), np.float32)
    X[:, 0] = PAT_FILL; X[:, 4] = np.log1p(LP_LEN + 1)
    if pat:
        j = np.searchsorted(pat["cand"], C); j = np.minimum(j, len(pat["cand"]) - 1); m = pat["cand"][j] == C
        X[m, 0] = pat["best"][j[m]]; X[m, 1] = pat["bte"][j[m]]; X[m, 2] = np.log1p(pat["npat"][j[m]]); X[m, 3] = 1.0
        X[m, 4] = np.log1p(np.minimum(pat["rank"][j[m]], LP_LEN) + 1)
    X[KC:, 5] = 1.0
    Xcov[:len(C)] = X.astype(np.float16)
    q["n_pat_block"] = len(C) - KC
    return Lpat, Ccov, Xcov, q


def _chunk(rows):
    outs = [query_lists(int(i)) for i in rows]
    return (np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]), np.stack([o[2] for o in outs]),
            {k: np.array([o[3][k] for o in outs]) for k in outs[0][3]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--schemes", default="orig,rn")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    out_dir = f"{WORK}/lists_{a.base}"; os.makedirs(out_dir, exist_ok=True)
    S = Splits(DS); N, R = S.n_ent, S.R
    Dd = np.load(DUMPS[a.base])
    te = TETables(f"{WORK}/te_TKGL_ICEWS.npz")
    summ = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)", workers=a.workers)
    for sch in a.schemes.split(","):
        rmap = np.arange(2 * R) if sch == "orig" else relmap2()
        Sv = S if sch == "orig" else SplitsView(S, rmap)
        allq = np.concatenate([Sv.q["train"], Sv.q["valid"], Sv.q["test"]])
        EI = EventIndex(allq, N, R, 1)
        log(sch, "event index", len(EI.src))
        for split in ("valid", "test"):
            t1 = time.time()
            sro = Dd[f"{split}_sro"].astype(np.int64)
            G.clear(); G.update(EI=EI, te=te, N=N, R=R, sro=sro, rmap=rmap, tki=Dd[f"{split}_topk_ids"], tks=Dd[f"{split}_topk_scores"],
                                pos_buf=np.full(N, -1, np.int64))
            n = len(sro)
            chunks = np.array_split(np.arange(n), max(1, n // 1000))
            res = [None] * len(chunks)
            with mp.get_context("fork").Pool(a.workers) as pool:
                for k, r_ in enumerate(pool.imap(_chunk, chunks, chunksize=1)):
                    res[k] = r_
                    if k % 100 == 0:
                        log(sch, split, k, "/", len(chunks))
            Lp = np.concatenate([r_[0] for r_ in res]); Cc = np.concatenate([r_[1] for r_ in res]); Xc = np.concatenate([r_[2] for r_ in res])
            qd = {k: np.concatenate([r_[3][k] for r_ in res]) for k in res[0][3]}
            assert (Cc[:, :KC] == Dd[f"{split}_topk_ids"][:, :KC]).all()
            np.save(f"{out_dir}/{split}_{sch}_Lpat.npy", Lp); np.save(f"{out_dir}/{split}_{sch}_Ccov.npy", Cc)
            np.save(f"{out_dir}/{split}_{sch}_Xcov.npy", Xc); np.savez(f"{out_dir}/{split}_{sch}_q.npz", **qd)
            gin = (Cc == sro[:, 2:3]).any(1)
            summ[f"{split}_{sch}"] = dict(n=int(n), seconds=round(time.time() - t1, 1), has_pat=float(qd["has_pat"].mean()),
                                         gold_in_pattern_union=float(qd["g_in_union"].mean()), gold_in_Lpat=float((qd["gpos_pat"] >= 0).mean()),
                                         gold_in_Ccov=float(gin.mean()), median_n_union=float(np.median(qd["n_union"])),
                                         mean_block=float(qd["n_pat_block"].mean()))
            log(sch, split, json.dumps(summ[f"{split}_{sch}"]))
        del EI
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"lists_summary_{a.base}.json"); jdump(summ, f"{out_dir}/lists_summary_{a.base}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
