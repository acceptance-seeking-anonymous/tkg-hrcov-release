#!/usr/bin/env python3
"""phase2_forward_holdout / fh_lists.py (derived from presub/g2_lists.py; changes: dataset via fh_common, --splits so that
Stage 1 touches valid only, summary names carry the split) -- per-base pattern-prior lists and the HRCOV_PAT layout (CPU).

Pattern prior = pilot_smallpedia/sp_lists.py's (pilot-1 pc_lists recipe with EXACT base probabilities from the full
rows): 1-2-hop entity-anonymised path patterns of s at t_snap = ts // step (M1 = 64, M2 = 32), utility
u = te_hier - log1p(mass_base(C_p) (e - 1)) from the early-train TE tables, candidates ranked by (best u, summed
positive u, base log-prob); L_pat keeps the first 300.  No bridge block, no static block (pre-registered).
Layout [W = 100] = base top-50 (log-prob order) ++ the first <= 50 L_pat candidates outside the top-50 (-1 pad).
Coverage features (6, the pattern part of COV_FEATS + is_cov_cand): pat_best_util, pat_best_te, pat_log_npat,
pat_reach, pat_log_pos, is_cov_cand (= column >= 50).
Output: work/lists_<base>{_smoke}/{split}_{Lpat,Lpat_u,CP,XPc}.npy + {split}_q.npz (rows = target rows of the split)
  python3 g2_lists.py --base tirgn --workers 8 [--smoke n]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fh_common import (DS, WORK, KC, BP, W_PAT, LP_LEN, KREF, COV6, STEP, N_ENT, Splits, split_queries, jdump, log_fn, fullrows,
                       module_provenance)
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
import multiprocessing as mp

G = {}
E1 = np.e - 1.0
PAT_FILL = -6.0


def lookup(sorted_c, C):
    if len(sorted_c) == 0:
        return np.zeros(len(C), np.int64), np.zeros(len(C), bool)
    j = np.minimum(np.searchsorted(sorted_c, C), len(sorted_c) - 1)
    return j, (sorted_c[j] == C) & (C >= 0)


def cov6(C, pat, ncov_from):
    """sp_lists.cov_feats restricted to the pattern columns + is_cov_cand (identical values)."""
    X = np.zeros((len(C), len(COV6)), np.float32)
    X[:, 0] = PAT_FILL; X[:, 4] = np.log1p(LP_LEN + 1)
    if pat:
        j, m = lookup(pat["cand"], C)
        X[m, 0] = pat["best"][j[m]]; X[m, 1] = pat["bte"][j[m]]; X[m, 2] = np.log1p(pat["npat"][j[m]]); X[m, 3] = 1.0
        X[m, 4] = np.log1p(np.minimum(pat["rank"][j[m]], LP_LEN) + 1)
    X[ncov_from:, 5] = 1.0
    X[C < 0] = 0.0
    return X


def query_lists(k):
    EI = G["EI"]; te = G["te"]; R = G["R"]
    s, r, o, t = (int(x) for x in G["sro"][k])
    lp = G["LPchunk"][k - G["lo"]].astype(np.float64)
    tki = np.argpartition(-lp, KREF)[:KREF]; tki = tki[np.argsort(-lp[tki], kind="stable")]
    top50 = tki[:KC]
    res = dict(Lpat=np.full(LP_LEN, -1, np.int32), Lpat_u=np.zeros(LP_LEN, np.float16), CP=np.full(W_PAT, -1, np.int32),
               XPc=np.zeros((W_PAT, len(COV6)), np.float16))
    q = dict(gpos_pat=-1, g_in_union=False, n_union=0, has_pat=False)
    pat = {}
    d = instantiate(EI, s, t // STEP, M1, M2)
    if d is not None:
        q["has_pat"] = True
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        mass = np.bincount(pp, weights=np.exp(lp[cc]), minlength=P_)
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = te.features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        util = te_h - np.log1p(mass * E1)
        cand, inv = np.unique(cc, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, util[pp])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pp])
        supp = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        order = np.lexsort((-lp[cand], -supp, -best))
        cand_o = cand[order]; rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        L = cand_o[:LP_LEN]; res["Lpat"][:len(L)] = L; res["Lpat_u"][:len(L)] = best[order][:LP_LEN]
        q["n_union"] = len(cand)
        gi = np.searchsorted(cand, o)
        if gi < len(cand) and cand[gi] == o:
            q["g_in_union"] = True
            if rank_o[gi] < LP_LEN: q["gpos_pat"] = int(rank_o[gi])
        pat = dict(cand=cand, best=best, bte=bte, npat=npat, rank=rank_o)
    C = list(top50.tolist()); seen = set(C)
    for e in res["Lpat"]:
        if len(C) >= W_PAT or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    CP = np.full(W_PAT, -1, np.int64); CP[:len(C)] = C
    res["CP"][:] = CP
    res["XPc"][:] = cov6(CP, pat, KC).astype(np.float16)
    q["n_pat_block"] = len(C) - KC
    return res, q


def _target_chunk(args):
    lo, hi = args
    G["LPchunk"] = np.asarray(G["LP"][G["tr"][lo:hi]]); G["lo"] = lo
    outs = [query_lists(k) for k in range(lo, hi)]
    res = {k: np.stack([o[0][k] for o in outs]) for k in outs[0][0]}
    q = {k: np.array([o[1][k] for o in outs]) for k in outs[0][1]}
    return lo, res, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--splits", required=True, help="Stage 1: valid; Stage 2: test")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    sfx = "_smoke" if a.smoke else ""
    out_dir = f"{WORK}/lists_{a.base}{sfx}"; os.makedirs(out_dir, exist_ok=True)
    S = Splits(DS); N, R = S.n_ent, S.R
    assert N == N_ENT
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    targets = {"valid": vidx, "test": np.arange(len(split_queries(S, "test")))}
    if a.smoke:
        targets = {k: v[:a.smoke] for k, v in targets.items()}
    targets = {k: v for k, v in targets.items() if k in a.splits.split(",")}
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    G.update(EI=EventIndex(allq, N, R, STEP), te=TETables(f"{WORK}/te_{DS}.npz"), R=R)
    summ = dict(ds=DS, base=a.base, host=os.environ.get("HD_HOST", "serverA (CPU, tkg-gpu:latest)"), workers=a.workers,
                layout="[top50 | <=50 pattern-prior]", W=W_PAT)
    for split, tr in targets.items():
        t1 = time.time(); n = len(tr)
        sro_all = split_queries(S, split)
        LPall = fullrows(a.base, split)
        assert LPall.shape[0] == len(sro_all), (split, LPall.shape, len(sro_all))
        G.update(sro=sro_all[tr], LP=LPall, tr=tr)
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        A = dict(Lpat=mm("Lpat", (n, LP_LEN), np.int32), Lpat_u=mm("Lpat_u", (n, LP_LEN), np.float16),
                 CP=mm("CP", (n, W_PAT), np.int32), XPc=mm("XPc", (n, W_PAT, len(COV6)), np.float16))
        step = 500
        tasks = [(lo, min(n, lo + step)) for lo in range(0, n, step)]
        Q = {}
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (lo, res, q) in enumerate(pool.imap(_target_chunk, tasks, chunksize=1)):
                hi = lo + len(next(iter(q.values())))
                for nm, arr in res.items():
                    A[nm][lo:hi] = arr
                for nm, arr in q.items():
                    Q.setdefault(nm, []).append(arr)
                if k % 50 == 0:
                    log(split, hi, "/", n)
        for v in A.values():
            v.flush()
        Q = {k: np.concatenate(v) for k, v in Q.items()}
        np.savez(f"{out_dir}/{split}_q.npz", rows=tr, **Q)
        summ[split] = dict(n=int(n), seconds=round(time.time() - t1, 1), has_pat=float(Q["has_pat"].mean()),
                           gold_in_union=float(Q["g_in_union"].mean()), gold_in_Lpat300=float((Q["gpos_pat"] >= 0).mean()),
                           median_union=float(np.median(Q["n_union"])), mean_pat_block=float(Q["n_pat_block"].mean()))
        log(split, json.dumps(summ[split]))
    summ["elapsed_s"] = round(time.time() - t0, 1); summ["modules"] = module_provenance()
    tag = "_".join(a.splits.split(","))
    jdump(summ, f"lists_summary_{a.base}_{tag}{sfx}.json"); jdump(summ, f"{out_dir}/lists_summary_{tag}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
