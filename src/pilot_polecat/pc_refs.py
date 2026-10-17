#!/usr/bin/env python3
"""pilot_polecat / pc_refs.py -- coverage-only reference arms on one base (CPU), exact ranks.

  PAT_BUMP : 6.1 non-learned pattern prior plug-in  s' = log p + delta * 1[c in top-B of L_pat]
             (B, delta) in B_PAT x D_GRID chosen by valid-sample MRR (retriever-track fixed combiner).
  BR_CS    : stage-2 bridge ranker list L_br as a backfill through the harness ConfidenceSwitch(q, M, B):
             trigger p1 <= tau_q (q-quantile of valid-sample p1), final list = base top-M ++ first B of L_br not
             in that head ++ base remainder; (q, M, B) chosen by valid-sample MRR.  M = K (200) is safe mode.
Exact: every list member's base log-prob comes from the full-row lookup (pc_lookup); ck from the same pass.
Output: work/refs_<base>/{ranks_test.npz, ranks_valid.npz}, refs_<base>.json (grids, selections).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (DS, DUMPS, WORK, LP_LEN, LB_LEN, D_GRID, B_PAT, Splits, BaseDump, metrics_from_ranks, jdump,
                       log_fn, bump_ranks_exact)

Q_GRID = [0.05, 0.10, 0.20, 0.30]
M_GRID = [0, 1, 3, 10, 200]
BCS_GRID = [10, 50, 200]


def load(base, split, S, D, allTrue):
    ld = f"{WORK}/lists_{base}"; lk = f"{WORK}/lookup_{base}"
    Lq = np.load(f"{ld}/{split}_q.npz"); rows = Lq["rows"]
    LK = np.load(f"{lk}/{split}.npz")
    sro = D.sro[rows].astype(np.int64); N = S.n_ent
    Lp = np.load(f"{ld}/{split}_Lpat.npy").astype(np.int64); Lb = np.load(f"{ld}/{split}_Lbr.npy").astype(np.int64)
    lp_req = LK["lp_req"]
    fk = S.fkey(sro[:, 0], sro[:, 1], sro[:, 3])
    def filt_of(L):
        code = fk[:, None] * N + np.where(L >= 0, L, 0)
        pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
        return (allTrue[pos] == code) & (L >= 0) & (L != sro[:, 2:3])
    return dict(rows=rows, sro=sro, gold=sro[:, 2], gold_lp=LK["gold_lp"].astype(np.float64), r_base=LK["rank_filt"].astype(np.int64),
                ck=LK["ck"].astype(np.int64), Lp=Lp, Lp_lp=lp_req[:, :LP_LEN], Lp_f=filt_of(Lp),
                Lb=Lb, Lb_lp=lp_req[:, LP_LEN:LP_LEN + LB_LEN], Lb_f=filt_of(Lb),
                tki=D.topk_ids[rows].astype(np.int64), p1=np.exp(D.topk_scores[rows, 0].astype(np.float64)),
                tk_f=filt_of(D.topk_ids[rows].astype(np.int64)))


def inhead_mask(V, M):
    """[n, LB_LEN] bool: L_br entry is inside the base top-M (cached per M)."""
    cache = V.setdefault("_inhead", {})
    if M not in cache:
        n = len(V["gold"]); m = np.zeros(V["Lb"].shape, bool)
        if M > 0:
            for lo in range(0, n, 20000):
                L = V["Lb"][lo:lo + 20000]; hd = V["tki"][lo:lo + 20000, :M]
                m[lo:lo + 20000] = (L[:, :, None] == hd[:, None, :]).any(2)
        cache[M] = m
    return cache[M]


def cs_ranks(V, trig, M, B):
    """ConfidenceSwitch exact filtered rank (see module doc)."""
    n = len(V["gold"]); r = V["r_base"].copy()
    idx = np.flatnonzero(trig)
    if len(idx) == 0:
        return r
    out = r.copy()
    for lo in range(0, len(idx), 20000):
        ii = idx[lo:lo + 20000]
        gold = V["gold"][ii]; rb = V["r_base"][ii]; glp = V["gold_lp"][ii]
        head = V["tki"][ii, :M]; head_f = V["tk_f"][ii, :M]
        g_in_head = (head == gold[:, None]).any(1) if M > 0 else np.zeros(len(ii), bool)
        head_nf = M - head_f.sum(1) - g_in_head.astype(np.int64)
        L = V["Lb"][ii]; Lf = V["Lb_f"][ii]; Llp = V["Lb_lp"][ii]
        inhead = inhead_mask(V, M)[ii]
        ok = (L >= 0) & ~inhead
        cum = np.cumsum(ok, 1)
        blk = ok & (cum <= B)                                   # first B not in head
        isg = blk & (L == gold[:, None])
        g_in_blk = isg.any(1)
        nf = blk & ~Lf & ~isg
        gp = np.where(g_in_blk, isg.argmax(1), L.shape[1])
        before = (nf & (np.arange(L.shape[1])[None, :] < gp[:, None])).sum(1)
        r_blk = 1 + head_nf + before
        r_out = rb + (nf & (Llp <= glp[:, None])).sum(1)
        out[ii] = np.where(g_in_head, rb, np.where(g_in_blk, r_blk, r_out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); F = S.filter_index(); N = S.n_ent
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    D = BaseDump(DUMPS[a.base], S)
    V = {sp: load(a.base, sp, S, D, allTrue) for sp in ("valid", "test")}
    log("loaded")
    res = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)")
    od = f"{WORK}/refs_{a.base}"; os.makedirs(od, exist_ok=True)
    rk_out = {sp: {"base": V[sp]["r_base"].astype(np.int32)} for sp in V}
    # ---------------- PAT_BUMP
    grids = {}
    for sp in ("valid", "test"):
        X = V[sp]
        grids[sp] = bump_ranks_exact(X["Lp"], X["Lp_lp"], X["Lp_f"], X["gold"], X["gold_lp"], X["r_base"], X["ck"])
    vm = (1.0 / grids["valid"]).mean(0); tm = (1.0 / grids["test"]).mean(0)
    bi, di = np.unravel_index(np.argmax(vm), vm.shape)
    res["PAT_BUMP"] = dict(selected=dict(B=B_PAT[bi], delta=D_GRID[di]), valid=metrics_from_ranks(grids["valid"][:, bi, di]),
                           test=metrics_from_ranks(grids["test"][:, bi, di]),
                           valid_grid_MRR=np.round(vm, 6).tolist(), test_grid_MRR_diagnostic=np.round(tm, 6).tolist(),
                           base_valid=metrics_from_ranks(V["valid"]["r_base"]), base_test=metrics_from_ranks(V["test"]["r_base"]),
                           B_grid=B_PAT, delta_grid=D_GRID,
                           identity_check_delta0=None)
    for sp in V:
        rk_out[sp]["PAT_BUMP"] = grids[sp][:, bi, di].astype(np.int32)
    # identity: delta -> 0 limit equals base (use smallest delta, B=10: must be >= base MRR - tiny)
    log("PAT_BUMP", json.dumps(res["PAT_BUMP"]["selected"]), res["PAT_BUMP"]["test"]["MRR"])
    # ---------------- BR_CS
    p1v = V["valid"]["p1"]
    grid = []
    for q in Q_GRID:
        tau = float(np.quantile(p1v, q))
        for M in M_GRID:
            for B in BCS_GRID:
                rv = cs_ranks(V["valid"], V["valid"]["p1"] <= tau, M, B)
                grid.append(dict(q=q, tau=tau, M=M, B=B, valid_MRR=float((1.0 / rv).mean())))
    best = max(grid, key=lambda g: g["valid_MRR"])
    rt = cs_ranks(V["test"], V["test"]["p1"] <= best["tau"], best["M"], best["B"])
    rv = cs_ranks(V["valid"], V["valid"]["p1"] <= best["tau"], best["M"], best["B"])
    # safe-mode best and unsafe-mode best as labelled diagnostics on test
    diag = {}
    for lab, cond in (("best_safe_M200", lambda g: g["M"] == 200), ("best_unsafe_M0", lambda g: g["M"] == 0)):
        g = max([x for x in grid if cond(x)], key=lambda x: x["valid_MRR"])
        r_ = cs_ranks(V["test"], V["test"]["p1"] <= g["tau"], g["M"], g["B"])
        diag[lab] = dict(cfg=g, test=metrics_from_ranks(r_))
    res["BR_CS"] = dict(selected=best, valid=metrics_from_ranks(rv), test=metrics_from_ranks(rt), grid=grid,
                        trigger_frac_test=float(np.mean(V["test"]["p1"] <= best["tau"])), diagnostics_test=diag)
    rk_out["test"]["BR_CS"] = rt.astype(np.int32); rk_out["valid"]["BR_CS"] = rv.astype(np.int32)
    log("BR_CS", json.dumps(best), res["BR_CS"]["test"]["MRR"])
    for sp in V:
        np.savez(f"{od}/ranks_{sp}.npz", **rk_out[sp])
    np.savez(f"{od}/patbump_grid_test.npz", grid=grids["test"].astype(np.int32))
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"refs_{a.base}.json"); jdump(res, f"{od}/refs_{a.base}.json")
    os.system(f"chmod -R a+rX {od}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
