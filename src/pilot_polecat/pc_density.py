#!/usr/bin/env python3
"""pilot_polecat / pc_density.py -- GraphBridge G5-style DENSITY CONTROLS for the coverage sets (CPU).

Is bridge / pattern reach relation-typed signal or just set size?  For every query of
  (a) the bridge-ranker training cohort (the exact 30,000 coverage-hard TRAIN rows pc_lists sampled, rng 0), and
  (b) a uniform 20,000-row TEST sample (rng 7, the augmentation-oracle sample),
compare reach of the natural candidate set C (typed 2-hop bridge; pattern union) with controls at the SAME |C|:
  uniform random admission  : E[reach] = |C| / N
  degree-matched random     : |C| entities drawn uniformly within train-degree bins, bin counts matched to C
                              -> P(gold in set) = c_bin(gold) / n_bin(gold)   (exact expectation)
  popularity (top-|C| by train degree)
and the same for the ranked lists L_br / L_pat truncated at B in {10, 50, 200}.  |C| is also reported as a fraction
of the entity count.  Also records dump-vs-lookup filtered-rank agreement (rate, max |diff|) for the base.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import DS, DUMPS, WORK, KC, Splits, BaseDump, split_queries, jdump, log_fn
from patterns import EventIndex, instantiate
from rt_common import M1, M2
import multiprocessing as mp

G = {}


def _chunk(args):
    rows, with_pat = args
    D = G["D"]; BI = G["BI"]; EI = G["EI"]; N = G["N"]; dbin = G["dbin"]; nbin = G["nbin"]; rpop = G["rpop"]; nb = len(nbin)
    out = []
    for i in rows:
        s, r, o, t = (int(x) for x in D.sro[i])
        rec = {}
        b = BI.build(s, r, o, D.topk_ids[i, :200], D.topk_scores[i, :200])
        sets = {"bridge": (b["cands"].astype(np.int64) if b is not None else np.zeros(0, np.int64))}
        if with_pat:
            d = instantiate(EI, s, t, M1, M2)
            sets["pattern"] = np.unique(d["pc_c"]).astype(np.int64) if d is not None else np.zeros(0, np.int64)
        for nm, C in sets.items():
            n = len(C)
            cb = np.bincount(dbin[C], minlength=nb) if n else np.zeros(nb, np.int64)
            rec[nm] = (n, bool((C == o).any()) if n else False, n / N, cb[dbin[o]] / nbin[dbin[o]], bool(rpop[o] < n))
        out.append(rec)
    return out


def summarize(recs, nm):
    a = np.array([r[nm] for r in recs], dtype=np.float64)
    return dict(n=int(len(a)), reach=round(float(a[:, 1].mean()), 4), median_size=float(np.median(a[:, 0])),
                median_size_frac_of_N=round(float(np.median(a[:, 2])), 4), mean_size_frac_of_N=round(float(a[:, 2].mean()), 4),
                uniform_random_same_size=round(float(a[:, 2].mean()), 4), degree_matched_random_same_size=round(float(a[:, 3].mean()), 4),
                popularity_top_same_size=round(float(a[:, 4].mean()), 4),
                reach_minus_degree_matched=round(float(a[:, 1].mean() - a[:, 3].mean()), 4))


def list_controls(L, gold, dbin, nbin, rpop, N, Bs):
    out = {}
    nb = len(nbin)
    for B in Bs:
        Lb = L[:, :B]; ok = Lb >= 0
        hit = ((Lb == gold[:, None]) & ok).any(1)
        size = ok.sum(1)
        gbin = dbin[gold]
        cnt_same = ((dbin[np.maximum(Lb, 0)] == gbin[:, None]) & ok).sum(1)
        out[f"B{B}"] = dict(reach=round(float(hit.mean()), 4), mean_size=float(size.mean()),
                            uniform_random_same_size=round(float((size / N).mean()), 5),
                            degree_matched_random_same_size=round(float((cnt_same / nbin[gbin]).mean()), 5),
                            popularity_top_same_size=round(float((rpop[gold] < size).mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=20000)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N = S.n_ent
    D = BaseDump(DUMPS[a.base], S)
    from bridge_features import BridgeIndex
    BI = BridgeIndex(S.q["train"], N, S.R)
    EI = EventIndex(np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]), N, S.R, 1)
    freq = S.freq
    dbin = np.where(freq == 0, 0, 1 + np.floor(np.log2(np.maximum(freq, 1)))).astype(np.int64)
    nbin = np.bincount(dbin).astype(np.float64)
    rpop = np.empty(N, np.int64); rpop[np.argsort(-freq, kind="stable")] = np.arange(N)
    G.update(D=D, BI=BI, EI=EI, N=N, dbin=dbin, nbin=nbin, rpop=rpop)
    rows = {sp: D.rows(sp) for sp in ("train", "test")}
    res = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)", n_entities=N,
               degree_bins="0 = no train fact; k = 1 + floor(log2(train degree))", n_degree_bins=int(len(nbin)))
    # (a) cohort: exactly pc_lists' sample
    gf = D.extra["gold_rank_filt"]
    hard = rows["train"][gf[rows["train"]] > KC]
    samp = np.random.default_rng(0).choice(hard, min(30000, len(hard)), replace=False)
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs = [x for part in pool.map(_chunk, [(c, False) for c in np.array_split(samp, a.workers * 8)], chunksize=1) for x in part]
    res["cohort_train_coverage_hard"] = dict(bridge=summarize(recs, "bridge"),
                                             note="30,000 train rows with gold outside base top-50 (pc_lists cohort sample)")
    log("cohort", json.dumps(res["cohort_train_coverage_hard"]))
    # (b) uniform test sample
    tsamp = np.sort(np.random.default_rng(7).choice(len(rows["test"]), min(a.n_test, len(rows["test"])), replace=False))
    trows = rows["test"][tsamp]
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs = [x for part in pool.map(_chunk, [(c, True) for c in np.array_split(trows, a.workers * 8)], chunksize=1) for x in part]
    gold = D.sro[trows, 2].astype(np.int64)
    pool_absent = gf[trows] > KC
    res["test_uniform_sample"] = dict(n=int(len(trows)), bridge=summarize(recs, "bridge"), pattern_union=summarize(recs, "pattern"),
                                      bridge_pool_absent=summarize([r for r, m in zip(recs, pool_absent) if m], "bridge"),
                                      pattern_pool_absent=summarize([r for r, m in zip(recs, pool_absent) if m], "pattern"))
    ld = f"{WORK}/lists_{a.base}"
    Lb = np.load(f"{ld}/test_Lbr.npy", mmap_mode="r")[tsamp].astype(np.int64)
    Lp = np.load(f"{ld}/test_Lpat.npy", mmap_mode="r")[tsamp].astype(np.int64)
    res["test_uniform_sample"]["bridge_stage2_list"] = list_controls(Lb, gold, dbin, nbin, rpop, N, (10, 50, 200))
    res["test_uniform_sample"]["pattern_prior_list"] = list_controls(Lp, gold, dbin, nbin, rpop, N, (10, 50, 300))
    res["test_uniform_sample"]["bridge_stage2_list_pool_absent"] = list_controls(Lb[pool_absent], gold[pool_absent], dbin, nbin, rpop, N, (10, 50, 200))
    res["test_uniform_sample"]["pattern_prior_list_pool_absent"] = list_controls(Lp[pool_absent], gold[pool_absent], dbin, nbin, rpop, N, (10, 50, 300))
    log("test", json.dumps(res["test_uniform_sample"]["bridge"]), json.dumps(res["test_uniform_sample"]["pattern_union"]))
    # dump vs lookup rank agreement
    lk = f"{WORK}/lookup_{a.base}"
    agr = {}
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    for sp, idx in (("valid", D.rows("valid")[vidx]), ("test", rows["test"])):
        if os.path.exists(f"{lk}/{sp}.npz"):
            L = np.load(f"{lk}/{sp}.npz")
            d_ = gf[idx].astype(np.int64); l_ = L["rank_filt"].astype(np.int64)
            diff = np.abs(d_ - l_)
            agr[sp] = dict(n=int(len(d_)), frac_equal=round(float((diff == 0).mean()), 6), max_abs_rank_diff=int(diff.max()),
                           n_differ=int((diff > 0).sum()), mean_abs_diff_when_differ=round(float(diff[diff > 0].mean()), 3) if (diff > 0).any() else 0.0,
                           frac_differ_with_rank_le_50=round(float(((diff > 0) & (d_ <= 50)).mean()), 6),
                           max_abs_gold_logprob_diff=float(np.abs(D.gold_score[idx] - L["gold_lp"]).max()),
                           MRR_dump=round(float((1 / d_).mean()), 6), MRR_lookup=round(float((1 / l_).mean()), 6))
    res["dump_vs_lookup_rank_agreement"] = agr
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"density_{a.base}.json"); jdump(res, f"{WORK}/density_{a.base}.json")
    log("ALL DONE", json.dumps(agr))


if __name__ == "__main__":
    main()
