#!/usr/bin/env python3
"""phase2_fresh_window / fh_prep.py (derived from presub/g2_prep.py) -- BASE-INDEPENDENT preparation for ICEWS18 / GDELT (CPU); pilot_polecat/pc_prep.py with:
  * dataset from g2_common (env G2_DS); pattern instantiation and event-index lookups at t_snap = ts // step
    (retriever-track convention for the RE-GCN-split datasets; polecat has step 1, so the code is otherwise unchanged)
  * no RecB-lite stage (G2 RecB = hd_recb.py, TGB-faithful re-implementation)
  1. valid sample (VALID_N uniform rows of the valid split, rng 12345; ICEWS18 -> full valid)   -> work/valid_sample_idx.npy
  2. pattern TE tables on the EARLY 70% of train timestamps (pc_prep recipe)                     -> work/te_<DS>.npz
  3. strata (EVAL ONLY) for the valid sample and every test query (pc_prep definitions)           -> work/strata_{valid,test}.npz
  python3 fh_prep.py --workers 8 --stage 1      # valid sample, TE tables, VALID strata (no test-set statistic)
  python3 fh_prep.py --workers 8 --stage 2      # TEST strata only (frozen TE table and valid sample untouched)
Forward-holdout changes vs g2_prep.py: dataset via fh_common (GDELTFW); --stage split so that nothing about the test
split is computed before the freeze; summary file names carry the stage.  All computations are otherwise verbatim.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fh_common import (H, P, DD, DS, WORK, VALID_N, STEP, Splits, split_queries, jdump, log_fn, module_provenance)
from patterns import EventIndex, instantiate, decode
from rt_common import te_keys, M1, M2
import multiprocessing as mp

G = {}


def _strata_chunk(args):
    lo, hi = args
    EI = G["EI"]; Q = G["Q"]; R = EI.R
    n = hi - lo
    out = {k: np.zeros(n, dt) for k, dt in (("nev_s", np.int32), ("nev_o", np.int32), ("dyad", bool), ("trip", bool),
                                              ("pat_any", bool), ("pat_hop1", bool), ("two_hop", bool), ("n_pat", np.int32),
                                              ("n_union", np.int32), ("nb_s", np.int32))}
    for k in range(n):
        s, r, o, t = (int(x) for x in Q[lo + k])
        t = t // STEP                                       # event-index time unit
        a, b = EI.indptr[s], EI.indptr[s + 1]
        ps = a + np.searchsorted(EI.ts[a:b], t, "left")
        a2, b2 = EI.indptr[o], EI.indptr[o + 1]
        po = a2 + np.searchsorted(EI.ts[a2:b2], t, "left")
        out["nev_s"][k] = ps - a; out["nev_o"][k] = po - a2
        ds_ = EI.dst[a:ps]; rs_ = EI.rel[a:ps]
        out["dyad"][k] = bool((ds_ == o).any())
        out["trip"][k] = bool(((ds_ == o) & (rs_ == r)).any())
        ns = np.unique(ds_); no = np.unique(EI.dst[a2:po])
        out["nb_s"][k] = len(ns)
        if len(ns) and len(no):
            com = np.intersect1d(ns, no, assume_unique=True)
            com = com[(com != s) & (com != o)]
            out["two_hop"][k] = len(com) > 0
        d = instantiate(EI, s, t, M1, M2)
        if d is not None:
            hit = d["pc_c"] == o
            out["pat_any"][k] = bool(hit.any())
            if hit.any():
                out["pat_hop1"][k] = bool((d["hop"][d["pc_p"][hit]] == 1).any())
            out["n_pat"][k] = len(d["pid"]); out["n_union"][k] = len(np.unique(d["pc_c"]))
    return lo, out


def _te_chunk(rows):
    EI = G["EI"]; Q = G["Qtr"]
    keys = {l: [] for l in ("full", "rr", "r2", "p")}; hits = {l: [] for l in keys}
    hop_h = np.zeros(2); hop_n = np.zeros(2)
    for i in rows:
        s, r, o, t = (int(x) for x in Q[i])
        d = instantiate(EI, s, t // STEP, M1, M2)
        if d is None:
            continue
        pid = d["pid"]; Pn = len(pid)
        hit = np.bincount(d["pc_p"][d["pc_c"] == o], minlength=Pn) > 0
        hop, rel1, lb1, rel2, lb2 = decode(pid, EI.R)
        for lvl, k in zip(("full", "rr", "r2", "p"), te_keys(r, pid, hop, rel1, rel2, EI.R)):
            keys[lvl].append(k); hits[lvl].append(hit)
        for h in (1, 2):
            m = hop == h; hop_h[h - 1] += hit[m].sum(); hop_n[h - 1] += m.sum()
    out = {}
    for lvl in keys:
        if not keys[lvl]:
            out[lvl] = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)); continue
        k = np.concatenate(keys[lvl]); h = np.concatenate(hits[lvl]).astype(np.float64)
        uk, inv = np.unique(k, return_inverse=True)
        out[lvl] = (uk, np.bincount(inv, weights=h), np.bincount(inv).astype(np.float64))
    return out, hop_h, hop_n


def _merge(parts):
    k = np.concatenate([p[0] for p in parts]); h = np.concatenate([p[1] for p in parts]); n = np.concatenate([p[2] for p in parts])
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=h), np.bincount(inv, weights=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--te_n", type=int, default=400000)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    os.makedirs(WORK, exist_ok=True)
    S = Splits(DS)
    summ = dict(ds=DS, step=STEP, stage=a.stage, host=os.environ.get("HD_HOST", "serverA (CPU, tkg-gpu:latest container)"), workers=a.workers)
    Qv = split_queries(S, "valid"); Qt = split_queries(S, "test"); Qtr = split_queries(S, "train")
    summ["n_rows"] = dict(train=len(Qtr), valid=len(Qv), test=len(Qt))
    rng = np.random.default_rng(12345)
    vidx = np.sort(rng.choice(len(Qv), min(VALID_N, len(Qv)), replace=False))
    if a.smoke:
        vidx = vidx[:a.smoke]; Qt = Qt[:a.smoke]          # first rows (fh_lists / fh_features smoke use the same rows)
    if a.stage == 1:
        np.save(f"{WORK}/valid_sample_idx.npy", vidx)
    else:
        assert (np.load(f"{WORK}/valid_sample_idx.npy") == vidx).all(), "valid sample differs from the frozen one"
    log("rows", summ["n_rows"], "valid sample", len(vidx))
    # ---------------- event index (all splits, both directions)
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, S.n_ent, S.R, STEP)
    G.update(EI=EI, Qtr=Qtr)
    # ---------------- TE tables (early 70% of train timestamps) -- Stage 1 only (frozen afterwards)
    if a.stage == 1:
        t1 = time.time()
        uts = np.unique(Qtr[:, 3]); cut = uts[int(0.7 * len(uts))]
        early = np.flatnonzero(Qtr[:, 3] < cut)
        te_rows = np.sort(np.random.default_rng(0).choice(early, min(a.te_n if not a.smoke else 3000, len(early)), replace=False))
        chunks = np.array_split(te_rows, a.workers * 8)
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_te_chunk, chunks, chunksize=1)
        arrs = {}
        for lvl in ("full", "rr", "r2", "p"):
            uk, hh, nn = _merge([p[0][lvl] for p in parts])
            arrs[f"{lvl}_k"] = uk; arrs[f"{lvl}_h"] = hh; arrs[f"{lvl}_n"] = nn
        hop_h = sum(p[1] for p in parts); hop_n = sum(p[2] for p in parts)
        arrs["prior"] = hop_h / np.maximum(hop_n, 1)
        np.savez(f"{WORK}/te_{DS}.npz", **arrs)
        summ["te"] = dict(n_queries=int(len(te_rows)), n_early_rows=int(len(early)), train_ts_cut=int(cut), prior_hit_rate_by_hop=arrs["prior"].tolist(),
                          n_keys={lvl: int(len(arrs[f"{lvl}_k"])) for lvl in ("full", "rr", "r2", "p")}, seconds=round(time.time() - t1, 1))
        log("te", json.dumps(summ["te"]))
    # ---------------- strata (EVAL ONLY): Stage 1 valid, Stage 2 test
    for split, Q in ((("valid", Qv[vidx]),) if a.stage == 1 else (("test", Qt),)):
        t1 = time.time()
        G["Q"] = Q
        n = len(Q); step = 2000
        tasks = [(lo, min(n, lo + step)) for lo in range(0, n, step)]
        res = {}
        with mp.get_context("fork").Pool(a.workers) as pool:
            for lo, o in pool.imap_unordered(_strata_chunk, tasks, chunksize=1):
                res[lo] = o
        keys = list(res[0].keys())
        st = {k: np.concatenate([res[lo][k] for lo, _ in tasks]) for k in keys}
        st["sro"] = Q; st["self_loop"] = Q[:, 0] == Q[:, 2]
        st["first_seen"] = (st["nev_s"] == 0) | (st["nev_o"] == 0)
        np.savez(f"{WORK}/strata_{split}.npz", **st)
        summ[f"strata_{split}"] = dict(n=n, seconds=round(time.time() - t1, 1),
                                       shares={k: float(np.mean(st[k])) for k in ("dyad", "trip", "pat_any", "pat_hop1", "two_hop", "self_loop", "first_seen")},
                                       no_pair_and_2hop=float(np.mean(~st["dyad"] & st["two_hop"])),
                                       no_pair_and_unreach=float(np.mean(~st["dyad"] & ~st["two_hop"])),
                                       median_n_pat=float(np.median(st["n_pat"])), median_n_union=float(np.median(st["n_union"])))
        log(split, json.dumps(summ[f"strata_{split}"]))
    summ["elapsed_s"] = round(time.time() - t0, 1); summ["modules"] = module_provenance()
    jdump(summ, f"prep_summary_stage{a.stage}.json"); jdump(summ, f"{WORK}/prep_summary_stage{a.stage}.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
