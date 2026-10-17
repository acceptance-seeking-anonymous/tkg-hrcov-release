#!/usr/bin/env python3
"""pilot_polecat / pc_prep.py -- BASE-INDEPENDENT preparation (CPU), run while the bases train.

  1. valid sample (VALID_N uniform rows of the valid split, rng 12345)        -> work/valid_sample_idx.npy
  2. RecB-lite per-query ranks (datasets-track profile_tkg_v2, patched only to save ranks; lambda on valid)
                                                                               -> work/recb_{valid,test}_rank.npy (split order)
  3. 6.1 pattern target-encoding tables on the EARLY 70% of train timestamps (retriever stageA recipe)
                                                                               -> work/te_POLECAT.npz
  4. gold-dependent per-query strata (EVAL ONLY) for the valid sample and every test query:
     n events of s / o before t (first-seen), dyadic & exact-triple online history, capped pattern reach
     (any / 1-hop; M1=64, M2=32), uncapped 2-hop common-neighbour reach, self-loop flag
                                                                               -> work/strata_{valid,test}.npz
  python3 pc_prep.py --workers 24
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (H, TRACKS, P, DD, DS, WORK, VALID_N, Splits, split_queries, profiler_to_split_order, jdump, log_fn)
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
        d = instantiate(EI, s, t, M1, M2)
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
    ap.add_argument("--skip_recb", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    os.makedirs(WORK, exist_ok=True)
    S = Splits(DS)
    summ = dict(host="serverA (CPU, tkg-gpu:latest container)", workers=a.workers)
    Qv = split_queries(S, "valid"); Qt = split_queries(S, "test"); Qtr = split_queries(S, "train")
    summ["n_rows"] = dict(train=len(Qtr), valid=len(Qv), test=len(Qt))
    rng = np.random.default_rng(12345)
    vidx = np.sort(rng.choice(len(Qv), min(VALID_N, len(Qv)), replace=False))
    if a.smoke:
        vidx = vidx[:a.smoke]; Qt = Qt[np.linspace(0, len(Qt) - 1, a.smoke).astype(np.int64)]
    np.save(f"{WORK}/valid_sample_idx.npy", vidx)
    log("rows", summ["n_rows"], "valid sample", len(vidx))
    # ---------------- RecB-lite (patched copy of the datasets-track profiler; only adds np.save of ranks)
    if not a.skip_recb:
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "profile_tkg_v2.py")).read()   # src/datasets
        anchor = "        G.clear()\n    out[\"profile_seconds_total\"]"
        assert anchor in src
        patched = src.replace(anchor, "        np.save(os.environ['RECB_PREFIX'] + '_test_rank_prof.npy', test_rank)\n"
                                      "        np.save(os.environ['RECB_PREFIX'] + '_val_rank_prof.npy', val_rank)\n" + anchor)
        open(f"{WORK}/profile_tkg_v2_saveranks.py", "w").write(patched)
        env = dict(os.environ, RECB_PREFIX=f"{WORK}/recb")
        t1 = time.time()
        subprocess.run([sys.executable, f"{WORK}/profile_tkg_v2_saveranks.py", "--name", "tkgl-polecat", "--dir", DD,
                        "--out", f"{WORK}/recb_profile.json", "--nproc", str(min(16, a.workers))], check=True, env=env)
        prof = json.load(open(f"{WORK}/recb_profile.json"))
        rb = prof["recb_lite"]; j = rb["lambda_grid"].index(rb["lambda_selected"])
        tr_ = np.load(f"{WORK}/recb_test_rank_prof.npy")[:, j]; vr_ = np.load(f"{WORK}/recb_val_rank_prof.npy")[:, j]
        np.save(f"{WORK}/recb_test_rank.npy", tr_[profiler_to_split_order(S, "test")])
        np.save(f"{WORK}/recb_valid_rank.npy", vr_[profiler_to_split_order(S, "valid")])
        summ["recb_lite"] = dict(lambda_selected=rb["lambda_selected"], valid=rb["valid"], test=rb["test"],
                                 valid_MRR_by_lambda=rb["valid_MRR_by_lambda"], seconds=round(time.time() - t1, 1),
                                 stratum_shares=prof.get("stratum_shares"))
        log("recb", json.dumps(summ["recb_lite"]["test"]))
    # ---------------- event index (all splits, both directions)
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, S.n_ent, S.R, 1)
    G.update(EI=EI, Qtr=Qtr)
    # ---------------- TE tables (early 70% of train timestamps)
    t1 = time.time()
    uts = np.unique(Qtr[:, 3]); cut = uts[int(0.7 * len(uts))]
    early = np.flatnonzero(Qtr[:, 3] < cut)
    te_rows = np.sort(np.random.default_rng(0).choice(early, min(a.te_n, len(early)), replace=False))
    chunks = np.array_split(te_rows, a.workers * 8)
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(_te_chunk, chunks, chunksize=1)
    arrs = {}
    for lvl in ("full", "rr", "r2", "p"):
        uk, hh, nn = _merge([p[0][lvl] for p in parts])
        arrs[f"{lvl}_k"] = uk; arrs[f"{lvl}_h"] = hh; arrs[f"{lvl}_n"] = nn
    hop_h = sum(p[1] for p in parts); hop_n = sum(p[2] for p in parts)
    arrs["prior"] = hop_h / np.maximum(hop_n, 1)
    np.savez(f"{WORK}/te_POLECAT.npz", **arrs)
    summ["te"] = dict(n_queries=int(len(te_rows)), train_ts_cut=int(cut), prior_hit_rate_by_hop=arrs["prior"].tolist(),
                      n_keys={lvl: int(len(arrs[f"{lvl}_k"])) for lvl in ("full", "rr", "r2", "p")}, seconds=round(time.time() - t1, 1))
    log("te", json.dumps(summ["te"]))
    # ---------------- strata
    for split, Q in (("valid", Qv[vidx]), ("test", Qt)):
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
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, "prep_summary.json"); jdump(summ, f"{WORK}/prep_summary.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
