#!/usr/bin/env python3
"""pilot_icews / pi_prep.py -- BASE-INDEPENDENT preparation for Stage A (CPU), run while the bases train.

  1. samples: valid (200,000 uniform, rng 12345) and test (10% uniform, rng 31415) query indices in split order
  2. RecB-lite per-query ranks for FULL valid and FULL test (datasets-track profile_tkg_v2, patched only to save ranks;
     lambda selected on full valid) -> work/recb_{valid,test}_rank.npy (split order) + sample subsets
  3. 6.1 pattern target-encoding tables on 400,000 queries of the EARLY 70% of train days (retriever stageA recipe,
     icews-safe keys) -> work/te_TKGL_ICEWS.npz   (shared by both relation schemes: relmap is identity on train relations)
  4. per-query strata (EVAL ONLY) for the valid and test samples from the full online history (residual-track Hist):
     exact recurrence, pair history, n events of s / o before t, first-seen, self-loop, relation unseen in train,
     relation is a merged (zero-stripped) code, gold full-train frequency stratum, harness hist=no
  python3 pi_prep.py --workers 22
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import (H, TRACKS, P, DD, DS, WORK, Splits, split_queries, valid_sample_idx, test_sample_idx, jdump, log_fn,
                       te_keys_icews, N_REL)
from patterns import EventIndex, instantiate, decode
from rt_common import M1, M2
from rs_features import Hist
import multiprocessing as mp

G = {}


def profiler_to_split_order(S, split):
    q = S.q[split]; n = len(q)
    assert (np.diff(q[:, 3]) >= 0).all(), "split file not time-sorted"
    ts = q[:, 3]; b = np.flatnonzero(np.diff(ts)) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [n]])
    perm = []
    for a, e in zip(st, en):
        idx = np.arange(a, e); perm.append(idx); perm.append(idx + n)
    return np.concatenate(perm)


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
        for lvl, k in zip(("full", "rr", "r2", "p"), te_keys_icews(r, pid, hop, rel1, rel2, EI.R)):
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
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--te_n", type=int, default=400000)
    ap.add_argument("--skip_recb", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    os.makedirs(WORK, exist_ok=True)
    S = Splits(DS); N, R = S.n_ent, S.R
    summ = dict(host="serverA (CPU, tkg-gpu:latest container)", workers=a.workers)
    nv, nt = 2 * len(S.q["valid"]), 2 * len(S.q["test"])
    vidx = valid_sample_idx(nv); tidx = test_sample_idx(nt)
    np.save(f"{WORK}/valid_sample_idx.npy", vidx); np.save(f"{WORK}/test_sample_idx.npy", tidx)
    Qv = split_queries(S, "valid"); Qt = split_queries(S, "test")
    summ["n_queries"] = dict(valid=int(nv), test=int(nt), valid_sample=int(len(vidx)), test_sample=int(len(tidx)))
    log("queries", summ["n_queries"])
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
        subprocess.run([sys.executable, f"{WORK}/profile_tkg_v2_saveranks.py", "--name", "tkgl-icews", "--dir", DD,
                        "--out", f"{WORK}/recb_profile.json", "--nproc", str(min(16, a.workers))], check=True, env=env)
        prof = json.load(open(f"{WORK}/recb_profile.json"))
        rb = prof["recb_lite"]; j = rb["lambda_grid"].index(rb["lambda_selected"])
        trk = np.load(f"{WORK}/recb_test_rank_prof.npy"); vrk = np.load(f"{WORK}/recb_val_rank_prof.npy")
        tr_ = trk[:, j][profiler_to_split_order(S, "test")]; vr_ = vrk[:, j][profiler_to_split_order(S, "valid")]
        np.save(f"{WORK}/recb_test_rank.npy", tr_); np.save(f"{WORK}/recb_valid_rank.npy", vr_)
        np.save(f"{WORK}/recb_test_rank_allLam.npy", trk[profiler_to_split_order(S, "test")])
        np.save(f"{WORK}/recb_test_sample_rank.npy", tr_[tidx]); np.save(f"{WORK}/recb_valid_sample_rank.npy", vr_[vidx])
        summ["recb_lite"] = dict(lambda_selected=rb["lambda_selected"], lambda_grid=rb["lambda_grid"], valid_full=rb["valid"],
                                 valid_MRR_by_lambda=rb["valid_MRR_by_lambda"], seconds=round(time.time() - t1, 1),
                                 note="test-side numbers are not printed here (read only at aggregation)")
        log("recb lambda", rb["lambda_selected"], "valid", rb["valid"]["MRR"])
    # ---------------- event index (all splits, both directions) + TE tables (early 70% of train days)
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, N, R, 1)
    Qtr = split_queries(S, "train")
    G.update(EI=EI, Qtr=Qtr)
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
    np.savez(f"{WORK}/te_TKGL_ICEWS.npz", **arrs)
    summ["te"] = dict(n_queries=int(len(te_rows)), train_day_cut=int(cut), prior_hit_rate_by_hop=arrs["prior"].tolist(),
                      n_keys={lvl: int(len(arrs[f"{lvl}_k"])) for lvl in ("full", "rr", "r2", "p")}, seconds=round(time.time() - t1, 1),
                      key_function="pi_common.te_keys_icews (multiplier 4e7)")
    log("te", json.dumps(summ["te"]))
    del EI; G.clear()
    # ---------------- strata (full online history)
    relmap = np.load(f"{P}/stage/relmap.npy")
    tr_rels = np.unique(S.q["train"][:, 1])
    Hh = Hist(S)
    for split, Q in (("valid", Qv[vidx]), ("test", Qt[tidx])):
        t1 = time.time()
        s, r, o, t = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
        st = {}
        for lo in range(0, len(Q), 200000):
            hi = min(len(Q), lo + 200000)
            hf = Hh.feats(s[lo:hi], r[lo:hi], t[lo:hi], o[lo:hi, None])
            for k in ("ntrip", "ndyad", "nent", "nquery", "nsubj"):
                st.setdefault(k, []).append(np.asarray(hf[k]).reshape(hi - lo, -1)[:, 0])
        st = {k: np.concatenate(v).astype(np.int64) for k, v in st.items()}
        out = dict(sro=Q, trip=st["ntrip"] > 0, dyad=st["ndyad"] > 0, nev_s=st["nsubj"], nev_o=st["nent"], n_sr=st["nquery"])
        out["first_seen"] = (out["nev_s"] == 0) | (out["nev_o"] == 0)
        out["self_loop"] = s == o
        rb_ = r % R
        out["rel_unseen_train"] = ~np.isin(rb_, tr_rels)
        out["rel_merged_code"] = relmap[rb_] != rb_
        out["freq_stratum"] = S.stratum(S.freq[o]); out["in_hist"] = S.in_hist(s, o)
        np.savez(f"{WORK}/strata_{split}.npz", **out)
        summ[f"strata_{split}"] = dict(n=int(len(Q)), seconds=round(time.time() - t1, 1),
                                       shares={k: float(np.mean(out[k])) for k in ("trip", "dyad", "first_seen", "self_loop", "rel_unseen_train",
                                                                                   "rel_merged_code", "in_hist")},
                                       head_share=float(np.mean(out["freq_stratum"] == 2)), cold_share=float(np.mean(out["freq_stratum"] == 0)),
                                       median_n_sr=float(np.median(out["n_sr"])))
        log(split, json.dumps(summ[f"strata_{split}"]))
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, "prep_summary.json"); jdump(summ, f"{WORK}/prep_summary.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
