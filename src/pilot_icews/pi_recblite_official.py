#!/usr/bin/env python3
"""pilot_icews / pi_recblite_official.py -- full-history RecB-lite on the test sample scored by py-tgb's Evaluator (CPU).

RecB-lite (datasets-track profile_tkg_v2): score(c) = sum_k lam^(t-k) over earlier (s, r', c, k) (FULL history, every
split before t), seen candidates above unseen, ties and unseen candidates broken by the static TRAIN answer frequency
xi(r', c), exact ties 1/2; lam = the value selected on full valid by pi_prep.  For each sampled test query this script
builds a full 87,856-entity score vector that realises exactly that lexicographic order:
   unseen c : y = xi(r', c)                      seen c : y = 1 + max xi + dense rank of (score, xi) among seen
masks the TGB time-filtered conflict set (shipped *_test_ns.pkl) to -inf and calls tgb Evaluator.eval on batches.
Per-query ranks (evaluator formula) are checked against the profiler's harness ranks (recb_test_sample_rank.npy).
  python3 pi_recblite_official.py --workers 12
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import (DS, WORK, P, Splits, split_queries, jdump, log_fn)
from pi_replay import load_official
import multiprocessing as mp

G = {}
BIG = np.int64(1 << 17)


def _work(rng_):
    lo, hi = rng_
    Q = G["Q"]; K = G["K"]; KO = G["KO"]; KT = G["KT"]; XI = G["XI"]; lam = G["lam"]; es = G["es"]; day2raw = G["day2raw"]
    ev = G["ev"]; N = XI.shape[1]; R2 = XI.shape[0]; off = float(XI.max()) + 1.0
    n = hi - lo
    rk = np.zeros(n); bm = 0.0
    B = 128
    for b0 in range(lo, hi, B):
        b1 = min(hi, b0 + B); nb = b1 - b0
        Y = np.empty((nb, N), np.float64)
        yp = np.empty(nb)
        cr, cc = [], []
        for j in range(nb):
            s, r, o, t = (int(x) for x in Q[b0 + j])
            key = s * R2 + r
            a = np.searchsorted(K, key * BIG, "left"); e = np.searchsorted(K, key * BIG + t, "left")
            y = XI[r].astype(np.float64)
            if e > a:
                objs = KO[a:e]; tk = KT[a:e]
                u, inv = np.unique(objs, return_inverse=True)
                sc = np.bincount(inv, weights=lam ** (t - tk).astype(np.float64), minlength=len(u))
                xi = XI[r, u].astype(np.float64)
                order = np.lexsort((xi, sc))
                ss, xx = sc[order], xi[order]
                newv = np.ones(len(u), bool); newv[1:] = (ss[1:] != ss[:-1]) | (xx[1:] != xx[:-1])
                dr = np.cumsum(newv).astype(np.float64)
                yv = np.empty(len(u)); yv[order] = off + dr
                y[u] = yv
            Y[j] = y; yp[j] = y[o]
            conf = np.asarray(es[(day2raw[t], s, r)], np.int64)
            cr.append(np.full(len(conf), j)); cc.append(conf)
        cr = np.concatenate(cr); cc = np.concatenate(cc)
        Y[cr, cc] = -np.inf
        m = ev.eval({"y_pred_pos": yp, "y_pred_neg": Y, "eval_metric": ["mrr"]})
        rkb = 1 + (Y > yp[:, None]).sum(1) + 0.5 * (Y == yp[:, None]).sum(1)
        rk[b0 - lo:b1 - lo] = rkb; bm += float(m["mrr"]) * nb
    return lo, rk, bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N, R = S.n_ent, S.R; R2 = 2 * R
    prep = json.load(open(f"{WORK}/prep_summary.json")); lam = float(prep["recb_lite"]["lambda_selected"])
    tidx = np.load(f"{WORK}/test_sample_idx.npy")
    Q = split_queries(S, "test")[tidx]
    q = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]).astype(np.int64)
    s = np.concatenate([q[:, 0], q[:, 2]]); r = np.concatenate([q[:, 1], q[:, 1] + R]); o = np.concatenate([q[:, 2], q[:, 0]])
    ts = np.concatenate([q[:, 3], q[:, 3]])
    c_ = (s * R2 + r) * BIG + ts; od = np.argsort(c_, kind="stable")
    K = c_[od]; KO = o[od].astype(np.int64); KT = ts[od].astype(np.int64)
    tr = S.q["train"].astype(np.int64)
    rr = np.concatenate([tr[:, 1], tr[:, 1] + R]); aa = np.concatenate([tr[:, 2], tr[:, 0]])
    XI = np.bincount(rr * N + aa, minlength=R2 * N).reshape(R2, N).astype(np.int32)
    log("index ready", len(K), "xi max", int(XI.max()))
    sampler, ev, ver = load_official()
    day2raw = {}
    for ln in open(f"{S.cfg['dd']}/timestamp2id.txt"):
        k, v = ln.split(); day2raw[int(v)] = int(k)
    G.update(Q=Q, K=K, KO=KO, KT=KT, XI=XI, lam=lam, es=sampler.eval_set["test"], day2raw=day2raw, ev=ev)
    n = len(Q); tasks = [(lo, min(n, lo + 4000)) for lo in range(0, n, 4000)]
    rk = np.zeros(n); bm = 0.0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (lo, r_, b_) in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            rk[lo:lo + len(r_)] = r_; bm += b_
            if k % 20 == 0:
                log(k, "/", len(tasks))
    prof = np.load(f"{WORK}/recb_test_sample_rank.npy").astype(np.float64)
    res = dict(host="serverA (CPU, tkg-gpu:latest)", py_tgb=ver, lam=lam, n=int(n),
               official=dict(MRR=float(np.mean(1 / rk)), H10=float(np.mean(rk <= 10)), evaluator_batchmean_MRR=bm / n),
               harness_profiler=dict(MRR=float(np.mean(1 / prof))), frac_rank_equal=float(np.mean(np.abs(rk - prof) < 1e-6)),
               max_abs_rank_diff=float(np.abs(rk - prof).max()), elapsed_s=round(time.time() - t0, 1))
    np.save(f"{WORK}/recblite_official_test_sample_rank.npy", rk); np.save("recblite_official_test_sample_rank.npy", rk)
    jdump(res, "recblite_official.json"); jdump(res, f"{WORK}/recblite_official.json")
    log("ALL DONE", json.dumps({k: res[k] for k in ("frac_rank_equal", "max_abs_rank_diff")}))


if __name__ == "__main__":
    main()
