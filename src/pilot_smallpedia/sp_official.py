#!/usr/bin/env python3
"""pilot_smallpedia / sp_official.py -- official py-tgb evaluation (CPU) of a base and of re-ranked systems.

Negatives: py-tgb TKGNegativeEdgeSampler.query_batch on the shipped tkgl-smallpedia_test_ns.pkl (time-filtered; the
verified entity/relation maps are identities).  Scores: base = full log-prob row; system = base row with its
promoted candidate list (final fused order) lifted above everything in that order (pc_lookup --mode systems
convention).  Each query is scored by tgb.linkproppred.evaluate.Evaluator.eval (MRR, hits@10).
  python3 sp_official.py --base regcn --systems HR_s0=<npy>,HRCOV_s0=<npy>
"""
import os, sys, json, time, argparse, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import (P, DD, DS, WORK, ZIP, N_ENT, TGB_NAME, Splits, split_queries, jdump, log_fn, fullrows, MODE_SFX)
import multiprocessing as mp

G = {}


def _work(args):
    lo, hi = args
    sro = G["sro"]; LP = G["LP"]; systems = G["systems"]; sampler = G["sampler"]; ev = G["ev"]; tsraw = G["tsraw"]
    res = {nm: (np.zeros(hi - lo, np.float32), np.zeros(hi - lo, np.float32), np.zeros(hi - lo, np.float32)) for nm in systems}
    row = np.asarray(LP[lo:hi])
    s, r, o, t = sro[lo:hi, 0], sro[lo:hi, 1], sro[lo:hi, 2], sro[lo:hi, 3]
    negs = sampler.query_batch(s, o, np.array([tsraw[int(x)] for x in t]), r, split_mode="test")
    for nm, arr in systems.items():
        Y = row
        if arr is not None:
            Y = row.copy()
            Cs = np.asarray(arr[lo:hi]).astype(np.int64)
            for qq in range(hi - lo):
                cc = Cs[qq][Cs[qq] >= 0]
                Y[qq, cc] = 1e6 - np.arange(len(cc), dtype=np.float32)
        rr, h10, rk = res[nm]
        for qq in range(hi - lo):
            yp = np.array([Y[qq, o[qq]]]); yn = Y[qq, negs[qq]][None, :]
            m = ev.eval({"y_pred_pos": yp, "y_pred_neg": yn, "eval_metric": ["mrr"]})
            rr[qq] = m["mrr"]; h10[qq] = m["hits@10"]; rk[qq] = 1 + (yn > yp).sum() + 0.5 * (yn == yp).sum()
    return lo, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--systems", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
    from tgb.linkproppred.evaluate import Evaluator
    import importlib.metadata as im
    S = Splits(DS)
    sro = split_queries(S, "test")
    n = len(sro) if not a.smoke else a.smoke
    tsraw = {}
    for ln in open(f"{DD}/timestamp2id.txt"):
        k, v = ln.split(); tsraw[int(v)] = int(k)
    nsdir = f"{WORK}/tgb_ns"; os.makedirs(nsdir, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        if not os.path.exists(f"{nsdir}/{TGB_NAME}_test_ns.pkl"):
            z.extract(f"{TGB_NAME}_test_ns.pkl", nsdir)
    sampler = TKGNegativeEdgeSampler(TGB_NAME, first_dst_id=0, last_dst_id=N_ENT - 1)
    sampler.load_eval_set(f"{nsdir}/{TGB_NAME}_test_ns.pkl", split_mode="test")
    systems = {"base": None}
    if a.systems:
        systems = {}
        for kv in a.systems.split(","):
            nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r")
    G.update(sro=sro, LP=fullrows(a.base, "test"), systems=systems, sampler=sampler, ev=Evaluator(name=TGB_NAME), tsraw=tsraw)
    tasks = [(lo, min(n, lo + 500)) for lo in range(0, n, 500)]
    out = {nm: [np.zeros(n, np.float32) for _ in range(3)] for nm in systems}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (lo, res) in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            for nm, (rr, h10, rk) in res.items():
                hi = lo + len(rr)
                out[nm][0][lo:hi] = rr; out[nm][1][lo:hi] = h10; out[nm][2][lo:hi] = rk
            if k % 50 == 0:
                log(k, "/", len(tasks))
    res = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)", py_tgb=im.version("py-tgb"), n=int(n),
               official={nm: dict(MRR=float(v[0].mean()), hits10=float(v[1].mean()), n=int(n)) for nm, v in out.items()},
               elapsed_s=round(time.time() - t0, 1))
    tag = ("base" if not a.systems else "systems") + MODE_SFX
    np.savez(f"official_{tag}_{a.base}_ranks.npz", **{nm: v[2] for nm, v in out.items()})
    jdump(res, f"official_{tag}_{a.base}.json")
    log("ALL DONE")


if __name__ == "__main__":
    main()
