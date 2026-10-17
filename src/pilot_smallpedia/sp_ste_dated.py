#!/usr/bin/env python3
"""fusion_followups step 3 / sp_ste_dated.py -- refit the static path-type precision table (sp_static.StaticTE) under
the DATED static graph: same 100,000 early-train queries as pilot-2 sp_prep (ts < 70% train quantile, rng 1), but every
static hop of every path only uses edges with valid_from < t of the fitting query.  -> work/ste_SMALLPEDIA_dt.npz
  SP_STATIC_MODE=dated python3 sp_ste_dated.py --workers 8"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import (DS, WORK, N_ENT, ST_DEG_CAP, Splits, split_queries, jdump, log_fn, load_static, DATED)
from patterns import EventIndex
import sp_static as SS
import multiprocessing as mp
G = {}


def _ste_chunk(rows):
    EI = G["EI"]; Q = G["Qtr"]; SI = G["SI"]; R2 = 2 * EI.R
    parts = []; n_reach = 0
    for i in rows:
        s, r, o, t = (int(x) for x in Q[i])
        pth = SS.enumerate_paths(SI, EI, s, t)
        f = SS.fit_counts(pth, o, r, R2, SI.D1)
        if f is not None:
            parts.append(f); n_reach += int((pth["cand"] == o).any())
    if not parts:
        return None, 0, len(rows)
    return {lvl: SS.merge_counts(parts, lvl) for lvl in ("full", "type", "shape")}, n_reach, len(rows)


def _merge(parts):
    k = np.concatenate([p[0] for p in parts]); h = np.concatenate([p[1] for p in parts]); n = np.concatenate([p[2] for p in parts])
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=h), np.bincount(inv, weights=n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--ste_n", type=int, default=100000)
    a = ap.parse_args(); t0 = time.time(); log = log_fn(t0)
    assert DATED, "run with SP_STATIC_MODE=dated"
    S = Splits(DS); Qtr = split_queries(S, "train")
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, S.n_ent, S.R, 1)
    SI = SS.StaticIndex(load_static(), N_ENT, ST_DEG_CAP); assert SI.dated
    uts = np.unique(Qtr[:, 3]); cut = uts[int(0.7 * len(uts))]
    early = np.flatnonzero(Qtr[:, 3] < cut)
    G.update(EI=EI, Qtr=Qtr, SI=SI)
    ste_rows = np.sort(np.random.default_rng(1).choice(early, min(a.ste_n, len(early)), replace=False))
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(_ste_chunk, np.array_split(ste_rows, a.workers * 8), chunksize=1)
    sarr = {}
    for lvl in ("full", "type", "shape"):
        uk, hh, zz = _merge([p[0][lvl] for p in parts if p[0] is not None])
        sarr[f"{lvl}_k"] = uk; sarr[f"{lvl}_h"] = hh; sarr[f"{lvl}_z"] = zz
    np.savez(f"{WORK}/ste_{DS}_dt.npz", **sarr)
    summ = dict(n_queries=int(len(ste_rows)), train_ts_cut=int(cut), gold_reached=float(sum(p[1] for p in parts) / max(1, sum(p[2] for p in parts))),
                n_types={lvl: int(len(sarr[f"{lvl}_k"])) for lvl in ("full", "type", "shape")}, seconds=round(time.time() - t0, 1),
                frac_static_entries_dated=float((SI.lb > -10**8).mean()))
    jdump(summ, "ste_dated_summary.json"); jdump(summ, f"{WORK}/ste_dated_summary.json")
    log(json.dumps(summ))


if __name__ == "__main__":
    main()
