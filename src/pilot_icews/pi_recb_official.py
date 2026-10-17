#!/usr/bin/env python3
"""pilot_icews / pi_recb_official.py -- TGB RecurrencyBaseline (Gastinger et al. IJCAI'24) on the Stage-A test sample (CPU).

Uses the TGB repository's UNMODIFIED predictor (modules/recurrencybaseline_predictor.baseline_predict, repo commit
3b83e5cf, copied to $P/stage/tgb_modules) with exactly the data preparation of examples/linkproppred/tkgl-icews/
recurrencybaseline.py (py-tgb LinkPropPredDataset, reformat_ts, group_by relation, create_basis_dict(train+valid),
negative sampler = dataset.negative_sampler with the shipped test ns), default mode (train_flag False:
lambda 0.1, alpha 0.99 for every relation) and window 100.  The script's own `test()` scores ALL test queries per
relation and returns an unlabelled list; here the same baseline_predict is called per relation on the time-sorted
sampled queries only (the predictor processes queries independently given (all_data, window, cur_ts)), in a fork pool
instead of ray, and every per-query MRR is mapped back to the harness test-sample order.
  python3 pi_recb_official.py --workers 20 --window 100
"""
import os, sys, json, time, argparse, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import (P, DS, DD, WORK, ZIP, TGB_NAME, Splits, split_queries, jdump, log_fn)
sys.path.insert(0, f"{P}/stage/tgb_modules")
import multiprocessing as mp

G = {}


def _work(task):
    rel, rows = task
    from modules.recurrencybaseline_predictor import baseline_predict
    td = G["test_data"][rows]
    perf, hits = baseline_predict(len(td), td, G["all_prel"][rel], G["window"], G["basis_dict"], G["num_nodes"], G["num_rels"],
                                  G["lmbda"], G["alpha"], G["evaluator"], G["first_ts"][rel], G["neg_sampler"], "test")
    return rows, np.asarray(perf, np.float64), np.asarray(hits, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--lmbda", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.99)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    from tgb.utils.info import PROJ_DIR
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    from modules.tkg_utils import create_basis_dict, group_by, reformat_ts
    import importlib.metadata as im
    root = os.path.join(PROJ_DIR, "datasets", TGB_NAME.replace("-", "_")); os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if not os.path.exists(os.path.join(root, n)):
                z.extract(n, root)
    dataset = LinkPropPredDataset(name=TGB_NAME, root="datasets", preprocess=True)
    relations = dataset.edge_type; num_rels = dataset.num_rels
    subjects = dataset.full_data["sources"]; objects = dataset.full_data["destinations"]
    num_nodes = dataset.num_nodes
    timestamps_orig = dataset.full_data["timestamps"]
    timestamps = reformat_ts(timestamps_orig, TGB_NAME)
    all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
    train_data = all_quads[dataset.train_mask]; val_data = all_quads[dataset.val_mask]; test_data = all_quads[dataset.test_mask]
    all_data = np.concatenate([train_data, val_data, test_data])
    log("dataset", len(train_data), len(val_data), len(test_data), "num_rels", num_rels)
    all_prel = group_by(all_data, 1)
    basis_dict = create_basis_dict(np.concatenate([train_data, val_data]))
    dataset.load_test_ns()
    # ---- map the harness test sample to TGB test rows (entity map verified identity by pi_verify)
    h2t = np.load(f"{P}/stage/h2t_entity.npy"); assert (h2t == np.arange(len(h2t))).all()
    S = Splits(DS); tidx = np.load(f"{WORK}/test_sample_idx.npy")
    Q = split_queries(S, "test")[tidx]
    day2raw = {}
    for ln in open(f"{DD}/timestamp2id.txt"):
        k, v = ln.split(); day2raw[int(v)] = int(k)
    Rn = S.R
    days = np.array(sorted(day2raw)); raws = np.array([day2raw[d] for d in days], np.int64)
    pr = np.searchsorted(raws, test_data[:, 4].astype(np.int64)); assert (raws[pr] == test_data[:, 4]).all()
    day_t = days[pr].astype(np.int64)
    key_t = ((day_t * (2 * Rn) + test_data[:, 1].astype(np.int64)) * S.n_ent + test_data[:, 0].astype(np.int64)) * S.n_ent + test_data[:, 2].astype(np.int64)
    key_q = ((Q[:, 3] * (2 * Rn) + Q[:, 1]) * S.n_ent + Q[:, 0]) * S.n_ent + Q[:, 2]
    od = np.argsort(key_t, kind="stable"); ks = key_t[od]
    pos = np.searchsorted(ks, key_q); assert (ks[np.minimum(pos, len(ks) - 1)] == key_q).all(), "sample query not in TGB test"
    tgb_row = od[pos]                                    # duplicates share a key -> identical per-query result
    log("mapped", len(tgb_row), "unique rows", len(np.unique(tgb_row)))
    sub = test_data[tgb_row]
    first_ts = {}
    for rel in np.unique(test_data[:, 1]):
        first_ts[int(rel)] = int(test_data[test_data[:, 1] == rel][0, 3])
    G.update(test_data=sub, all_prel=all_prel, window=a.window, basis_dict=basis_dict, num_nodes=num_nodes, num_rels=num_rels,
             lmbda=a.lmbda, alpha=a.alpha, evaluator=Evaluator(name=TGB_NAME), first_ts=first_ts, neg_sampler=dataset.negative_sampler)
    tasks = []
    for rel in np.unique(sub[:, 1]):
        rows = np.flatnonzero(sub[:, 1] == rel)
        rows = rows[np.argsort(sub[rows, 3], kind="stable")]
        for ch in np.array_split(rows, max(1, len(rows) // 1500)):
            tasks.append((int(rel), ch))
    tasks.sort(key=lambda x: -len(x[1]))
    rr = np.full(len(sub), np.nan); h10 = np.full(len(sub), np.nan)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (rows, pf, ht) in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            rr[rows] = pf; h10[rows] = ht
            if k % 50 == 0:
                log(k, "/", len(tasks))
    assert not np.isnan(rr).any()
    res = dict(host="serverA (CPU, tkg-gpu:latest)", py_tgb=im.version("py-tgb"), tgb_repo_commit="3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb",
               mode="default (train_flag False)", lmbda=a.lmbda, alpha=a.alpha, window=a.window, n=int(len(rr)),
               official=dict(MRR=float(rr.mean()), H10=float(h10.mean())), num_rels_tgb=int(num_rels), elapsed_s=round(time.time() - t0, 1))
    np.save(f"recb_official_w{a.window}_test_sample_rr.npy", rr); np.save(f"{WORK}/recb_official_w{a.window}_test_sample_rr.npy", rr)
    jdump(res, f"recb_official_w{a.window}.json"); jdump(res, f"{WORK}/recb_official_w{a.window}.json")
    log("ALL DONE")


if __name__ == "__main__":
    main()
