#!/usr/bin/env python3
"""pilot_polecat / pc_tgb_verify.py -- TGB (py-tgb) <-> harness bridge for tkgl-polecat.

1. Loads tkgl-polecat through py-tgb's own LinkPropPredDataset (files copied from the raw zip into the
   package datasets dir; no download).
2. Builds the harness-id -> TGB-internal-id map for entities (and checks relations / timestamps), by
   replaying both id assignments on the same CSV and checking TGB's full_data arrays row by row.
3. Verifies that TGB's time-filtered negative set equals the harness time-aware filter: for EVERY val/test
   key (ts, src, rel) the TGB conflict set == the harness set of true answers at (s, r', t) (both
   directions), and negatives = arange(min_dst, max_dst+1) minus the conflict set covers all entities.
4. Writes id maps to the pilot dir and a JSON report to the job cwd.
"""
import os, sys, json, time, shutil, zipfile, csv, io
import numpy as np
t0 = time.time()
H = os.path.expandvars("$WORK_DIR")
P = f"{H}/tracks/pilot_polecat"
DD = f"{H}/tracks/datasets/harness_data/tkgl-polecat"
ZIP = f"{H}/tracks/datasets/raw/tkgl-polecat.zip"
OUT = {"host": "serverA (CPU, tkg-gpu:latest container)"}
log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)

import tgb
from tgb.utils.info import PROJ_DIR
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import importlib.metadata as im
OUT["py_tgb_version"] = im.version("py-tgb")
root = os.path.join(PROJ_DIR, "datasets", "tkgl_polecat")
os.makedirs(root, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if not os.path.exists(os.path.join(root, n)):
            z.extract(n, root)
log("files in", root, os.listdir(root))
ds = LinkPropPredDataset(name="tkgl-polecat", root="datasets", preprocess=True)
fd = ds.full_data
src = fd["sources"]; dst = fd["destinations"]; ts = fd["timestamps"]; et = fd["edge_type"]
E2 = len(src); E = E2 // 2
OUT["tgb"] = dict(n_rows_with_inverse=int(E2), num_nodes=int(ds.num_nodes), num_rels=int(ds.num_rels),
                  min_dst=int(ds.min_dst_idx), max_dst=int(ds.max_dst_idx),
                  n_train=int(ds.train_mask.sum()), n_val=int(ds.val_mask.sum()), n_test=int(ds.test_mask.sum()))
log(OUT["tgb"])

# ---- replay both id assignments on the CSV ----
ent_h = {}
for ln in open(f"{DD}/entity2id.txt"):
    k, v = ln.rstrip("\n").split("\t"); ent_h[k] = int(v)
N = len(ent_h)
with zipfile.ZipFile(ZIP) as z:
    name = [n for n in z.namelist() if n.endswith(".csv") and "edgelist" in n][0]
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8")); hdr = next(rd)
        rows = list(rd)
OUT["csv_header"] = hdr
tgb_ids = {}; h2t = np.full(N, -1, np.int64); hs = np.zeros(len(rows), np.int64); ho = np.zeros(len(rows), np.int64)
ts_raw = np.zeros(len(rows), np.int64); rel_raw = np.zeros(len(rows), np.int64)
for i, row in enumerate(rows):
    a, b = int(row[1]), int(row[2])
    if a not in tgb_ids: tgb_ids[a] = len(tgb_ids)
    if b not in tgb_ids: tgb_ids[b] = len(tgb_ids)
    ha, hb = ent_h[row[1]], ent_h[row[2]]
    for hh, tt in ((ha, tgb_ids[a]), (hb, tgb_ids[b])):
        if h2t[hh] < 0: h2t[hh] = tt
        elif h2t[hh] != tt: raise SystemExit(f"non-functional map at row {i}")
    hs[i] = ha; ho[i] = hb; ts_raw[i] = int(row[0]); rel_raw[i] = int(row[3])
assert (h2t >= 0).all()
OUT["idmap"] = dict(n_entities_harness=N, n_entities_tgb=len(tgb_ids), bijective=bool(len(np.unique(h2t)) == N),
                    identity=bool((h2t == np.arange(N)).all()), frac_identity=float((h2t == np.arange(N)).mean()),
                    n_raw_string_vs_int_collisions=int(N - len(tgb_ids)))
t2h = np.empty(N, np.int64); t2h[h2t] = np.arange(N)
# TGB full_data first half must be the CSV rows in order (inverse appended)
OUT["tgb_rows_match_csv_order"] = dict(src=bool((src[:E] == h2t[hs]).all()), dst=bool((dst[:E] == h2t[ho]).all()),
                                       ts=bool((ts[:E] == ts_raw).all()), rel=bool((et[:E] == rel_raw).all()),
                                       inverse_src=bool((src[E:] == h2t[ho]).all()), inverse_rel=bool((et[E:] == rel_raw + 16).all()))
log(OUT["idmap"], OUT["tgb_rows_match_csv_order"])
# timestamps -> harness day index
tsmap = {}
for ln in open(f"{DD}/timestamp2id.txt"):
    k, v = ln.split(); tsmap[int(k)] = int(v)
np.save(f"{P}/stage/h2t_entity.npy", h2t); np.save(f"{P}/stage/t2h_entity.npy", t2h)
np.save("h2t_entity.npy", h2t)
json.dump({str(k): v for k, v in tsmap.items()}, open(f"{P}/stage/tsraw2day.json", "w"))

# ---- harness filter (both directions, all splits) ----
def lq(fp):
    return np.loadtxt(fp, dtype=np.int64).reshape(-1, 4)
Q = {nm: lq(f"{DD}/{nm}.txt") for nm in ("train", "valid", "test")}
R = 16
from collections import defaultdict
def hfilter(q):
    d = defaultdict(set)
    for s, r, o, t in q:
        d[(int(s), int(r), int(t))].add(int(o)); d[(int(o), int(r) + R, int(t))].add(int(s))
    return d
for split, nm in (("val", "valid"), ("test", "test")):
    (ds.load_val_ns if split == "val" else ds.load_test_ns)()
    es = ds.ns_sampler.eval_set[split]
    hf = hfilter(Q[nm])
    n_keys = len(es); n_eq = 0; n_missing = 0; ex = []; sizes = []
    k0 = next(iter(es.keys())); v0 = es[k0]
    for (t_, s_, r_), conf in es.items():
        key = (int(t2h[int(s_)]), int(r_), tsmap[int(t_)])
        conf = np.asarray(conf).astype(np.int64)
        sizes.append(len(conf))
        got = hf.get(key)
        if got is None:
            n_missing += 1
            if len(ex) < 5: ex.append(("missing", key))
            continue
        hset = set(int(h2t[x]) for x in got)
        if hset == set(conf.tolist()): n_eq += 1
        elif len(ex) < 5: ex.append(("diff", key, sorted(hset)[:10], sorted(conf.tolist())[:10]))
    OUT[f"ns_{split}"] = dict(n_keys_tgb=n_keys, n_keys_harness=len(hf), n_equal_sets=n_eq, n_missing_in_harness=n_missing,
                              frac_equal=n_eq / max(n_keys, 1), examples=ex, conflict_size_mean=float(np.mean(sizes)),
                              conflict_size_max=int(np.max(sizes)), example_key=str(k0), example_val_type=str(type(v0)))
    log(split, OUT[f"ns_{split}"])
OUT["negatives_cover_all_entities"] = bool(ds.min_dst_idx == 0 and ds.max_dst_idx == N - 1)
ev = Evaluator(name="tkgl-polecat")
rng = np.random.default_rng(0)
yp = rng.normal(size=(50, 1)); yn = rng.normal(size=(50, 300)); yn[:, :3] = yp
m = ev.eval({"y_pred_pos": yp, "y_pred_neg": yn, "eval_metric": ["mrr"]})
rk = 1 + (yn > yp[:, :1]).sum(1) + 0.5 * (yn == yp[:, :1]).sum(1)
OUT["evaluator_formula_check"] = dict(tgb=float(m["mrr"]), mine=float(np.mean(1 / rk)))
OUT["elapsed_s"] = round(time.time() - t0, 1)
json.dump(OUT, open("tgb_verify.json", "w"), indent=1)
os.system(f"chmod -R a+rX {P}/stage")
log("DONE")
