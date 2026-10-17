#!/usr/bin/env python3
"""pilot_smallpedia / sp_verify.py -- TGB (py-tgb) <-> harness bridge for tkgl-smallpedia + static-edge emission.

1. Loads tkgl-smallpedia through py-tgb's own LinkPropPredDataset (files extracted from the raw zip into the package
   datasets dir; no download), including ds.static_data (TGB's own static-edge processing, shared node-id dict).
2. Entity map: harness entity2id (Q-id -> id) vs TGB node-id dict (Q-id -> id) -> h2t; relation map likewise.
   TGB full_data rows (first half) vs the harness split files concatenated (train, valid, test).
3. Conflict sets: for EVERY val/test key (ts, src, rel) the TGB conflict set == the harness time-aware filter
   set at (s, r', t); negatives = arange(min_dst, max_dst + 1) must be exactly the temporal entities.
4. Static edges: TGB static (head, tail, edge_type) -> shared id space: temporal entities -> harness id (via t2h),
   static-only nodes keep a fresh id >= N_ENT (TGB id if the map is the identity); static relation index p in
   [0, n_prel) (TGB edge_type minus its offset).  Written to $P/stage/static_edges.npz + names json.
   Also: static-edge stats, overlap with temporal pairs (train / valid / test), static-only node count.
"""
import os, sys, json, time, shutil, zipfile, csv, io, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from sp_common import H, P, DD, ZIP, N_ENT, N_REL, TGB_NAME, STATIC
t0 = time.time()
OUT = {"host": "serverA (CPU, tkg-gpu:latest container)"}
log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
os.makedirs(f"{P}/stage", exist_ok=True)

import tgb
from tgb.utils.info import PROJ_DIR
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import importlib.metadata as im
OUT["py_tgb_version"] = im.version("py-tgb")
root = os.path.join(PROJ_DIR, "datasets", TGB_NAME.replace("-", "_"))
os.makedirs(root, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if not os.path.exists(os.path.join(root, n)):
            z.extract(n, root)
try:
    shutil.copy(os.path.join(os.path.dirname(list(tgb.__path__)[0] + "/"), "datasets", "tkgl_smallpedia", "smallpedia_remove_conflict.py"),
                "tgb_smallpedia_remove_conflict.py")
except Exception as e:
    log("copy remove_conflict failed", e)
ds = LinkPropPredDataset(name=TGB_NAME, root="datasets", preprocess=True)
fd = ds.full_data
src = fd["sources"]; dst = fd["destinations"]; ts = fd["timestamps"]; et = fd["edge_type"]
E2 = len(src); E = E2 // 2
OUT["tgb"] = dict(n_rows_with_inverse=int(E2), num_nodes=int(ds.num_nodes), num_rels=int(ds.num_rels),
                  min_dst=int(ds.min_dst_idx), max_dst=int(ds.max_dst_idx),
                  n_train=int(ds.train_mask.sum()), n_val=int(ds.val_mask.sum()), n_test=int(ds.test_mask.sum()))
log(OUT["tgb"])
node_id = ds._node_id; etid = ds._edge_type_ids              # Q-id -> TGB id ; P-id -> TGB rel id (temporal)
OUT["tgb"]["n_node_ids_after_temporal"] = int(len(node_id))
ent_h = {}
for ln in open(f"{DD}/entity2id.txt"):
    k, v = ln.rstrip("\n").split("\t"); ent_h[k] = int(v)
rel_h = {}
for ln in open(f"{DD}/relation2id.txt"):
    k, v = ln.rstrip("\n").split("\t"); rel_h[k] = int(v)
assert len(ent_h) == N_ENT and len(rel_h) == N_REL, (len(ent_h), len(rel_h))
h2t = np.full(N_ENT, -1, np.int64)
for k, v in ent_h.items():
    h2t[v] = node_id[k]
assert (h2t >= 0).all()
r2t = np.full(N_REL, -1, np.int64)
for k, v in rel_h.items():
    r2t[v] = etid[k]
OUT["idmap"] = dict(n_entities_harness=N_ENT, bijective=bool(len(np.unique(h2t)) == N_ENT), identity=bool((h2t == np.arange(N_ENT)).all()),
                    frac_identity=float((h2t == np.arange(N_ENT)).mean()), max_tgb_id=int(h2t.max()),
                    relations_identity=bool((r2t == np.arange(N_REL)).all()), n_rel_tgb_temporal=int(len(etid)))
t2h = np.full(max(int(h2t.max()) + 1, N_ENT), -1, np.int64); t2h[h2t] = np.arange(N_ENT)
log(OUT["idmap"])
# harness split files concatenated must equal TGB full_data first half (same CSV, time-sorted)
def lq(fp):
    return np.loadtxt(fp, dtype=np.int64).reshape(-1, 4)
Q = {nm: lq(f"{DD}/{nm}.txt") for nm in ("train", "valid", "test")}
allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
tsmap = {}
for ln in open(f"{DD}/timestamp2id.txt"):
    k, v = ln.split(); tsmap[int(k)] = int(v)
inv_ts = {v: k for k, v in tsmap.items()}
OUT["tgb_rows_match_harness_order"] = dict(n_equal_len=bool(len(allq) == E), src=bool((src[:E] == h2t[allq[:, 0]]).all()),
                                           dst=bool((dst[:E] == h2t[allq[:, 2]]).all()),
                                           ts=bool((ts[:E] == np.array([inv_ts[x] for x in allq[:, 3]])).all()),
                                           rel=bool((et[:E] == r2t[allq[:, 1]]).all()),
                                           inverse_src=bool((src[E:] == h2t[allq[:, 2]]).all()),
                                           inverse_rel=bool((et[E:] == r2t[allq[:, 1]] + len(etid)).all()),
                                           split_sizes=[int(len(Q[k])) for k in ("train", "valid", "test")])
log(OUT["tgb_rows_match_harness_order"])
np.save(f"{P}/stage/h2t_entity.npy", h2t); np.save(f"{P}/stage/t2h_entity.npy", t2h)
json.dump({str(k): v for k, v in tsmap.items()}, open(f"{P}/stage/tsraw2id.json", "w"))
# ---- conflict sets
R = N_REL
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
    for (t_, s_, r_), conf in es.items():
        rr_ = int(r_)
        # TGB relation id -> harness relation id (identity expected; inverse = + n_rel)
        key = (int(t2h[int(s_)]), rr_, tsmap[int(t_)])
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
                              conflict_size_max=int(np.max(sizes)))
    log(split, OUT[f"ns_{split}"])
OUT["negatives_are_exactly_temporal_entities"] = bool(ds.min_dst_idx == 0 and ds.max_dst_idx == N_ENT - 1)
ev = Evaluator(name=TGB_NAME)
rng = np.random.default_rng(0)
yp = rng.normal(size=(50, 1)); yn = rng.normal(size=(50, 300)); yn[:, :3] = yp
m = ev.eval({"y_pred_pos": yp, "y_pred_neg": yn, "eval_metric": ["mrr"]})
rk = 1 + (yn > yp[:, :1]).sum(1) + 0.5 * (yn == yp[:, :1]).sum(1)
OUT["evaluator_formula_check"] = dict(tgb=float(m["mrr"]), mine=float(np.mean(1 / rk)))
# ---- static edges through TGB's own processing (shares and EXTENDS the node-id dict)
sd = ds.static_data
hS = np.asarray(sd["head"]).astype(np.int64); tS = np.asarray(sd["tail"]).astype(np.int64); eS = np.asarray(sd["edge_type"]).astype(np.int64)
n_nodes = int(len(ds._node_id))
off = int(eS.min()); prel_ids = np.unique(eS)
p_local = np.searchsorted(prel_ids, eS)
inv_et = {v: k for k, v in ds._edge_type_ids.items()}
prel_names = [inv_et.get(int(x), str(x)) for x in prel_ids]
# shared id space: temporal entities -> harness id; static-only -> TGB id (>= N_ENT when the temporal map is the identity)
def to_h(x):
    out = x.copy()
    m_ = x < len(t2h)
    out[m_] = np.where(t2h[np.minimum(x[m_], len(t2h) - 1)] >= 0, t2h[np.minimum(x[m_], len(t2h) - 1)], x[m_])
    return out
hh = to_h(hS); th = to_h(tS)
assert OUT["idmap"]["identity"], "non-identity entity map: static-only id assignment would need a remap"
np.savez(STATIC, h=hh.astype(np.int32), t=th.astype(np.int32), p=p_local.astype(np.int32), n_nodes=np.int64(n_nodes),
         n_prel=np.int64(len(prel_ids)), tgb_edge_type=prel_ids)
inv_node = {v: k for k, v in ds._node_id.items()}
json.dump(dict(prel_names=prel_names, tgb_static_edge_type_offset=off,
               static_only_qids=[inv_node[i] for i in range(N_ENT, n_nodes)][:200000]), open(f"{P}/stage/static_names.json", "w"))
deg = np.bincount(np.concatenate([hh, th]), minlength=n_nodes)
both_tem = (hh < N_ENT) & (th < N_ENT)
pairs_st = set((np.minimum(hh, th)[both_tem] * n_nodes + np.maximum(hh, th)[both_tem]).tolist())
ov = {}
for nm in ("train", "valid", "test"):
    q = Q[nm]; key = np.minimum(q[:, 0], q[:, 2]) * n_nodes + np.maximum(q[:, 0], q[:, 2])
    ov[nm] = float(np.mean([k in pairs_st for k in key.tolist()]))
tem_rel_names = set(rel_h.keys())
pcount = np.bincount(p_local, minlength=len(prel_ids))
top = np.argsort(-pcount)[:25]
OUT["static"] = dict(n_edges=int(len(hh)), n_nodes_total=n_nodes, n_static_only_nodes=int(n_nodes - N_ENT), n_prel=int(len(prel_ids)),
                     tgb_edge_type_offset=off, n_edges_both_temporal=int(both_tem.sum()),
                     n_edges_one_static_only=int(((hh >= N_ENT) ^ (th >= N_ENT)).sum()),
                     n_edges_both_static_only=int(((hh >= N_ENT) & (th >= N_ENT)).sum()),
                     n_self_loops=int((hh == th).sum()),
                     temporal_entities_with_static_edge=float((deg[:N_ENT] > 0).mean()),
                     static_degree_temporal_entities=dict(mean=float(deg[:N_ENT].mean()), median=float(np.median(deg[:N_ENT])),
                                                          p99=float(np.percentile(deg[:N_ENT], 99)), max=int(deg[:N_ENT].max())),
                     static_degree_all=dict(max=int(deg.max()), n_gt_1000=int((deg > 1000).sum()), n_gt_10000=int((deg > 10000).sum())),
                     top_relations=[(prel_names[i], int(pcount[i])) for i in top],
                     static_relations_also_temporal=sorted(set(prel_names) & tem_rel_names),
                     frac_split_facts_whose_pair_is_a_static_edge=ov)
log(json.dumps(OUT["static"])[:2000])
OUT["elapsed_s"] = round(time.time() - t0, 1)
json.dump(OUT, open("tgb_verify.json", "w"), indent=1)
os.system(f"chmod -R a+rX {P}/stage")
log("DONE")
