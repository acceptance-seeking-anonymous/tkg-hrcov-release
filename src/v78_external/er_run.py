#!/usr/bin/env python3
"""v78_external / er_run.py -- EdgeReMIND (BryantPollard/EdgeReMIND @ 41c0db2) faithful training + per-query score dumps.

Training/selection code path is copied VERBATIM from examples/linkproppred/tkgl/edgeremind.py (same seed, gamma=2, n=3,
30 epochs, Adam lr 1e-3, K=20, smoothing window 3, max_train_queries 2M, decay 1/(7*86400)); the authors' own
evaluate_parallel() is then run on the test split (faithful test MRR, position cutoffs = TGB/TGM streaming protocol,
batch 200 events).  After that, every val/test query is re-scored over ALL entities with the trained theta:
  * 'strict' cutoff  : history = all edges with timestamp < t (index position of the first edge at t) == HRcov boundary
  * 'faithful' cutoff: the authors' cutoff (split offset + #events in earlier 200-event batches), optional (--faithful-perquery)
Per query the official py-tgb Evaluator.eval is called on (gold score, scores of the shipped negatives).
Dumps (strict variant) are written in the HARNESS query order (split_queries of harness_data/<ds>/<split>.txt).
"""
import argparse, os, sys, json, time, warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
import numpy as np
import torch
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, TGBTKGNegativeEdgeSamplerHook
from tgm.nn import (EdgeReMINDPredictor, EdgeReMINDTrainer, build_full_index, calibrate_bank_lambdas,
                    evaluate_parallel, train_learned)
from tgm.util.logging import enable_logging, log_latency, log_metric
from tgm.util.seed import seed_everything

TR = os.path.expandvars("$WORK_DIR/tracks")
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1337)
ap.add_argument('--dataset', type=str, default='tkgl-smallpedia')
ap.add_argument('--num-workers', type=int, default=20)
ap.add_argument('--out', type=str, required=True)
ap.add_argument('--topk', type=int, default=1000)
ap.add_argument('--splits', type=str, default='val,test')
ap.add_argument('--faithful-perquery', action='store_true')
ap.add_argument('--theta-from', type=str, default=None, help='resume: load theta (skip training + authors test eval)')
ap.add_argument('--min-avail-gb', type=float, default=80.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
META = dict(method='EdgeReMIND', repo='https://github.com/BryantPollard/EdgeReMIND', commit='41c0db2df31bf4ab60e6e4fec8176932314e3ec8',
            dataset=args.dataset, seed=args.seed, num_workers=args.num_workers, topk=args.topk,
            config=dict(gamma=2.0, n_timescales=3, epochs=30, lr=1e-3, opt_bsize=1024, k=20, feature_bsize=200,
                        max_train_queries=2_000_000, smooth_window=3, patience=0, min_delta=1e-3, shrinkage=0.0,
                        decay_rate_srd=1.0 / (7 * 86400), init_weights=[1.0, 1e-3, 1e-2, 2.0, 1e-2, 0.0], loader_batch=200))
def save_meta():
    json.dump(META, open(f"{args.out}/meta.json", "w"), indent=1, default=float)
def log(*x): print(f"[{time.time()-T0:8.1f}s]", *x, flush=True)

# ----------------------------------------------------------------- verbatim example setup
enable_logging(log_file_path=None)
seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)
data = DGData.from_tgb(args.dataset)
min_dst_node = data.edge_index[:, 1].min().int()
max_dst_node = data.edge_index[:, 1].max().int()
num_relations = int(data.edge_type.max().item()) + 1
train_data, val_data, test_data = data.split()
train_dg = DGraph(train_data); val_dg = DGraph(val_data); test_dg = DGraph(test_data)
train_mat = train_dg.materialize(materialize_features=False)
hm = HookManager(keys=['val', 'test'])
val_hook = TGBTKGNegativeEdgeSamplerHook(args.dataset, split_mode='val', first_dst_id=min_dst_node, last_dst_id=max_dst_node)
test_hook = TGBTKGNegativeEdgeSamplerHook(args.dataset, split_mode='test', first_dst_id=min_dst_node, last_dst_id=max_dst_node)
if args.dataset in ('tkgl-wikidata',):
    val_hook.neg_sampler.strategy = 'dst-time-filtered'; test_hook.neg_sampler.strategy = 'dst-time-filtered'
hm.register('val', val_hook); hm.register('test', test_hook)
val_loader = DGDataLoader(val_dg, 200, hook_manager=hm)
test_loader = DGDataLoader(test_dg, 200, hook_manager=hm)
bank_lambdas = calibrate_bank_lambdas(train_mat.edge_src.cpu().numpy(), train_mat.edge_dst.cpu().numpy(),
                                      train_mat.edge_time.cpu().numpy(), train_mat.edge_type.cpu().numpy(),
                                      n_timescales=3, geometric_factor=2.0)
META['bank_half_lives'] = (np.log(2.0) / bank_lambdas).tolist()
model = EdgeReMINDPredictor(num_relations=num_relations, decay_rate_srd=1.0 / (7 * 86400), decay_rate_d=None,
                            init_weights=(1.0, 1e-3, 1e-2, 2.0, 1e-2, 0.0), bank_lambdas=bank_lambdas)
trainer = EdgeReMINDTrainer(model, first_dst=int(min_dst_node), last_dst=int(max_dst_node), k=20, feature_bsize=200,
                            max_train_queries=2_000_000, seed=args.seed)
val_mat = val_dg.materialize(materialize_features=False)
test_mat = test_dg.materialize(materialize_features=False)
index, n_train, n_val = build_full_index(train_mat, val_mat, test_mat, 1.0 / (7 * 86400), None, bank_lambdas=bank_lambdas)
log('index built', n_train, n_val)
if args.theta_from is None:
    t_tr = time.time()
    with hm.activate('val'):
        best_label, val_mrr = train_learned(model, trainer, index, train_mat, val_loader, num_relations=num_relations,
                                            first_dst=int(min_dst_node), last_dst=int(max_dst_node), num_workers=args.num_workers,
                                            epochs=30, lr=1e-3, opt_bsize=1024, n_train=n_train, bsize=200, patience=0,
                                            min_delta=1e-3, shrinkage=0.0, smooth_window=3,
                                            val_curve_path=f"{args.out}/val_curve.json")
    theta = model.theta.detach().cpu().numpy()
    np.save(f"{args.out}/theta.npy", theta)
    META.update(best_epoch=best_label, val_mrr_authors=float(val_mrr), train_select_s=round(time.time() - t_tr, 1))
    save_meta(); log('validation MRR (authors, faithful cutoffs)', val_mrr)
    t_te = time.time()
    with hm.activate('test'):
        test_mrr = evaluate_parallel(test_loader, index, theta, n_train + n_val, 200, args.num_workers)
    META.update(test_mrr_authors=float(test_mrr), authors_test_eval_s=round(time.time() - t_te, 1))
    save_meta(); log('test MRR (authors, faithful cutoffs)', test_mrr)
else:
    theta = np.load(args.theta_from).astype(np.float32)
    prev = os.path.join(os.path.dirname(os.path.abspath(args.theta_from)), 'meta.json')
    if os.path.exists(prev):
        pm = json.load(open(prev))
        for k in ('best_epoch', 'val_mrr_authors', 'train_select_s', 'test_mrr_authors', 'authors_test_eval_s'):
            if k in pm: META[k] = pm[k]
    META['theta_from'] = args.theta_from; save_meta()
theta = np.asarray(theta, dtype=np.float32)

# ----------------------------------------------------------------- per-query full scoring
def arrs(m):
    return [x.cpu().numpy().astype(np.int64) for x in (m.edge_src, m.edge_dst, m.edge_time, m.edge_type)]
ALL_T = np.concatenate([arrs(train_mat)[2], arrs(val_mat)[2], arrs(test_mat)[2]])
assert np.all(np.diff(ALL_T) >= 0), "index stream not time-sorted"
N = int(data.edge_index.max().item()) + 1
K = args.topk
_G = {}

def _score_rows(args_):
    """worker: rows of one 200-event batch -> per-query outputs (strict + optional faithful)."""
    import psutil
    (b0, s_, r_, t_, g_, cf_, cs_, negs, do_f) = args_
    index, theta, ev = _G['index'], _G['theta'], _G['ev']
    cand = np.arange(N, dtype=np.int64)
    n = len(s_)
    o = dict(top_ids=np.empty((n, K), np.int32), top_scores=np.empty((n, K), np.float32), floor=np.empty(n, np.float32),
             tail_min=np.empty(n, np.float32), n_gt_floor=np.empty(n, np.int32), gold_score=np.empty(n, np.float32),
             rr=np.empty(n, np.float64), rank=np.empty(n, np.float64), maxdev=0.0)
    if do_f:
        o['rr_f'] = np.empty(n, np.float64); o['rank_f'] = np.empty(n, np.float64)
    for i in range(n):
        s, r, t, g = int(s_[i]), int(r_[i]), int(t_[i]), int(g_[i])
        ng = negs[i]
        for var in (('s', 'f') if do_f else ('s',)):
            cut = int(cs_[i]) if var == 's' else int(cf_[i])
            sc = index.query_features(s, r, t, cut, cand) @ theta[r]
            pos = sc[g]; neg = sc[ng]
            m = ev.eval({'y_pred_pos': np.array([pos]), 'y_pred_neg': neg, 'eval_metric': ['mrr']})
            opt = int((neg > pos).sum()); pes = int((neg >= pos).sum()); rk = 0.5 * (opt + pes) + 1.0
            o['maxdev'] = max(o['maxdev'], abs(float(m['mrr']) - 1.0 / rk))
            if var == 'f':
                o['rr_f'][i] = float(m['mrr']); o['rank_f'][i] = rk; continue
            o['rr'][i] = float(m['mrr']); o['rank'][i] = rk; o['gold_score'][i] = pos
            part = np.argpartition(-sc, K - 1)[:K]
            vK = sc[part].min()
            gt = np.flatnonzero(sc > vK); eq = np.flatnonzero(sc == vK)
            need = K - len(gt)
            sel = np.concatenate([gt, eq[:need]])
            sel = sel[np.lexsort((sel, -sc[sel]))]
            o['top_ids'][i] = sel; o['top_scores'][i] = sc[sel]
            if len(eq) > need:
                fl = vK
            else:
                below = sc[sc < vK]; fl = below.max() if len(below) else vK
            o['floor'][i] = fl; o['tail_min'][i] = sc.min(); o['n_gt_floor'][i] = int((sc > fl).sum())
    return b0, o

def harness_order(split):
    hs = {'val': 'valid', 'test': 'test'}[split]
    dd = f"{TR}/datasets/harness_data/{args.dataset}"
    R = int(open(f"{dd}/stat.txt").read().split()[1])
    q = np.loadtxt(f"{dd}/{hs}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q)); out = []
    for u, b, e in zip(uts, st, en):
        m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        out.append(m.copy()); out.append(inv)
    sro = np.concatenate(out).astype(np.int64)
    id2raw = {}
    for ln in open(f"{dd}/timestamp2id.txt"):
        k, v = ln.split(); id2raw[int(v)] = int(k)
    raw = np.array([id2raw[int(x)] for x in sro[:, 3]], dtype=np.int64)
    csr_f = f"{TR}/hardening/v76/recb_xf/recb_csr_{hs}_{args.dataset}.npz"
    csr_eq = bool(np.array_equal(np.load(csr_f)["sro"], sro)) if os.path.exists(csr_f) else None
    return sro, raw, csr_eq, R

import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import psutil
for split in args.splits.split(','):
    t_sp = time.time()
    loader = {'val': val_loader, 'test': test_loader}[split]
    offset = n_train if split == 'val' else n_train + n_val
    nq = n_val if split == 'val' else len(test_mat.edge_src)
    do_f = bool(args.faithful_perquery)
    tmp = f"{args.out}/_tmp_{split}"; os.makedirs(tmp, exist_ok=True)
    MM = dict(top_ids=np.lib.format.open_memmap(f"{tmp}/top_ids.npy", 'w+', np.int32, (nq, K)),
              top_scores=np.lib.format.open_memmap(f"{tmp}/top_scores.npy", 'w+', np.float32, (nq, K)))
    V = dict(floor=np.zeros(nq, np.float32), tail_min=np.zeros(nq, np.float32), n_gt_floor=np.zeros(nq, np.int32),
             gold_score=np.zeros(nq, np.float32), rr=np.zeros(nq), rank=np.zeros(nq))
    if do_f: V['rr_f'] = np.zeros(nq); V['rank_f'] = np.zeros(nq)
    KEY = np.zeros((nq, 4), np.int64)   # TGB stream keys: s, r, o, raw t
    CUT = np.zeros((nq, 2), np.int64)
    _G.update(index=index, theta=theta, ev=Evaluator(name=args.dataset))
    seen = 0; maxdev = 0.0; done = 0
    ctx = mp.get_context('fork')
    with hm.activate(split), ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as ex:
        fut = deque()
        def drain(f):
            global done, maxdev
            b0, o = f.result(); n = len(o['rr'])
            for k in MM: MM[k][b0:b0 + n] = o[k]
            for k in V: V[k][b0:b0 + n] = o[k]
            maxdev = max(maxdev, o['maxdev']); done += n
        for bi, batch in enumerate(loader):
            ne = batch.edge_src.numel()
            if ne == 0: continue
            s_ = batch.edge_src.cpu().numpy().astype(np.int64); g_ = batch.edge_dst.cpu().numpy().astype(np.int64)
            t_ = batch.edge_time.cpu().numpy().astype(np.int64); r_ = batch.edge_type.cpu().numpy().astype(np.int64)
            cf = np.full(ne, offset + seen, np.int64)
            cs = np.searchsorted(ALL_T, t_, side='left').astype(np.int64)
            negs = [batch.neg_batch_list[i].cpu().numpy().astype(np.int64) for i in range(ne)]
            KEY[seen:seen + ne] = np.stack([s_, r_, g_, t_], 1); CUT[seen:seen + ne, 0] = cf; CUT[seen:seen + ne, 1] = cs
            fut.append(ex.submit(_score_rows, (seen, s_, r_, t_, g_, cf, cs, negs, do_f)))
            seen += ne
            while len(fut) >= 2 * args.num_workers: drain(fut.popleft())
            if bi % 100 == 0:
                av = psutil.virtual_memory().available / 2**30
                log(split, 'batch', bi, 'queued', seen, 'done', done, f'host_avail_GB={av:.0f}')
                if av < args.min_avail_gb: raise SystemExit(f'host MemAvailable {av:.0f} GB below guard -- aborting to protect co-tenants')
        while fut: drain(fut.popleft())
    assert seen == nq == done, (seen, nq, done)
    # strict (first edge of t) can lie AFTER the faithful cutoff (batch start) when a 200-event batch straddles a timestamp
    # boundary: the faithful protocol then hides earlier-timestamp edges of the same batch. Record both directions.
    cut_stats = dict(frac_strict_lt_faithful=float((CUT[:, 1] < CUT[:, 0]).mean()), frac_strict_gt_faithful=float((CUT[:, 1] > CUT[:, 0]).mean()),
                     frac_equal=float((CUT[:, 1] == CUT[:, 0]).mean()))
    assert np.all(ALL_T[CUT[:, 1] - 1] < KEY[:, 3]) and np.all(ALL_T[CUT[:, 1]] == KEY[:, 3]), "strict cutoff not at timestamp boundary"
    # ---------------- harness order mapping
    sro, raw, csr_eq, R = harness_order(split)
    hk = np.stack([sro[:, 0], sro[:, 1], sro[:, 2], raw], 1)
    assert len(hk) == nq, (len(hk), nq)
    oh = np.lexsort(hk.T[::-1]); ot = np.lexsort(KEY.T[::-1])
    assert np.array_equal(hk[oh], KEY[ot]), "multiset of (s,r,o,t) differs between TGB stream and harness"
    perm = np.empty(nq, np.int64); perm[oh] = ot          # harness row h  <-  TGB stream position perm[h]
    assert np.array_equal(KEY[perm], hk)
    D = f"{args.out}/{split}"; os.makedirs(D, exist_ok=True)
    for k, mm in MM.items():
        dst = np.lib.format.open_memmap(f"{D}/{k}.npy", 'w+', mm.dtype, mm.shape)
        for c0 in range(0, nq, 20000): dst[c0:c0 + 20000] = mm[perm[c0:c0 + 20000]]
        dst.flush(); del dst
    for k, v in V.items():
        nm = {'rr': 'rr_official', 'rank': 'rank_official', 'rr_f': 'rr_official_faithful_cutoff', 'rank_f': 'rank_official_faithful_cutoff'}.get(k, k)
        np.save(f"{D}/{nm}.npy", v[perm])
    np.save(f"{D}/gold.npy", sro[:, 2].astype(np.int32)); np.save(f"{D}/keys_harness.npy", sro)
    np.save(f"{D}/tgb_stream_pos.npy", perm); np.save(f"{D}/cutoffs_strict_faithful.npy", CUT[perm])
    MM.clear()
    import shutil; shutil.rmtree(tmp)
    # ---------------- checks + metrics
    gold_chk = np.load(f"{D}/top_ids.npy", mmap_mode='r')
    rk = V['rank'][perm]; rr = V['rr'][perm]
    gs = V['gold_score'][perm]; fl = V['floor'][perm]
    ingold = np.zeros(nq, bool); rank_in_top = np.full(nq, -1)
    for c0 in range(0, nq, 20000):
        ti = np.asarray(gold_chk[c0:c0 + 20000]); hit = ti == sro[c0:c0 + 20000, 2][:, None]
        ingold[c0:c0 + 20000] = hit.any(1)
    res = dict(n=int(nq), MRR=float(rr.mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean()),
               evaluator_vs_formula_max_abs_dev=float(maxdev), gold_in_topk_frac=float(ingold.mean()),
               gold_outside_topk_score_le_floor=bool(np.all(gs[~ingold] <= fl[~ingold] + 0.0)),
               tail_exact_frac=float((V['tail_min'][perm] == fl).mean()),
               median_n_gt_floor=float(np.median(V['n_gt_floor'])),
               alignment=dict(rows=int(nq), gold_equals_harness_gold=bool(np.array_equal(KEY[perm][:, 2], sro[:, 2])),
                              keys_equal_harness_keys=True, harness_split_queries_equals_v76_recb_csr_sro=csr_eq,
                              tgb_stream_order_is_harness_order=bool(np.array_equal(perm, np.arange(nq))), harness_R=R),
               cutoff_stats=cut_stats, elapsed_s=round(time.time() - t_sp, 1))
    if do_f:
        rkf = V['rank_f']; res['faithful_cutoff'] = dict(MRR=float(V['rr_f'].mean()), H1=float((rkf <= 1).mean()),
                                                     H3=float((rkf <= 3).mean()), H10=float((rkf <= 10).mean()))
    META.setdefault('splits', {})[split] = res
    META['peak_rss_main_gb'] = psutil.Process().memory_info().rss / 2**30
    save_meta(); log(split, json.dumps(res))
META['total_s'] = round(time.time() - T0, 1); save_meta()
log('DONE')
