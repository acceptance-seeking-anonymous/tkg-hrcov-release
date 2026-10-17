#!/usr/bin/env python3
"""v78_external / ct_score.py -- official py-tgb 2.3.0 scoring + per-query dumps of CountTRuCoLa rankings files.

Input: the rankings files written by the faithful CountTRuCoLa run (rule_based/main.py, JuliaGast/counttrucola_submission
@ 5d9aedf): pairs of lines  "head rel ? t_ct"  /  "cand score cand score ..."  (t_ct = CountTRuCoLa's own timestamp id).
Every entity not listed scores 0 (rule_utils.create_scores_array).  Each query is scored with the official
tgb.linkproppred.evaluate.Evaluator on (gold score, scores of the shipped negatives; TKGNegativeEdgeSampler.query_batch),
i.e. ties are averaged (optimistic/pessimistic), not randomly broken as in the authors' EVAL_TYPE=random.
Dumps are written in the HARNESS query order (split_queries of harness_data/<ds>/<split>.txt).
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

TR = os.path.expandvars("$WORK_DIR/tracks")
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--rankings-val', required=True); ap.add_argument('--rankings-test', required=True)
ap.add_argument('--topk', type=int, default=1000); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--own-results', default=None, help='authors results csv (copied into meta)')
ap.add_argument('--splits', default='val,test')
a = ap.parse_args()
T0 = time.time(); os.makedirs(a.out, exist_ok=True)
def log(*x): print(f"[{time.time()-T0:8.1f}s]", *x, flush=True)

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
import importlib.metadata as im
ds = LinkPropPredDataset(name=a.dataset, root="datasets", preprocess=True)
N = int(ds.num_nodes); K = a.topk
ts_orig = ds.full_data["timestamps"]

# ---- CountTRuCoLa timestamp ids: verbatim logic of rule_dataset.RuleDataset.timestamps_to_id / reformat_ts
def tkg_granularity_lookup(dataset_name, ts_distmean, ts_distmin):
    if 'icews' in dataset_name or 'polecat' in dataset_name:
        if '18' in dataset_name or '14' in dataset_name: return ts_distmean
        else: return 86400
    elif 'wiki' in dataset_name or 'yago' in dataset_name: return 31536000
    else: return ts_distmin
def reformat_ts(timestamps, dataset_name):
    all_ts = sorted(set(timestamps)); ts_min = np.min(all_ts)
    if 'tkgl' in dataset_name:
        d = np.diff(np.array(all_ts, dtype=np.float64)); ts_distmax, ts_distmin, ts_distmean = np.max(d), np.min(d), np.mean(d)
        if ts_distmean != ts_distmin:
            ts_dist = tkg_granularity_lookup(dataset_name, ts_distmean, ts_distmin)
            if ts_dist - ts_distmean > 0.1 * ts_distmean: ts_dist = ts_distmin
        else: ts_dist = ts_distmean
    return np.ceil((timestamps - ts_min) / ts_dist).astype(int)
uts = np.unique(ts_orig); new = reformat_ts(uts, a.dataset)
raw2ct = dict(zip(uts.tolist(), new.tolist()))
assert len(set(new.tolist())) == len(uts), "CountTRuCoLa ts-id map not injective"

def harness_order(split):
    hs = {'val': 'valid', 'test': 'test'}[split]
    dd = f"{TR}/datasets/harness_data/{a.dataset}"
    R = int(open(f"{dd}/stat.txt").read().split()[1])
    q = np.loadtxt(f"{dd}/{hs}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    u_, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q)); out = []
    for u, b, e in zip(u_, st, en):
        m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        out.append(m.copy()); out.append(inv)
    sro = np.concatenate(out).astype(np.int64)
    id2raw = {}
    for ln in open(f"{dd}/timestamp2id.txt"):
        k, v = ln.split(); id2raw[int(v)] = int(k)
    raw = np.array([id2raw[int(x)] for x in sro[:, 3]], dtype=np.int64)
    csr_f = f"{TR}/hardening/v76/recb_xf/recb_csr_{hs}_{a.dataset}.npz"
    csr_eq = bool(np.array_equal(np.load(csr_f)["sro"], sro)) if os.path.exists(csr_f) else None
    return sro, raw, csr_eq, R

def read_rankings(path):
    keys, ptr, cand, score = [], [0], [], []
    with open(path) as f:
        while True:
            h = f.readline()
            if not h: break
            b = f.readline()
            s, r, _, t = h.split()
            keys.append((int(s), int(r), int(t)))
            tok = b.split()
            if tok:
                c = np.array(tok[0::2], dtype=np.int64); v = np.array(tok[1::2], dtype=np.float64)
                cand.append(c); score.append(v); ptr.append(ptr[-1] + len(c))
            else:
                ptr.append(ptr[-1])
    cand = np.concatenate(cand) if cand else np.zeros(0, np.int64); score = np.concatenate(score) if score else np.zeros(0)
    kd = {k: i for i, k in enumerate(keys)}
    return kd, np.array(ptr), cand, score, len(keys)

G = {}
def work(rng):
    lo, hi = rng
    sro, raw, kidx, ptr, cand, score, sampler, ev, split = (G[k] for k in ('sro', 'raw', 'kidx', 'ptr', 'cand', 'score', 'sampler', 'ev', 'split'))
    n = hi - lo
    o = dict(top_ids=np.empty((n, K), np.int32), top_scores=np.empty((n, K), np.float64), floor=np.empty(n), tail_min=np.empty(n),
             n_gt_floor=np.empty(n, np.int32), gold_score=np.empty(n), rr=np.empty(n), rank=np.empty(n), n_cand=np.empty(n, np.int32), maxdev=0.0)
    sc = np.zeros(N, np.float64)
    for j, h in enumerate(range(lo, hi)):
        s, r, g, t = int(sro[h, 0]), int(sro[h, 1]), int(sro[h, 2]), int(raw[h])
        i = kidx[h]
        sc[:] = 0.0
        if i >= 0:
            c = cand[ptr[i]:ptr[i + 1]]; sc[c] = score[ptr[i]:ptr[i + 1]]; o['n_cand'][j] = len(c)
        else:
            o['n_cand'][j] = 0
        neg = sampler.query_batch(np.array([s]), np.array([g]), np.array([t]), np.array([r]), split_mode=split)[0]
        neg = np.asarray(neg, dtype=np.int64)
        pos = sc[g]; ns = sc[neg]
        m = ev.eval({'y_pred_pos': np.array([pos]), 'y_pred_neg': ns, 'eval_metric': ['mrr']})
        opt = int((ns > pos).sum()); pes = int((ns >= pos).sum()); rk = 0.5 * (opt + pes) + 1.0
        o['maxdev'] = max(o['maxdev'], abs(float(m['mrr']) - 1.0 / rk))
        o['rr'][j] = float(m['mrr']); o['rank'][j] = rk; o['gold_score'][j] = pos
        part = np.argpartition(-sc, K - 1)[:K]; vK = sc[part].min()
        gt = np.flatnonzero(sc > vK); eq = np.flatnonzero(sc == vK); need = K - len(gt)
        sel = np.concatenate([gt, eq[:need]]); sel = sel[np.lexsort((sel, -sc[sel]))]
        o['top_ids'][j] = sel; o['top_scores'][j] = sc[sel]
        if len(eq) > need: fl = vK
        else:
            below = sc[sc < vK]; fl = below.max() if len(below) else vK
        o['floor'][j] = fl; o['tail_min'][j] = sc.min(); o['n_gt_floor'][j] = int((sc > fl).sum())
    return lo, o

META = dict(method='CountTRuCoLa', repo='https://github.com/JuliaGast/counttrucola_submission', commit='5d9aedf4106926d13befa141e0855d684f3f5f9d',
            dataset=a.dataset, py_tgb=im.version('py-tgb'), topk=K, rankings=dict(val=a.rankings_val, test=a.rankings_test),
            score_floor_rule="entities absent from the rankings line score 0 (rule_utils.create_scores_array)",
            evaluator="official py-tgb Evaluator.eval per query; negatives TKGNegativeEdgeSampler(first_dst_id=0,last_dst_id=N-1).query_batch on shipped ns")
if a.own_results and os.path.exists(a.own_results):
    META['authors_results_csv_tail'] = open(a.own_results).read().splitlines()[-3:]
for split in a.splits.split(','):
    t_sp = time.time()
    sro, raw, csr_eq, R = harness_order(split)
    kd, ptr, cand, score, nkeys = read_rankings(a.rankings_val if split == 'val' else a.rankings_test)
    log(split, 'rankings parsed', nkeys, 'queries,', len(cand), 'candidate entries')
    kidx = np.array([kd.get((int(s), int(r), raw2ct[int(t)]), -1) for s, r, t in zip(sro[:, 0], sro[:, 1], raw)], dtype=np.int64)
    sampler = TKGNegativeEdgeSampler(a.dataset, first_dst_id=0, last_dst_id=N - 1)
    nsfile = os.path.join(ds.root, f"{a.dataset}_{'val' if split == 'val' else 'test'}_ns.pkl")
    sampler.load_eval_set(nsfile, split_mode=split)
    G.update(sro=sro, raw=raw, kidx=kidx, ptr=ptr, cand=cand, score=score, sampler=sampler, ev=Evaluator(name=a.dataset), split=split)
    nq = len(sro)
    D = f"{a.out}/{split}"; os.makedirs(D, exist_ok=True)
    TI = np.lib.format.open_memmap(f"{D}/top_ids.npy", 'w+', np.int32, (nq, K))
    TS = np.lib.format.open_memmap(f"{D}/top_scores.npy", 'w+', np.float64, (nq, K))
    V = {k: np.zeros(nq, dt) for k, dt in (('floor', np.float64), ('tail_min', np.float64), ('n_gt_floor', np.int32), ('gold_score', np.float64),
                                           ('rr', np.float64), ('rank', np.float64), ('n_cand', np.int32))}
    chunks = [(lo, min(lo + 500, nq)) for lo in range(0, nq, 500)]
    maxdev = 0.0; done = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as ex:
        for lo, o in ex.map(work, chunks):
            hi = lo + len(o['rr']); TI[lo:hi] = o['top_ids']; TS[lo:hi] = o['top_scores']
            for k in V: V[k][lo:hi] = o[k]
            maxdev = max(maxdev, o['maxdev']); done += hi - lo
            if (lo // 500) % 100 == 0: log(split, 'done', done, '/', nq)
    TI.flush(); TS.flush()
    for k, v in V.items(): np.save(f"{D}/{ {'rr': 'rr_official', 'rank': 'rank_official'}.get(k, k)}.npy", v)
    np.save(f"{D}/gold.npy", sro[:, 2].astype(np.int32)); np.save(f"{D}/keys_harness.npy", sro)
    ingold = np.zeros(nq, bool)
    for c0 in range(0, nq, 20000):
        ingold[c0:c0 + 20000] = (np.asarray(TI[c0:c0 + 20000]) == sro[c0:c0 + 20000, 2][:, None]).any(1)
    rk = V['rank']; rr = V['rr']
    used = np.zeros(nkeys, bool); used[kidx[kidx >= 0]] = True
    res = dict(n=int(nq), MRR=float(rr.mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean()),
               evaluator_vs_formula_max_abs_dev=float(maxdev), gold_in_topk_frac=float(ingold.mean()),
               gold_outside_topk_score_le_floor=bool(np.all(V['gold_score'][~ingold] <= V['floor'][~ingold])),
               tail_exact_frac=float((V['tail_min'] == V['floor']).mean()),
               queries_without_ranking_line=int((kidx < 0).sum()), ranking_lines_unmatched=int((~used).sum()),
               n_cand_max=int(V['n_cand'].max()), n_cand_median=float(np.median(V['n_cand'])),
               alignment=dict(rows=int(nq), gold_equals_harness_gold=True, harness_split_queries_equals_v76_recb_csr_sro=csr_eq, harness_R=R,
                              note="rows are built directly from the harness split_queries order; gold.npy == harness gold by construction; each row's ranking line is looked up by (s, r, CountTRuCoLa ts id)"),
               elapsed_s=round(time.time() - t_sp, 1))
    META.setdefault('splits', {})[split] = res
    json.dump(META, open(f"{a.out}/meta.json", "w"), indent=1, default=float)
    log(split, json.dumps(res))
    del TI, TS
META['total_s'] = round(time.time() - T0, 1)
json.dump(META, open(f"{a.out}/meta.json", "w"), indent=1, default=float)
log('DONE')
