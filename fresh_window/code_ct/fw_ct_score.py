#!/usr/bin/env python3
"""phase2_fresh_window / fw_ct_score.py -- v78_external/ct_score.py adapted to GDELTFW (no TGB negative files).
Rankings file of the released CountTRuCoLa run: pairs of lines "head rel ? t_ct" / "cand score cand score ...".  Every
entity not listed scores 0 (rule_utils.create_scores_array).  Per harness query (split_queries order) the rank is the
py-tgb Evaluator formula over all entities with the time-aware filter (negatives = all entities except the gold and the
other true objects of (s, r, t) in the harness data): rank = 1 + #(neg > pos) + 0.5 #(neg == pos); harness strict rank
1 + #(neg > pos) alongside.  Top-1000 dump (score desc, ties by id asc), floor, tail_min, n_gt_floor, n_cand, gold_score
as in ct_score.py.  CountTRuCoLa ts id = ceil((unix_day - unix_min) / 86400) (reformat_ts with granularity 86400 s).
  python3 fw_ct_score.py --dd D --rankings FILE --split val|test --out DIR [--topk 1000] [--workers 12]
"""
import os, sys, json, time, argparse, calendar
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

ap = argparse.ArgumentParser()
ap.add_argument("--dd", required=True); ap.add_argument("--rankings", required=True); ap.add_argument("--split", required=True, choices=["val", "test"])
ap.add_argument("--out", required=True); ap.add_argument("--topk", type=int, default=1000); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--check_evaluator", type=int, default=2000, help="queries re-scored with the vendored tgb Evaluator (formula check)")
a = ap.parse_args(); T0 = time.time(); os.makedirs(a.out, exist_ok=True)
def log(*x): print(f"[{time.time()-T0:8.1f}s]", *x, flush=True)

st = open(f"{a.dd}/stat.txt").read().split(); N, R = int(st[0]), int(st[1]); K = a.topk
def load(sp): return np.loadtxt(f"{a.dd}/{sp}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
hs = {"val": "valid", "test": "test"}[a.split]
q = load(hs); order = np.argsort(q[:, 3], kind="stable"); q = q[order]
u_, st_ = np.unique(q[:, 3], return_index=True); en_ = np.append(st_[1:], len(q)); blocks = []
for u, b, e in zip(u_, st_, en_):
    m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R; blocks.append(m.copy()); blocks.append(inv)
sro = np.concatenate(blocks).astype(np.int64)
# time-aware filter: true objects of (s, r, t) in the split (facts of a timestamp live in one split)
from collections import defaultdict
true_o = defaultdict(set)
for s, r, o, t in sro: true_o[(int(s), int(r), int(t))].add(int(o))
id2day = {}
for ln in open(f"{a.dd}/timestamp2id.txt"):
    k, v = ln.split(); id2day[int(v)] = k
unix = np.array([calendar.timegm(time.strptime(id2day[t], "%Y%m%d")) for t in range(len(id2day))], np.int64)
ct_ts = np.ceil((unix - unix.min()) / 86400).astype(np.int64)
assert len(set(ct_ts.tolist())) == len(ct_ts) and (ct_ts == np.arange(len(ct_ts))).all(), "CountTRuCoLa ts ids != harness ids"

def read_rankings(path):
    keys, ptr, cand, score = [], [0], [], []
    with open(path) as f:
        while True:
            h = f.readline()
            if not h: break
            b = f.readline(); s, r, _, t = h.split(); keys.append((int(s), int(r), int(t))); tok = b.split()
            if tok:
                c = np.array(tok[0::2], dtype=np.int64); v = np.array(tok[1::2], dtype=np.float64)
                cand.append(c); score.append(v); ptr.append(ptr[-1] + len(c))
            else: ptr.append(ptr[-1])
    cand = np.concatenate(cand) if cand else np.zeros(0, np.int64); score = np.concatenate(score) if score else np.zeros(0)
    return {k: i for i, k in enumerate(keys)}, np.array(ptr), cand, score, len(keys)

kd, ptr, cand, score, nkeys = read_rankings(a.rankings); log(a.split, "rankings parsed", nkeys, "lines,", len(cand), "entries")
kidx = np.array([kd.get((int(s), int(r), int(ct_ts[t])), -1) for s, r, t in zip(sro[:, 0], sro[:, 1], sro[:, 3])], np.int64)
G = dict(sro=sro, kidx=kidx, ptr=ptr, cand=cand, score=score)

def work(rng):
    lo, hi = rng; n = hi - lo
    o = dict(top_ids=np.empty((n, K), np.int32), top_scores=np.empty((n, K), np.float64), floor=np.empty(n), tail_min=np.empty(n),
             n_gt_floor=np.empty(n, np.int32), gold_score=np.empty(n), rr=np.empty(n), rank=np.empty(n), hrank=np.empty(n), n_cand=np.empty(n, np.int32),
             n_filtered=np.empty(n, np.int32))
    sc = np.zeros(N, np.float64)
    for j, h in enumerate(range(lo, hi)):
        s, r, g, t = (int(x) for x in G["sro"][h]); i = G["kidx"][h]; sc[:] = 0.0
        if i >= 0:
            c = G["cand"][G["ptr"][i]:G["ptr"][i + 1]]; sc[c] = G["score"][G["ptr"][i]:G["ptr"][i + 1]]; o["n_cand"][j] = len(c)
        else: o["n_cand"][j] = 0
        neg = np.ones(N, bool); neg[g] = False
        fo = [x for x in true_o[(s, r, t)] if x != g]; neg[fo] = False; o["n_filtered"][j] = len(fo)
        pos = sc[g]; ns = sc[neg]
        opt = int((ns > pos).sum()); pes = int((ns >= pos).sum()); rk = 0.5 * (opt + pes) + 1.0
        o["rr"][j] = 1.0 / rk; o["rank"][j] = rk; o["hrank"][j] = opt + 1.0; o["gold_score"][j] = pos
        part = np.argpartition(-sc, K - 1)[:K] if K < N else np.arange(N); vK = sc[part].min()
        gt = np.flatnonzero(sc > vK); eq = np.flatnonzero(sc == vK); need = K - len(gt)
        sel = np.concatenate([gt, eq[:need]]); sel = sel[np.lexsort((sel, -sc[sel]))]
        o["top_ids"][j] = sel; o["top_scores"][j] = sc[sel]
        if len(eq) > need: fl = vK
        else:
            below = sc[sc < vK]; fl = below.max() if len(below) else vK
        o["floor"][j] = fl; o["tail_min"][j] = sc.min(); o["n_gt_floor"][j] = int((sc > fl).sum())
    return lo, o

nq = len(sro); D = a.out
TI = np.lib.format.open_memmap(f"{D}/top_ids.npy", "w+", np.int32, (nq, K)); TS = np.lib.format.open_memmap(f"{D}/top_scores.npy", "w+", np.float64, (nq, K))
V = {k: np.zeros(nq, dt) for k, dt in (("floor", np.float64), ("tail_min", np.float64), ("n_gt_floor", np.int32), ("gold_score", np.float64), ("rr", np.float64),
                                       ("rank", np.float64), ("hrank", np.float64), ("n_cand", np.int32), ("n_filtered", np.int32))}
chunks = [(lo, min(lo + 500, nq)) for lo in range(0, nq, 500)]
with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
    for lo, o in ex.map(work, chunks):
        hi = lo + len(o["rr"]); TI[lo:hi] = o["top_ids"]; TS[lo:hi] = o["top_scores"]
        for k in V: V[k][lo:hi] = o[k]
TI.flush(); TS.flush()
for k, v in V.items(): np.save(f"{D}/{ {'rr': 'rr_official', 'rank': 'rank_official', 'hrank': 'rank_harness_strict'}.get(k, k)}.npy", v)
np.save(f"{D}/gold.npy", sro[:, 2].astype(np.int32)); np.save(f"{D}/keys_harness.npy", sro)
chk = None
if a.check_evaluator:
    try:
        from tgb.linkproppred.evaluate import Evaluator
        ev = Evaluator(name="tkgl-polecat"); dev = 0.0; nchk = min(a.check_evaluator, nq); sc = np.zeros(N)
        for h in range(nchk):
            s, r, g, t = (int(x) for x in sro[h]); i = kidx[h]; sc[:] = 0.0
            if i >= 0: sc[cand[ptr[i]:ptr[i + 1]]] = score[ptr[i]:ptr[i + 1]]
            neg = np.ones(N, bool); neg[g] = False; neg[[x for x in true_o[(s, r, t)] if x != g]] = False
            m = ev.eval({"y_pred_pos": np.array([sc[g]]), "y_pred_neg": sc[neg], "eval_metric": ["mrr"]})
            dev = max(dev, abs(float(m["mrr"]) - V["rr"][h]))
        chk = dict(n=nchk, max_abs_dev_vs_vendored_tgb_Evaluator=dev, evaluator_instance="tgb.linkproppred.evaluate.Evaluator(name='tkgl-polecat') used only for its MRR formula")
    except Exception as e:
        chk = dict(error=repr(e))
ingold = np.zeros(nq, bool)
for c0 in range(0, nq, 20000): ingold[c0:c0 + 20000] = (np.asarray(TI[c0:c0 + 20000]) == sro[c0:c0 + 20000, 2][:, None]).any(1)
used = np.zeros(nkeys, bool); used[kidx[kidx >= 0]] = True
meta = dict(split=a.split, n=int(nq), N=N, R=R, topk=K, rankings=a.rankings, rankings_size=os.path.getsize(a.rankings),
            evaluator="all-entity time-aware filtered rank, ties averaged (py-tgb formula); harness strict alongside",
            gold_in_topk_frac=float(ingold.mean()), gold_outside_topk_score_le_floor=bool(np.all(V["gold_score"][~ingold] <= V["floor"][~ingold])),
            tail_exact_frac=float((V["tail_min"] == V["floor"]).mean()), queries_without_ranking_line=int((kidx < 0).sum()),
            ranking_lines_unmatched=int((~used).sum()), n_cand_max=int(V["n_cand"].max()), n_cand_median=float(np.median(V["n_cand"])),
            evaluator_check=chk, elapsed_s=round(time.time() - T0, 1))
if a.split == "val":     # validation metrics only; nothing is printed for test
    rk = V["rank"]; meta.update(MRR=float(V["rr"].mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean()))
json.dump(meta, open(f"{D}/meta.json", "w"), indent=1, default=float)
log(json.dumps({k: v for k, v in meta.items() if a.split == "val" or k not in ("MRR", "H1", "H3", "H10")}))
