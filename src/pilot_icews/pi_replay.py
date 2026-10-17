#!/usr/bin/env python3
"""pilot_icews / pi_replay.py -- GPU replay of a frozen Stage-A base (selected checkpoint), two modes.

Replays the base exactly as pi_base.py's dump (ground-truth online history: valid = train + rolling valid, test =
train + valid + rolling test; RE-GCN last 10 snapshots; LogCL global copy-history over every earlier fact) and, for the
VALID sample and the TEST sample, works on the FULL 87,856-entity log-prob row of each query.

mode lookup  : per query  lp_req [n,300] exact base log-prob of the requested candidates (C_cov columns 50..199 of the
               'orig' and 'rn' lists, -1 pad), rk_req raw base rank of each (1 + #entities with higher log-prob),
               gold_lp, rank_filt (1 + #non-filter entities with lp > gold), tie_n.  For TEST also the OFFICIAL
               py-tgb evaluation of the base on the sample (see below).
mode systems : official py-tgb evaluation of re-ranked systems on the test sample.  --systems name=path.npy,...  each npy
               is [n_test_sample, W] int32: the promoted list in final fused order (-1 pad); the system's full score row =
               base log-prob row with those entities lifted above everything in that order (pc_lookup convention).
Official evaluation: py-tgb Evaluator.eval (numpy path) on batches whose y_pred_neg is the full score row (TGB id order)
with the TGB time-filtered conflict set (TKGNegativeEdgeSampler eval_set, loaded from the shipped *_test_ns.pkl) masked to
-inf, i.e. exactly the negatives query_batch returns (all entities minus the conflict set; -inf never counts as > or >=).
Per-query ranks are the evaluator's formula 1 + #(neg > pos) + 0.5 #(neg == pos); the size-weighted mean of Evaluator.eval's
batch mrr is recorded beside it (evaluator_vs_formula_absdiff; not asserted, checked at aggregation).  A uniform 20,000-query subsample is additionally scored through the per-query query_batch path.
  python3 pi_replay.py --base regcn --mode lookup
  python3 pi_replay.py --base regcn --mode systems --systems HR_s0=<npy>,...
"""
import os, sys, json, time, argparse, copy, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from pi_common import (fast_global_hist, DUMPS, H, P, DS, WORK, ZIP, TGB_NAME, CKPT, N_ENT, Splits, split_queries, filter_pairs, jdump, log_fn, KC)
sys.path.insert(0, H)
from regcn import make_edges, REGCN
from logcl import LogCL, GlobalHist
from pi_base import groups_of


def load_official():
    from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
    from tgb.linkproppred.evaluate import Evaluator
    import importlib.metadata as im
    nsdir = f"{WORK}/tgb_ns"; os.makedirs(nsdir, exist_ok=True)
    fn = f"{nsdir}/{TGB_NAME}_test_ns.pkl"
    if not os.path.exists(fn):
        with zipfile.ZipFile(ZIP) as z:
            z.extract(f"{TGB_NAME}_test_ns.pkl", nsdir)
    sampler = TKGNegativeEdgeSampler(TGB_NAME, first_dst_id=0, last_dst_id=N_ENT - 1)
    sampler.load_eval_set(fn, split_mode="test")
    return sampler, Evaluator(name=TGB_NAME), im.version("py-tgb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", default="lookup")
    ap.add_argument("--systems", default="")
    ap.add_argument("--no_official", action="store_true")
    ap.add_argument("--perquery_n", type=int, default=20000)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    from pi_common import DEV as dev
    S = Splits(DS); N, R = S.n_ent, S.R
    ck = torch.load(CKPT[a.base], map_location="cpu", weights_only=False)
    L = 10
    model = (REGCN if a.base == "regcn" else LogCL)(N, R, 200, 2, 0.2, L).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    tr_pairs = groups_of(S.q["train"]); va_pairs = groups_of(S.q["valid"]); te_pairs = groups_of(S.q["test"])
    F_ = S.filter_index()
    vidx = np.load(f"{WORK}/valid_sample_idx.npy"); tidx = np.load(f"{WORK}/test_sample_idx.npy")
    want = {"valid": vidx, "test": tidx} if a.mode == "lookup" else {"test": tidx}
    Dd = np.load(DUMPS[a.base])
    req = {}
    if a.mode == "lookup":
        for sp in want:
            parts = []
            for sch in ("orig", "rn"):
                C = np.load(f"{WORK}/lists_{a.base}/{sp}_{sch}_Ccov.npy", mmap_mode="r")
                parts.append(np.asarray(C[:, KC:]))
            req[sp] = np.concatenate(parts, 1).astype(np.int64)
            assert len(req[sp]) == len(want[sp])
    systems = {}
    if a.mode == "systems":
        for kv in a.systems.split(","):
            nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r")
            assert len(systems[nm]) == len(tidx), nm
    official = not a.no_official
    if official:
        sampler, evaluator, tgb_version = load_official()
        h2t = np.load(f"{P}/stage/h2t_entity.npy"); assert (h2t == np.arange(N)).all(), "non-identity entity map not supported here"
        day2raw = {}
        for ln in open(f"{S.cfg['dd']}/timestamp2id.txt"):
            k, v = ln.split(); day2raw[int(v)] = int(k)
        es = sampler.eval_set["test"]
        rng = np.random.default_rng(7)
        pq_sel = set(np.sort(rng.choice(len(tidx), min(a.perquery_n, len(tidx)), replace=False)).tolist())
    names = (["base"] if a.mode == "lookup" else list(systems))
    out = {}
    for sp in want:
        n = len(want[sp])
        O = dict(gold_lp=np.zeros(n, np.float32), rank_filt=np.zeros(n, np.int32), tie_n=np.zeros(n, np.int32))
        if a.mode == "lookup":
            O.update(lp_req=np.zeros((n, req[sp].shape[1]), np.float32), rk_req=np.zeros((n, req[sp].shape[1]), np.int32))
        if sp == "test" and official:
            for nm in names:
                O[f"tgb_rank_{nm}"] = np.zeros(n, np.float64); O[f"tgb_pq_rank_{nm}"] = np.full(n, np.nan)
        out[sp] = O
    eval_chk = {nm: [0.0, 0.0, 0] for nm in names}       # sum of evaluator batch-mean*n, sum of my rr, n
    hist_edges = [make_edges(g, R, dev) for _, g in tr_pairs[-L:]]
    gh = None
    if a.base == "logcl":
        gh = fast_global_hist()(N)
        for _, g in tr_pairs:
            gh.update(g, R)
    log("history ready")
    t_off = 0.0
    with torch.no_grad():
        for sp, pairs in (("valid", va_pairs), ("test", te_pairs)):
            pos0 = 0; wset = want.get(sp); slot0 = 0
            for k, (ts, triples) in enumerate(pairs):
                nq = 2 * len(triples)
                if wset is not None and len(triples):
                    lo_, hi_ = np.searchsorted(wset, [pos0, pos0 + nq])
                    loc = wset[lo_:hi_] - pos0
                    if len(loc):
                        s_all = np.concatenate([triples[:, 0], triples[:, 2]]); r_all = np.concatenate([triples[:, 1], triples[:, 1] + R])
                        o_all = np.concatenate([triples[:, 2], triples[:, 0]])
                        for b0 in range(0, len(loc), 2048):
                            lb = loc[b0:b0 + 2048]; sl = np.arange(lo_ + b0, lo_ + b0 + len(lb))
                            s_q, r_q, o_q = s_all[lb], r_all[lb], o_all[lb]
                            st = torch.tensor(s_q, device=dev); rt = torch.tensor(r_q, device=dev)
                            if a.base == "regcn":
                                sc, _, _ = model(hist_edges[-L:], st, rt)
                            else:
                                sc, _, _, _, _ = model(hist_edges[-L:], st, rt, gh.freq_vec(s_q, r_q, dev))
                            lp = F.log_softmax(sc, dim=1); del sc
                            gold = torch.tensor(o_q, device=dev)
                            glp = lp.gather(1, gold.unsqueeze(1))
                            rows, cols = filter_pairs(F_, S.fkey(s_q, r_q, np.full(len(s_q), ts)), o_q)
                            lf = lp.clone()
                            if len(rows):
                                lf[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = float("-inf")
                            O = out[sp]
                            O["gold_lp"][sl] = glp.squeeze(1).cpu().numpy()
                            O["rank_filt"][sl] = ((lf > glp).sum(1) + 1).cpu().numpy()
                            O["tie_n"][sl] = ((lf == glp).sum(1) - 1).cpu().numpy()
                            if a.mode == "lookup":
                                rq = torch.tensor(req[sp][sl], device=dev)
                                lpr = lp.gather(1, rq.clamp(min=0))
                                srt, _ = torch.sort(lp, dim=1)
                                cle = torch.searchsorted(srt, lpr.contiguous(), right=True)
                                O["lp_req"][sl] = torch.where(rq >= 0, lpr, torch.full_like(lpr, -1e30)).cpu().numpy()
                                O["rk_req"][sl] = torch.where(rq >= 0, N - cle + 1, torch.zeros_like(cle)).cpu().numpy()
                                del srt, cle, lpr
                            if sp == "test" and official:
                                t1 = time.time()
                                lpc = lp.cpu().numpy()
                                traw = day2raw[int(ts)]
                                conf = [np.asarray(es[(traw, int(s_q[j]), int(r_q[j]))], np.int64) for j in range(len(lb))]
                                cr = np.repeat(np.arange(len(lb)), [len(x) for x in conf]); cc = np.concatenate(conf)
                                for nm in names:
                                    Y = lpc
                                    if nm != "base":
                                        Y = lpc.copy()
                                        Cs = np.asarray(systems[nm][sl]).astype(np.int64)
                                        for qq in range(len(lb)):
                                            ccq = Cs[qq][Cs[qq] >= 0]
                                            Y[qq, ccq] = 1e6 - np.arange(len(ccq), dtype=np.float32)
                                    yp = Y[np.arange(len(lb)), o_q].copy()
                                    Yn = Y.copy(); Yn[cr, cc] = -np.inf
                                    m = evaluator.eval({"y_pred_pos": yp, "y_pred_neg": Yn, "eval_metric": ["mrr"]})
                                    rk = 1 + (Yn > yp[:, None]).sum(1) + 0.5 * (Yn == yp[:, None]).sum(1)
                                    O[f"tgb_rank_{nm}"][sl] = rk
                                    eval_chk[nm][0] += float(m["mrr"]) * len(lb); eval_chk[nm][1] += float((1.0 / rk).sum()); eval_chk[nm][2] += len(lb)
                                    for qq in range(len(lb)):
                                        if int(sl[qq]) in pq_sel:
                                            negs = sampler.query_batch(np.array([s_q[qq]]), np.array([o_q[qq]]), np.array([traw]),
                                                                       np.array([r_q[qq]]), split_mode="test")[0]
                                            ypq = np.array([Y[qq, o_q[qq]]]); ynq = Y[qq, negs][None, :]
                                            mq = evaluator.eval({"y_pred_pos": ypq, "y_pred_neg": ynq, "eval_metric": ["mrr"]})
                                            O[f"tgb_pq_rank_{nm}"][sl[qq]] = 1.0 / float(mq["mrr"])
                                t_off += time.time() - t1
                            del lp, lf
                if len(triples):
                    hist_edges.append(make_edges(triples, R, dev)); hist_edges = hist_edges[-L:]
                    if gh is not None:
                        gh.update(triples, R)
                pos0 += nq
                if k % 200 == 0:
                    log(sp, "snapshot", k, "/", len(pairs), "official_s", round(t_off))
    res = dict(base=a.base, mode=a.mode, host="serverA (GPU replay + CPU official evaluator, tkg-gpu:latest)",
               gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"), checkpoint=CKPT[a.base], ckpt_epoch=int(ck.get("epoch", -1)), official_eval_s=round(t_off, 1))
    stats = {}
    for sp, O in out.items():
        dg = Dd[f"{sp}_gold_lp"]; dr = Dd[f"{sp}_rank_filt"]
        stats[sp] = dict(n=int(len(O["rank_filt"])), max_abs_gold_lp_diff_vs_dump=float(np.abs(O["gold_lp"] - dg).max()),
                         rank_agree_vs_dump=float(np.mean(O["rank_filt"] == dr)), max_abs_rank_diff_vs_dump=int(np.abs(O["rank_filt"].astype(np.int64) - dr).max()))
    res["replay_vs_dump"] = stats
    if official:
        res["py_tgb"] = tgb_version
        res["official"] = {}
        for nm in names:
            rk = out["test"][f"tgb_rank_{nm}"]; pq = out["test"][f"tgb_pq_rank_{nm}"]; mpq = ~np.isnan(pq)
            res["official"][nm] = dict(MRR=float(np.mean(1.0 / rk)), H10=float(np.mean(rk <= 10)), n=int(len(rk)),
                                       evaluator_batchmean_MRR=eval_chk[nm][0] / max(eval_chk[nm][2], 1),
                                       evaluator_vs_formula_absdiff=abs(eval_chk[nm][0] - eval_chk[nm][1]) / max(eval_chk[nm][2], 1),
                                       perquery_path_n=int(mpq.sum()), perquery_path_max_abs_rank_diff=float(np.abs(pq[mpq] - rk[mpq]).max()))
        rf = out["test"]["rank_filt"].astype(np.float64); tr_ = out["test"]["tgb_rank_base"] if "tgb_rank_base" in out["test"] else None
        if tr_ is not None:
            res["official_vs_harness_base"] = dict(harness_MRR=float(np.mean(1 / rf)), tgb_MRR=float(np.mean(1 / tr_)),
                                                   frac_rank_equal=float(np.mean(np.abs(rf - tr_) < 1e-6)),
                                                   max_abs_rank_diff=float(np.abs(rf - tr_).max()), n_with_ties=int((out["test"]["tie_n"] > 0).sum()))
    tagd = f"{WORK}/{'lookup' if a.mode == 'lookup' else 'official'}_{a.base}"
    os.makedirs(tagd, exist_ok=True)
    for sp, O in out.items():
        np.savez(f"{tagd}/{sp}.npz", **O)
        if sp == "test" and official:
            np.savez(f"official_{a.mode}_{a.base}_ranks.npz", **{k: v for k, v in O.items() if k.startswith("tgb_")})
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"{a.mode}_{a.base}.json"); jdump(res, f"{tagd}/{a.mode}_{a.base}.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE", json.dumps(res.get("replay_vs_dump")))


if __name__ == "__main__":
    main()
