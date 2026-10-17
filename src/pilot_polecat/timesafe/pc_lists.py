#!/usr/bin/env python3
"""pilot_polecat / pc_lists.py -- per-base coverage candidate lists + coverage features (CPU).

For one frozen base (regcn | logcl) and the valid sample + EVERY test query:
  * 6.1 pattern prior (non-learned): instantiate the subject's 1-2-hop entity-anonymised path patterns
    (M1=64, M2=32, online history < t), score each pattern by the predicted residual utility
    u = te_hier - log1p(mass_base(C_p) (e-1)) (retriever-track `confpat`), rank candidates by the best
    utility of any pattern containing them -> L_pat (top LP_LEN).
  * GraphBridge typed bridge (harness BridgeIndex over the TRAIN graph, anchors = base top-1 and s,
    relation-expanded first hop) scored by a stage-2 ranker (G12 arm5 features minus the event-LM column:
    X16 + cascade, 3-seed ensemble) trained here on 8,000 coverage-hard TRAIN queries -> L_br (top LB_LEN).
  * C_cov = base top-50 ++ first BP of L_pat outside the top-50 ++ first BB of L_br outside both; per
    candidate the 11 coverage features COV_FEATS.
Outputs under work/lists_<base>/ : {valid,test}_{Lpat,Lpat_u,Lbr,Lbr_s,Ccov,Xcov}.npy + {split}_q.npz
(gold-dependent diagnostics only: positions of the gold in the lists, natural bridge reach / size).
  python3 pc_lists.py --base regcn --workers 24
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (DS, DUMPS, WORK, KC, BP, BB, W_COV, LP_LEN, LB_LEN, COV_FEATS, SEEDS, Splits, BaseDump,
                       split_queries, jdump, log_fn)
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
import multiprocessing as mp

G = {}
E1 = np.e - 1.0
PAT_FILL = -6.0
KREF = 200


def make_net(nf):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(nf, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                               torch.nn.ReLU(), torch.nn.Linear(64, 1))


def np_mlp(W, X):
    h = np.maximum(X @ W[0] + W[1], 0.0); h = np.maximum(h @ W[2] + W[3], 0.0)
    return (h @ W[4] + W[5])[:, 0]


def bridge_score(X18):
    s = np.zeros(len(X18))
    for (W, mu, sd) in G["nets"]:
        s += np_mlp(W, (X18 - mu) / sd)
    return s / len(G["nets"])


def bridge_one(s, r, o, tki, tks):
    b = G["BI"].build(s, r, o, tki, tks)
    if b is None:
        return None
    X18 = np.concatenate([b["X"], b["casc"]], 1).astype(np.float32)
    return b, X18


def _cohort_chunk(rows):
    D = G["D"]; out = []
    for i in rows:
        s, r, o, t = (int(x) for x in D.sro[i])
        z = bridge_one(s, r, o, D.topk_ids[i, :KREF], D.topk_scores[i, :KREF])
        if z is None or not z[0]["reached"]:
            out.append((int(i), None, -1)); continue
        b, X18 = z
        out.append((int(i), X18, int(np.flatnonzero(b["cands"] == o)[0])))
    return out


def query_lists(i):
    """one dump row -> lists, C_cov, coverage features, gold diagnostics."""
    D = G["D"]; EI = G["EI"]; te = G["te"]; N = G["N"]; R = G["R"]
    s, r, o, t = (int(x) for x in D.sro[i])
    tki = D.topk_ids[i, :KREF].astype(np.int64); tks = D.topk_scores[i, :KREF].astype(np.float64)
    p_top = np.exp(tks); tail = max(1.0 - p_top.sum(), 1e-12); lp_tail = np.log(tail / max(N - KREF, 1))
    top50 = tki[:KC]
    res = dict(Lpat=np.full(LP_LEN, -1, np.int32), Lpat_u=np.zeros(LP_LEN, np.float16), Lbr=np.full(LB_LEN, -1, np.int32),
               Lbr_s=np.zeros(LB_LEN, np.float32), Ccov=np.full(W_COV, -1, np.int32), Xcov=np.zeros((W_COV, len(COV_FEATS)), np.float16))
    q = dict(gpos_pat=-1, gpos_br=-1, g_in_union=False, g_in_bridge=False, br_size=0, n_union=0, has_pat=False, has_br=False)
    # ---------------- pattern prior
    pat = {}
    d = instantiate(EI, s, t, M1, M2)
    if d is not None:
        q["has_pat"] = True
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        # base prob of each pair candidate (top-K exact, tail-uniform elsewhere)
        pos_in_top = G["pos_buf"]; pos_in_top[tki] = np.arange(KREF)
        pk = pos_in_top[cc]
        lpc = np.where(pk >= 0, tks[np.maximum(pk, 0)], lp_tail)
        pos_in_top[tki] = -1
        mass = np.bincount(pp, weights=np.exp(lpc), minlength=P_)
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = te.features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        util = te_h - np.log1p(mass * E1)
        cand, inv = np.unique(cc, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, util[pp])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pp])
        supp = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        lpcand = np.zeros(len(cand)); lpcand[inv] = lpc
        order = np.lexsort((-lpcand, -supp, -best))
        cand_o = cand[order]
        rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        L = cand_o[:LP_LEN]; res["Lpat"][:len(L)] = L; res["Lpat_u"][:len(L)] = best[order][:LP_LEN]
        q["n_union"] = len(cand)
        gi = np.searchsorted(cand, o)
        if gi < len(cand) and cand[gi] == o:
            q["g_in_union"] = True
            if rank_o[gi] < LP_LEN:
                q["gpos_pat"] = int(rank_o[gi])
        pat = dict(cand=cand, best=best, bte=bte, npat=npat, rank=rank_o)
    # ---------------- bridge
    br = {}
    z = bridge_one(s, r, o, tki, tks)
    if z is not None:
        b, X18 = z
        q["has_br"] = True
        sc = bridge_score(X18)
        order = np.argsort(-sc, kind="stable")
        bc = b["cands"].astype(np.int64)
        res["Lbr"][:min(LB_LEN, len(bc))] = bc[order][:LB_LEN]; res["Lbr_s"][:min(LB_LEN, len(bc))] = sc[order][:LB_LEN]
        q["br_size"] = len(bc); q["g_in_bridge"] = bool(b["reached"])
        rank_b = np.empty(len(bc), np.int64); rank_b[order] = np.arange(len(bc))
        if b["reached"]:
            gp = int(rank_b[np.flatnonzero(bc == o)[0]])
            if gp < LB_LEN:
                q["gpos_br"] = gp
        srt = np.argsort(bc)
        br = dict(cand=bc[srt], score=sc[srt], rank=rank_b[srt], pm_subj=b["X"][srt, 1], ninterm=b["X"][srt, 2],
                  smin=float(sc.min()))
    # ---------------- C_cov
    C = list(top50.tolist()); seen = set(C)
    for e in res["Lpat"]:
        if len(C) >= KC + BP or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    n_pb = len(C)
    for e in res["Lbr"]:
        if len(C) >= KC + BP + BB or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    C = np.array(C, np.int64); res["Ccov"][:len(C)] = C
    X = np.zeros((len(C), len(COV_FEATS)), np.float32)
    X[:, 0] = PAT_FILL; X[:, 4] = np.log1p(LP_LEN + 1); X[:, 7] = np.log1p(LB_LEN + 1)
    if pat:
        j = np.searchsorted(pat["cand"], C); j = np.minimum(j, len(pat["cand"]) - 1); m = pat["cand"][j] == C
        X[m, 0] = pat["best"][j[m]]; X[m, 1] = pat["bte"][j[m]]; X[m, 2] = np.log1p(pat["npat"][j[m]]); X[m, 3] = 1.0
        X[m, 4] = np.log1p(np.minimum(pat["rank"][j[m]], LP_LEN) + 1)
    if br:
        j = np.searchsorted(br["cand"], C); j = np.minimum(j, len(br["cand"]) - 1); m = br["cand"][j] == C
        X[:, 6] = br["smin"] - 1.0
        X[m, 5] = 1.0; X[m, 6] = br["score"][j[m]]; X[m, 7] = np.log1p(np.minimum(br["rank"][j[m]], LB_LEN) + 1)
        X[m, 8] = np.log1p(br["pm_subj"][j[m]]); X[m, 9] = np.log1p(br["ninterm"][j[m]])
    X[KC:, 10] = 1.0
    res["Xcov"][:len(C)] = X.astype(np.float16)
    q["n_pat_block"] = n_pb - KC; q["n_br_block"] = len(C) - n_pb
    return res, q


def _ts_bucket(task):
    """time-safe cohort bucket (runs in a pool worker): build the snapshot index over train facts with ts < cut
    (cut <= t of every query in the bucket) and compute the cohort features of the bucket's queries with it."""
    from bridge_features import BridgeIndex
    cut, rows = task
    tb = time.time()
    sub = G["trq"][G["ts_tr"] < cut]
    BIk = BridgeIndex(sub, G["N"], G["R"])
    tb = time.time() - tb
    G["BI"] = BIk                      # worker-local; the parent keeps the all-train index for the target splits
    out = _cohort_chunk(rows)
    info = dict(cut=int(cut), n_facts=int(len(sub)), max_fact_ts=int(sub[:, 3].max()) if len(sub) else -1,
                n_queries=int(len(rows)), build_s=round(tb, 2), DEG_CAP=int(BIk.DEG_CAP))
    del BIk; G["BI"] = None
    return info, out


def _target_chunk(rows):
    outs = [query_lists(int(i)) for i in rows]
    res = {k: np.stack([o[0][k] for o in outs]) for k in outs[0][0]}
    qk = outs[0][1].keys()
    q = {k: np.array([o[1][k] for o in outs]) for k in qk}
    return res, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--cohort_sample", type=int, default=30000)
    ap.add_argument("--cohort_n", type=int, default=8000)
    ap.add_argument("--timesafe", type=int, default=0,
                    help="K>0: build the stage-2 TRAINING cohort features from snapshot indices (train facts with ts < cut_k, "
                         "cut_k <= t of the query; K cuts at cohort-timestamp quantiles); K<0: exact, one snapshot per "
                         "distinct query timestamp (facts with ts < t); 0: static all-train index (original)")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    import torch
    torch.set_num_threads(16)
    out_dir = f"{WORK}/lists_{a.base}" + ("_smoke" if a.smoke else ""); os.makedirs(out_dir, exist_ok=True)
    S = Splits(DS); N, R = S.n_ent, S.R
    D = BaseDump(DUMPS[a.base], S); log("dump", D.N, D.K, D.load_s)
    summ = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)", workers=a.workers, work=WORK)
    rows = {sp: D.rows(sp) for sp in ("train", "valid", "test")}
    for sp in ("valid", "test"):
        assert (D.sro[rows[sp]].astype(np.int64) == split_queries(S, sp)).all(), f"dump row order != split order ({sp})"
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    targets = {"valid": rows["valid"][vidx], "test": rows["test"]}
    if a.smoke:
        targets = {k: v[:a.smoke] for k, v in targets.items()}
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, N, R, 1)
    te = TETables(f"{WORK}/te_POLECAT.npz")
    from bridge_features import BridgeIndex
    BI = BridgeIndex(S.q["train"], N, R); log("bridge index", BI.build_s, "s; DEG_CAP", BI.DEG_CAP)
    G.update(D=D, EI=EI, te=te, BI=BI, N=N, R=R, pos_buf=np.full(N, -1, np.int64))
    # ---------------- stage-2 bridge ranker on coverage-hard TRAIN queries
    gf = D.extra["gold_rank_filt"]
    hard = rows["train"][gf[rows["train"]] > KC]
    rng = np.random.default_rng(0)
    samp = rng.choice(hard, min(a.cohort_sample if not a.smoke else 3000, len(hard)), replace=False)
    t1 = time.time()
    ts_info = None
    if a.timesafe:
        # snapshot-before-t cohort: query at time t sees only train facts with ts < cut_k <= t (never its own fact,
        # never a later fact).  Cuts = K quantiles of the cohort timestamps (each bucket ~|samp|/K queries).
        trq = S.q["train"]; ts_tr = trq[:, 3].astype(np.int64)
        tq = D.sro[samp, 3].astype(np.int64)
        if a.timesafe < 0:                                  # exact: one snapshot per distinct query timestamp (cut = t)
            cuts = np.unique(tq)
        else:
            cuts = np.unique(np.quantile(tq, np.linspace(0, 1, a.timesafe, endpoint=False), method="lower").astype(np.int64))
        bk = np.searchsorted(cuts, tq, side="right") - 1
        assert (bk >= 0).all() and (cuts[bk] <= tq).all()
        G.update(trq=trq, ts_tr=ts_tr)
        tasks = [(int(cut), samp[bk == k]) for k, cut in enumerate(cuts) if (bk == k).any()]
        tasks.sort(key=lambda x: -x[0])                     # largest snapshots first (load balance)
        coh_map = {}; builds = []
        with mp.get_context("fork").Pool(a.workers) as pool:
            for j, (info, out) in enumerate(pool.imap_unordered(_ts_bucket, tasks, chunksize=1)):
                tmin = int(D.sro[[x[0] for x in out], 3].min())
                assert info["max_fact_ts"] < tmin, f"snapshot leaks a fact at/after t: {info} tmin={tmin}"
                for x in out:
                    coh_map[x[0]] = x
                builds.append(info)
                if j % 100 == 0:
                    log("snapshot", j, "/", len(tasks), json.dumps(info))
        builds.sort(key=lambda x: x["cut"])
        G["BI"] = BI                                   # targets (valid / test) keep the all-train index, as before
        coh = [coh_map[int(i)] for i in samp]          # original sample order -> same "first n reached" protocol
        stale = (tq - cuts[bk]).astype(np.float64)
        ts_info = dict(K_requested=int(a.timesafe), mode="exact per-timestamp (cut = t)" if a.timesafe < 0 else "quantile buckets",
                       n_snapshots=int(len(cuts)), build_s_total=round(float(sum(x["build_s"] for x in builds)), 1),
                       snapshots=builds if len(builds) <= 64 else builds[::max(1, len(builds) // 64)],
                       staleness_t_minus_cut=dict(mean=float(stale.mean()), median=float(np.median(stale)),
                                                  p90=float(np.percentile(stale, 90)), max=float(stale.max()),
                                                  frac_exact=float((stale == 0).mean())),
                       cohort_t_range=[int(tq.min()), int(tq.max())])
        np.savez(f"{out_dir}/cohort_timesafe.npz", samp=samp, t=tq, bucket=bk, cut=cuts[bk],
                 reached=np.array([x[1] is not None for x in coh]),
                 cset=np.array([len(x[1]) if x[1] is not None else 0 for x in coh]))
    else:
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_cohort_chunk, np.array_split(samp, a.workers * 8), chunksize=1)
        coh = [x for p in parts for x in p]
    keep = [x for x in coh if x[1] is not None][:(a.cohort_n if not a.smoke else 1000)]
    summ["bridge_cohort"] = dict(n_sampled=len(coh), reach_rate=float(np.mean([x[1] is not None for x in coh])),
                                 n_train=len(keep), median_cset=float(np.median([len(x[1]) for x in keep])),
                                 median_cset_frac_of_N=float(np.median([len(x[1]) for x in keep]) / N),
                                 index="snapshot-before-t (time-safe)" if a.timesafe else "static all-train (transductive)",
                                 timesafe=ts_info, seconds=round(time.time() - t1, 1))
    log("cohort", json.dumps(summ["bridge_cohort"]))
    nets = []; ntr = []
    for sd_ in SEEDS:
        rg = np.random.default_rng(sd_); torch.manual_seed(sd_)
        Xp, Xn = [], []
        for (_, X, gj) in keep:
            n = len(X)
            if n <= 1: continue
            negj = np.delete(np.arange(n), gj)
            take = rg.choice(negj, size=min(64, len(negj)), replace=False)
            Xp.append(np.repeat(X[gj:gj + 1], len(take), 0)); Xn.append(X[take])
        Xp = np.concatenate(Xp); Xn = np.concatenate(Xn)
        allx = np.concatenate([Xp, Xn]); mu = allx.mean(0); sdv = allx.std(0) + 1e-6
        Pp = torch.tensor((Xp - mu) / sdv); Pn = torch.tensor((Xn - mu) / sdv)
        net = make_net(Xp.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        nn_ = len(Pp)
        for ep in range(40 if not a.smoke else 3):
            perm = torch.randperm(nn_)
            for b0 in range(0, nn_, 4096):
                idx = perm[b0:b0 + 4096]
                loss = -torch.nn.functional.logsigmoid(net(Pp[idx]).squeeze(1) - net(Pn[idx]).squeeze(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        Wt = [net[0].weight.detach().numpy().T, net[0].bias.detach().numpy(), net[2].weight.detach().numpy().T,
              net[2].bias.detach().numpy(), net[4].weight.detach().numpy().T, net[4].bias.detach().numpy()]
        nets.append((Wt, mu.astype(np.float32), sdv.astype(np.float32))); ntr.append(int(nn_))
        torch.save(dict(state_dict=net.state_dict(), mu=mu, sd=sdv, seed=sd_), f"{out_dir}/bridge_ranker_s{sd_}.pt")
    G["nets"] = nets
    summ["bridge_ranker"] = dict(features="X16 + casc (G12 arm5 without event-LM)", seeds=SEEDS, n_pairs=ntr, epochs=40, n_neg=64)
    log("bridge ranker trained")
    # ---------------- target splits
    for split, tr in targets.items():
        t1 = time.time(); n = len(tr)
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        A = dict(Lpat=mm("Lpat", (n, LP_LEN), np.int32), Lpat_u=mm("Lpat_u", (n, LP_LEN), np.float16),
                 Lbr=mm("Lbr", (n, LB_LEN), np.int32), Lbr_s=mm("Lbr_s", (n, LB_LEN), np.float32),
                 Ccov=mm("Ccov", (n, W_COV), np.int32), Xcov=mm("Xcov", (n, W_COV, len(COV_FEATS)), np.float16))
        step = 1000
        tasks = [tr[lo:lo + step] for lo in range(0, n, step)]
        Q = {}
        done = 0
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (res, q) in enumerate(pool.imap(_target_chunk, tasks, chunksize=1)):
                lo = k * step; hi = lo + len(tasks[k])
                for nm, arr in res.items():
                    A[nm][lo:hi] = arr
                for nm, arr in q.items():
                    Q.setdefault(nm, []).append(arr)
                done = hi
                if k % 50 == 0:
                    log(split, done, "/", n)
        for v in A.values():
            v.flush()
        Q = {k: np.concatenate(v) for k, v in Q.items()}
        np.savez(f"{out_dir}/{split}_q.npz", rows=tr, **Q)
        summ[split] = dict(n=n, seconds=round(time.time() - t1, 1),
                           has_pat=float(Q["has_pat"].mean()), has_br=float(Q["has_br"].mean()),
                           gold_in_union=float(Q["g_in_union"].mean()), gold_in_bridge=float(Q["g_in_bridge"].mean()),
                           gold_in_Lpat300=float((Q["gpos_pat"] >= 0).mean()), gold_in_Lbr200=float((Q["gpos_br"] >= 0).mean()),
                           median_br_size=float(np.median(Q["br_size"])), median_union=float(np.median(Q["n_union"])),
                           mean_pat_block=float(Q["n_pat_block"].mean()), mean_br_block=float(Q["n_br_block"].mean()))
        log(split, json.dumps(summ[split]))
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"lists_summary_{a.base}{'_smoke' if a.smoke else ''}.json"); jdump(summ, f"{out_dir}/lists_summary.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
