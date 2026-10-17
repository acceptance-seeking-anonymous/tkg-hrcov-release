#!/usr/bin/env python3
"""pilot_polecat / pc_augarms.py -- 6.5 utility-gated augmentation arms on one base (GPU), run because the
pre-registered stop rule fired (oracle-gate m=3 estimate >= 0.002).  Lean port of tracks/augmentation/track_main.py:

  pools      : target queries = base top-1 prob p1 <= tau, tau = 0.3-quantile of valid-sample p1 (harness q_max 0.3);
               V = valid-sample pool split by time (70% quantile) -> V1 (gate fitting, <= --n_v1 queries) / V2 (selection,
               <= --n_v2 queries); T = EVERY test query in the pool.  Non-pool queries keep the base rank in every arm.
  generators : structural pool = stage-2 bridge top-20 UNION pattern-prior top-20 (pc_lists); the 6.5 peer-import
               and LLM generators are not used (see report).
  S1         : every V1 (query, evidence) pair injected ALONE (m=1) -> realised u_ll = dlog p(gold), u_rr = d(1/rank)
  S2         : utility gate = HistGradientBoostingRegressor on 19 GOLD-FREE pair features, 3 seeds (80% query subsample);
               primary target u_ll (predicted delta log-likelihood), sensitivity target u_rr
  S3         : V2 grid theta x per-query cap B in {1,3} x multiplicity m in {1,3}; selection = best V2 MRR among configs
               admitting >= 1% of V2 queries (valid only)
  S4         : test: selected utility gates + MATCHED budget (same #pairs, same B, same m) random / triadic-closure
               (Adamic-Adar, train graph) / uncertainty 4u(1-u) gates, ungated completion (same B, m), perturbation control
               (same admitted queries, evidence replaced by uniform random entities)
Injection mechanism: rescore_lib.Rescorer (last-snapshot, per-query block-diagonal copy; LogCL also global copy counts).
Output: work/augarms_<base>/ranks_test.npz (full-test ranks per arm), augarms_<base>.json
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (TRACKS, P, DS, DUMPS, WORK, KC, LP_LEN, Splits, BaseDump, split_queries, metrics_from_ranks, jdump, log_fn)
sys.path.insert(0, f"{TRACKS}/augmentation/code"); sys.path.insert(0, f"{TRACKS}/residual/code")
import rescore_lib
rescore_lib.HARNESS = P
from rescore_lib import Rescorer, make_filter
from rs_features import Hist

THETAS = {"ll": (0.0, 0.02, 0.05, 0.1, 0.2, 0.4), "rr": (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)}
BS = (1, 3); MS = (1, 3); SEEDS = (0, 1, 2); GK = 20
FEATS = ["rk_br", "rk_pat", "sc_br", "sc_pat", "n_gens", "lf_c", "lf_s", "ltrip", "lpair", "lsr", "aa", "lrel", "inv",
         "b_lrank", "b_lp", "b_p1", "b_ent", "b_margin", "b_unc"]


def within_rank(q, score):
    o = np.lexsort((-score, q)); qs = q[o]
    if len(qs) == 0:
        return np.zeros(0, np.int64)
    start = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    gs = np.repeat(start, np.diff(np.r_[start, len(qs)]))
    r = np.empty(len(q), np.int64); r[o] = np.arange(len(q)) - gs
    return r


def group_pairs(P, adm):
    """admitted pair indices -> (unique query slots, list of entity arrays), O(n log n)."""
    adm = np.asarray(adm, np.int64)
    if len(adm) == 0:
        return np.zeros(0, np.int64), []
    adm = adm[np.argsort(P["pq"][adm], kind="stable")]
    qq = P["pq"][adm]
    qa, start = np.unique(qq, return_index=True)
    return qa, np.split(P["pc"][adm], start[1:])


class Batch:
    """collect many (arm, copies) request sets, rescore them in ONE Rescorer.run (LogCL history roll once)."""

    def __init__(self):
        self.names, self.q, self.c, self.off = [], [], [], []

    def add(self, name, qidx, ents_list, m):
        self.names.append((name, len(self.q), len(qidx)))
        for j, ents in zip(qidx, ents_list):
            self.q.append(int(j)); self.c.append(np.repeat(np.asarray(ents, np.int64), m))

    def run(self, RS, Qt, filt, tag):
        if not self.q:
            return {}
        off = np.zeros(len(self.q) + 1, np.int64); off[1:] = np.cumsum([len(x) for x in self.c])
        cc = np.concatenate(self.c) if off[-1] else np.zeros(0, np.int64)
        out = RS.run(np.array(self.q), Qt[:, 0], Qt[:, 1], Qt[:, 2], Qt[:, 3], off, cc, filt, log_every=500, tag=tag)
        res = {}
        for nm, a, n in self.names:
            res[nm] = dict(q=np.array(self.q[a:a + n]), rank=out["rank_filt"][a:a + n], lp=out["logp_gold"][a:a + n])
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n_v1", type=int, default=8000)
    ap.add_argument("--n_v2", type=int, default=6000)
    ap.add_argument("--q_pool", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N, R = S.n_ent, S.R
    D = BaseDump(DUMPS[a.base], S)
    Hh = Hist(S)
    from bridge_features import BridgeIndex
    BI = BridgeIndex(S.q["train"], N, R)
    # sorted unique train neighbours per entity (Adamic-Adar)
    rowid = np.repeat(np.arange(N), np.diff(BI.uindptr)); o = np.lexsort((BI.unb, rowid)); NB = BI.unb[o].astype(np.int64)
    invlogdeg = 1.0 / np.log(np.maximum(BI.deg, 2)).astype(np.float64)
    rfreq = np.bincount(np.concatenate([S.q["train"][:, 1], S.q["train"][:, 1] + R]), minlength=2 * R)
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    sfx = "_smoke" if a.smoke else ""
    ld = f"{WORK}/lists_{a.base}{sfx}"; lk = f"{WORK}/lookup_{a.base}{sfx}"
    sp_rows = {"valid": D.rows("valid")[vidx], "test": D.rows("test")}
    if a.smoke:
        sp_rows = {k: v[:a.smoke] for k, v in sp_rows.items()}
    p1v = np.exp(D.topk_scores[sp_rows["valid"], 0].astype(np.float64)); tau = float(np.quantile(p1v, a.q_pool))
    res = dict(base=a.base, host="serverA (GPU, tkg-gpu:latest)", tau=tau, q_pool=a.q_pool, feats=FEATS)

    def pool_pairs(split, pos):
        """pairs (query-slot, entity) for split positions pos (indices into the split arrays)."""
        rows = sp_rows[split][pos]; Q = D.sro[rows].astype(np.int64)
        Lb = np.load(f"{ld}/{split}_Lbr.npy", mmap_mode="r")[pos][:, :GK].astype(np.int64)
        Lp = np.load(f"{ld}/{split}_Lpat.npy", mmap_mode="r")[pos][:, :GK].astype(np.int64)
        Sb = np.load(f"{ld}/{split}_Lbr_s.npy", mmap_mode="r")[pos][:, :GK].astype(np.float64)
        Sp = np.load(f"{ld}/{split}_Lpat_u.npy", mmap_mode="r")[pos][:, :GK].astype(np.float64)
        LK = np.load(f"{lk}/{split}.npz")
        lpr = LK["lp_req"][pos]; rkr = LK["rk_req"][pos]
        reqL = np.concatenate([np.load(f"{ld}/{split}_Lpat.npy", mmap_mode="r")[pos], np.load(f"{ld}/{split}_Lbr.npy", mmap_mode="r")[pos]], 1)
        pq, pc, rb, rp, sb, spp, blp, brk = [], [], [], [], [], [], [], []
        for j in range(len(pos)):
            d = {}
            for k in range(GK):
                e = int(Lb[j, k])
                if e >= 0 and e != Q[j, 0]: d.setdefault(e, [0, 0, 0.0, 0.0]); d[e][0] = k + 1; d[e][2] = Sb[j, k]
                e = int(Lp[j, k])
                if e >= 0 and e != Q[j, 0]: d.setdefault(e, [0, 0, 0.0, 0.0]); d[e][1] = k + 1; d[e][3] = Sp[j, k]
            rq = {int(x): kk for kk, x in enumerate(reqL[j]) if x >= 0}
            for e, v in d.items():
                pq.append(j); pc.append(e); rb.append(v[0]); rp.append(v[1]); sb.append(v[2]); spp.append(v[3])
                kk = rq[e]; blp.append(lpr[j, kk]); brk.append(rkr[j, kk])
        pq = np.array(pq, np.int64); pc = np.array(pc, np.int64)
        s, r, t = Q[pq, 0], Q[pq, 1], Q[pq, 3]
        hf = Hh.feats(s, r, t, pc[:, None])
        aa = np.zeros(len(pq))
        for i in range(len(pq)):
            a1 = NB[BI.uindptr[s[i]]:BI.uindptr[s[i] + 1]]; a2 = NB[BI.uindptr[pc[i]]:BI.uindptr[pc[i] + 1]]
            if len(a1) and len(a2):
                z = np.intersect1d(a1, a2, assume_unique=True); aa[i] = invlogdeg[z].sum()
        tks = D.topk_scores[rows].astype(np.float64)
        p1 = np.exp(tks[:, 0]); pn = np.exp(tks[:, :KC]); pn /= pn.sum(1, keepdims=True); ent = -(pn * np.log(pn + 1e-12)).sum(1)
        X = np.stack([np.array(rb), np.array(rp), np.array(sb), np.array(spp), (np.array(rb) > 0).astype(float) + (np.array(rp) > 0),
                      np.log1p(S.freq[pc]), np.log1p(S.freq[s]), np.log1p(hf["ntrip"][:, 0]), np.log1p(hf["ndyad"][:, 0]),
                      np.log1p(hf["nquery"]), aa, np.log1p(rfreq[r]), (r >= R).astype(float),
                      np.log(np.maximum(np.array(brk, np.float64), 1)), np.maximum(np.array(blp, np.float64), -40), p1[pq], ent[pq],
                      (tks[:, 0] - tks[:, 1])[pq], (4 * p1 * (1 - p1))[pq]], 1).astype(np.float32)
        return dict(Q=Q, rows=rows, pq=pq, pc=pc, X=X, aa=aa, gr=np.where(np.array(rb) > 0, np.array(rb), 99) + np.where(np.array(rp) > 0, np.array(rp), 99),
                    p1=p1, gold=Q[:, 2], r_dump=D.extra["gold_rank_filt"][rows].astype(np.int64))

    # ---------------- pools
    vts = D.sro[sp_rows["valid"], 3]; uts = np.unique(vts); cut = uts[int(0.7 * len(uts))]
    vpool = np.flatnonzero(p1v <= tau)
    rng = np.random.default_rng(11)
    v1 = vpool[vts[vpool] < cut]; v2 = vpool[vts[vpool] >= cut]
    if a.smoke:
        v1, v2 = vpool[::2], vpool[1::2]
    v1 = np.sort(rng.choice(v1, min(a.n_v1, len(v1)), replace=False)); v2 = np.sort(rng.choice(v2, min(a.n_v2, len(v2)), replace=False))
    p1t = np.exp(D.topk_scores[sp_rows["test"], 0].astype(np.float64)); tpool = np.flatnonzero(p1t <= tau)
    res["pools"] = dict(valid_pool_frac=float(len(vpool) / len(p1v)), n_v1=int(len(v1)), n_v2=int(len(v2)),
                        test_pool_n=int(len(tpool)), test_pool_frac=float(len(tpool) / len(p1t)))
    log("pools", json.dumps(res["pools"]))
    PV1 = pool_pairs("valid", v1); PV2 = pool_pairs("valid", v2); PT = pool_pairs("test", tpool)
    log("pairs", len(PV1["pq"]), len(PV2["pq"]), len(PT["pq"]))
    for nm_, PP in (("V1", PV1), ("V2", PV2), ("T", PT)):
        g = PP["pc"] == PP["gold"][PP["pq"]]
        res["pools"][f"{nm_}_pairs"] = int(len(PP["pq"])); res["pools"][f"{nm_}_gold_pair_frac"] = float(g.mean())
        res["pools"][f"{nm_}_queries_with_gold_in_pool"] = float(np.bincount(PP["pq"][g], minlength=len(PP["Q"])).astype(bool).mean())
    import torch
    RS = Rescorer(a.base, DS, device=("cuda" if torch.cuda.is_available() else "cpu"), copies_budget=6e6)
    filt = make_filter(DS)
    ck_dir = f"{WORK}/augarms_{a.base}{sfx}"; os.makedirs(ck_dir, exist_ok=True)
    ck1 = f"{ck_dir}/S1_cache.npz"
    # ---------------- S1: V1 singles + no-injection copies (V1, V2)
    if os.path.exists(ck1):
        z = np.load(ck1); o1 = {"v1_noinj": dict(rank=z["r0v1"], lp=z["lp0v1"]), "v1_single": dict(rank=z["r1"], lp=z["lp1"])}
        o2 = {"v2_noinj": dict(rank=z["r0v2"])}; log("S1 loaded from cache")
    else:
        o1 = o2 = None
    bt = Batch()
    if o1 is None:
        bt.add("v1_noinj", np.arange(len(PV1["Q"])), [[]] * len(PV1["Q"]), 1)
        bt.add("v1_single", PV1["pq"], [[c] for c in PV1["pc"]], 1)
        o1 = bt.run(RS, PV1["Q"], filt, "S1")
        bt2 = Batch(); bt2.add("v2_noinj", np.arange(len(PV2["Q"])), [[]] * len(PV2["Q"]), 1); o2 = bt2.run(RS, PV2["Q"], filt, "S1v2")
        np.savez(ck1, r0v1=o1["v1_noinj"]["rank"], lp0v1=o1["v1_noinj"]["lp"], r1=o1["v1_single"]["rank"], lp1=o1["v1_single"]["lp"],
                 r0v2=o2["v2_noinj"]["rank"])
    r0v1 = o1["v1_noinj"]["rank"]; lp0v1 = o1["v1_noinj"]["lp"]
    u_ll = o1["v1_single"]["lp"] - lp0v1[PV1["pq"]]; u_rr = 1.0 / o1["v1_single"]["rank"] - 1.0 / r0v1[PV1["pq"]]
    r0v2 = o2["v2_noinj"]["rank"]
    res["S1"] = dict(n_pairs=int(len(u_ll)), mean_u_ll=float(u_ll.mean()), mean_u_ll_gold=float(u_ll[PV1["pc"] == PV1["gold"][PV1["pq"]]].mean()) if (PV1["pc"] == PV1["gold"][PV1["pq"]]).any() else None,
                     mean_u_ll_nongold=float(u_ll[PV1["pc"] != PV1["gold"][PV1["pq"]]].mean()),
                     repro_noinj_vs_dump_V1=float(np.mean(r0v1 == PV1["r_dump"])), repro_noinj_vs_dump_V2=float(np.mean(r0v2 == PV2["r_dump"])))
    log("S1", json.dumps(res["S1"]))
    # ---------------- S2: gates
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    preds = {}
    gold2 = (PV2["pc"] == PV2["gold"][PV2["pq"]]).astype(int)
    res["S2"] = {}
    for var, y in (("ll", u_ll), ("rr", u_rr)):
        for sd in SEEDS:
            rg = np.random.default_rng(sd); qs = np.unique(PV1["pq"]); keepq = rg.choice(qs, int(0.8 * len(qs)), replace=False)
            m = np.isin(PV1["pq"], keepq)
            mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=sd)
            mdl.fit(PV1["X"][m], y[m])
            preds[(var, sd, "V2")] = mdl.predict(PV2["X"]); preds[(var, sd, "T")] = mdl.predict(PT["X"])
            res["S2"][f"{var}_s{sd}"] = dict(gold_identity_AUC_V2=float(roc_auc_score(gold2, preds[(var, sd, "V2")])) if 0 < gold2.sum() < len(gold2) else None)
    log("S2", json.dumps(res["S2"]))
    # ---------------- S3: V2 grid
    def admit(P, score, theta, B):
        sel = np.flatnonzero(score > theta)
        wr = within_rank(P["pq"][sel], score[sel])
        return sel[wr < B]
    bt = Batch(); cfgs = []
    for var in ("ll", "rr"):
        for sd in SEEDS:
            for th in THETAS[var]:
                for B in BS:
                    adm = admit(PV2, preds[(var, sd, "V2")], th, B)
                    if len(adm) == 0: continue
                    qa, ents = group_pairs(PV2, adm)
                    if len(qa) < 0.01 * len(PV2["Q"]): continue
                    for m in MS:
                        nm = f"{var}|{sd}|{th}|{B}|{m}"; cfgs.append((nm, var, sd, th, B, m, len(adm), len(qa)))
                        bt.add(nm, qa, ents, m)
    ck3 = f"{ck_dir}/S3_cache.npz"
    if os.path.exists(ck3):
        z = np.load(ck3, allow_pickle=True); o3 = z["o3"].item(); log("S3 loaded from cache")
    else:
        o3 = bt.run(RS, PV2["Q"], filt, "S3")
        np.savez(ck3, o3=np.array({k: dict(q=v["q"], rank=v["rank"]) for k, v in o3.items()}, dtype=object))
    grid = []
    for (nm, var, sd, th, B, m, npairs, nq) in cfgs:
        r = r0v2.copy(); r[o3[nm]["q"]] = o3[nm]["rank"]
        grid.append(dict(var=var, seed=sd, theta=th, B=B, m=m, n_pairs=int(npairs), n_queries=int(nq),
                         dMRR_V2=float((1.0 / r).mean() - (1.0 / r0v2).mean())))
    res["S3_grid"] = grid
    sel = {}
    for var in ("ll", "rr"):
        for sd in SEEDS:
            gg = [g for g in grid if g["var"] == var and g["seed"] == sd]
            if gg:
                sel[(var, sd)] = max(gg, key=lambda g: g["dMRR_V2"])
    res["S3_selected"] = {f"{k[0]}_s{k[1]}": v for k, v in sel.items()}
    log("S3", json.dumps(res["S3_selected"]))
    # ---------------- S4: test arms
    bt = Batch(); arms = {}
    nTQ = len(PT["Q"]); rngT = np.random.default_rng(5)
    def ents_of(adm):
        return group_pairs(PT, adm)
    glT = PT["pc"] == PT["gold"][PT["pq"]]
    def add_arm(name, adm, m):
        qa, ents = ents_of(adm)
        arms[name] = dict(n_pairs=int(len(adm)), n_queries=int(len(qa)), m=m, gold_frac_admitted=float(glT[adm].mean()) if len(adm) else None)
        bt.add(name, qa, ents, m)
    for (var, sd), g in sel.items():
        if var == "rr" and sd != SEEDS[0]:        # sensitivity target: one seed on test (copy budget)
            continue
        adm = admit(PT, preds[(var, sd, "T")], g["theta"], g["B"]); add_arm(f"util_{var}_s{sd}", adm, g["m"])
        if var != "ll" or sd != SEEDS[0]:          # matched comparators against the seed-0 primary gate (copy budget)
            continue
        K_ = len(adm); B = g["B"]; m = g["m"]
        # random matched: random scores, cap B, take K_ best
        rs_ = rngT.random(len(PT["pq"])); wr = within_rank(PT["pq"], rs_); cand = np.flatnonzero(wr < B)
        add_arm(f"random_matched_s{sd}", cand[np.argsort(-rs_[cand])[:K_]], m)
        wr = within_rank(PT["pq"], PT["aa"]); cand = np.flatnonzero(wr < B)
        add_arm(f"aa_matched_s{sd}", cand[np.argsort(-PT["aa"][cand], kind="stable")[:K_]], m)
        unc = (4 * PT["p1"] * (1 - PT["p1"]))[PT["pq"]]; wr = within_rank(PT["pq"], -PT["gr"].astype(float)); cand = np.flatnonzero(wr < B)
        add_arm(f"unc_matched_s{sd}", cand[np.lexsort((PT["gr"][cand], -unc[cand]))[:K_]], m)
        # perturbation: same admitted (query, slot) structure, random entities
        qa, ents = ents_of(adm)
        pert = [rngT.integers(0, N, len(e)) for e in ents]
        arms[f"perturb_s{sd}"] = dict(n_pairs=int(K_), n_queries=int(len(qa)), m=m); bt.add(f"perturb_s{sd}", qa, pert, m)
        wr = within_rank(PT["pq"], -PT["gr"].astype(float)); add_arm("ungated_sameBm", np.flatnonzero(wr < B), m)
    o4 = bt.run(RS, PT["Q"], filt, "S4")
    Dte = sp_rows["test"]; r_base_full = D.extra["gold_rank_filt"][Dte].astype(np.int64)
    ranks = dict(base=r_base_full.astype(np.int32))
    for nm in arms:
        r = r_base_full.copy(); r[tpool[o4[nm]["q"]]] = o4[nm]["rank"]; ranks[nm] = r.astype(np.int32)
        arms[nm]["test"] = metrics_from_ranks(r); arms[nm]["dMRR_vs_base"] = float((1.0 / r).mean() - (1.0 / r_base_full).mean())
        gl = PT["pc"] == PT["gold"][PT["pq"]]
    res["S4_arms"] = arms
    od = f"{WORK}/augarms_{a.base}{sfx}"; os.makedirs(od, exist_ok=True)
    np.savez(f"{od}/ranks_test.npz", tpool=tpool, **ranks)
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"augarms_{a.base}{sfx}.json"); jdump(res, f"{od}/augarms_{a.base}.json")
    os.system(f"chmod -R a+rX {od}")
    log("ALL DONE", json.dumps({k: v["dMRR_vs_base"] for k, v in arms.items()}))


if __name__ == "__main__":
    main()
