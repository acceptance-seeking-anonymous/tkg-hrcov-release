"""
harness / fulltest_eval.py -- FULL-TEST stratum-mix evaluator.

Scores EVERY test query (both directions; RE-GCN extrapolation split) with a frozen base, optionally
fused with a pluggable backfill (backfill.py interface) through a gold-free CONFIDENCE-SWITCH rule,
and reports time-aware filtered MRR / H@1 / H@3 / H@10, two-axis strata, the head-damage guardrail
and a paired bootstrap against the base.

ConfidenceSwitch(q, M, B):
  trigger : base top-1 probability p1 <= tau_q, tau_q = q-quantile of p1 over ALL valid queries
            (gold-free; tau fixed on valid, never on test)
  fusion  : final list = base top-M  ++  first B backfill candidates not already in that head
            ++  the rest of the base ranking (minus anything already placed)
  (M = K, the base window size, is the "safe" mode: H@k for k<=K can never change.)
Fused filtered rank is computed exactly when the gold is inside the base top-K or the block, and
conservatively (only visible filter entities subtracted) otherwise -- the same convention as the base.

Selection: (q, M, B) is chosen by MRR on a uniform VALID sample (valid_pool cache); the test grid is
reported only as a labelled diagnostic.

  python3 fulltest_eval.py --dataset ICEWS18 --base tirgn [--backfill g12arm5]
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from oih_common import (Splits, BaseDump, DATASETS, HARNESS, metrics_from_ranks, paired_bootstrap,
                        filter_lookup, jdump)

Q_GRID = [0.05, 0.10, 0.20, 0.30]
M_GRID = [0, 1, 3, 10, None]          # None -> K (safe mode)
B_GRID = [10, 50, 200]
STRATA = {0: "cold", 1: "mid", 2: "head"}


class BaseView:
    """Rows of one split of a base dump, truncated to kref columns, with filtered ranks + strata."""

    def __init__(self, S, D, rows, F, kref):
        self.S = S; self.D = D; self.rows = rows; self.kref = kref
        self.T = D.topk_ids[rows, :kref]; self.Ts = D.topk_scores[rows, :kref]
        self.g = D.sro[rows, 2].astype(np.int64); self.s = D.sro[rows, 0].astype(np.int64)
        self.kg = D.gold_rank_raw[rows].astype(np.int64); self.gs = D.gold_score[rows]
        self.p1 = np.exp(self.Ts[:, 0].astype(np.float64))
        self.fkeys = S.fkey(D.sro[rows, 0], D.sro[rows, 1], D.sro[rows, 3])
        self.F = F
        if "gold_rank_filt" in D.extra:
            # new-style dumps carry the EXACT filtered rank computed from the full score matrix
            self.base_rank = D.extra["gold_rank_filt"][rows].astype(np.int64)
            self.rank_source = "exact gold_rank_filt from dump"
        else:
            # conservative filtered base rank, recomputed with the kref window (== dump K for test rows)
            self.base_rank = np.array([fused_rank(self.T[n], self.Ts[n], int(self.g[n]), int(self.kg[n]), float(self.gs[n]),
                                                  self.others(n), (), 0, 0) for n in range(len(rows))], np.int64)
            self.rank_source = "conservative from top-K window"
            if kref == D.K:   # self-test: fusion code with an empty block must reproduce the base filtered rank
                rc, _, _ = D.filtered_ranks(rows, F)
                assert np.array_equal(rc, self.base_rank), "fused_rank(M=0,B=0) != base filtered rank"
        self.stratum = Splits.stratum(S.freq[self.g])
        self.in_hist = S.in_hist(self.s, self.g)
        self.in_top50 = self.base_rank <= 50

    def others(self, n):
        fs = filter_lookup(self.F, self.fkeys[n])
        return fs[fs != self.g[n]]


def fused_rank(T, Ts, g, kg, gs, others, block_src, M, Bsz):
    """Filtered rank of gold g in  base top-M ++ block ++ base remainder  (see module doc)."""
    T = [int(x) for x in T]
    head = T[:M]; hset = set(head)
    block = []
    if Bsz > 0:
        for b in block_src:
            b = int(b)
            if b < 0:
                break
            if b in hset:
                continue
            block.append(b)
            if len(block) >= Bsz:
                break
    Fs = set(int(x) for x in others)
    if g in hset:
        p = head.index(g)
        return 1 + sum(1 for x in head[:p] if x not in Fs)
    nh = sum(1 for x in head if x not in Fs)
    if g in block:
        j = block.index(g)
        return 1 + nh + sum(1 for x in block[:j] if x not in Fs)
    nb = sum(1 for x in block if x not in Fs)
    U = (hset | set(block) | Fs) - {g}
    pos = {e: i for i, e in enumerate(T)}
    above = sum(1 for e in U if e in pos and Ts[pos[e]] > gs)
    return 1 + nh + nb + (kg - 1) - above


def stratum_table(view, rank_a, rank_b, legacy=None):
    out = {}
    def cell(mask):
        if mask.sum() == 0:
            return None
        ma = metrics_from_ranks(rank_a[mask]); mb = metrics_from_ranks(rank_b[mask])
        return dict(n=int(mask.sum()), base=ma, fused=mb, dMRR=round(mb["MRR"] - ma["MRR"], 6),
                    dH1=round(mb["H1"] - ma["H1"], 6), dH10=round(mb["H10"] - ma["H10"], 6))
    out["ALL"] = cell(np.ones(len(rank_a), bool))
    for sv, sn in STRATA.items():
        out[sn] = cell(view.stratum == sv)
    for pool in (True, False):
        for hist in (True, False):
            key = f"pool={'in' if pool else 'out'}_top50|hist={'yes' if hist else 'no'}"
            m = (view.in_top50 == pool) & (view.in_hist == hist)
            out[key] = cell(m)
            for sv, sn in STRATA.items():
                out[f"{sn}|{key}"] = cell(m & (view.stratum == sv))
    if legacy is not None:
        lm, lname = legacy
        for v in (0, 1):
            out[f"legacy_{lname}={v} (forward only)"] = cell(lm == v)
    return out


def head_damage(view, rr_a, rr_b, B=2000, material=0.001):
    """Per-stratum paired bootstrap.  flag: DAMAGE = CI entirely < 0 AND |delta| >= material;
    'neg-negligible' = significant but below the materiality threshold; HELP / neutral analogous."""
    rows = []
    for sv, sn in list(STRATA.items()) + [(None, "ALL")]:
        m = np.ones(len(rr_a), bool) if sv is None else (view.stratum == sv)
        bt = paired_bootstrap(rr_a[m], rr_b[m], B=B)
        lo, hi = bt["CI95"]
        if hi < 0:
            flag = "DAMAGE" if bt["delta"] <= -material else "neg-negligible"
        elif lo > 0:
            flag = "HELP" if bt["delta"] >= material else "pos-negligible"
        else:
            flag = "neutral"
        rows.append(dict(stratum=sn, n=int(m.sum()), base_MRR=round(float(rr_a[m].mean()), 6),
                         fused_MRR=round(float(rr_b[m].mean()), 6), delta=bt["delta"], CI95=bt["CI95"], flag=flag))
    head = [r for r in rows if r["stratum"] == "head"][0]
    return dict(table=rows, guardrail_pass=bool(head["flag"] != "DAMAGE"),
                rule=f"guardrail passes unless the head-stratum paired-bootstrap 95% CI of dMRR lies entirely below 0 "
                     f"AND |dMRR| >= {material} (materiality threshold)")


def grid_ranks(view, idx_trig, bf_ids, cfgs):
    """rank per config for the triggered subset idx_trig (positions into view rows)."""
    out = {}
    for (M, Bsz) in cfgs:
        Mv = view.kref if M is None else M
        r = np.empty(len(idx_trig), np.int64)
        for n, (vi, bi) in enumerate(idx_trig):
            r[n] = fused_rank(view.T[vi], view.Ts[vi], int(view.g[vi]), int(view.kg[vi]), float(view.gs[vi]),
                              view.others(vi), bf_ids[bi], Mv, Bsz)
        out[(M, Bsz)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="tirgn")
    ap.add_argument("--backfill", default="g12arm5")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    cfg = DATASETS[a.dataset]
    S = Splits(a.dataset); F = S.filter_index()
    D = BaseDump(cfg["dumps"][a.base], S); kref = D.K
    te_rows = D.rows("test")
    TV = BaseView(S, D, te_rows, F, kref)
    res = dict(dataset=a.dataset, base=a.base, dump=cfg["dumps"][a.base], kref=int(kref),
               protocol="time-aware filtered (others at same (s,r',ts) removed), both directions, RE-GCN split; "
                        "conservative for golds beyond the base top-K (visible filter entities only)")
    ma = metrics_from_ranks(TV.base_rank)
    res["row_a"] = dict(name=f"{a.base} alone", **ma, rank_source=TV.rank_source, reference=cfg["ref"].get(a.base),
                        matches_reference=bool(cfg["ref"].get(a.base) and abs(ma["MRR"] - cfg["ref"][a.base]["MRR"]) <= cfg["ref"][a.base].get("tol", 0.0005)))
    # legacy two-axis pool (FeatureBuilder / AbC pool, forward queries only)
    legacy = None
    try:
        if a.dataset == "ICEWS18":
            lp = f"{HARNESS}/../runs/twoaxis_ICEWS18_v2.npz"; lk = "in_pool"
        else:
            lp = f"{HARNESS}/../gdelt_armAD/twoaxis_GDELT.npz"; lk = "in_pool_grounded"
        L = np.load(lp); te_q = S.q["test"]
        key2row = {}
        R = S.R
        for n, i in enumerate(te_rows):
            s_, r_, o_, ts_ = (int(x) for x in D.sro[i])
            if r_ < R:
                key2row.setdefault((s_, r_, o_, ts_), n)
        lm = np.full(len(te_rows), -1, np.int64)
        for qi, v in zip(L["qi"], L[lk]):
            h, r, t, ts = (int(x) for x in te_q[int(qi)])
            n = key2row.get((h, r, t, ts))
            if n is not None:
                lm[n] = int(v)
        legacy = (lm, lk); res["legacy_twoaxis"] = dict(path=lp, key=lk, n_mapped=int((lm >= 0).sum()))
    except Exception as e:
        res["legacy_twoaxis"] = dict(error=repr(e))

    if a.backfill:
        # ---------------- valid tuning ----------------
        vd_path = cfg["dumps"].get(f"{a.base}_valid")
        DV = BaseDump(vd_path, S) if vd_path else D
        vb = np.load(f"{HARNESS}/results/backfill_{a.backfill}_{a.dataset}_valid_pool.npz", allow_pickle=False)
        vsamp = np.load(f"{HARNESS}/cache/{a.dataset}_{a.base}_valid_pool/valid_sample_rows.npy")
        vall = DV.rows("valid"); vp1_all = np.exp(DV.topk_scores[vall, 0].astype(np.float64))
        qmax_pool = json.load(open(f"{HARNESS}/cache/{a.dataset}_{a.base}_test_pool/meta.json"))["q_max"]
        QG = [q for q in Q_GRID if q <= qmax_pool + 1e-9]
        taus = {q: float(np.quantile(vp1_all, q)) for q in QG}
        VV = BaseView(S, DV, vsamp, F, kref)
        vpos = {int(r): n for n, r in enumerate(vsamp)}
        vtrig = [(vpos[int(r)], bi) for bi, r in enumerate(vb["rows"]) if int(r) in vpos]
        cfgs = [(M, Bsz) for M in M_GRID for Bsz in B_GRID]
        vg = grid_ranks(VV, vtrig, vb["ids"], cfgs)
        vi_arr = np.array([t[0] for t in vtrig], np.int64)
        grid = []
        base_v = metrics_from_ranks(VV.base_rank)["MRR"]
        for q in QG:
            sel = VV.p1[vi_arr] <= taus[q]
            for (M, Bsz) in cfgs:
                rk = VV.base_rank.copy(); rk[vi_arr[sel]] = vg[(M, Bsz)][sel]
                mv = metrics_from_ranks(rk)
                grid.append(dict(q=q, tau=taus[q], M=("K" if M is None else M), B=Bsz, valid_MRR=mv["MRR"],
                                 valid_dMRR=round(mv["MRR"] - base_v, 6), valid_dH1=round(mv["H1"] - metrics_from_ranks(VV.base_rank)["H1"], 6),
                                 n_trig_valid=int(sel.sum())))
        best = max(grid, key=lambda g: g["valid_MRR"])
        res["valid_tuning"] = dict(n_valid_sample=int(len(vsamp)), n_valid_pool=int(len(vtrig)), base_valid_MRR=base_v,
                                   taus=taus, grid=grid, selected=best,
                                   any_config_improves_valid=bool(best["valid_dMRR"] > 0))
        # ---------------- test ----------------
        tb = np.load(f"{HARNESS}/results/backfill_{a.backfill}_{a.dataset}_test_pool.npz", allow_pickle=False)
        tpos = {int(r): n for n, r in enumerate(te_rows)}
        ttrig = [(tpos[int(r)], bi) for bi, r in enumerate(tb["rows"]) if int(r) in tpos]
        ti_arr = np.array([t[0] for t in ttrig], np.int64)
        tg = grid_ranks(TV, ttrig, tb["ids"], cfgs)
        test_grid = []
        for q in QG:
            sel = TV.p1[ti_arr] <= taus[q]
            for (M, Bsz) in cfgs:
                rk = TV.base_rank.copy(); rk[ti_arr[sel]] = tg[(M, Bsz)][sel]
                mt = metrics_from_ranks(rk)
                test_grid.append(dict(q=q, M=("K" if M is None else M), B=Bsz, **mt, dMRR=round(mt["MRR"] - ma["MRR"], 6),
                                      n_trig=int(sel.sum())))
        res["test_grid_DIAGNOSTIC_not_for_selection"] = test_grid
        Msel = None if best["M"] == "K" else best["M"]
        sel = TV.p1[ti_arr] <= taus[best["q"]]
        rank_b = TV.base_rank.copy(); rank_b[ti_arr[sel]] = tg[(Msel, best["B"])][sel]
        mb = metrics_from_ranks(rank_b)
        res["row_b"] = dict(name=f"{a.base} + {a.backfill} backfill, confidence-switch (q={best['q']}, M={best['M']}, B={best['B']})",
                            **mb, dMRR=round(mb["MRR"] - ma["MRR"], 6), n_triggered=int(sel.sum()),
                            frac_triggered=round(float(sel.sum() / len(te_rows)), 4))
        # safe-mode companion row (M=K, largest pool, B=200)
        sel_s = TV.p1[ti_arr] <= taus[max(QG)]
        rank_s = TV.base_rank.copy(); rank_s[ti_arr[sel_s]] = tg[(None, 200)][sel_s]
        ms = metrics_from_ranks(rank_s)
        res["row_b_safe_mode"] = dict(name=f"{a.base} + {a.backfill}, safe mode (q={max(QG)}, M=K, B=200)", **ms,
                                      dMRR=round(ms["MRR"] - ma["MRR"], 6))
        rr_a = 1.0 / TV.base_rank; rr_b = 1.0 / rank_b
        res["bootstrap_b_vs_a"] = paired_bootstrap(rr_a, rr_b, B=a.boot)
        res["bootstrap_safe_vs_a"] = paired_bootstrap(rr_a, 1.0 / rank_s, B=a.boot)
        res["head_damage"] = head_damage(TV, rr_a, rr_b, B=a.boot)
        res["strata"] = stratum_table(TV, TV.base_rank, rank_b, legacy)
        # backfill-alone diagnostics on the triggered test pool
        wgr = tb["wide_gold_rank"]; bz = TV.base_rank[ti_arr] > kref
        res["backfill_diagnostics"] = dict(
            n_pool=int(len(ttrig)), frac_pool_gold_beyond_base_K=round(float(bz.mean()), 4),
            wide_reach_pool=round(float((wgr > 0).mean()), 4),
            wide_reach_base_beyond_K=round(float((wgr[bz] > 0).mean()), 4) if bz.any() else None,
            backfill_H50_base_beyond_K=round(float(((wgr[bz] > 0) & (wgr[bz] <= 50)).mean()), 4) if bz.any() else None,
            backfill_H200_base_beyond_K=round(float(((wgr[bz] > 0) & (wgr[bz] <= 200)).mean()), 4) if bz.any() else None)
    else:
        res["strata"] = stratum_table(TV, TV.base_rank, TV.base_rank, legacy)
    res["elapsed_s"] = round(time.time() - t0, 1)
    np.savez(f"fulltest_ranks_{a.dataset}_{a.base}.npz", rows=te_rows, base_rank=TV.base_rank,
             **({"fused_rank": rank_b} if a.backfill else {}), stratum=TV.stratum, in_top50=TV.in_top50, in_hist=TV.in_hist)
    jdump(res, f"fulltest_{a.dataset}_{a.base}.json")
    print(json.dumps({k: res.get(k) for k in ("row_a", "row_b", "row_b_safe_mode")}), flush=True)


if __name__ == "__main__":
    main()
