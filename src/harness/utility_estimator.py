"""
harness / utility_estimator.py -- the SHARED conditional-utility estimator.

    u_gold(e | q) = log p(gold | base + e) - log p(gold | base)          (frozen base, default TiRGN)

"Evidence" e is an additive logit bump `bump` on a set of entities: either an abstract PATTERN
(granularity role / country / role_country of the event-LM vocab -> every entity with that signature)
or an explicit ENTITY SET (e.g. a backfill block).  Two partitions:
  mode="window" : softmax over the base top-W window (W=50) + the gold appended if outside it.
                  == build_utility_signal.py "score-splice dlog-lik" (u_ll); reproduced exactly on ICEWS18.
  mode="full"   : the base's own normalised distribution over ALL entities (dump scores are log-probs):
                  exact p for the top-K entities and the gold, tail-uniform p for everything else,
                  u = bump*[gold in e] - log(1 + sum_{x in e} p_x (exp(bump)-1)).   (recommended for
                  coverage evidence, because it prices entities the window cannot see.)
The legacy rank-shift u_rank of build_utility_signal.py is reproduced too (window only; note that it
does NOT bump a gold that lies outside the window, unlike u_ll -- kept only for continuity).

u_pred(e | q): gold-free predictor (sklearn HistGradientBoosting) over the 16 gold-free query+pattern
features of extract_goldfree_features.py (base distribution shape, subject/relation frequency, pattern
mass / match count / first position inside the base window).  Classifier = P(pattern is the gold's own
pattern), regressor = E[u_ll averaged over the bump grid].  Trained on TRAIN-split forward queries,
validated on TEST slices.

CLI (builds the per-dataset signal + features + predictor, writes utility_estimator_<DS>.json):
    python3 utility_estimator.py --dataset GDELT
Outputs under <HARNESS>/cache/utility_<DS>_<base>/ :
    rows.npy (dump rows, forward queries, all splits), split.npy, gold_freq.npy, gold_in_top50.npy,
    u_ll_pos/neg.npy [n,4,3], u_rank_pos/neg.npy [n,4,3], u_full_pos/neg.npy [n,4,3],
    X_pos/X_neg.npy [n,3,16], decoy_role/decoy_country.npy, predictor.pkl
"""
import os, sys, json, time, argparse, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from collections import Counter
from oih_common import Splits, BaseDump, DATASETS, HARNESS, jdump

BUMP_GRID = np.array([0.5, 1.0, 2.0, 4.0])
GRANS = ["role", "country", "role_country"]
FEAT_NAMES = ["base_entropy", "top1", "margin12", "margin1_10", "score_std", "eff_support",
              "log_subj_freq", "log_rel_freq", "role_div", "country_div",
              "is_role", "is_country", "is_role_country", "pattern_mass", "pattern_nmatch_frac", "pattern_top_pos"]


class UtilityEstimator:
    def __init__(self, dataset, base="tirgn", window=50, dump=None, splits=None):
        cfg = DATASETS[dataset]
        self.dataset = dataset; self.base = base; self.W = window
        self.S = splits or Splits(dataset)
        self.D = dump or BaseDump(cfg["dumps"][base], self.S)
        v = np.load(cfg["lm_vocab"], allow_pickle=True)
        self.role = v["ent_role"].astype(np.int64); self.country = v["ent_country"].astype(np.int64)
        self.N = self.S.n_ent; self.K = self.D.K
        self.role_hist = np.bincount(self.role); self.country_hist = np.bincount(self.country)
        rc = self.role * (self.country.max() + 1) + self.country
        self.rc_code = rc; self.rc_hist = Counter(rc.tolist())
        tr = self.S.q["train"]
        self.sfreq = np.bincount(tr[:, 0], minlength=self.N)          # subject-only freq (as extract_goldfree)
        self.rfreq = np.bincount(tr[:, 1], minlength=2 * self.S.R)
        self.predictor = None

    # ------------------------------------------------------------------ masks
    def pattern_mask(self, ids, gran, role, country):
        """ids [n,k] entity ids; role/country [n] pattern signature -> bool [n,k]."""
        r = self.role[ids]; c = self.country[ids]
        if gran == 0: return r == role[:, None]
        if gran == 1: return c == country[:, None]
        return (r == role[:, None]) & (c == country[:, None])

    def pattern_count(self, gran, role, country):
        """number of entities in the vocabulary matching each pattern [n]."""
        if gran == 0: return self.role_hist[np.clip(role, 0, len(self.role_hist) - 1)] * (role >= 0)
        if gran == 1: return self.country_hist[np.clip(country, 0, len(self.country_hist) - 1)] * (country >= 0)
        code = role * (self.country.max() + 1) + country
        return np.array([self.rc_hist.get(int(x), 0) for x in code]) * (role >= 0)

    # ------------------------------------------------------------------ u_gold
    def u_gold(self, rows, gran, role, country, bumps=BUMP_GRID, mode="window", gold_matches=None):
        """Pattern evidence. rows [n] dump rows. Returns u [n, len(bumps)].
        gold_matches: bool [n], whether the gold carries the pattern (computed if None)."""
        D = self.D; W = self.W
        ids = D.topk_ids[rows].astype(np.int64); sc = D.topk_scores[rows].astype(np.float64)
        g = D.sro[rows, 2].astype(np.int64); gs = D.gold_score[rows].astype(np.float64)
        gm = (self.pattern_mask(g[:, None], gran, role, country)[:, 0] if gold_matches is None else gold_matches).astype(np.float64)
        if mode == "window":
            idw = ids[:, :W]; scw = sc[:, :W]
            m = self.pattern_mask(idw, gran, role, country).astype(np.float64)
            inwin = (idw == g[:, None]); gin = inwin.any(1)
            # append gold column when it is outside the window (u_ll convention)
            logits = np.concatenate([scw, np.where(gin, -np.inf, gs)[:, None]], 1)
            mm = np.concatenate([m, np.where(gin, 0.0, gm)[:, None]], 1)
            gidx = np.where(gin, inwin.argmax(1), W)
            mx = logits.max(1, keepdims=True)
            base_lse = np.log(np.exp(logits - mx).sum(1)) + mx[:, 0]
            base_ll = logits[np.arange(len(rows)), gidx] - base_lse
            out = np.zeros((len(rows), len(bumps)))
            for bi, b in enumerate(bumps):
                sl = logits + b * mm
                mx2 = sl.max(1, keepdims=True)
                lse = np.log(np.exp(sl - mx2).sum(1)) + mx2[:, 0]
                out[:, bi] = (sl[np.arange(len(rows)), gidx] - lse) - base_ll
            return out
        # full partition
        p = np.exp(sc)                                            # exact probs of the top-K
        m = self.pattern_mask(ids, gran, role, country)
        intop = (ids == g[:, None]).any(1)
        tail_mass = np.clip(1.0 - p.sum(1) - np.where(intop, 0.0, np.exp(gs)), 0.0, 1.0)
        n_tail = self.N - self.K - (~intop).astype(np.int64)
        p_tail = tail_mass / np.maximum(n_tail, 1)
        n_match_all = self.pattern_count(gran, role, country).astype(np.float64)
        n_match_tail = np.maximum(n_match_all - m.sum(1) - np.where(intop, 0.0, gm), 0.0)
        mass_top = (p * m).sum(1)
        mass_e = mass_top + n_match_tail * p_tail + np.where(intop, 0.0, gm * np.exp(gs))
        out = np.zeros((len(rows), len(bumps)))
        for bi, b in enumerate(bumps):
            out[:, bi] = b * gm - np.log1p(mass_e * (np.exp(b) - 1.0))
        return out

    def u_gold_entities(self, rows, ent_lists, bump=1.0):
        """Entity-set evidence (full partition). ent_lists [n,B] ids (-1 pad). Returns u [n]."""
        D = self.D
        ids = D.topk_ids[rows].astype(np.int64); sc = D.topk_scores[rows].astype(np.float64)
        g = D.sro[rows, 2].astype(np.int64); gs = D.gold_score[rows].astype(np.float64)
        p = np.exp(sc); intop = (ids == g[:, None]).any(1)
        tail_mass = np.clip(1.0 - p.sum(1) - np.where(intop, 0.0, np.exp(gs)), 0.0, 1.0)
        p_tail = tail_mass / np.maximum(self.N - self.K - (~intop), 1)
        out = np.zeros(len(rows))
        for n in range(len(rows)):
            e = ent_lists[n]; e = e[e >= 0]
            if len(e) == 0:
                continue
            pos = {int(x): j for j, x in enumerate(ids[n])}
            pe = np.array([p[n, pos[int(x)]] if int(x) in pos else (np.exp(gs[n]) if int(x) == g[n] else p_tail[n]) for x in e])
            gm = float(np.any(e == g[n]))
            out[n] = bump * gm - np.log1p(pe.sum() * (np.exp(bump) - 1.0))
        return out

    def u_rank_legacy(self, rows, gran, role, country, bumps=BUMP_GRID):
        """build_utility_signal.rank_shift, vectorised (window W)."""
        D = self.D; W = self.W
        ids = D.topk_ids[rows, :W].astype(np.int64); scs = D.topk_scores[rows, :W].astype(np.float64)
        g = D.sro[rows, 2].astype(np.int64); gs = D.gold_score[rows].astype(np.float64)
        raw = np.maximum(1, D.gold_rank_raw[rows].astype(np.int64)); base_recip = 1.0 / raw
        m = self.pattern_mask(ids, gran, role, country).astype(np.float64)
        inwin = ids == g[:, None]; gin = inwin.any(1); gpos = inwin.argmax(1)
        ar = np.arange(len(rows))
        out = np.zeros((len(rows), len(bumps)))
        above_base = (scs > gs[:, None]).sum(1) - np.where(gin, (scs[ar, gpos] > gs), False)
        for bi, b in enumerate(bumps):
            new = scs + b * m
            gold_new = gs + np.where(gin, b * m[ar, gpos], 0.0)
            above_new = (new > gold_new[:, None]).sum(1) - np.where(gin, new[ar, gpos] > gold_new, False)
            fr_in = np.maximum(1, raw - above_base + above_new)
            added = (new > gs[:, None]).sum(1) - (scs > gs[:, None]).sum(1)
            fr_out = np.maximum(1, raw + added)
            fr = np.where(gin, fr_in, fr_out)
            out[:, bi] = 1.0 / fr - base_recip
        return out

    # ------------------------------------------------------------------ gold-free features
    def goldfree_features(self, rows, gran, role, country):
        """[n,16] gold-free query+pattern features (extract_goldfree_features.py, vectorised, window W)."""
        D = self.D; W = self.W
        scs = D.topk_scores[rows, :W].astype(np.float64); ids = D.topk_ids[rows, :W].astype(np.int64)
        x = scs - scs.max(1, keepdims=True); e = np.exp(x); p = e / e.sum(1, keepdims=True)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        top1 = scs[:, 0]; m12 = scs[:, 0] - scs[:, 1]; m110 = scs[:, 0] - scs[:, 9]
        sstd = scs.std(1); eff = np.exp(ent)
        h = D.sro[rows, 0].astype(np.int64); r = D.sro[rows, 1].astype(np.int64)
        qsf = np.log1p(self.sfreq[h]); qrf = np.log1p(self.rfreq[r])
        cr = self.role[ids]; cc = self.country[ids]
        def ndistinct(a):
            s_ = np.sort(a, 1); return 1 + (np.diff(s_, axis=1) != 0).sum(1)
        rdiv = ndistinct(cr) / 50.0; cdiv = ndistinct(cc) / 50.0
        mask = self.pattern_mask(ids, gran, role, country)
        mass = (p * mask).sum(1); nm = mask.sum(1) / 50.0
        anym = mask.any(1); topm = np.where(anym, mask.argmax(1) / 50.0, 1.0)
        oh = np.zeros((len(rows), 3)); oh[:, gran] = 1.0
        return np.column_stack([ent, top1, m12, m110, sstd, eff, qsf, qrf, rdiv, cdiv, oh, mass, nm, topm]).astype(np.float32)

    # ------------------------------------------------------------------ decoys
    def decoy_patterns(self, rows, seed=0):
        """One decoy (role, country) per row: a role (country) present in the window but != the gold's,
        drawn uniformly over DISTINCT values (as build_utility_signal / extract_goldfree), shared by
        the utility target and the feature row (fixes the rng(0)/rng(1) mismatch of the originals)."""
        rng = np.random.default_rng(seed)
        D = self.D; W = self.W
        ids = D.topk_ids[rows, :W].astype(np.int64); g = D.sro[rows, 2].astype(np.int64)
        out = []
        for vals, gval in ((self.role[ids], self.role[g]), (self.country[ids], self.country[g])):
            eq_prev = np.zeros(vals.shape, bool)
            for j in range(1, W):
                eq_prev[:, j] = (vals[:, :j] == vals[:, j:j + 1]).any(1)
            ok = (~eq_prev) & (vals != gval[:, None])
            key = rng.random(vals.shape); key[~ok] = -1
            j = key.argmax(1); has = ok.any(1)
            out.append(np.where(has, vals[np.arange(len(rows)), j], -1))
        return out[0], out[1]

    # ------------------------------------------------------------------ predictor
    def fit_predictor(self, X_pos, X_neg, u_pos, u_neg, tr_mask, max_train=300000, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        rng = np.random.default_rng(seed)
        tr = np.where(tr_mask)[0]
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        Xa = np.concatenate([X_pos[tr].reshape(-1, 16), X_neg[tr].reshape(-1, 16)])
        ya = np.concatenate([np.ones(len(tr) * 3), np.zeros(len(tr) * 3)]).astype(np.int8)
        ua = np.concatenate([u_pos[tr].mean(1).reshape(-1), u_neg[tr].mean(1).reshape(-1)])
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(Xa, ya)
        reg = HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=0).fit(Xa, ua)
        self.predictor = dict(clf=clf, reg=reg, n_train_queries=int(len(tr)), feat_names=FEAT_NAMES)
        return self.predictor

    def u_pred(self, rows, gran, role, country):
        """gold-free prediction for pattern evidence: (P(helpful/gold-pattern), E[u_ll])."""
        X = self.goldfree_features(rows, gran, role, country)
        return self.predictor["clf"].predict_proba(X)[:, 1], self.predictor["reg"].predict(X)

    def load_predictor(self, path=None):
        path = path or f"{HARNESS}/cache/utility_{self.dataset}_{self.base}/predictor.pkl"
        self.predictor = pickle.load(open(path, "rb")); return self.predictor


def build(dataset, base="tirgn", chunk=200000):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    t0 = time.time()
    U = UtilityEstimator(dataset, base)
    S = U.S; D = U.D; R = S.R
    rows = np.where(D.sro[:, 1] < R)[0]                      # forward queries, all splits in the dump
    n = len(rows)
    out_dir = f"{HARNESS}/cache/utility_{dataset}_{base}"; os.makedirs(out_dir, exist_ok=True)
    nB = len(BUMP_GRID)
    arrs = {k: np.zeros((n, nB, 3), np.float32) for k in ("u_ll_pos", "u_ll_neg", "u_rank_pos", "u_rank_neg", "u_full_pos", "u_full_neg")}
    Xp = np.zeros((n, 3, 16), np.float32); Xn = np.zeros((n, 3, 16), np.float32)
    dr_all = np.zeros(n, np.int64); dc_all = np.zeros(n, np.int64)
    for lo in range(0, n, chunk):
        rr = rows[lo:lo + chunk]
        g = D.sro[rr, 2].astype(np.int64)
        rt = U.role[g]; ct = U.country[g]
        dr, dc = U.decoy_patterns(rr, seed=lo)
        dr_all[lo:lo + len(rr)] = dr; dc_all[lo:lo + len(rr)] = dc
        for gi in range(3):
            arrs["u_ll_pos"][lo:lo + len(rr), :, gi] = U.u_gold(rr, gi, rt, ct, mode="window")
            arrs["u_full_pos"][lo:lo + len(rr), :, gi] = U.u_gold(rr, gi, rt, ct, mode="full")
            arrs["u_rank_pos"][lo:lo + len(rr), :, gi] = U.u_rank_legacy(rr, gi, rt, ct)
            arrs["u_ll_neg"][lo:lo + len(rr), :, gi] = U.u_gold(rr, gi, dr, dc, mode="window", gold_matches=np.zeros(len(rr), bool))
            arrs["u_full_neg"][lo:lo + len(rr), :, gi] = U.u_gold(rr, gi, dr, dc, mode="full", gold_matches=np.zeros(len(rr), bool))
            arrs["u_rank_neg"][lo:lo + len(rr), :, gi] = U.u_rank_legacy(rr, gi, dr, dc)
            Xp[lo:lo + len(rr), gi] = U.goldfree_features(rr, gi, rt, ct)
            Xn[lo:lo + len(rr), gi] = U.goldfree_features(rr, gi, dr, dc)
        print(f"  {lo + len(rr)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    split = D.split[rows]; gold = D.sro[rows, 2].astype(np.int64)
    gfreq = S.freq[gold]; intop50 = (D.topk_ids[rows, :50] == gold[:, None]).any(1)
    np.save(f"{out_dir}/rows.npy", rows); np.save(f"{out_dir}/split.npy", split)
    np.save(f"{out_dir}/gold_freq.npy", gfreq); np.save(f"{out_dir}/gold_in_top50.npy", intop50)
    np.save(f"{out_dir}/X_pos.npy", Xp); np.save(f"{out_dir}/X_neg.npy", Xn)
    np.save(f"{out_dir}/decoy_role.npy", dr_all); np.save(f"{out_dir}/decoy_country.npy", dc_all)
    for k, v in arrs.items():
        np.save(f"{out_dir}/{k}.npy", v)
    summ = dict(dataset=dataset, base=base, dump=D.path, window=U.W, n_rows=int(n),
                n_by_split={nm: int((split == c).sum()) for nm, c in (("train", 0), ("valid", 1), ("test", 2))},
                bump_grid=BUMP_GRID.tolist(), granularity=GRANS, cache_dir=out_dir)
    for k in ("u_ll", "u_full", "u_rank"):
        summ[f"mean_{k}_pos_by_gran"] = {gn: round(float(arrs[f"{k}_pos"][:, :, j].mean()), 5) for j, gn in enumerate(GRANS)}
        summ[f"mean_{k}_neg_by_gran"] = {gn: round(float(arrs[f"{k}_neg"][:, :, j].mean()), 5) for j, gn in enumerate(GRANS)}
    summ["frac_pos_helpful_u_full_b1"] = {gn: round(float((arrs["u_full_pos"][:, 1, j] > 0).mean()), 4) for j, gn in enumerate(GRANS)}
    summ["role_reusability"] = dict(entities_per_role=round(float(len(U.role) / len(np.unique(U.role))), 3),
                                    frac_entities_sharing_role=round(float((U.role_hist[U.role] > 1).mean()), 4))
    # ---------------- reproduction check vs the existing ICEWS18 signal ----------------
    if dataset == "ICEWS18":
        try:
            old = np.load(os.path.expandvars("$EXT_DIR/logcl_redun/utility_signal_ICEWS18.npz"))
            pos = {int(r): j for j, r in enumerate(rows)}
            jj = np.array([pos.get(int(q), -1) for q in old["qi"]]); ok = jj >= 0
            d_ll = np.abs(old["u_ll_pos"][ok] - arrs["u_ll_pos"][jj[ok]]).max()
            d_rk = np.abs(old["u_rank_pos"][ok] - arrs["u_rank_pos"][jj[ok]]).max()
            gfo = np.load(os.path.expandvars("$EXT_DIR/logcl_redun/goldfree_features_ICEWS18.npz"))
            gsel = gfo["is_gold_pattern"] == 1
            ri = gfo["row_idx"][gsel]; gid = gfo["gran_id"][gsel]; Xo = gfo["X"][gsel]
            jx = np.array([pos.get(int(q), -1) for q in ri]); okx = jx >= 0
            d_x = np.abs(Xo[okx] - Xp[jx[okx], gid[okx]]).max(0)
            summ["reproduction_vs_existing_ICEWS18"] = dict(
                n_rows_matched=int(ok.sum()), n_rows_old=int(len(old["qi"])),
                max_abs_diff_u_ll_pos=float(d_ll), max_abs_diff_u_rank_pos=float(d_rk),
                goldpattern_feature_rows_matched=int(okx.sum()),
                max_abs_diff_goldfree_features_per_feature=dict(zip(FEAT_NAMES, [round(float(x), 6) for x in d_x])))
        except Exception as e:
            summ["reproduction_vs_existing_ICEWS18"] = dict(error=repr(e))
    # ---------------- gold-free learnability (honest: features never see the gold) ----------------
    pred = U.fit_predictor(Xp, Xn, arrs["u_ll_pos"], arrs["u_ll_neg"], split == 0)
    pickle.dump(pred, open(f"{out_dir}/predictor.pkl", "wb"))
    te = np.where(split == 2)[0]
    strat = Splits.stratum(gfreq)
    table = []
    slices = [("all", np.ones(len(te), bool)), ("cold", strat[te] == 0), ("mid", strat[te] == 1), ("head", strat[te] == 2),
              ("gold_in_top50", intop50[te]), ("gold_not_in_top50", ~intop50[te]),
              ("cold_gold_not_in_top50", (strat[te] == 0) & ~intop50[te])]
    Xt = np.concatenate([Xp[te], Xn[te]], 1)                     # [nt, 6, 16]
    yt = np.concatenate([np.ones((len(te), 3)), np.zeros((len(te), 3))], 1)
    ut = np.concatenate([arrs["u_ll_pos"][te].mean(1), arrs["u_ll_neg"][te].mean(1)], 1)
    pc = pred["clf"].predict_proba(Xt.reshape(-1, 16))[:, 1].reshape(len(te), 6)
    pr = pred["reg"].predict(Xt.reshape(-1, 16)).reshape(len(te), 6)
    # query-only ablation: features 0..9 (+gran one-hot) carry no pattern information -> must be ~0.5
    from sklearn.ensemble import HistGradientBoostingClassifier
    trq = np.where(split == 0)[0][:200000]
    Xq = np.concatenate([Xp[trq][:, :, :13].reshape(-1, 13), Xn[trq][:, :, :13].reshape(-1, 13)])
    yq = np.concatenate([np.ones(len(trq) * 3), np.zeros(len(trq) * 3)])
    clf_q = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=0).fit(Xq, yq)
    pq = clf_q.predict_proba(Xt[:, :, :13].reshape(-1, 13))[:, 1].reshape(len(te), 6)
    for nm, m in slices:
        if m.sum() == 0:
            continue
        auc = roc_auc_score(yt[m].reshape(-1), pc[m].reshape(-1))
        aucq = roc_auc_score(yt[m].reshape(-1), pq[m].reshape(-1))
        rho = spearmanr(pr[m].reshape(-1), ut[m].reshape(-1)).correlation
        table.append(dict(slice=nm, n_queries=int(m.sum()), classify_AUC=round(float(auc), 4),
                          query_only_AUC_control=round(float(aucq), 4), splice_Spearman=round(float(rho), 4)))
    summ["learnability_goldfree"] = dict(table=table, n_train_queries=pred["n_train_queries"],
                                         features=FEAT_NAMES, note="classifier separates the gold's own pattern from a decoy; "
                                         "regressor predicts u_ll averaged over the bump grid; features are gold-free")
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"{out_dir}/summary.json"); jdump(summ, f"utility_estimator_{dataset}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    print(json.dumps(summ["learnability_goldfree"]["table"]), flush=True)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="tirgn")
    a = ap.parse_args()
    build(a.dataset, a.base)
