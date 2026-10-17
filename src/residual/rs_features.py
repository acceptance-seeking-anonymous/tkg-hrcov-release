#!/usr/bin/env python3
"""
Residual-objective track (idea 6.2) -- step 1: candidate sets, features, oracle ceilings.

For one (dataset, frozen base) this writes, under TRACK/feat/<DS>_<base>/ :
  {valid,test}_X.npy      [n, W, 29] float16   per-(query, candidate) features (names in meta.json)
  {valid,test}_logp.npy   [n, W]     float32   base log-prob used in the fused score (tail value if invisible)
  {valid,test}_cand.npy   [n, W]     int32     candidate ids (-1 pad): base top-50 (+ <=50 bridge cands, TiRGN)
  {valid,test}_filt.npy   [n, W]     bool      other true answers of the same (s, r', ts)  (time-aware filter)
  {valid,test}_babove.npy [n, W]     bool      EVAL ONLY: candidate visible in the dump top-K with score > gold
  {valid,test}_greuse.npy [n, W]     float16   reusability gate g(c) = 1 - 1/rolecount(c)
  {valid,test}_q.npz                 per-query arrays (rows, sro, gold_pos, r_base, p1, u_bm, strata, ...)
and feat_summary_<DS>_<base>.json (oracle ceilings, identity checks) in the job cwd.

Candidate set C(q) = base top-50 (all queries)  UNION  first 50 G12-arm5 backfill candidates not in the top-50
(TiRGN trigger-pool queries only; the shared bridge caches exist only for TiRGN).
History features use every fact (train+valid+test) with ts < t_q (standard online extrapolation setting).
"""
import os, sys, json, time, argparse, pickle
H = os.path.expandvars("$WORK_DIR")
sys.path.insert(0, H)
import numpy as np
from oih_common import Splits, BaseDump, DATASETS, metrics_from_ranks, jdump
from fulltest_eval import BaseView

TRACK = f"{H}/tracks/residual"
KC, BBR = 50, 50
BM_WIN, BM_MAXEV, BM_TAU, BM_LOOK, NLB = 65, 8, 100.0, 16, 7
BIG = np.int64(1 << 17)
REC_NONE = float(np.log1p(10000.0))

FEAT_NAMES = [
    "logp", "logp_minus_top1", "log_base_rank", "is_topKc", "q_p1", "q_entropy50", "q_margin12", "q_log_tailmass",
    "upred_help_role", "upred_help_country", "upred_help_rc", "upred_u_role", "upred_u_country", "upred_u_rc",
    "ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
    "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj",
    "br_in_bridge", "br_arm5_score", "br_log_pos", "br_q_has_bridge"]
GROUPS = dict(base=list(range(0, 8)), upred=list(range(8, 14)), evid=list(range(14, 25)), bridge=list(range(25, 29)))


def lagbin(lag):
    lag = np.maximum(np.asarray(lag, np.float64), 1.0)
    return np.clip(np.ceil(np.log2(lag)), 0, NLB - 1).astype(np.int64)


class Hist:
    """Sorted (key, ts) arrays over ALL facts in both directions -> counts / last-time strictly before t."""

    def __init__(self, S):
        q = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]).astype(np.int64)
        R = S.R; N = S.n_ent; self.N = N; self.R2 = 2 * R; self.step = int(S.cfg["step_unit"])
        s = np.concatenate([q[:, 0], q[:, 2]]); r = np.concatenate([q[:, 1], q[:, 1] + R])
        o = np.concatenate([q[:, 2], q[:, 0]]); ts = np.concatenate([q[:, 3], q[:, 3]])
        assert ts.max() < BIG
        def mk(k):
            c = k.astype(np.int64) * BIG + ts
            od = np.argsort(c, kind="stable")
            return c[od], od
        self.trip, _ = mk((s * self.R2 + r) * N + o)
        self.dyad, od = mk(s * N + o); self.dyad_rel = r[od].astype(np.int64)
        self.objrel, _ = mk(r * N + o)
        self.ent, _ = mk(s)
        self.query, oq = mk(s * self.R2 + r); self.query_obj = o[oq]

    @staticmethod
    def cnt(arr, key, t):
        key = np.asarray(key, np.int64); t = np.asarray(t, np.int64)
        lo = np.searchsorted(arr, key * BIG, "left"); hi = np.searchsorted(arr, key * BIG + t, "left")
        return hi - lo, hi

    def feats(self, s, r, t, cand):
        """s,r,t [n]; cand [n,W] (-1 pad) -> dict of [n,W] / [n] arrays."""
        n, W = cand.shape; N = self.N; st = self.step
        c = np.where(cand >= 0, cand, 0).astype(np.int64)
        S_ = s[:, None].astype(np.int64); Rr = r[:, None].astype(np.int64); T = np.broadcast_to(t[:, None].astype(np.int64), (n, W))
        out = {}
        k_trip = (S_ * self.R2 + Rr) * N + c
        ntrip, hi = self.cnt(self.trip, k_trip, T)
        last = np.where(ntrip > 0, self.trip[np.maximum(hi - 1, 0)] - k_trip * BIG, -1)
        out["ntrip"] = ntrip
        out["rec_trip"] = np.where(ntrip > 0, np.log1p((T - last) / st), REC_NONE)
        lo30 = np.searchsorted(self.trip, k_trip * BIG + np.maximum(T - 30 * st, 0), "left")
        out["ntrip30"] = hi - lo30
        k_dy = S_ * N + c
        ndy, hi = self.cnt(self.dyad, k_dy, T)
        last = np.where(ndy > 0, self.dyad[np.maximum(hi - 1, 0)] - k_dy * BIG, -1)
        out["ndyad"] = ndy; out["dyad_hi"] = hi
        out["rec_dyad"] = np.where(ndy > 0, np.log1p((T - last) / st), REC_NONE)
        k_or = Rr * N + c
        nor, hi = self.cnt(self.objrel, k_or, T)
        last = np.where(nor > 0, self.objrel[np.maximum(hi - 1, 0)] - k_or * BIG, -1)
        out["nobjrel"] = nor
        out["rec_objrel"] = np.where(nor > 0, np.log1p((T - last) / st), REC_NONE)
        out["nent"], _ = self.cnt(self.ent, c, T)
        kq = s.astype(np.int64) * self.R2 + r.astype(np.int64)
        nq, hiq = self.cnt(self.query, kq, t)
        out["nquery"] = nq
        out["last_obj"] = np.where(nq > 0, self.query_obj[np.maximum(hiq - 1, 0)], -1)
        out["nsubj"], _ = self.cnt(self.ent, s.astype(np.int64), t)
        return out


class DyadReader:
    """BridgeMem-style entity-anonymised dyadic transition count reader (base-agnostic).
    Context of a prior event on the ordered pair (a, c) = (relation incl. direction, log-lag bin);
    target = the next event's relation (incl. direction).  Fitted on TRAIN facts only.  Back-off
    P(r|r_e,lb) -> P(r|r_e) -> P(r) with additive pseudo-count tau (BridgeMem's rho = tau/(n+tau))."""

    def __init__(self, S, H_):
        R2 = 2 * S.R; N = S.n_ent; st = int(S.cfg["step_unit"]); self.R2 = R2; self.st = st; self.H = H_
        tr = S.q["train"].astype(np.int64); R = S.R
        s = np.concatenate([tr[:, 0], tr[:, 2]]); r = np.concatenate([tr[:, 1], tr[:, 1] + R])
        o = np.concatenate([tr[:, 2], tr[:, 0]]); ts = np.concatenate([tr[:, 3], tr[:, 3]])
        key = s * N + o; c = key * BIG + ts; od = np.argsort(c, kind="stable")
        key = key[od]; ts = ts[od]; r = r[od]; n = len(key)
        C = np.zeros(R2 * NLB * R2, np.int64); taken = np.zeros(n, np.int64)
        for k in range(1, BM_LOOK + 1):
            i = np.arange(k, n); p = i - k
            ok = (key[p] == key[i]) & (ts[p] < ts[i]) & ((ts[i] - ts[p]) <= BM_WIN * st) & (taken[i] < BM_MAXEV)
            if not ok.any():
                continue
            ii = i[ok]; pp = p[ok]; taken[ii] += 1
            lb = lagbin((ts[ii] - ts[pp]) // st)
            C += np.bincount((r[pp] * NLB + lb) * R2 + r[ii], minlength=len(C))
        C = C.reshape(R2 * NLB, R2).astype(np.float64)
        tot_t = C.sum(0); P0 = (tot_t + 1.0) / (tot_t.sum() + R2)
        C1 = C.reshape(R2, NLB, R2).sum(1); n1 = C1.sum(1, keepdims=True)
        P1 = (C1 + BM_TAU * P0[None, :]) / (n1 + BM_TAU)
        nc = C.sum(1, keepdims=True)
        P = (C + BM_TAU * np.repeat(P1, NLB, axis=0)) / (nc + BM_TAU)
        self.P = P.astype(np.float64); self.logP0 = np.log(P0); self.n_transitions = int(C.sum())

    def score(self, s, r, t, cand, dyad_hi):
        """A(q,c) = log mean_j P(r_q | ctx_j) - log P0(r_q) over <=8 most recent (a,c) events in 65 snapshots."""
        n, W = cand.shape; N = self.H.N; st = self.st
        c = np.where(cand >= 0, cand, 0).astype(np.int64)
        key = s[:, None].astype(np.int64) * N + c; T = t[:, None].astype(np.int64)
        rq = np.broadcast_to(r[:, None].astype(np.int64), (n, W))
        sumP = np.zeros((n, W)); m = np.zeros((n, W), np.int64)
        dy = self.H.dyad; dr = self.H.dyad_rel
        for k in range(1, BM_LOOK + 1):
            p = dyad_hi - k
            okp = p >= 0
            pc = np.maximum(p, 0)
            ev = dy[pc]; kev = ev // BIG; tev = ev - kev * BIG
            ok = okp & (kev == key) & ((T - tev) <= BM_WIN * st) & (m < BM_MAXEV)
            if not ok.any():
                break
            lb = lagbin((T - tev) // st)
            ctx = dr[pc] * NLB + lb
            sumP += np.where(ok, self.P[ctx, rq], 0.0); m += ok
        A = np.where(m > 0, np.log(np.maximum(sumP, 1e-300) / np.maximum(m, 1)) - self.logP0[rq], 0.0)
        A = np.where(cand >= 0, A, 0.0)
        return A, m


def upred_features(U, ids50, sc50, h, r, cand):
    """Vectorised re-implementation of UtilityEstimator.goldfree_features for every candidate's own
    (role / country / role_country) pattern.  Returns X [n, W, 3, 16] float32."""
    sc = sc50.astype(np.float64); ids = ids50.astype(np.int64); n = len(sc); W = cand.shape[1]
    x = sc - sc.max(1, keepdims=True); e = np.exp(x); p = e / e.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    top1 = sc[:, 0]; m12 = sc[:, 0] - sc[:, 1]; m110 = sc[:, 0] - sc[:, 9]; sstd = sc.std(1); eff = np.exp(ent)
    qsf = np.log1p(U.sfreq[h]); qrf = np.log1p(U.rfreq[r])
    cr = U.role[ids]; cc = U.country[ids]
    def nd(a):
        s_ = np.sort(a, 1); return 1 + (np.diff(s_, axis=1) != 0).sum(1)
    Q = np.column_stack([ent, top1, m12, m110, sstd, eff, qsf, qrf, nd(cr) / 50.0, nd(cc) / 50.0])
    X = np.zeros((n, W, 3, 16), np.float32)
    X[:, :, :, :10] = Q[:, None, None, :]
    cl = np.where(cand >= 0, cand, 0).astype(np.int64)
    for g in range(3):
        X[:, :, g, 10 + g] = 1.0
        for j in range(W):
            rj = U.role[cl[:, j]]; cj = U.country[cl[:, j]]
            if g == 0: mk = cr == rj[:, None]
            elif g == 1: mk = cc == cj[:, None]
            else: mk = (cr == rj[:, None]) & (cc == cj[:, None])
            X[:, j, g, 13] = (p * mk).sum(1); X[:, j, g, 14] = mk.sum(1) / 50.0
            anym = mk.any(1); X[:, j, g, 15] = np.where(anym, mk.argmax(1) / 50.0, 1.0)
    return X


def load_strik(S, path):
    d = {}
    if not os.path.exists(path):
        return d
    st = int(S.cfg["step_unit"]); R = S.R
    for ln in open(path):
        p = ln.split()
        if len(p) != 5:
            continue
        s, r, o, t = map(int, p[:4]); v = float(p[4])
        d[(s, r, o, t * st)] = v; d[(o, r + R, s, t * st)] = v
    return d


def fused_ranks_np(s, cand, filt, babove, gold_pos, r_base):
    """Filtered rank of the gold in:  [C_promoted sorted by fused score s]  ++  base remainder.
    C_promoted = base top-KC  UNION  {bridge cand b : s(b) > min_{x in top-KC} s(x)}  (gold-free rule).
    gold in C_promoted : 1 + #{x in C_prom, not filtered, s(x) > s(gold)}
    otherwise          : base filtered rank + #{promoted bridge b, not filtered, not already above the gold in base}
                         (conservative for invisible b, = harness fused_rank convention)."""
    n, W = s.shape
    valid = cand >= 0
    s = np.where(valid, s, -np.inf)
    mint = s[:, :KC].min(1)
    jj = np.arange(W)[None, :]
    prom = valid & ((jj < KC) | (s > mint[:, None]))
    unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0)
    sg = s[ar, g]
    gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & (s > sg[:, None]); above[ar, g] = False
    r_in = 1 + above.sum(1)
    add = (unf & (jj >= KC) & ~babove).sum(1)
    r_out = r_base + add
    return np.where(gin, r_in, r_out).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="tirgn")
    ap.add_argument("--valid_n", type=int, default=200000, help="valid rows kept (all if fewer)")
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--smoke", type=int, default=0, help=">0: only this many rows per split (smoke test)")
    a = ap.parse_args()
    t0 = time.time(); DS = a.dataset; cfg = DATASETS[DS]
    out_dir = f"{TRACK}/feat/{DS}_{a.base}" + ("_smoke" if a.smoke else ""); os.makedirs(out_dir, exist_ok=True)
    S = Splits(DS); F = S.filter_index(); N = S.n_ent; R = S.R
    log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
    D = BaseDump(cfg["dumps"][a.base], S); log("dump", a.base, D.N, D.K, D.load_s)
    if DS == "ICEWS18" and a.base == "tirgn":
        DV = BaseDump(cfg["dumps"]["tirgn_valid"], S); kv = 50          # truncate valid (K=200) to 50 cols
    else:
        DV = D; kv = D.K
    Hh = Hist(S); log("hist index")
    BM = DyadReader(S, Hh); log("dyad reader fitted on", BM.n_transitions, "train transitions")
    from utility_estimator import UtilityEstimator
    U = UtilityEstimator(DS, "tirgn", dump=D, splits=S); U.load_predictor()
    clf, reg = U.predictor["clf"], U.predictor["reg"]
    v = np.load(cfg["lm_vocab"], allow_pickle=True); ent_role = v["ent_role"].astype(np.int64)
    role_count = np.bincount(ent_role, minlength=int(ent_role.max()) + 1)
    greuse_ent = (1.0 - 1.0 / np.maximum(role_count[ent_role], 1)).astype(np.float32)
    strik = load_strik(S, f"rsmf_{DS}_test_extend.txt")
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    summary = dict(dataset=DS, base=a.base, dump=cfg["dumps"][a.base], KC=KC, BBR=BBR, feat_names=FEAT_NAMES, groups=GROUPS,
                   host="serverA (CPU, tkg-gpu:latest)", n_strik_keys=len(strik),
                   dyad_reader=dict(n_train_transitions=BM.n_transitions, window_snapshots=BM_WIN, max_events=BM_MAXEV,
                                    tau=BM_TAU, lag_bins=NLB),
                   greuse_frac_zero=float((greuse_ent == 0).mean()))
    rng = np.random.default_rng(12345)
    for split, DD, kref in (("valid", DV, kv), ("test", D, D.K)):
        rows = DD.rows(split)
        if split == "valid":
            keep = rows
            vs_path = f"{H}/cache/{DS}_tirgn_valid_pool/valid_sample_rows.npy"
            if len(rows) > a.valid_n:
                must = np.load(vs_path) if (a.base == "tirgn" and os.path.exists(vs_path)) else np.zeros(0, np.int64)
                rest = np.setdiff1d(rows, must)
                keep = np.union1d(must, rng.choice(rest, a.valid_n - len(must), replace=False))
            rows = np.sort(keep)
        if a.smoke:
            rows = rows[np.linspace(0, len(rows) - 1, a.smoke).astype(np.int64)]
        n = len(rows)
        V = BaseView(S, DD, rows, F, kref); log(split, "baseview", n, V.rank_source)
        bf = None
        if a.base == "tirgn":
            coh = "valid_pool" if split == "valid" else "test_pool"
            bfz = np.load(f"{H}/results/backfill_g12arm5_{DS}_{coh}.npz", allow_pickle=False)
            bf = {int(rr): j for j, rr in enumerate(bfz["rows"])}; bf_ids = bfz["ids"]; bf_sc = bfz["scores"]
        W = KC + (BBR if a.base == "tirgn" else 0)
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        X_ = mm("X", (n, W, len(FEAT_NAMES)), np.float16); LP = mm("logp", (n, W), np.float32)
        CA = mm("cand", (n, W), np.int32); FI = mm("filt", (n, W), np.bool_); BA = mm("babove", (n, W), np.bool_)
        GR = mm("greuse", (n, W), np.float16)
        qa = {k: np.zeros(n, dt) for k, dt in (("gold_pos", np.int16), ("u_bm", np.float32), ("is_pool", np.bool_),
                                                ("rec_nonrepeat", np.bool_), ("rec_top1_miss", np.bool_), ("gold_dyad_hist", np.bool_),
                                                ("gold_dyad_win", np.bool_), ("strik", np.float32), ("strik_found", np.bool_))}
        sro = DD.sro[rows].astype(np.int64)
        tki = DD.topk_ids[rows, :kref].astype(np.int64); tks = DD.topk_scores[rows, :kref].astype(np.float64)
        gs = DD.gold_score[rows].astype(np.float64)
        Kd = kref
        for lo in range(0, n, a.chunk):
            hi_ = min(n, lo + a.chunk); b = np.arange(lo, hi_); nb = len(b)
            s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
            ids = tki[b]; sc = tks[b]
            cand = np.full((nb, W), -1, np.int64); cand[:, :KC] = ids[:, :KC]
            lp = np.full((nb, W), -np.inf); lp[:, :KC] = sc[:, :KC]
            brank = np.full((nb, W), Kd + 1.0); brank[:, :KC] = np.arange(1, KC + 1)[None, :]
            tail_mass = np.clip(1.0 - np.exp(sc).sum(1), 1e-12, 1.0); lp_tail = np.log(tail_mass / max(N - Kd, 1))
            bab = np.zeros((nb, W), bool)
            Xb = np.zeros((nb, W, 4), np.float32)
            if bf is not None:
                Xb[:, :KC, 2] = np.log1p(201.0)
                for i in range(nb):
                    j = bf.get(int(rows[b[i]]))
                    if j is None:
                        continue
                    qa["is_pool"][b[i]] = True; Xb[i, :, 3] = 1.0
                    lst = bf_ids[j]; ok = lst >= 0; lst = lst[ok]; scl = bf_sc[j][ok]
                    posmap = {int(e): k for k, e in enumerate(lst)}
                    for k in range(KC):
                        pk = posmap.get(int(ids[i, k]))
                        if pk is not None:
                            Xb[i, k, 0] = 1.0; Xb[i, k, 1] = scl[pk]; Xb[i, k, 2] = np.log1p(pk + 1)
                    top = set(int(e) for e in ids[i, :KC])
                    extra = [(k, int(e)) for k, e in enumerate(lst) if int(e) not in top][:BBR]
                    vis = {int(e): k for k, e in enumerate(ids[i])}
                    for m_, (k, e) in enumerate(extra):
                        jj = KC + m_; cand[i, jj] = e
                        Xb[i, jj, 0] = 1.0; Xb[i, jj, 1] = scl[k]; Xb[i, jj, 2] = np.log1p(k + 1)
                        kv_ = vis.get(e)
                        if kv_ is not None:                       # visible in the dump top-K (ranks KC+1..K)
                            lp[i, jj] = sc[i, kv_]; brank[i, jj] = kv_ + 1; bab[i, jj] = sc[i, kv_] > gs[b[i]]
                        else:
                            lp[i, jj] = lp_tail[i]
            valid = cand >= 0
            fk = S.fkey(s, r, t)
            code = fk[:, None] * N + np.where(valid, cand, 0)
            pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
            istrue = (allTrue[pos] == code) & valid
            isgold = (cand == o[:, None]) & valid
            filt = istrue & ~isgold
            gp = np.where(isgold.any(1), isgold.argmax(1), -1)
            qa["gold_pos"][b] = gp
            hf = Hh.feats(s, r, t, np.where(valid, cand, -1))
            A, mev = BM.score(s, r, t, np.where(valid, cand, -1), hf["dyad_hi"])
            p_top = np.exp(sc[:, :KC])
            u_bm = (p_top * (mev[:, :KC] > 0)).sum(1); qa["u_bm"][b] = u_bm
            gH = Hh.feats(s, r, t, o[:, None])
            qa["rec_nonrepeat"][b] = gH["ntrip"][:, 0] == 0
            qa["rec_top1_miss"][b] = gH["last_obj"] != o
            qa["gold_dyad_hist"][b] = gH["ndyad"][:, 0] > 0
            _, mg = BM.score(s, r, t, o[:, None], gH["dyad_hi"]); qa["gold_dyad_win"][b] = mg[:, 0] > 0
            if split == "test" and strik:
                for i in range(nb):
                    vv = strik.get((int(s[i]), int(r[i]), int(o[i]), int(t[i])))
                    if vv is not None:
                        qa["strik"][b[i]] = vv; qa["strik_found"][b[i]] = True
            Xu = upred_features(U, ids[:, :50], sc[:, :50], s, r, np.where(valid, cand, 0))
            flat = Xu.reshape(-1, 16)
            ph = clf.predict_proba(flat)[:, 1].reshape(nb, W, 3); uh = reg.predict(flat).reshape(nb, W, 3)
            del Xu, flat
            pn = p_top / p_top.sum(1, keepdims=True); ent50 = -(pn * np.log(pn + 1e-12)).sum(1)
            Xf = np.zeros((nb, W, len(FEAT_NAMES)), np.float32)
            lpf = np.where(valid, lp, lp_tail[:, None])
            Xf[:, :, 0] = lpf; Xf[:, :, 1] = lpf - sc[:, :1]; Xf[:, :, 2] = np.log(brank)
            Xf[:, :KC, 3] = 1.0
            Xf[:, :, 4] = np.exp(sc[:, :1]); Xf[:, :, 5] = ent50[:, None]; Xf[:, :, 6] = (sc[:, 0] - sc[:, 1])[:, None]
            Xf[:, :, 7] = np.log(tail_mass)[:, None]
            Xf[:, :, 8:11] = ph; Xf[:, :, 11:14] = uh
            Xf[:, :, 14] = np.log1p(hf["ntrip"]); Xf[:, :, 15] = np.log1p(hf["ndyad"]); Xf[:, :, 16] = hf["rec_trip"]
            Xf[:, :, 17] = hf["rec_dyad"]; Xf[:, :, 18] = np.log1p(hf["nobjrel"]); Xf[:, :, 19] = np.log1p(hf["nent"])
            Xf[:, :, 20] = hf["rec_objrel"]; Xf[:, :, 21] = np.log1p(hf["ntrip30"]); Xf[:, :, 22] = A
            Xf[:, :, 23] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 24] = np.log1p(hf["nsubj"])[:, None]
            Xf[:, :, 25:29] = Xb
            Xf[~valid] = 0.0
            X_[lo:hi_] = Xf.astype(np.float16); LP[lo:hi_] = np.where(valid, lp, -np.inf).astype(np.float32)
            CA[lo:hi_] = cand.astype(np.int32); FI[lo:hi_] = filt; BA[lo:hi_] = bab
            GR[lo:hi_] = np.where(valid, greuse_ent[np.where(valid, cand, 0)], 0.0).astype(np.float16)
            log(split, f"{hi_}/{n}")
        X_.flush(); LP.flush(); CA.flush(); FI.flush(); BA.flush(); GR.flush()
        chk = {}
        if DD is D:
            sub = np.arange(min(2000, n))
            mine_all = upred_features(U, tki[sub, :50], tks[sub, :50], sro[sub, 0], sro[sub, 1], np.maximum(np.asarray(CA[sub]), 0).astype(np.int64))
            for g in range(3):
                for j in (0, 7):
                    cj = np.asarray(CA[sub, j]).astype(np.int64)
                    ref = U.goldfree_features(rows[sub], g, U.role[cj], U.country[cj])
                    chk[f"g{g}_c{j}"] = float(np.abs(ref - mine_all[:, j, g]).max())
        LPn = np.asarray(LP); CAn = np.asarray(CA); FIn = np.asarray(FI); BAn = np.asarray(BA)
        gp = qa["gold_pos"].astype(np.int64)
        rk0 = fused_ranks_np(LPn, CAn, FIn, BAn, gp, V.base_rank)
        ident = float(np.mean(rk0 == V.base_rank))
        mism = np.where(rk0 != V.base_rank)[0]
        np.savez(f"{out_dir}/{split}_q.npz", rows=rows, sro=sro, r_base=V.base_rank, p1=V.p1, stratum=V.stratum,
                 in_top50=V.in_top50, in_hist=V.in_hist, gs=gs, kg=V.kg, **qa)
        rb = V.base_rank.astype(np.float64)
        intop = (gp >= 0) & (gp < KC); inC = gp >= 0; inbr = gp >= KC
        def m_(rk): return metrics_from_ranks(rk, ks=(1, 3, 10, 50))
        orc = dict(base=m_(rb), inpool_at1=m_(np.where(intop, 1, rb)), C_at1=m_(np.where(inC, 1, rb)),
                   bridge_only_at1=m_(np.where(inbr, 1, rb)))
        if W > KC:
            nun_top = (~FIn[:, :KC] & (CAn[:, :KC] >= 0)).sum(1)
            bpos = np.where(inbr, gp - KC, 0)
            nun_before = np.zeros(n, np.int64)
            for i in np.where(inbr)[0]:
                nun_before[i] = int((~FIn[i, KC:KC + bpos[i]] & (CAn[i, KC:KC + bpos[i]] >= 0)).sum())
            orc["bridge_block_after_top50_realised"] = m_(np.where(inbr, 1 + nun_top + nun_before, rb))
        frac = dict(gold_in_top50=float(intop.mean()), gold_in_bridge_only=float(inbr.mean()), gold_in_C=float(inC.mean()),
                    gold_dyad_hist=float(qa["gold_dyad_hist"].mean()), gold_dyad_window=float(qa["gold_dyad_win"].mean()),
                    rec_nonrepeat=float(qa["rec_nonrepeat"].mean()), rec_top1_miss=float(qa["rec_top1_miss"].mean()),
                    is_pool=float(qa["is_pool"].mean()))
        if split == "test":
            frac["strik_found"] = float(qa["strik_found"].mean())
        per = {}
        for nm_, msk in (("cold", V.stratum == 0), ("mid", V.stratum == 1), ("head", V.stratum == 2), ("hist_no", ~V.in_hist),
                         ("hist_yes", V.in_hist), ("nonrepeat", qa["rec_nonrepeat"]), ("pool", qa["is_pool"])):
            if msk.sum():
                per[nm_] = dict(n=int(msk.sum()), base_MRR=round(float((1 / rb[msk]).mean()), 6),
                                inpool_at1_MRR=round(float((1 / np.where(intop, 1, rb)[msk]).mean()), 6),
                                C_at1_MRR=round(float((1 / np.where(inC, 1, rb)[msk]).mean()), 6))
        summary[split] = dict(n=int(n), W=int(W), kref=int(kref), rank_source=V.rank_source, identity_f0=ident,
                              n_identity_mismatch=int(len(mism)), mismatch_examples=[(int(i), int(rk0[i]), int(V.base_rank[i])) for i in mism[:10]],
                              upred_selfcheck_maxabs=chk, oracle=orc, fractions=frac, oracle_per_stratum=per)
        log(split, "done", json.dumps(summary[split]["oracle"]))
    meta = dict(feat_names=FEAT_NAMES, groups=GROUPS, KC=KC, BBR=BBR, dataset=DS, base=a.base)
    jdump(meta, f"{out_dir}/meta.json")
    summary["elapsed_s"] = round(time.time() - t0, 1); summary["out_dir"] = out_dir
    jdump(summary, f"feat_summary_{DS}_{a.base}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
