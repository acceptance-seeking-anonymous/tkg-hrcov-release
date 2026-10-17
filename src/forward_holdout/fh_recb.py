#!/usr/bin/env python3
"""phase2_forward_holdout / fh_recb.py = presub hd_recb.py + --only_valid (Stage 1 builds the valid blend CSR without the test split) -- vectorised re-implementation of the TGB Recurrency Baseline (Gastinger et al., IJCAI 2024,
TGB modules/recurrencybaseline_predictor.py + examples/linkproppred/<ds>/recurrencybaseline.py) on harness-format data.

Score of candidate o for query (s, r', ?, t)   (r' >= R are the inverse relations, TGB add_inverse_quadruples):
    score(o) = (1000*alpha) * psi(o) + (1000*(1-alpha)) * xi(o)
    psi(o)   = sum_{(s, r', o, k), k < t} 2^(lam*(k - t)) / D,   D = sum_{i = min_ts}^{max_ts - 1} 2^(lam*(i - t))
               (min_ts / max_ts over relation r' facts with ts < t; D == 0 -> no division, as TGB)
    xi(o)    = #{(s', r', o, k), k < t} / #{(s', r', o', k), k < t}
Ranking = official py-tgb Evaluator formula over negatives = all entities minus the true answers of (s, r', t):
    rank = 1 + #(neg > pos) + 0.5 * #(neg == pos)          (exact, computed without materialising score rows)
Variants:
  clean    : psi row of a query without strict-recurrency candidates is 0.
  faithful : the TGB code never resets `predictions_psi`, so such a query inherits the psi row of the most recent earlier
             query (same relation, same ray chunk) that had candidates; chunking = TGB predict() with --num_processes P.
Modes (TGB): default lam 0.1 / alpha 0.99 for every relation; trained = per relation lam_r (alpha = 1) from LAMS, then
alpha_r from ALPHAS at lam_r, both by that relation's valid MRR (strict '>' updates, TGB init values); relations without
valid queries keep (1.0001, 0.99999).
Stages:
  --stage valid : ranks of every valid query for all 14 x 13 (lam, alpha) configs, both variants -> selection json
  --stage test  : ranks of every test query for default + trained configs (both variants) and the blend CSR files
                  (clean psi for default and clean-trained lam per query) for valid and test.
  python3 hd_recb.py --data_dir D --name tkgl-polecat --stage valid --workers 16 --P 16 --out OUT
"""
import os, sys, json, time, argparse
import numpy as np
import multiprocessing as mp

LAMS = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.9, 1.0001]
ALPHAS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
I_DEF_L, I_DEF_A = LAMS.index(0.1), ALPHAS.index(0.99)
I_TRDEF_L, I_TRDEF_A = len(LAMS) - 1, len(ALPHAS) - 2      # TGB train(): defaults for relations without valid queries
G = {}


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def reformat_ts_map(dd, name):
    """harness ts id -> TGB reformat_ts value (tkg_utils.reformat_ts)."""
    raw = {}
    for ln in open(f"{dd}/timestamp2id.txt"):
        k, v = ln.split(); raw[int(v)] = int(k)
    ids = np.array(sorted(raw)); rv = np.array([raw[i] for i in ids], np.int64)
    all_ts = np.sort(np.unique(rv)); d = np.diff(all_ts)
    dmax, dmin, dmean = d.max(), d.min(), d.mean()
    if dmean != dmin:
        if "icews" in name or "polecat" in name: ts_dist = 86400
        elif "wiki" in name or "yago" in name: ts_dist = 31536000
        else: ts_dist = dmean
        if ts_dist - dmean > 0.1 * dmean:
            ts_dist = dmean
    else:
        ts_dist = dmean
    new = np.ceil((rv - all_ts.min()) / ts_dist).astype(np.int64)
    out = np.zeros(ids.max() + 1, np.int64); out[ids] = new
    return out, dict(ts_dist=float(ts_dist), dmin=int(dmin), dmax=int(dmax), dmean=float(dmean),
                     identity=bool((out == np.arange(len(out))).all()))


def split_queries(q, R, split):
    """harness/base_dump order: per timestamp, forward triples (file order) then inverses.  (first TRAIN snapshot not scored)"""
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    ts = q[:, 3]; out = []
    uts, st = np.unique(ts, return_index=True); en = np.append(st[1:], len(q))
    for k, (u, b, e) in enumerate(zip(uts, st, en)):
        if split == "train" and k == 0:
            continue
        m = q[b:e]
        out.append(m.copy()); inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R; out.append(inv)
    return np.concatenate(out).astype(np.int64)


# ----------------------------------------------------------------------------------------------- per-task worker
def walk_pairs(rel, s_arr, t_arr):
    """strict-recurrency pairs for queries (s_arr, t_arr) of relation rel.
    returns pq (query idx), pc (candidate), psi [np, nL] (normalised by D of each query's t), has [m] bool."""
    A = G["A"]; Tm = G["TT"]
    Ks, Fo2, Ft2 = A["Ks"][rel], A["Fo2"][rel], A["Ft2"][rel]
    m = len(s_arr)
    if len(Ks) == 0 or not G["basis"][rel]:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, len(LAMS))), np.zeros(m, bool)
    lo = np.searchsorted(Ks, s_arr * Tm, "left"); hi = np.searchsorted(Ks, s_arr * Tm + t_arr, "left")
    ln = hi - lo
    has = ln > 0
    if ln.sum() == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, len(LAMS))), has
    eq = np.repeat(np.arange(m), ln)
    idx = np.repeat(lo - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())
    eo = Fo2[idx]; et = Ft2[idx]
    tq = t_arr[eq].astype(np.float64)
    L = np.asarray(LAMS)[None, :]
    w = np.power(2.0, L * (et.astype(np.float64)[:, None] - tq[:, None]))          # [nE, nL]   TGB score_delta
    key = eq * G["N"] + eo
    uk, inv = np.unique(key, return_inverse=True)
    raw = np.stack([np.bincount(inv, weights=w[:, k], minlength=len(uk)) for k in range(len(LAMS))], 1)
    pq = uk // G["N"]; pc = uk % G["N"]
    D = np.stack([delta_sum(rel, int(t)) for t in t_arr])                           # [m, nL]
    Dq = D[pq]
    psi = np.where(Dq > 0, raw / np.where(Dq > 0, Dq, 1.0), raw)
    return pq, pc, psi, has


def delta_sum(rel, t):
    c = G["Dcache"]
    k = (rel, t)
    if k not in c:
        ft = G["A"]["Ft1"][rel]
        hi = np.searchsorted(ft, t, "left")
        if hi == 0:
            c[k] = np.zeros(len(LAMS))
        else:
            mn, mx = int(ft[0]), int(ft[hi - 1])
            timesteps = np.arange(mn, mx); now = np.ones(len(timesteps)) * t
            c[k] = np.array([np.sum(np.power(2.0, lam * (timesteps - now))) for lam in LAMS])
    return c[k]


def seg_sum(v, q_sorted, m):
    """sum of rows of v grouped by the SORTED query index q_sorted -> [m, ...] (exact for counts)."""
    v = v.astype(np.int64) if v.dtype == bool else v
    cs = np.concatenate([np.zeros((1,) + v.shape[1:], v.dtype), np.cumsum(v, 0)], 0)
    b = np.searchsorted(q_sorted, np.arange(m + 1), "left")
    return (cs[b[1:]] - cs[b[:-1]]).astype(np.float64)


def rank_group(m, gold, pq, pc, psi, fq, fo, xi_vec, xs_sorted, lam_idx=None, alpha_idx=None):
    """exact official ranks for one (relation, t) group.
    lam_idx/alpha_idx None -> all configs [m, nL, nA]; else per-query config arrays (len m) -> [m]."""
    N = G["N"]
    # gold psi
    gkey = np.arange(m) * N + gold
    pkey = pq * N + pc
    o = np.argsort(pkey); pks = pkey[o]
    pos = np.searchsorted(pks, gkey); pos = np.clip(pos, 0, max(len(pks) - 1, 0))
    gin = (len(pks) > 0) & (pks[pos] == gkey) if len(pks) else np.zeros(m, bool)
    gpsi = np.zeros((m, psi.shape[1]))
    if len(pks):
        gpsi[gin] = psi[o[pos[gin]]]
    # exclusion: pairs whose cand is a true answer (filter) or the gold
    fkey = fq * N + fo
    excl = np.isin(pkey, np.concatenate([fkey, gkey]))
    # U = P u F u {gold}
    ukey = np.unique(np.concatenate([pkey, fkey, gkey]))
    uq = ukey // N; ue = ukey % N
    xi_c = xi_vec[pc]; xi_g = xi_vec[gold]; xi_u = xi_vec[ue]
    nnz = len(xs_sorted); z = N - nnz
    ok = ~excl
    okc = ok[:, None]
    if lam_idx is None:
        out = np.zeros((m, len(LAMS), len(ALPHAS)), np.float32)
        for ai, al in enumerate(ALPHAS):
            a = 1000 * al; c = 1000 * (1 - al)
            S = a * psi + c * xi_c[:, None]                       # [np, nL]
            g = a * gpsi + c * xi_g[:, None]                      # [m, nL]
            if len(pq):
                gp = g[pq]
                gt = seg_sum((S > gp) & okc, pq, m); eqc = seg_sum((S == gp) & okc, pq, m)
            else:
                gt = np.zeros((m, len(LAMS))); eqc = np.zeros((m, len(LAMS)))
            cv = c * xs_sorted
            r_ = np.searchsorted(cv, g, "right"); l_ = np.searchsorted(cv, g, "left")
            gt_all = nnz - r_; eq_all = (r_ - l_) + np.where(g == 0, z, 0)
            su = c * xi_u
            gu = g[uq]
            gtU = seg_sum(su[:, None] > gu, uq, m); eqU = seg_sum(su[:, None] == gu, uq, m)
            out[:, :, ai] = 1 + gt + gt_all - gtU + 0.5 * (eqc + eq_all - eqU)
        return out
    # per-query single config
    lam_idx = np.asarray(lam_idx); alpha_idx = np.asarray(alpha_idx)
    al = np.asarray(ALPHAS)[alpha_idx]
    a = 1000 * al; c = 1000 * (1 - al)
    g = a * gpsi[np.arange(m), lam_idx] + c * xi_g
    gt = np.zeros(m); eqc = np.zeros(m)
    if len(pq):
        S = a[pq] * psi[np.arange(len(pq)), lam_idx[pq]] + c[pq] * xi_c
        gt = np.bincount(pq[ok], weights=(S[ok] > g[pq][ok]).astype(np.float64), minlength=m)
        eqc = np.bincount(pq[ok], weights=(S[ok] == g[pq][ok]).astype(np.float64), minlength=m)
    gt_all = np.zeros(m); eq_all = np.zeros(m)
    for cc in np.unique(c):
        mm = c == cc
        cv = cc * xs_sorted
        r_ = np.searchsorted(cv, g[mm], "right"); l_ = np.searchsorted(cv, g[mm], "left")
        gt_all[mm] = nnz - r_; eq_all[mm] = (r_ - l_) + np.where(g[mm] == 0, z, 0)
    su = c[uq] * xi_u
    gtU = np.bincount(uq, weights=(su > g[uq]).astype(np.float64), minlength=m)
    eqU = np.bincount(uq, weights=(su == g[uq]).astype(np.float64), minlength=m)
    return (1 + gt + gt_all - gtU + 0.5 * (eqc + eq_all - eqU)).astype(np.float32)


def task(args):
    """(split, rel, row_lo, row_hi): rows [row_lo, row_hi) of this relation's query list (relation order)."""
    split, rel, lo_, hi_ = args
    Qr = G["Qrel"][split][rel]                     # indices into the split's sro, relation order
    rows = Qr[lo_:hi_]
    sro = G["sro"][split][rows]
    tq = G["tsmap"][sro[:, 3]]
    A = G["A"]; N = G["N"]
    ft1, fo1 = A["Ft1"][rel], A["Fo1"][rel]
    stage = G["stage"]
    src = G["src"][split][rel][lo_:hi_]            # faithful psi source (index into relation list, -1 none)
    res = {}
    if stage == "valid":
        res["clean"] = np.zeros((len(rows), len(LAMS), len(ALPHAS)), np.float32)
        res["faith"] = np.zeros((len(rows), len(LAMS), len(ALPHAS)), np.float32)
    else:
        for k in ("clean_def", "clean_tr", "faith_def", "faith_tr"):
            res[k] = np.zeros(len(rows), np.float32)
        csr = dict(n=np.zeros(len(rows), np.int64), cand=[], psi_def=[], psi_tr=[])
    # xi counts at the first query time
    uts, st = np.unique(tq, return_index=True); en = np.append(st[1:], len(tq))
    h0 = np.searchsorted(ft1, uts[0], "left")
    cnt = np.bincount(fo1[:h0], minlength=N).astype(np.int64); hprev = h0
    Fi = G["F"]
    for u, b, e in zip(uts, st, en):
        h = np.searchsorted(ft1, u, "left")
        if h > hprev:
            cnt += np.bincount(fo1[hprev:h], minlength=N); hprev = h
        tot = h
        xi_vec = cnt / tot if tot > 0 else np.zeros(N)
        nzv = np.flatnonzero(cnt)
        xs_sorted = np.sort(xi_vec[nzv]) if tot > 0 else np.zeros(0)
        m = e - b
        s_ = sro[b:e, 0]; g_ = sro[b:e, 2]; t_ = np.full(m, u, np.int64)
        # filter: other true answers at (s, r', ts)  (harness ts id in the key)
        fk = (sro[b:e, 3] * (2 * G["R"]) + rel) * N + s_
        i0 = np.searchsorted(Fi["keys"], fk); i0c = np.clip(i0, 0, len(Fi["keys"]) - 1)
        okk = Fi["keys"][i0c] == fk
        fq_l, fo_l = [], []
        for j in np.flatnonzero(okk):
            ob = Fi["objs"][Fi["start"][i0c[j]]:Fi["end"][i0c[j]]]
            ob = ob[ob != g_[j]]
            if len(ob):
                fq_l.append(np.full(len(ob), j)); fo_l.append(ob)
        fq = np.concatenate(fq_l) if fq_l else np.zeros(0, np.int64); fo = np.concatenate(fo_l) if fo_l else np.zeros(0, np.int64)
        # clean pairs
        pq, pc, psi, has = walk_pairs(rel, s_, t_)
        # faithful pairs: from the source query (its own s and t)
        sg = src[b:e]
        sel = sg >= 0
        if sel.any():
            srow = Qr[sg[sel]]
            ss = G["sro"][split][srow, 0]; st_ = G["tsmap"][G["sro"][split][srow, 3]]
            pq2, pc2, psi2, _ = walk_pairs(rel, ss, st_)
            qmap = np.flatnonzero(sel)
            pq2 = qmap[pq2]
        else:
            pq2, pc2, psi2 = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, len(LAMS)))
        if stage == "valid":
            res["clean"][b:e] = rank_group(m, g_, pq, pc, psi, fq, fo, xi_vec, xs_sorted)
            res["faith"][b:e] = rank_group(m, g_, pq2, pc2, psi2, fq, fo, xi_vec, xs_sorted)
        else:
            Lc, Ac = G["sel"]["clean"]; Lf, Af = G["sel"]["faith"]
            dl = np.full(m, I_DEF_L); da = np.full(m, I_DEF_A)
            res["clean_def"][b:e] = rank_group(m, g_, pq, pc, psi, fq, fo, xi_vec, xs_sorted, dl, da)
            res["clean_tr"][b:e] = rank_group(m, g_, pq, pc, psi, fq, fo, xi_vec, xs_sorted, np.full(m, Lc[rel]), np.full(m, Ac[rel]))
            res["faith_def"][b:e] = rank_group(m, g_, pq2, pc2, psi2, fq, fo, xi_vec, xs_sorted, dl, da)
            res["faith_tr"][b:e] = rank_group(m, g_, pq2, pc2, psi2, fq, fo, xi_vec, xs_sorted, np.full(m, Lf[rel]), np.full(m, Af[rel]))
            # blend CSR (clean psi): per query its strict-recurrency candidates, psi at lam 0.1 and at clean-trained lam_r
            nq_ = np.bincount(pq, minlength=m)
            csr["n"][b:e] = nq_
            o = np.lexsort((pc, pq))
            csr["cand"].append(pc[o].astype(np.int32)); csr["psi_def"].append(psi[o, I_DEF_L].astype(np.float32))
            csr["psi_tr"].append(psi[o, Lc[rel]].astype(np.float32))
    if stage != "valid":
        for k in ("cand", "psi_def", "psi_tr"):
            csr[k] = np.concatenate(csr[k]) if csr[k] else np.zeros(0, np.float32)
        res["csr"] = csr
    G["Dcache"].clear()
    return split, rel, lo_, hi_, res


def faithful_sources(has, n, P):
    """TGB predict(): chunk boundaries for --num_processes P, then psi source = last query <= j in the same chunk with
    candidates (-1 if none)."""
    if P > 1:
        nq = n // P
        if nq < P:
            chunk = np.zeros(n, np.int64)
        else:
            chunk = np.minimum(np.arange(n) // max(nq, 1), P - 1)
    else:
        chunk = np.zeros(n, np.int64)
    idx = np.where(has, np.arange(n), -1)
    last = np.maximum.accumulate(idx) if n else idx
    first_of_chunk = np.zeros(n, np.int64)
    if n:
        starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
        first_of_chunk = starts[np.searchsorted(starts, np.arange(n), "right") - 1]
    return np.where(last >= first_of_chunk, last, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--stage", required=True, choices=["valid", "test"])
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--out", default="."); ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--ts_step", type=int, default=0, help="presub G2: RE-GCN-split datasets with raw timestamps (no "
                    "timestamp2id.txt): reformat ts -> (ts - min_ts) // ts_step (TGB reformat_ts with unit steps)")
    ap.add_argument("--only_valid", action="store_true", help="forward holdout Stage 1: with --stage test, build the VALID blend CSR "
                    "(and valid default/trained ranks) only; the test split is not touched")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    os.makedirs(a.out, exist_ok=True)
    Q = {sp: load_quads(f"{a.data_dir}/{sp}.txt") for sp in ("train", "valid", "test")}
    st = open(f"{a.data_dir}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
    if a.ts_step > 0:
        uts_ = np.unique(np.concatenate([Q[sp][:, 3] for sp in Q]))
        tsmap = np.zeros(int(uts_.max()) + 1, np.int64); tsmap[uts_] = (uts_ - uts_.min()) // a.ts_step
        dd_ = np.diff(uts_)
        tsinfo = dict(ts_step=a.ts_step, dmin=int(dd_.min()), dmax=int(dd_.max()), all_multiples=bool((uts_ % a.ts_step == 0).all()),
                      n_ts=int(len(uts_)), identity=False)
    else:
        tsmap, tsinfo = reformat_ts_map(a.data_dir, a.name)
    log("N", N, "R", R, "ts reformat", tsinfo)
    allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
    s = np.concatenate([allq[:, 0], allq[:, 2]]); r = np.concatenate([allq[:, 1], allq[:, 1] + R])
    o = np.concatenate([allq[:, 2], allq[:, 0]]); t = tsmap[np.concatenate([allq[:, 3], allq[:, 3]])]
    TT = int(t.max()) + 2
    A = {k: {} for k in ("Ft1", "Fo1", "Ks", "Fo2", "Ft2")}
    ordr = np.argsort(r, kind="stable")
    rb = np.searchsorted(r[ordr], np.arange(2 * R + 1))
    for rel in range(2 * R):
        ix = ordr[rb[rel]:rb[rel + 1]]
        o1 = np.argsort(t[ix], kind="stable"); A["Ft1"][rel] = t[ix][o1]; A["Fo1"][rel] = o[ix][o1]
        k2 = s[ix] * TT + t[ix]; o2 = np.argsort(k2, kind="stable")
        A["Ks"][rel] = k2[o2]; A["Fo2"][rel] = o[ix][o2]; A["Ft2"][rel] = t[ix][o2]
    # TGB create_basis_dict(train_val_data incl. inverse): relations present in train+valid
    tv = np.concatenate([Q["train"], Q["valid"]])
    basis = np.zeros(2 * R, bool); basis[np.unique(np.concatenate([tv[:, 1], tv[:, 1] + R]))] = True
    # filter index (harness ts ids)
    h_, r_, t2_, ts_ = allq.T
    s_all = np.concatenate([h_, t2_]); r_all = np.concatenate([r_, r_ + R]); o_all = np.concatenate([t2_, h_]); ts_all = np.concatenate([ts_, ts_])
    key = (ts_all * (2 * R) + r_all) * N + s_all
    od = np.lexsort((o_all, key)); key = key[od]; o_all = o_all[od]
    uk, stt = np.unique(key, return_index=True); enn = np.append(stt[1:], len(key))
    F = dict(keys=uk, start=stt, end=enn, objs=o_all)
    splits = ("valid",) if (a.stage == "valid" or a.only_valid) else ("valid", "test")
    sro = {sp: split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
    Qrel = {sp: {} for sp in sro}; src = {sp: {} for sp in sro}
    G.update(A=A, TT=TT, N=N, R=R, basis=basis, F=F, sro=sro, tsmap=tsmap, stage=a.stage, Dcache={})
    for sp in sro:
        rr = sro[sp][:, 1]
        for rel in range(2 * R):
            Qrel[sp][rel] = np.flatnonzero(rr == rel)          # split order restricted to rel == TGB per-relation order
    G["Qrel"] = Qrel
    # has-candidates flags (cheap, vectorised) -> faithful sources per relation
    for sp in sro:
        for rel in range(2 * R):
            rows = Qrel[sp][rel]
            if len(rows) == 0:
                src[sp][rel] = np.zeros(0, np.int64); continue
            ss = sro[sp][rows, 0]; tt_ = tsmap[sro[sp][rows, 3]]
            Ks = A["Ks"][rel]
            if len(Ks) and basis[rel]:
                has = np.searchsorted(Ks, ss * TT + tt_, "left") > np.searchsorted(Ks, ss * TT, "left")
            else:
                has = np.zeros(len(rows), bool)
            src[sp][rel] = faithful_sources(has, len(rows), a.P)
    G["src"] = src
    meta = dict(name=a.name, N=N, R=R, ts=tsinfo, P_emulated=a.P, LAMS=LAMS, ALPHAS=ALPHAS, workers=a.workers,
                n_queries={sp: int(len(sro[sp])) for sp in sro},
                host=os.environ.get("HD_HOST", "serverA (CPU)"))
    if a.stage == "test":
        selj = json.load(open(f"{a.out}/recb_selection_{a.name}.json"))
        G["sel"] = {v: (np.array(selj[v]["lam_idx"]), np.array(selj[v]["alpha_idx"])) for v in ("clean", "faith")}
    for sp in splits:
        tasks = []
        for rel in range(2 * R):
            n = len(Qrel[sp][rel])
            if a.smoke: n = min(n, a.smoke)
            for lo_ in range(0, n, a.chunk):
                tasks.append((sp, rel, lo_, min(n, lo_ + a.chunk)))
        tasks.sort(key=lambda x: -(x[3] - x[2]))
        nrow = len(sro[sp])
        if a.stage == "valid":
            out = {k: np.zeros((nrow, len(LAMS), len(ALPHAS)), np.float32) for k in ("clean", "faith")}
        else:
            out = {k: np.zeros(nrow, np.float32) for k in ("clean_def", "clean_tr", "faith_def", "faith_tr")}
            csr_parts = {}
        done = np.zeros(nrow, bool)
        t1 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (sp_, rel, lo_, hi_, res) in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
                rows = Qrel[sp][rel][lo_:hi_]
                for kk in out:
                    out[kk][rows] = res[kk]
                done[rows] = True
                if a.stage == "test":
                    csr_parts[(rel, lo_)] = (rows, res["csr"])
                if k % 50 == 0:
                    log(sp, "task", k, "/", len(tasks), "elapsed", round(time.time() - t1))
        meta[f"{sp}_seconds"] = round(time.time() - t1, 1); meta[f"{sp}_scored"] = int(done.sum())
        if a.stage == "valid":
            np.save(f"{a.out}/recb_valid_ranks_clean_{a.name}.npy", out["clean"]); np.save(f"{a.out}/recb_valid_ranks_faith_{a.name}.npy", out["faith"])
            # TGB train(): per relation lambda (alpha = 1) then alpha
            sel = {}
            for v in ("clean", "faith"):
                Lr = np.full(2 * R, I_TRDEF_L); Ar = np.full(2 * R, I_TRDEF_A); detail = {}
                for rel in range(2 * R):
                    rows = Qrel[sp][rel]
                    if a.smoke: rows = rows[:a.smoke]
                    if len(rows) == 0:
                        continue
                    rr_ = (1.0 / out[v][rows]).astype(np.float32).astype(np.float64)       # official per-query float32 MRR
                    best_l, best_m = LAMS.index(0.1), 0.0
                    mL = rr_[:, :, len(ALPHAS) - 1].mean(0)
                    for li in range(len(LAMS)):
                        if mL[li] > best_m: best_m, best_l = mL[li], li
                    best_a, best_ma = ALPHAS.index(0.99), 0.0
                    mA = rr_[:, best_l, :].mean(0)
                    for ai in range(len(ALPHAS)):
                        if mA[ai] > best_ma: best_ma, best_a = mA[ai], ai
                    Lr[rel] = best_l; Ar[rel] = best_a
                    detail[rel] = dict(n=int(len(rows)), lam=LAMS[best_l], alpha=ALPHAS[best_a], mrr_lam=float(best_m), mrr_alpha=float(best_ma))
                sel[v] = dict(lam_idx=Lr.tolist(), alpha_idx=Ar.tolist(), lam=[LAMS[i] for i in Lr], alpha=[ALPHAS[i] for i in Ar],
                              detail=detail)
                ranks_def = out[v][:, I_DEF_L, I_DEF_A]
                ranks_tr = out[v][np.arange(nrow), Lr[sro[sp][:, 1]], Ar[sro[sp][:, 1]]]
                sel[v]["valid_MRR_default"] = float(np.mean((1.0 / ranks_def).astype(np.float32).astype(np.float64)))
                sel[v]["valid_MRR_trained"] = float(np.mean((1.0 / ranks_tr).astype(np.float32).astype(np.float64)))
                sel[v]["valid_H10_trained"] = float(np.mean(ranks_tr <= 10))
            sel["meta"] = meta
            json.dump(sel, open(f"{a.out}/recb_selection_{a.name}.json", "w"), indent=1)
            log("valid", {v: (sel[v]["valid_MRR_default"], sel[v]["valid_MRR_trained"]) for v in ("clean", "faith")})
        else:
            np.savez(f"{a.out}/recb_{sp}_ranks_{a.name}.npz", **out)
            # CSR in split order
            n_ = np.zeros(nrow, np.int64); parts = {}
            for (rel, lo_), (rows, c) in csr_parts.items():
                n_[rows] = c["n"]
            ptr = np.concatenate([[0], np.cumsum(n_)])
            cand = np.zeros(ptr[-1], np.int32); pd_ = np.zeros(ptr[-1], np.float32); pt_ = np.zeros(ptr[-1], np.float32)
            for (rel, lo_), (rows, c) in csr_parts.items():
                off = np.concatenate([[0], np.cumsum(c["n"])])
                for j, rw in enumerate(rows):
                    a0, a1 = off[j], off[j + 1]
                    cand[ptr[rw]:ptr[rw + 1]] = c["cand"][a0:a1]; pd_[ptr[rw]:ptr[rw + 1]] = c["psi_def"][a0:a1]
                    pt_[ptr[rw]:ptr[rw + 1]] = c["psi_tr"][a0:a1]
            selj = json.load(open(f"{a.out}/recb_selection_{a.name}.json"))
            np.savez(f"{a.out}/recb_csr_{sp}_{a.name}.npz", ptr=ptr, cand=cand, psi_def=pd_, psi_tr=pt_, sro=sro[sp],
                     alpha_tr=np.array(selj["clean"]["alpha"]), lam_tr=np.array(selj["clean"]["lam"]), alpha_def=0.99, lam_def=0.1)
            mets = {}
            for kk in out:
                rk = out[kk].astype(np.float64)
                mets[kk] = dict(MRR=float(np.mean((1.0 / out[kk]).astype(np.float32).astype(np.float64))), H1=float(np.mean(rk <= 1)),
                                H3=float(np.mean(rk <= 3)), H10=float(np.mean(rk <= 10)), n=int(len(rk)))
            meta[f"{sp}_metrics"] = mets
            meta[f"{sp}_csr_pairs"] = int(ptr[-1])
            log(sp, "metrics", json.dumps(mets) if sp == "valid" else "(written)")
    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta["only_valid"] = bool(a.only_valid)
    json.dump(meta, open(f"{a.out}/recb_{'validcsr' if a.only_valid else a.stage}_meta_{a.name}.json", "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
