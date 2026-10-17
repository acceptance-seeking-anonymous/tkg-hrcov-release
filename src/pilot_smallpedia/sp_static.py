"""pilot_smallpedia / sp_static.py -- static Wikidata KG index + typed static/mixed path enumeration (CPU, numpy).

Union graph = temporal history before t (EventIndex, both directions) U static Wikidata edges (time-less, both
directions, typed by pd = p (head->tail) or p + NP (tail->head)).  For a query (s, r, t), candidate o must be a temporal
entity (< N_ENT).  Path shapes:
   S  : s -pd-> o                                    type (0, pd, 0)
   SS : s -pd1-> x -pd2-> o      (x any node, static deg(x) <= cap)       type (1, pd1, pd2)
   ST : s -pd-> x -r'-> o        (x temporal entity; x's M_T most recent distinct events before t)  type (2, pd, r')
   TS : s -r'-> x -pd-> o        (x in s's M_T most recent distinct temporal partners before t; deg(x) <= cap) type (3, r', pd)
Type codes: ((shape * D1 + a) * D1 + b) with D1 = 2 * NP (>= 2R).  Path-type precision prior (target encoding, fitted on
early-train queries): P(c = gold | c in C_type), hierarchical back-off full=(r, type) -> type -> shape.
"""
import numpy as np

SH_S, SH_SS, SH_ST, SH_TS = 0, 1, 2, 3
M_T = 64          # temporal partners / events per node considered on a temporal hop
MAX_X = 128       # static intermediates expanded per query on the ST hop (lowest static degree first)
TE_K = 20.0       # pseudo-count (in candidate units) of the hierarchical smoothing


class StaticIndex:
    def __init__(self, st, n_ent, cap):
        h, t, p = st["h"], st["t"], st["p"]
        self.NT = st["n_nodes"]; self.NP = st["n_prel"]; self.D1 = 2 * self.NP; self.N = n_ent; self.cap = cap
        src = np.concatenate([h, t]); dst = np.concatenate([t, h]); pd = np.concatenate([p, p + self.NP])
        o = np.lexsort((pd, dst, src))
        self.dst = dst[o].astype(np.int64); self.pd = pd[o].astype(np.int64)
        self.indptr = np.zeros(self.NT + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=self.NT), out=self.indptr[1:])
        self.deg = np.diff(self.indptr)
        # inverse direction code of pd (for "shared attribute": s -pd-> x <-pd- o  <=>  path pd1 = pd, pd2 = inv(pd))
        self.inv = lambda q: np.where(q < self.NP, q + self.NP, q - self.NP)

    def nbrs(self, u):
        a, b = self.indptr[u], self.indptr[u + 1]
        return self.dst[a:b], self.pd[a:b]

    def multi_nbrs(self, us):
        """concatenated neighbour lists of an array of nodes -> (owner_index, dst, pd)."""
        a = self.indptr[us]; b = self.indptr[us + 1]; ln = b - a
        if ln.sum() == 0:
            z = np.zeros(0, np.int64); return z, z, z
        own = np.repeat(np.arange(len(us)), ln)
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a, b) if y > x])
        return own, self.dst[idx], self.pd[idx]


def temporal_partners(EI, u, t, M=M_T):
    """M most recent distinct (rel, dst) of u before t (rel in [0, 2R))."""
    if u >= EI.N:
        z = np.zeros(0, np.int64); return z, z
    rel, dst, _ = EI.recent(u, t, M)
    return rel, dst


def enumerate_paths(SI, EI, s, t, cap=None, uncapped=False, nodirect=False):
    """All typed paths s -> o (o temporal entity, o != s).  Returns dict of arrays (one row per DISTINCT
    (candidate, type) with path count): cand, shape, code, cnt  and the set sizes per shape.
    nodirect=True drops shape S (direct static edge s-o, any property / direction); SS / ST / TS are unchanged."""
    cap = SI.cap if cap is None else cap
    N = SI.N; D1 = SI.D1
    C, SH, CO, CT = [], [], [], []
    x1, p1 = SI.nbrs(s)
    # ---- S
    m = (x1 < N) & (x1 != s)
    if m.any() and not nodirect:
        C.append(x1[m]); SH.append(np.full(m.sum(), SH_S)); CO.append((SH_S * D1 + p1[m]) * D1); CT.append(np.ones(m.sum(), np.int64))
    # ---- SS (intermediate x with deg <= cap unless uncapped)
    keep = np.ones(len(x1), bool) if uncapped else (SI.deg[x1] <= cap)
    xs, ps = x1[keep], p1[keep]
    if len(xs):
        own, d2, q2 = SI.multi_nbrs(xs)
        m = (d2 < N) & (d2 != s)
        if m.any():
            C.append(d2[m]); SH.append(np.full(m.sum(), SH_SS)); CO.append((SH_SS * D1 + ps[own[m]]) * D1 + q2[m]); CT.append(np.ones(m.sum(), np.int64))
    # ---- ST: s -pd-> x (temporal entity) -r'-> o
    m = (x1 < N) & (x1 != s)
    if m.any():
        xt, pt = x1[m], p1[m]
        if len(xt) > MAX_X:
            o_ = np.argsort(SI.deg[xt], kind="stable")[:MAX_X]; xt, pt = xt[o_], pt[o_]
        for x, pdx in zip(xt.tolist(), pt.tolist()):
            rr, dd = temporal_partners(EI, x, t)
            mm = (dd != s)
            if mm.any():
                C.append(dd[mm]); SH.append(np.full(mm.sum(), SH_ST)); CO.append((SH_ST * D1 + pdx) * D1 + rr[mm]); CT.append(np.ones(mm.sum(), np.int64))
    # ---- TS: s -r'-> x -pd-> o
    rr, dd = temporal_partners(EI, s, t)
    if len(dd):
        k = np.ones(len(dd), bool) if uncapped else (SI.deg[dd] <= cap)
        dd, rr = dd[k], rr[k]
        if len(dd):
            own, d2, q2 = SI.multi_nbrs(dd)
            m = (d2 < N) & (d2 != s)
            if m.any():
                C.append(d2[m]); SH.append(np.full(m.sum(), SH_TS)); CO.append((SH_TS * D1 + rr[own[m]]) * D1 + q2[m]); CT.append(np.ones(m.sum(), np.int64))
    if not C:
        z = np.zeros(0, np.int64)
        return dict(cand=z, shape=z, code=z, cnt=z)
    c = np.concatenate(C); sh = np.concatenate(SH); co = np.concatenate(CO)
    key = co * N + c
    uk, cnt = np.unique(key, return_counts=True)
    cand = uk % N; code = uk // N; shape = code // (D1 * D1)
    return dict(cand=cand, shape=shape, code=code, cnt=cnt)


def type_levels(r, code, shape, R2, D1):
    """TE keys: full = (r, code), type = code, shape."""
    return (r * (4 * D1 * D1) + code), code, shape


class StaticTE:
    """hierarchical candidate-level precision of static path types: hits / sizes (summed over fit instances)."""
    def __init__(self, path=None, arrs=None):
        if path is not None:
            z = np.load(path); arrs = {k: z[k] for k in z.files}
        self.a = arrs

    def _lk(self, lvl, keys):
        K = self.a[f"{lvl}_k"]; Hh = self.a[f"{lvl}_h"]; Z = self.a[f"{lvl}_z"]
        if len(K) == 0:
            return np.zeros(len(keys)), np.zeros(len(keys))
        pos = np.clip(np.searchsorted(K, keys), 0, len(K) - 1); ok = K[pos] == keys
        return np.where(ok, Hh[pos], 0.0), np.where(ok, Z[pos], 0.0)

    def prec(self, r, code, shape, R2, D1):
        kf, kt, ks = type_levels(r, code, shape, R2, D1)
        hs, zs = self._lk("shape", ks); p_s = (hs + 1e-3) / (zs + 1.0)
        ht, zt = self._lk("type", kt); p_t = (ht + TE_K * p_s) / (zt + TE_K)
        hf, zf = self._lk("full", kf); p_f = (hf + TE_K * p_t) / (zf + TE_K)
        return p_f, np.log1p(zf)


def fit_counts(paths, gold, r, R2, D1):
    """per (query, type): size |C_type| and hit = gold in C_type -> keys + (h, z) for the three levels."""
    code = paths["code"]; cand = paths["cand"]; shape = paths["shape"]
    if len(code) == 0:
        return None
    uc, inv, size = np.unique(code, return_inverse=True, return_counts=True)
    hit = np.bincount(inv, weights=(cand == gold).astype(np.float64), minlength=len(uc))
    ush = uc // (D1 * D1)
    kf, kt, ks = type_levels(r, uc, ush, R2, D1)
    return dict(full=(kf, hit, size), type=(kt, hit, size), shape=(ks, hit, size))


def merge_counts(parts, lvl):
    k = np.concatenate([p[lvl][0] for p in parts]); h = np.concatenate([p[lvl][1] for p in parts]); z = np.concatenate([p[lvl][2] for p in parts])
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=h), np.bincount(inv, weights=z.astype(np.float64))


def cand_stats(paths, r, TE, SI, R2):
    """per distinct candidate: counts by shape, best log-precision, summed precision, shared-attribute count."""
    cand = paths["cand"]
    if len(cand) == 0:
        return None
    p_f, _ = TE.prec(r, paths["code"], paths["shape"], R2, SI.D1)
    uc, inv = np.unique(cand, return_inverse=True)
    n = len(uc)
    cnt = paths["cnt"].astype(np.float64)
    by = np.zeros((n, 4))
    np.add.at(by, (inv, paths["shape"]), cnt)
    best = np.full(n, -np.inf); np.maximum.at(best, inv, np.log(p_f))
    ssum = np.bincount(inv, weights=p_f * np.minimum(cnt, 5.0), minlength=n)
    # shared attribute: SS with pd2 == inverse(pd1)
    code = paths["code"]; D1 = SI.D1
    a = (code // D1) % D1; b = code % D1
    sa = (paths["shape"] == SH_SS) & (b == SI.inv(a))
    shared = np.bincount(inv, weights=np.where(sa, cnt, 0.0), minlength=n)
    return dict(cand=uc, by=by, best=best, ssum=ssum, shared=shared)
