"""
retriever track (idea 6.1) / patterns.py -- entity-anonymised multi-hop path PATTERNS.

A pattern is the label sequence of a path that starts at the query subject s, with every
entity on the path anonymised:
    1-hop : (rel1, lag1)                      s --rel1@lag1--> c
    2-hop : (rel1, lag1, rel2, lag2)          s --rel1@lag1--> x --rel2@lag2--> c
rel in [0, 2R) (direction folded in: r' >= R is the inverse direction), lag = log2-bucket of
(t_query - t_event) in snapshots, measured from the QUERY time for both hops.  Only events with
t_event < t_query are used (all splits: the frozen bases are also evaluated with ground-truth
history before t).  Instantiating a pattern on s's history yields its candidate set C_p(q).

Caps (tractability, documented in the report): hop-1 keeps the M1 most recent DISTINCT (rel1, x)
pairs of s; hop-2 keeps, for every distinct intermediate x, its M2 most recent distinct (rel2, c)
pairs with c != s.  The most recent occurrence defines the lag bucket of a distinct pair.
"""
import numpy as np

L_BUCKETS = 7          # lag buckets: 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | >=64 snapshots


def lag_bucket(lag):
    lag = np.maximum(np.asarray(lag, np.int64), 1)
    return np.minimum(np.floor(np.log2(lag)).astype(np.int64), L_BUCKETS - 1)


class EventIndex:
    """Per-entity time-sorted event lists over ALL splits, both directions."""

    def __init__(self, quads_all, n_ent, R, step):
        q = np.asarray(quads_all, np.int64)
        h, r, t, ts = q[:, 0], q[:, 1], q[:, 2], q[:, 3] // step
        src = np.concatenate([h, t]); rel = np.concatenate([r, r + R])
        dst = np.concatenate([t, h]); tt = np.concatenate([ts, ts])
        o = np.lexsort((tt, src))
        self.src = src[o]; self.rel = rel[o].astype(np.int32); self.dst = dst[o].astype(np.int32)
        self.ts = tt[o].astype(np.int32)
        self.indptr = np.zeros(n_ent + 1, np.int64)
        np.cumsum(np.bincount(self.src, minlength=n_ent), out=self.indptr[1:])
        self.N = n_ent; self.R = R; self.step = step

    def recent(self, u, t_snap, M, exclude=-1, scan_mult=6):
        """M most recent DISTINCT (rel, dst) pairs of u strictly before t_snap.
        Returns rel[int64], dst[int64], lag[int64] (most-recent-first)."""
        lo, hi = self.indptr[u], self.indptr[u + 1]
        if hi == lo:
            return _E, _E, _E
        pos = lo + np.searchsorted(self.ts[lo:hi], t_snap, "left")
        if pos == lo:
            return _E, _E, _E
        a = max(lo, pos - scan_mult * M)
        rel = self.rel[a:pos][::-1].astype(np.int64); dst = self.dst[a:pos][::-1].astype(np.int64)
        lag = t_snap - self.ts[a:pos][::-1].astype(np.int64)
        if exclude >= 0:
            k = dst != exclude
            rel, dst, lag = rel[k], dst[k], lag[k]
        key = rel * self.N + dst
        _, first = np.unique(key, return_index=True)
        first.sort()
        first = first[:M]
        return rel[first], dst[first], lag[first]

    def n_events_before(self, u, t_snap):
        lo, hi = self.indptr[u], self.indptr[u + 1]
        return int(np.searchsorted(self.ts[lo:hi], t_snap, "left"))

    def dyad_events(self, s, c_arr, t_snap, window, max_ev=8):
        """BridgeMem-style dyadic history: for each candidate c, the <=max_ev most recent events on
        the ordered pair (s, c) within `window` snapshots before t.  Returns dict c -> (rel, lag)."""
        lo, hi = self.indptr[s], self.indptr[s + 1]
        pos = lo + np.searchsorted(self.ts[lo:hi], t_snap, "left")
        a = lo + np.searchsorted(self.ts[lo:hi], t_snap - window, "left")
        return self.rel[a:pos], self.dst[a:pos], t_snap - self.ts[a:pos]


_E = np.zeros(0, np.int64)


def instantiate(EI, s, t_snap, M1=64, M2=32):
    """All 1-hop and 2-hop patterns of subject s at query time t_snap.
    Returns dict:
      pid   [P] int64   pattern ids (1-hop: rel1*L+lb1 ; 2-hop: OFF2 + ((rel1*L+lb1)*2R+rel2)*L+lb2)
      hop   [P] int8
      pc_p  [Q] int32   pattern index of each (pattern, candidate) pair  (unique pairs)
      pc_c  [Q] int32   candidate entity of each pair
      npath [P] int32   number of distinct paths realising the pattern
      ninterm [P] int32 distinct intermediates (1 for hop-1)
      n_hop1 int        distinct hop-1 pairs of s
    or None if s has no history before t."""
    L = L_BUCKETS; R2 = 2 * EI.R
    rel1, x, lag1 = EI.recent(s, t_snap, M1, exclude=s)
    if len(rel1) == 0:
        return None
    lb1 = lag_bucket(lag1)
    p1 = rel1 * L + lb1                              # hop-1 pattern id (< R2*L)
    OFF2 = R2 * L
    # hop-2 expansion per distinct intermediate
    ux, inv = np.unique(x, return_inverse=True)
    pp, cc, mm = [p1], [x], [x]                      # paths: pattern id, candidate, intermediate (hop1: interm = c itself)
    for j, xv in enumerate(ux):
        rel2, c2, lag2 = EI.recent(int(xv), t_snap, M2, exclude=s)
        if len(rel2) == 0:
            continue
        q2 = rel2 * L + lag_bucket(lag2)             # (rel2, lb2) code < R2*L
        e1 = np.where(inv == j)[0]                  # hop-1 edges into xv
        P = (p1[e1][:, None] * (R2 * L) + q2[None, :]).ravel() + OFF2
        pp.append(P); cc.append(np.tile(c2, len(e1))); mm.append(np.full(len(P), xv, np.int64))
    pp = np.concatenate(pp); cc = np.concatenate(cc); mm = np.concatenate(mm)
    pid, pinv = np.unique(pp, return_inverse=True)
    # unique (pattern, candidate) pairs
    key = pinv.astype(np.int64) * EI.N + cc
    ukey, kfirst, kcount = np.unique(key, return_index=True, return_counts=True)
    pc_p = (ukey // EI.N).astype(np.int32); pc_c = (ukey % EI.N).astype(np.int32)
    npath = np.bincount(pinv, minlength=len(pid)).astype(np.int32)
    # distinct intermediates per pattern
    mkey = np.unique(pinv.astype(np.int64) * EI.N + mm)
    ninterm = np.bincount((mkey // EI.N).astype(np.int64), minlength=len(pid)).astype(np.int32)
    hop = np.where(pid >= OFF2, 2, 1).astype(np.int8)
    return dict(pid=pid, hop=hop, pc_p=pc_p, pc_c=pc_c, npath=npath, ninterm=ninterm, n_hop1=int(len(rel1)))


def decode(pid, R, L=L_BUCKETS):
    """pattern id -> (hop, rel1, lb1, rel2, lb2) arrays; rel2/lb2 = -1 for hop-1."""
    pid = np.asarray(pid, np.int64); R2 = 2 * R; OFF2 = R2 * L
    h2 = pid >= OFF2
    a = np.where(h2, (pid - OFF2) // (R2 * L), pid)
    b = np.where(h2, (pid - OFF2) % (R2 * L), -1)
    rel1 = a // L; lb1 = a % L
    rel2 = np.where(h2, b // L, -1); lb2 = np.where(h2, b % L, -1)
    return np.where(h2, 2, 1), rel1, lb1, rel2, lb2
