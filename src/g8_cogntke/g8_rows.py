"""G8 / g8_rows.py -- row source for hd_blend.py / hd_lookup.py (copies in this directory, --source cogn): exact CognTKE
log-prob rows rebuilt from the sparse store written by g8_dump.py (lp = -lse for unreached entities, logit - lse for reached;
bitwise identical to the dump's lp). G8_NINF=1: unreached entities at -1e30 (= -inf for ranking; finite so the official-negative mask (-inf) stays below it)
(base-only sensitivity, not in the decision).
Yields (split, ts, sro [n,4] harness split order, fn(s_q, r_q) -> [b, N] cuda rows, facts)."""
import os, numpy as np, torch


class Store:
    def __init__(self, d, sp, dev):
        self.ptr = np.load(f"{d}/rows_{sp}_ptr.npy"); self.ids = np.load(f"{d}/rows_{sp}_ids.npy", mmap_mode='r')
        self.val = np.load(f"{d}/rows_{sp}_val16.npy", mmap_mode='r'); self.lse = np.load(f"{d}/rows_{sp}_lse.npy")
        self.sro = np.load(f"{d}/rows_{sp}_sro.npy"); self.dev = dev; self.ninf = os.environ.get('G8_NINF', '0') == '1'
    def dense(self, rows, N):
        rows = np.asarray(rows, np.int64); B = len(rows)
        lse = torch.as_tensor(self.lse[rows], device=self.dev)
        out = (torch.full((B, N), -1e30, device=self.dev) if self.ninf else (-lse).unsqueeze(1).repeat(1, N))
        st, en = self.ptr[rows], self.ptr[rows + 1]; ln = en - st
        idx = np.concatenate([np.arange(s, e) for s, e in zip(st, en)]) if B else np.zeros(0, np.int64)
        rr = torch.as_tensor(np.repeat(np.arange(B), ln), device=self.dev)
        cc = torch.as_tensor(np.asarray(self.ids[idx]).astype(np.int64), device=self.dev)
        vv = torch.as_tensor(np.asarray(self.val[idx]), device=self.dev).float()
        out[rr, cc] = vv - lse[rr]
        return out


def source(cfg, store_dir, splits_needed, dev):
    N, R = cfg["N"], cfg["R"]
    def lq(fp):
        return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
    Q = {sp: lq(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    S = {sp: Store(store_dir, sp, dev) for sp in splits_needed}
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        uts, st, en = *np.unique(q[:, 3], return_index=True), None
        en = np.append(st[1:], len(q)); ptr = 0
        for u, b, e in zip(uts, st, en):
            trip = q[b:e]
            if sp in splits_needed:
                inv = trip[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
                sro = np.concatenate([trip, inv]).astype(np.int64); n = len(sro)
                assert (S[sp].sro[ptr:ptr + n].astype(np.int64) == sro).all(), "store order != harness split order"
                key = {}
                for j in range(n): key.setdefault((int(sro[j, 0]), int(sro[j, 1])), ptr + j)
                def fn(s_q, r_q, key=key, st_=S[sp]):
                    return st_.dense([key[(int(s), int(r))] for s, r in zip(s_q, r_q)], N)
                yield sp, int(u), sro, fn, trip
                ptr += n
            else:
                yield sp, int(u), None, None, trip
