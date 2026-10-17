#!/usr/bin/env python3
"""phase2_fresh_window / fh_blend.py = presub hd_blend.py + the gdeltfw dataset entry (no other change) --
recurrency-blend controls and scoring of every claim-bearing arm.

BASE(+)RecB : full-row blend of the base probabilities p with q = RecB / sum(RecB)  (RecB = Gastinger psi-Delta-xi,
              clean variant, default or per-relation trained; psi from hd_recb.py CSR files, xi recomputed here from the
              online fact counts):  MIX  s = (1-beta) p + beta q      LOGLIN  s = log p + w log(q + eps)
RecB-POOL   : pool = base top-K  U  RecB top-K (RecB > 0), ranked by the blend s; everything else after, in base order.
--stage valid : harness exact filtered ranks of every valid row for all 42 blend + 126 pool configurations.
--stage test  : the valid-selected blend / pool configuration only (selection json written by hd_select.py), plus the
                base and the re-ranker systems (promoted lists lifted above the base row, pilot convention); every arm is
                scored by the official tgb.linkproppred.evaluate.Evaluator per batch with the official negatives
                (TKGNegativeEdgeSampler.query_batch on the shipped test_ns.pkl; non-negatives masked to -inf, which the
                evaluator's optimistic/pessimistic counts ignore).  Per-query RR = the evaluator's own formula on the same
                arrays; the batch mean is compared with Evaluator.eval's batch MRR (max deviation recorded, not asserted).  Harness strict ranks alongside.
Row sources: replay (RE-GCN / LogCL checkpoints, ground-truth online history, as pc_lookup.py), fullrows (float32 memmaps
written by sp_fullrows.py / hd_fullrows), tirgn (hd_tirgn.py checkpoint replay).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

HAR = os.environ.get("HD_HAR", "/home/anon/cs_scratch/open_ideas_harness")          # serverB: a mirror of the needed serverA files
TR = f"{HAR}/tracks"
HD = os.environ.get("HD_DIR", f"{TR}/hardening")
SCR = os.environ.get("HD_SCR", "/home/anon/cs_scratch")
DSCFG = {
    "polecat": dict(name="tkgl-polecat", dd=f"{TR}/datasets/harness_data/tkgl-polecat", N=150931, R=16,
                    valid_rows=f"{TR}/pilot_polecat/work/valid_sample_idx.npy", ns=f"{TR}/pilot_polecat/work/tgb_ns/tkgl-polecat_test_ns.pkl",
                    zip=f"{TR}/datasets/raw/tkgl-polecat.zip", stage=f"{TR}/pilot_polecat/stage"),
    # presub G2 (RE-GCN split, no TGB evaluator): harness exact filtered ranks for every arm (official=False)
    "icews18": dict(name="ICEWS18", dd=f"{SCR}/logcl_redun/data/ICEWS18", N=23033, R=256, valid_rows=None, ns=None, zip=None,
                    stage=None, official=False),
    "gdelt": dict(name="GDELT", dd=f"{SCR}/gdelt_armAD/data/GDELT", N=7691, R=240,
                  valid_rows=os.environ.get("G2_VALID_ROWS", f"{HD}/presub_g2/GDELT/work/valid_sample_idx.npy"), ns=None, zip=None,
                  stage=None, official=False),
    # forward holdout (phase2_fresh_window): GDELTFW post-cutoff slice, harness exact filtered ranks (no TGB evaluator)
    "gdeltfw": dict(name="GDELTFW", dd=os.environ.get("FH_DD", f"{TR}/phase2_fresh_window/data/GDELTFW"), N=1209, R=215,
                     valid_rows=os.environ.get("G2_VALID_ROWS", f"{TR}/phase2_fresh_window/GDELTFW/work/valid_sample_idx.npy"),
                     ns=None, zip=None, stage=None, official="allent"),
    "smallpedia": dict(name="tkgl-smallpedia", dd=f"{TR}/datasets/harness_data/tkgl-smallpedia", N=47433, R=283,
                       valid_rows=None, ns=f"{TR}/pilot_smallpedia/work/tgb_ns/tkgl-smallpedia_test_ns.pkl",
                       zip=f"{TR}/datasets/raw/tkgl-smallpedia.zip", stage=f"{TR}/pilot_smallpedia/stage"),
}
BETAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
LOGW = [(w, e) for w in (0.25, 0.5, 1.0, 2.0) for e in (1e-5, 1e-3)]
BLEND = [(v, "MIX", b) for v in ("def", "tr") for b in BETAS] + [(v, "LOGLIN", we) for v in ("def", "tr") for we in LOGW]
KS = [20, 50, 100]
POOL = [(K, ci) for K in KS for ci in range(len(BLEND))]
PIDX = {kc: j for j, kc in enumerate(POOL)}


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def split_queries(q, R, split):
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q)); out = []
    for k, (u, b, e) in enumerate(zip(uts, st, en)):
        if split == "train" and k == 0: continue
        m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        out.append(m.copy()); out.append(inv)
    return np.concatenate(out).astype(np.int64)


class Filt:
    def __init__(self, Q, N, R):
        allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
        h, r, t, ts = allq.T
        s = np.concatenate([h, t]); rr = np.concatenate([r, r + R]); o = np.concatenate([t, h]); tt = np.concatenate([ts, ts])
        key = (tt * (2 * R) + rr) * N + s
        od = np.lexsort((o, key)); key = key[od]; o = o[od]
        self.keys, self.start = np.unique(key, return_index=True); self.end = np.append(self.start[1:], len(key)); self.objs = o
        self.N, self.R = N, R

    def pairs(self, sro):
        """(row, col) of OTHER true answers for each query row (gold excluded)."""
        fk = (sro[:, 3] * (2 * self.R) + sro[:, 1]) * self.N + sro[:, 0]
        i0 = np.searchsorted(self.keys, fk); i0c = np.clip(i0, 0, len(self.keys) - 1)
        ok = self.keys[i0c] == fk
        ri, ci = [], []
        for j in np.flatnonzero(ok):
            ob = self.objs[self.start[i0c[j]]:self.end[i0c[j]]]
            ob = ob[ob != sro[j, 2]]
            if len(ob): ri.append(np.full(len(ob), j)); ci.append(ob)
        if not ri: return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(ri), np.concatenate(ci)


class XiCounts:
    """online object counts per relation (both directions) of all facts with ts < current t (GPU)."""
    def __init__(self, N, R, dev):
        self.C = torch.zeros((2 * R, N), device=dev); self.tot = torch.zeros(2 * R, device=dev); self.R = R

    def add(self, trip):
        if len(trip) == 0: return
        tt = torch.as_tensor(trip, device=self.C.device)
        h, r, o = tt[:, 0], tt[:, 1], tt[:, 2]
        self.C.index_put_((r, o), torch.ones(len(r), device=self.C.device), accumulate=True)
        self.C.index_put_((r + self.R, h), torch.ones(len(r), device=self.C.device), accumulate=True)
        self.tot.index_add_(0, r, torch.ones(len(r), device=self.C.device)); self.tot.index_add_(0, r + self.R, torch.ones(len(r), device=self.C.device))

    def xi(self, rel_t):
        return self.C[rel_t] / self.tot[rel_t].clamp(min=1.0).unsqueeze(1)


class RecbCSR:
    def __init__(self, path):
        z = np.load(path)
        self.ptr = z["ptr"]; self.cand = z["cand"]; self.psi = {"def": z["psi_def"], "tr": z["psi_tr"]}
        self.alpha = {"def": None, "tr": np.asarray(z["alpha_tr"], np.float64)}; self.sro = z["sro"]

    def rows(self, idx, B, N, dev, variant, rel):
        a, b = self.ptr[idx], self.ptr[idx + 1]; ln = b - a
        q = np.repeat(np.arange(len(idx)), ln)
        pos = np.repeat(a - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())
        Psi = torch.zeros((B, N), device=dev)
        if len(pos):
            Psi[torch.as_tensor(q, device=dev), torch.as_tensor(self.cand[pos].astype(np.int64), device=dev)] = \
                torch.as_tensor(self.psi[variant][pos], device=dev)
        al = np.full(len(idx), 0.99) if variant == "def" else self.alpha["tr"][rel]
        return Psi, torch.as_tensor(al, device=dev, dtype=torch.float32)


def recb_row(csr, idx, rel, xi, dev, variant):
    B, N = xi.shape
    Psi, al = csr.rows(idx, B, N, dev, variant, rel)
    S = (1000 * al).unsqueeze(1) * Psi + (1000 * (1 - al)).unsqueeze(1) * xi
    return S


def blend_scores(lp, q, cfg):
    v, form, par = cfg
    if form == "MIX":
        return (1 - par) * lp.exp() + par * q
    w, e = par
    return lp + w * torch.log(q + e)


def frank(s, fr, fc, gold):
    """harness strict filtered rank: 1 + #(non-filtered entities with score > gold score)."""
    ar = torch.arange(len(gold), device=s.device)
    g = s[ar, gold]
    lf = s.clone()
    if len(fr): lf[fr, fc] = float("-inf")
    lf[ar, gold] = float("-inf")
    return 1 + (lf > g.unsqueeze(1)).sum(1)


def pool_rank(lp, s, pool, fr, fc, gold):
    ar = torch.arange(len(gold), device=lp.device)
    filt = torch.zeros_like(pool)
    if len(fr): filt[fr, fc] = True
    filt[ar, gold] = True
    unf = pool & ~filt
    gin = pool[ar, gold]
    g_s = s[ar, gold]; g_lp = lp[ar, gold]
    r_in = 1 + (unf & (s > g_s.unsqueeze(1))).sum(1)
    r_out = 1 + unf.sum(1) + ((~pool) & ~filt & (lp > g_lp.unsqueeze(1))).sum(1)
    return torch.where(gin, r_in, r_out)


def pool_mask(lp, R_, K):
    B, N = lp.shape
    M = torch.zeros((B, N), dtype=torch.bool, device=lp.device)
    M.scatter_(1, torch.topk(lp, K, dim=1).indices, True)
    tv, ti = torch.topk(R_, K, dim=1)
    M2 = torch.zeros_like(M); M2.scatter_(1, ti, tv > 0)
    return M | M2


def lift(lp, ids):
    """promoted ids [B, W] (-1 pad, final order) lifted above everything (pilot convention)."""
    Y = lp.clone()
    W = ids.shape[1]
    val = (1e6 - torch.arange(W, device=lp.device, dtype=torch.float32)).unsqueeze(0).expand_as(ids)
    m = ids >= 0
    rr = torch.arange(len(ids), device=lp.device).unsqueeze(1).expand_as(ids)
    Y[rr[m], ids[m]] = val[m]
    return Y


def pool_lift(lp, s, pool):
    key = torch.where(pool, s, torch.full_like(s, float("-inf")))
    kmax = int(pool.sum(1).max())
    tv, ti = torch.topk(key, max(kmax, 1), dim=1)
    ids = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
    return lift(lp, ids)


# --------------------------------------------------------------------------------------------- row sources
def replay_source(cfg, base, ckpt, L, splits_needed, dev, log):
    """yields (split, ts, sro[n,4] in split order, lp [n,N] cuda) per snapshot, plus the snapshot facts for xi."""
    sys.path.insert(0, HAR)
    from regcn import read_quads, group_by_time, make_edges, REGCN
    from logcl import LogCL, GlobalHist
    N, R = cfg["N"], cfg["R"]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = (REGCN if base.startswith("regcn") else LogCL)(N, R, 200, 2, 0.2, L).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    pairs = {sp: group_by_time(read_quads(os.path.join(cfg["dd"], f"{sp}.txt"))) for sp in ("train", "valid", "test")}
    hist = []; gh = GlobalHist(N) if base.startswith("logcl") else None
    last = max(("train", "valid", "test").index(s) for s in splits_needed)
    with torch.no_grad():
        for spi, sp in enumerate(("train", "valid", "test")):
            if spi > last: break
            for ts, trip in pairs[sp]:
                if sp in splits_needed and len(trip):
                    s_all = np.concatenate([trip[:, 0], trip[:, 2]]); r_all = np.concatenate([trip[:, 1], trip[:, 1] + R])
                    o_all = np.concatenate([trip[:, 2], trip[:, 0]])
                    sro = np.stack([s_all, r_all, o_all, np.full(len(s_all), ts)], 1).astype(np.int64)
                    yield sp, ts, sro, (lambda s_q, r_q, h=hist[-L:]: F.log_softmax(
                        model(h, torch.as_tensor(s_q, device=dev), torch.as_tensor(r_q, device=dev))[0] if gh is None else
                        model(h, torch.as_tensor(s_q, device=dev), torch.as_tensor(r_q, device=dev), gh.freq_vec(s_q, r_q, dev))[0], dim=1)), trip
                else:
                    yield sp, ts, None, None, trip
                if len(trip):
                    hist.append(make_edges(trip, R, dev))
                    if gh is not None: gh.update(trip, R)


def fullrow_source(cfg, paths, splits_needed, dev):
    Q = {sp: load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    R = cfg["R"]
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
        if sp in splits_needed:
            MM = np.memmap(paths[sp], dtype=np.float32, mode="r").reshape(-1, cfg["N"])
            ptr = 0
        for u, b, e in zip(uts, st, en):
            trip = q[b:e]
            if sp in splits_needed:
                inv = trip[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
                sro = np.concatenate([trip, inv]).astype(np.int64)
                n = len(sro); lo = ptr
                yield sp, int(u), sro, (lambda s_q, r_q, lo=lo, n=n, cache={}: None), trip, (lo, lo + n)
                ptr += n
            else:
                yield sp, int(u), None, None, trip, None


def recb_source(cfg, splits_needed):
    """presub G1: snapshot iterator without a base model; the score row is RecB itself (set in the batch loop)."""
    Q = {sp: load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    R = cfg["R"]
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
        for u, b, e in zip(uts, st, en):
            trip = q[b:e]
            if sp in splits_needed:
                inv = trip[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
                yield sp, int(u), np.concatenate([trip, inv]).astype(np.int64), None, trip, None
            else:
                yield sp, int(u), None, None, trip, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DSCFG)); ap.add_argument("--base", required=True)
    ap.add_argument("--stage", required=True, choices=["valid", "test"]); ap.add_argument("--source", required=True)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--fullrows", default="", help="valid_path,test_path")
    ap.add_argument("--recb_dir", default=f"{HD}/work/recb"); ap.add_argument("--sel", default="")
    ap.add_argument("--systems", default="", help="name=path.npy,...  (test split order)")
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--out", default=".")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--tirgn_tree", default=f"{HD}/stage/TiRGN")
    ap.add_argument("--no_controls", action="store_true", help="test stage: base row + systems only (G1: --source recb)")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    cfg = DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    OFFICIAL = cfg.get("official", True)
    torch.set_num_threads(a.threads)
    Q = {sp: load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = Filt(Q, N, R)
    xi = XiCounts(N, R, dev)
    sp_need = ("valid",) if a.stage == "valid" else ("test",)
    sro_full = {sp: split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
    if a.stage == "valid":
        rows_req = np.load(cfg["valid_rows"]) if cfg["valid_rows"] else np.arange(len(sro_full["valid"]))
    else:
        rows_req = np.arange(len(sro_full["test"]))
    if a.smoke: rows_req = rows_req[:a.smoke]
    req = np.zeros(len(sro_full[sp_need[0]]), bool); req[rows_req] = True
    slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(len(rows_req))
    csr = RecbCSR(f"{a.recb_dir}/recb_csr_{sp_need[0]}_{cfg['name']}.npz")
    assert (csr.sro == sro_full[sp_need[0]]).all(), "RecB CSR row order != split order"
    n = len(rows_req)
    meta = dict(ds=a.ds, base=a.base, stage=a.stage, source=a.source, n=int(n), device=dev,
                gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None), host=os.environ.get("HD_HOST", ""),
                blend_cfgs=[list(map(str, c)) for c in BLEND], pool_cfgs=[[K, ci] for K, ci in POOL])
    if a.stage == "valid":
        RK = np.zeros((n, len(BLEND) + len(POOL)), np.float32); RB = np.zeros(n, np.float32)
    else:
        if not a.no_controls:
            sel = json.load(open(a.sel))[a.base]
            bsel = BLEND[sel["blend_idx"]]; psel = POOL[sel["pool_idx"]]
            meta["selected"] = dict(blend=list(map(str, bsel)), pool=[psel[0], list(map(str, BLEND[psel[1]]))])
        systems = {}
        if a.systems:
            for kv in a.systems.split(","):
                nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r"); assert len(systems[nm]) == n or a.smoke
        arms = ["base"] + ([] if a.no_controls else ["BLEND", "POOL"]) + list(systems)
        OUT = {k: np.zeros(n, np.float32) for nm in arms for k in (f"rr_{nm}", f"rank_{nm}", f"hrank_{nm}")}
        max_dev = 0.0
    if a.stage == "test" and OFFICIAL is True:
        from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
        from tgb.linkproppred.evaluate import Evaluator
        import importlib.metadata as im
        import zipfile
        ns = cfg["ns"]
        if not os.path.exists(ns):
            os.makedirs(os.path.dirname(ns), exist_ok=True)
            with zipfile.ZipFile(cfg["zip"]) as z: z.extract(os.path.basename(ns), os.path.dirname(ns))
        sampler = TKGNegativeEdgeSampler(cfg["name"], first_dst_id=0, last_dst_id=N - 1)
        sampler.load_eval_set(ns, split_mode="test")
        evaluator = Evaluator(name=cfg["name"]); meta["py_tgb"] = im.version("py-tgb")
        h2t = np.load(f"{cfg['stage']}/h2t_entity.npy"); t2h = np.load(f"{cfg['stage']}/t2h_entity.npy")
        assert (h2t == np.arange(N)).all() and (t2h == np.arange(N)).all(), "non-identity TGB id map (handled by pilots; not here)"
        tsraw = {}
        for ln in open(f"{cfg['dd']}/timestamp2id.txt"):
            k, v = ln.split(); tsraw[int(v)] = int(k)
        max_dev = 0.0
    # ------------------------------------------------------------------ iterate snapshots
    if a.source == "replay":
        it = ((sp, ts, sro, fn, trip, None) for sp, ts, sro, fn, trip in replay_source(cfg, a.base, a.ckpt, a.L, sp_need, dev, log))
    elif a.source == "fullrows":
        vp, tp = a.fullrows.split(",")
        it = fullrow_source(cfg, {"valid": vp, "test": tp}, sp_need, dev)
        MMr = np.memmap({"valid": vp, "test": tp}[sp_need[0]], dtype=np.float32, mode="r").reshape(-1, N)
    elif a.source == "recb":
        it = recb_source(cfg, sp_need)
    elif a.source == "tirgn":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hd_tirgn_replay as TRP
        it = ((sp, ts, sro, fn, trip, None) for sp, ts, sro, fn, trip in TRP.source(cfg, a.ckpt, a.L, sp_need, dev, a.tirgn_tree))
    pos = {"valid": 0, "test": 0}
    tq = time.time(); nsnap = 0
    for sp, ts, sro, fn, trip, rng in it:
        if sro is not None:
            p0 = pos[sp]; nq = len(sro); pos[sp] += nq
            idx_all = np.arange(p0, p0 + nq)
            loc = np.flatnonzero(req[idx_all])
            if len(loc):
                assert (sro[loc] == sro_full[sp][idx_all[loc]]).all()
                for b0 in range(0, len(loc), a.bs):
                    lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[rows]
                    s_q, r_q, o_q = sro[lb, 0], sro[lb, 1], sro[lb, 2]
                    if a.source == "fullrows":
                        lp = torch.as_tensor(np.asarray(MMr[rows]), device=dev)
                    elif a.source == "recb":
                        lp = None
                    else:
                        lp = fn(s_q, r_q)
                    B = len(lb)
                    gold = torch.as_tensor(o_q, device=dev)
                    fr_, fc_ = Fi.pairs(sro[lb])
                    fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                    X = xi.xi(torch.as_tensor(r_q, device=dev))
                    Rv = {v: recb_row(csr, rows, r_q, X, dev, v) for v in ("def", "tr")}
                    qv = {}
                    for v, S in Rv.items():
                        ssum = S.sum(1, keepdim=True)
                        qv[v] = torch.where(ssum > 0, S / ssum.clamp(min=1e-30), torch.zeros_like(S))
                    if a.source == "recb":     # G1: the generator row is RecB trained (clean), as probabilities (ties stay ties)
                        lp = qv["tr"]
                    if a.stage == "valid":
                        RB[sl] = frank(lp, fr, fc, gold).float().cpu().numpy()
                        pm = {(K, v): pool_mask(lp, Rv[v], K) for K in KS for v in ("def", "tr")}
                        res_ = torch.zeros((B, len(BLEND) + len(POOL)), device=dev)
                        for ci, c in enumerate(BLEND):
                            s = blend_scores(lp, qv[c[0]], c)
                            res_[:, ci] = frank(s, fr, fc, gold).float()
                            for K in KS:
                                res_[:, len(BLEND) + PIDX[(K, ci)]] = pool_rank(lp, s, pm[(K, c[0])], fr, fc, gold).float()
                            del s
                        RK[sl] = res_.cpu().numpy(); del pm, res_
                    else:
                        Ys = {"base": lp}
                        if not a.no_controls:
                            Ys["BLEND"] = blend_scores(lp, qv[bsel[0]], bsel)
                            pc_ = BLEND[psel[1]]
                            sp_ = blend_scores(lp, qv[pc_[0]], pc_)
                            Ys["POOL"] = pool_lift(lp, sp_, pool_mask(lp, Rv[pc_[0]], psel[0]))
                        for nm, arr in systems.items():
                            ids = torch.as_tensor(np.asarray(arr[rows]).astype(np.int64), device=dev)
                            Ys[nm] = lift(lp, ids)
                        if OFFICIAL == "allent":   # phase2_fresh_window (v79): all-entity time-aware filter, ties averaged + strict alongside
                            arA = torch.arange(len(gold), device=dev)
                            MA = torch.ones((len(gold), N), dtype=torch.bool, device=dev); MA[arA, gold] = False
                            if len(fr): MA[fr, fc] = False
                            for nm, Y in Ys.items():
                                pos_ = Y[arA, gold]; Yn = torch.where(MA, Y, torch.full_like(Y, float("-inf")))
                                opt = (Yn > pos_.unsqueeze(1)).sum(1); pes = (Yn >= pos_.unsqueeze(1)).sum(1)
                                rk = 0.5 * (opt + pes).double() + 1
                                OUT[f"rr_{nm}"][sl] = (1.0 / rk).float().cpu().numpy(); OUT[f"rank_{nm}"][sl] = rk.float().cpu().numpy()
                                OUT[f"hrank_{nm}"][sl] = frank(Y, fr, fc, gold).float().cpu().numpy()
                                del Yn
                            del Ys, lp, X, Rv, qv, MA
                            continue
                        if not OFFICIAL:      # presub G2: harness exact filtered rank for every arm (same tie convention)
                            for nm, Y in Ys.items():
                                hk = frank(Y, fr, fc, gold).float()
                                OUT[f"rr_{nm}"][sl] = (1.0 / hk).cpu().numpy(); OUT[f"rank_{nm}"][sl] = hk.cpu().numpy()
                                OUT[f"hrank_{nm}"][sl] = hk.cpu().numpy()
                            del Ys, lp, X, Rv, qv
                            continue
                        # official negatives (TGB ids == harness ids, asserted)
                        negs = sampler.query_batch(h2t[s_q], h2t[o_q], np.array([tsraw[int(ts)]] * B), r_q, split_mode="test")
                        ln_ = np.array([len(x) for x in negs])
                        M = torch.zeros((B, N), dtype=torch.bool, device=dev)
                        M[torch.as_tensor(np.repeat(np.arange(B), ln_), device=dev), torch.as_tensor(np.concatenate(negs).astype(np.int64), device=dev)] = True
                        ar = torch.arange(B, device=dev)
                        for nm, Y in Ys.items():
                            pos_ = Y[ar, gold]
                            Yn = torch.where(M, Y, torch.full_like(Y, float("-inf")))
                            opt = (Yn > pos_.unsqueeze(1)).sum(1); pes = (Yn >= pos_.unsqueeze(1)).sum(1)
                            rk = 0.5 * (opt + pes).double() + 1
                            m = evaluator.eval({"y_pred_pos": pos_.cpu().numpy(), "y_pred_neg": Yn.cpu().numpy(), "eval_metric": ["mrr"]})
                            rr = (1.0 / rk.float()).cpu().numpy()
                            max_dev = max(max_dev, abs(float(m["mrr"]) - float(rr.mean())))
                            OUT[f"rr_{nm}"][sl] = rr; OUT[f"rank_{nm}"][sl] = rk.cpu().numpy()
                            OUT[f"hrank_{nm}"][sl] = frank(Y, fr, fc, gold).float().cpu().numpy()
                            del Yn
                        del Ys, M
                    del lp, X, Rv, qv
        xi.add(trip)
        nsnap += 1
        if nsnap % 100 == 0:
            log(sp, "snapshot", nsnap, "pos", dict(pos), "elapsed_scoring", round(time.time() - tq))
    meta["elapsed_s"] = round(time.time() - t0, 1)
    if a.stage == "valid":
        assert (RB > 0).all() and (RK > 0).all(), "unscored rows"
        np.savez(f"{a.out}/blend_valid_{a.ds}_{a.base}.npz", ranks=RK, base=RB, rows=rows_req)
        mrr = (1.0 / RK.astype(np.float64)).mean(0)
        meta["valid_MRR_base"] = float((1.0 / RB.astype(np.float64)).mean())
        meta["valid_MRR_blend"] = mrr[:len(BLEND)].tolist(); meta["valid_MRR_pool"] = mrr[len(BLEND):].tolist()
        log("valid base", meta["valid_MRR_base"], "best blend", float(mrr[:len(BLEND)].max()), "best pool", float(mrr[len(BLEND):].max()))
    else:
        assert all((OUT[f"rank_{nm}"] > 0).all() for nm in arms) or a.smoke, "unscored rows"
        meta["official_vs_formula_max_batch_dev"] = max_dev if OFFICIAL is True else None
        meta["evaluator"] = ("all-entity time-aware filtered rank, ties averaged (py-tgb Evaluator formula); harness strict frank in hrank_" if OFFICIAL == "allent"
                             else "official py-tgb Evaluator" if OFFICIAL else "harness exact filtered rank (frank) for every arm")
        res = {}
        for nm in arms:
            rk = OUT[f"rank_{nm}"].astype(np.float64); hr = OUT[f"hrank_{nm}"].astype(np.float64)
            res[nm] = dict(official=dict(MRR=float(OUT[f"rr_{nm}"].astype(np.float64).mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()),
                                         H10=float((rk <= 10).mean())),
                           harness=dict(MRR=float((1 / hr).mean()), H1=float((hr <= 1).mean()), H3=float((hr <= 3).mean()), H10=float((hr <= 10).mean())))
        meta["metrics"] = res
        np.savez(f"{a.out}/official_test_{a.ds}_{a.base}.npz", rows=rows_req, **OUT)
    json.dump(meta, open(f"{a.out}/blend_{a.stage}_{a.ds}_{a.base}.json", "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
