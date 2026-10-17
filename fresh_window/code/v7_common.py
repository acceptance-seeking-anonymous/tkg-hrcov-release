"""v7 common loaders + statistics (local analysis of stored outputs)."""
import json, numpy as np
D = "v7/data"
B = 2000


def boot_query(d, B=B, seed=0):
    d = np.asarray(d, np.float64); n = len(d); rng = np.random.default_rng(seed)
    nz = np.flatnonzero(d); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 100):
        bb = min(100, B - b0); w = rng.poisson(1.0, size=(bb, len(nz))); bs[b0:b0 + bb] = (w * dn).sum(1) / n
    return bs


def boot_cluster(d, cid, B=B, seed=0, chunk=50):
    d = np.asarray(d, np.float64); u, inv = np.unique(cid, return_inverse=True); nC = len(u)
    S = np.bincount(inv, weights=d, minlength=nC); M = np.bincount(inv, minlength=nC).astype(np.float64)
    rng = np.random.default_rng(seed); bs = np.empty(B); pv = np.full(nC, 1.0 / nC)
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0); cnt = rng.multinomial(nC, pv, size=bb).astype(np.float64); bs[b0:b0 + bb] = (cnt @ S) / (cnt @ M)
    return bs


def summ(d, ts, bar=0.005, B=B):
    d = np.asarray(d, np.float64)
    bq = boot_query(d, B); bt = boot_cluster(d, ts, B)
    out = dict(delta=float(d.mean()), n=int(len(d)))
    for nm, bs in (("query", bq), ("ts", bt)):
        out[nm] = dict(CI95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], L=float(np.percentile(bs, 5)), U=float(np.percentile(bs, 95)),
                       p_one_sided_le0=float((bs <= 0).mean()), p_one_sided_ge0=float((bs >= 0).mean()), SE=float(bs.std(ddof=1)))
    out["MDE80_ts"] = float(2.486 * out["ts"]["SE"])
    out["present_pos"] = bool(out["query"]["L"] > 0 and out["ts"]["L"] > 0)
    out["present_neg"] = bool(out["query"]["U"] < 0 and out["ts"]["U"] < 0)
    out["excluded_gain_ge_bar"] = bool(out["ts"]["U"] < bar)        # signed delta: gain >= bar excluded
    out["excluded_loss_ge_bar"] = bool(out["ts"]["L"] > -bar)       # loss >= bar excluded
    return out


def hits(rank, ks=(1, 3, 10)):
    rank = np.asarray(rank, np.float64)
    return {f"H{k}": float(np.mean(rank <= k)) for k in ks}


def mets_seeds(Z, prefix, seeds=(0, 1, 2), harness=False):
    """seed-mean metrics + per-query seed-mean RR from an official npz (rr_/rank_/hrank_)."""
    names = [f"{prefix}_s{s}" for s in seeds if f"rr_{prefix}_s{s}" in Z.files] or ([prefix] if f"rr_{prefix}" in Z.files else [])
    assert names, prefix
    per = []
    for nm in names:
        if harness:
            rk = Z[f"hrank_{nm}"].astype(np.float64); rr = 1.0 / rk
        else:
            rk = Z[f"rank_{nm}"].astype(np.float64); rr = Z[f"rr_{nm}"].astype(np.float64)
        per.append(dict(MRR=float(rr.mean()), **hits(rk), rr=rr))
    out = {k: float(np.mean([p[k] for p in per])) for k in ("MRR", "H1", "H3", "H10")}
    out["per_seed_MRR"] = [p["MRR"] for p in per]; out["n_seeds"] = len(per)
    return out, np.mean([p["rr"] for p in per], 0)


def loadx(key, split):
    return dict(np.load(f"{D}/x/x_{key}_{split}.npz"))


def boot_query_multi(Dm, Mm=None, B=B, seed=0, chunk=50):
    """Poisson(1) weights shared across columns: Dm [n, m] per-query deltas (already masked), Mm [n, m] subset masks
    (None = all queries). Returns draws [B, m] of sum(w d) / sum(w m)."""
    Dm = np.asarray(Dm, np.float32); n, m = Dm.shape
    Mm = np.ones_like(Dm) if Mm is None else np.asarray(Mm, np.float32)
    rng = np.random.default_rng(seed); out = np.empty((B, m))
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0); w = rng.poisson(1.0, size=(bb, n)).astype(np.float32)
        out[b0:b0 + bb] = (w @ Dm).astype(np.float64) / np.maximum((w @ Mm).astype(np.float64), 1e-9)
    return out


def boot_cluster_multi(Dm, Mm, cid, B=B, seed=0, chunk=50):
    Dm = np.asarray(Dm, np.float64); Mm = np.ones_like(Dm) if Mm is None else np.asarray(Mm, np.float64)
    u, inv = np.unique(cid, return_inverse=True); nC = len(u)
    S = np.stack([np.bincount(inv, weights=Dm[:, j], minlength=nC) for j in range(Dm.shape[1])], 1)
    M = np.stack([np.bincount(inv, weights=Mm[:, j], minlength=nC) for j in range(Dm.shape[1])], 1)
    rng = np.random.default_rng(seed); out = np.empty((B, Dm.shape[1])); pv = np.full(nC, 1.0 / nC)
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0); cnt = rng.multinomial(nC, pv, size=bb).astype(np.float64)
        out[b0:b0 + bb] = (cnt @ S) / np.maximum(cnt @ M, 1e-9)
    return out


def summ_from(delta, bq, bt, bar=0.005):
    out = dict(delta=float(delta))
    for nm, bs in (("query", bq), ("ts", bt)):
        out[nm] = dict(CI95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], L=float(np.percentile(bs, 5)), U=float(np.percentile(bs, 95)),
                       p_one_sided_le0=float((bs <= 0).mean()), SE=float(bs.std(ddof=1)))
    out["MDE80_ts"] = float(2.486 * out["ts"]["SE"])
    out["present_pos"] = bool(out["query"]["L"] > 0 and out["ts"]["L"] > 0); out["present_neg"] = bool(out["query"]["U"] < 0 and out["ts"]["U"] < 0)
    out["excluded_gain_ge_bar"] = bool(out["ts"]["U"] < bar); out["excluded_loss_ge_bar"] = bool(out["ts"]["L"] > -bar)
    return out


def summ_multi(cols, ts, masks=None, bar=0.005):
    """cols: dict name -> per-query delta array; masks: dict name -> bool mask (subset). Shared draws."""
    names = list(cols); n = len(next(iter(cols.values())))
    Mm = np.stack([np.ones(n) if masks is None or masks.get(k) is None else masks[k].astype(np.float64) for k in names], 1)
    Dm = np.stack([cols[k] for k in names], 1) * Mm
    bq = boot_query_multi(Dm, Mm); bt = boot_cluster_multi(Dm, Mm, ts)
    res = {}
    for j, k in enumerate(names):
        dlt = Dm[:, j].sum() / max(Mm[:, j].sum(), 1e-9)
        res[k] = summ_from(dlt, bq[:, j], bt[:, j], bar); res[k]["n"] = int(Mm[:, j].sum())
    return res
