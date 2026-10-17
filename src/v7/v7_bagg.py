"""v7 / v7_bagg.py -- aggregation of B/C/A5 items (tois_v7_rule.json): seed-mean per-query RR / hits from the official
per-system arrays, deltas with query Poisson + timestamp-block bootstraps (shared draws), Holm within items, bars, MDE."""
import numpy as np, json
import v7_common as VC


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()


def arm_from(Z, name, seeds=(0, 1, 2), official=True):
    """Z: dict of arrays rr_<sys>, rank_<sys>, hrank_<sys>; name without _s<seed>."""
    ks = [f"{name}_s{s}" for s in seeds if f"rr_{name}_s{s}" in Z] or ([name] if f"rr_{name}" in Z else [])
    assert ks, name
    rr = np.mean([np.asarray(Z[f"rr_{k}"], np.float64) for k in ks], 0)
    H = {h: np.mean([(np.asarray(Z[f"rank_{k}"], np.float64) <= h).astype(np.float64) for k in ks], 0) for h in (1, 3, 10)}
    hr = np.mean([1.0 / np.asarray(Z[f"hrank_{k}"], np.float64) for k in ks], 0) if all(f"hrank_{k}" in Z for k in ks) else None
    per = [float(np.mean(Z[f"rr_{k}"])) for k in ks]
    return dict(rr=rr, H=H, per_seed=per, n_seeds=len(ks), harness_MRR=(float(hr.mean()) if hr is not None else None))


def metrics(a):
    return dict(MRR=float(a["rr"].mean()), H1=float(a["H"][1].mean()), H3=float(a["H"][3].mean()), H10=float(a["H"][10].mean()),
                per_seed_MRR=a["per_seed"], n_seeds=a["n_seeds"], harness_MRR=a["harness_MRR"])


def compare(arms, pairs, ts, bar=0.005):
    """pairs: list of (label, a, b) -> delta a - b for MRR and H@1/3/10 with both bootstraps (shared draws)."""
    cols = {}
    for lab, x, y in pairs:
        cols[f"{lab}|MRR"] = arms[x]["rr"] - arms[y]["rr"]
        for h in (1, 3, 10): cols[f"{lab}|H{h}"] = arms[x]["H"][h] - arms[y]["H"][h]
    S = VC.summ_multi(cols, ts, bar=bar)
    out = {}
    for lab, x, y in pairs:
        out[lab] = dict(a=x, b=y, **{m: S[f"{lab}|{m}"] for m in ("MRR", "H1", "H3", "H10")})
    return out
