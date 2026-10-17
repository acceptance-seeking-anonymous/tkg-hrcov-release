#!/usr/bin/env python3
"""presub / g7_cluster_bootstrap.py -- G7 of a_presub_rule.json: timestamp-block and fact-pair cluster bootstraps for every
hardening decision delta, from the stored official-evaluator per-query reciprocal ranks (no model is re-run).

For each dataset: test queries in hd_agg split order (per timestamp: forward block, then inverse block), clusters
  ts   = all queries sharing a timestamp (both directions)        -- primary G7 statistic
  fact = the two directions of one test fact
Cluster bootstrap: clusters resampled with replacement (multinomial), B = 2000, seed 0, statistic = sum of resampled
per-query deltas / number of resampled queries; percentile 95% CI; one-sided p = fraction of draws <= 0.
The query-level Poisson(1) bootstrap of hd_agg.py is recomputed with the same seed and checked against the verdict.
  python3 g7_cluster_bootstrap.py --cfg agg_cfg.json --verdict hardening_verdict_raw.json --out g7_cluster_bootstrap.json
(run from the directory that agg_cfg.json's relative paths refer to)
"""
import json, argparse, time
import numpy as np

B = 2000
MARGIN = 0.005


def boot_query(d, B=B, seed=0):          # identical to hd_agg.boot
    d = np.asarray(d, np.float64); n = len(d); rng = np.random.default_rng(seed)
    nz = np.flatnonzero(d); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 100):
        bb = min(100, B - b0)
        w = rng.poisson(1.0, size=(bb, len(nz)))
        bs[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), CI95=[float(lo), float(hi)], p_one_sided=float((bs <= 0).mean()),
                p_le_margin=float((bs <= MARGIN).mean()), n=int(n))


def boot_cluster(d, cid, B=B, seed=0, chunk=50):
    d = np.asarray(d, np.float64)
    u, inv = np.unique(cid, return_inverse=True); nC = len(u)
    S = np.bincount(inv, weights=d, minlength=nC); M = np.bincount(inv, minlength=nC).astype(np.float64)
    rng = np.random.default_rng(seed); bs = np.empty(B); pv = np.full(nC, 1.0 / nC)
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        cnt = rng.multinomial(nC, pv, size=bb).astype(np.float64)
        bs[b0:b0 + bb] = (cnt @ S) / (cnt @ M)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), CI95=[float(lo), float(hi)], p_one_sided=float((bs <= 0).mean()),
                p_le_margin=float((bs <= MARGIN).mean()), n_clusters=int(nC), mean_cluster_size=float(M.mean()),
                max_cluster_size=int(M.max()))


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def split_order(q):
    """hd_agg.split_queries order: per timestamp [forward block; inverse block] -> per-query ts, fact id, gold."""
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
    ts, fid, gold = [], [], []
    for u, b, e in zip(uts, st, en):
        m = e - b; f = np.arange(b, e)
        ts.append(np.full(2 * m, u)); fid.append(np.concatenate([f, f])); gold.append(np.concatenate([q[b:e, 2], q[b:e, 0]]))
    return np.concatenate(ts), np.concatenate(fid), np.concatenate(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--verdict", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    C = json.load(open(a.cfg)); V = json.load(open(a.verdict))
    OUT = dict(name="g7_cluster_bootstrap", rule="a_presub_rule.json (G7_cluster_bootstrap)", B=B, seed=0,
               note=("post-hoc robustness of the registered hardening decision; the registered outcome is not altered. "
                     "ts = timestamp-block clusters (primary G7 statistic), fact = both directions of one test fact."), datasets={})
    for ds, D in C.items():
        VD = V["datasets"][ds]
        Q = {sp: load_quads(f"{D['dd']}/{sp}.txt") for sp in ("train", "test")}
        N = int(open(f"{D['dd']}/stat.txt").read().split()[0])
        ts, fid, gold = split_order(Q["test"])
        S = np.load(D["strata"]); sro_ok = bool((S["sro"][:, 2] == gold).all())
        assert sro_ok, "query order differs from the hardening strata order"
        freq = np.bincount(Q["train"][:, 0], minlength=N) + np.bincount(Q["train"][:, 2], minlength=N)
        head = freq[gold] >= 500
        RB = np.load(D["recb_ranks"]); rr_recb = 1.0 / RB["clean_tr"].astype(np.float64)
        rr = {}
        for b, Bc in D["bases"].items():
            Z = np.load(Bc["official"]); r = {}
            for arm in ("base", "BLEND", "POOL"):
                r[arm] = Z[f"rr_{arm}"].astype(np.float64)
            for pre in ("HR", "HRCOV"):
                ks = [f"rr_{pre}_s{s}" for s in (0, 1, 2) if f"rr_{pre}_s{s}" in Z.files]
                r[pre] = np.mean([Z[k].astype(np.float64) for k in ks], 0)
            assert len(r["base"]) == len(ts)
            rr[b] = r
        bb = VD["strongest_blend_overall"]["base"]
        res = dict(n_test=int(len(ts)), n_timestamps=int(len(np.unique(ts))), n_facts=int(len(np.unique(fid))),
                   two_strongest=VD["two_strongest"], strongest_blend_overall=bb, decision_deltas={}, other_bases={})

        def three(d, mask=None):
            if mask is not None:
                d, t_, f_ = d[mask], ts[mask], fid[mask]
            else:
                t_, f_ = ts, fid
            return dict(query=boot_query(d), ts=boot_cluster(d, t_), fact=boot_cluster(d, f_))

        for b in VD["two_strongest"]:
            r = rr[b]; dec = VD["decision"]["per_base"][b]; better = dec["better_comparator"]
            k1, k2, k3, k4 = "C1 HRCOV - BASE(+)RecB", "C1 HRCOV - RecB-POOL", f"C2 HRCOV - {bb}(+)RecB", f"C3 head HRCOV - {better}"
            E = {k1: three(r["HRCOV"] - r["BLEND"]), k2: three(r["HRCOV"] - r["POOL"]), k3: three(r["HRCOV"] - rr[bb]["BLEND"]),
                 k4: three(r["HRCOV"] - r[better], head),
                 "secondary HRCOV - HR": three(r["HRCOV"] - r["HR"]),
                 f"secondary HR - {better}": three(r["HR"] - r[better]),
                 "secondary HRCOV - RecB trained (clean)": three(r["HRCOV"] - rr_recb)}
            chk = {k1: dec["C1"]["BLEND"], k2: dec["C1"]["POOL"], k3: dec["C2"]}
            repro = {k: dict(delta_absdiff=abs(E[k]["query"]["delta"] - v["delta"]),
                             CI_absdiff=max(abs(E[k]["query"]["CI95"][i] - v["CI95"][i]) for i in (0, 1))) for k, v in chk.items()}
            repro["C3 head"] = dict(delta_absdiff=abs(E[k4]["query"]["delta"] - dec["C3_head"]["delta"]))
            flips = {}
            for cl in ("ts", "fact"):
                c1 = all(E[k][cl]["delta"] >= MARGIN and E[k][cl]["p_one_sided"] < 0.05 for k in (k1, k2))
                c2 = E[k3][cl]["delta"] > 0 and E[k3][cl]["p_one_sided"] < 0.05
                dmg = E[k4][cl]["CI95"][1] < 0 and abs(E[k4][cl]["delta"]) >= 0.001
                flips[cl] = dict(C1_pass=bool(c1), C2_pass=bool(c2), C3_damage=bool(dmg), PASS=bool(c1 and c2 and not dmg),
                                 registered_PASS=bool(dec["PASS"]), changes=bool((c1 and c2 and not dmg) != dec["PASS"]))
            width = {k: {cl: (v[cl]["CI95"][1] - v[cl]["CI95"][0]) / max(v["query"]["CI95"][1] - v["query"]["CI95"][0], 1e-12)
                         for cl in ("ts", "fact")} for k, v in E.items()}
            res["decision_deltas"][b] = dict(better_comparator=better, deltas=E, reproduction_vs_verdict=repro,
                                             decision_under_cluster_bootstrap=flips, CI_width_ratio_cluster_over_query=width)
        for b, r in rr.items():
            if b in VD["two_strongest"]: continue
            better = "BLEND" if r["BLEND"].mean() >= r["POOL"].mean() else "POOL"
            res["other_bases"][b] = {f"HRCOV - {better}": three(r["HRCOV"] - r[better]), "HRCOV - HR": three(r["HRCOV"] - r["HR"])}
        oc = [res["decision_deltas"][b]["decision_under_cluster_bootstrap"] for b in VD["two_strongest"]]
        res["outcome_registered"] = VD["decision"]["outcome"]
        for cl in ("ts", "fact"):
            res[f"outcome_if_{cl}_cluster_bootstrap"] = "PRACTICAL_ADVANCE" if all(x[cl]["PASS"] for x in oc) else "CHANGED (see per-base flags)"
        OUT["datasets"][ds] = res
        print(ds, res["n_timestamps"], res["outcome_if_ts_cluster_bootstrap"], res["outcome_if_fact_cluster_bootstrap"], round(time.time() - t0, 1), flush=True)
    OUT["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(OUT, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
