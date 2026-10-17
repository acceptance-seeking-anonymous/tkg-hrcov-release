"""v7 / v7_b4c3_agg.py -- B4 (G1 on tkgl-icews Stage B) and C3 (relation normalisation) aggregation, run on serverA (CPU).
Per-query arrays: official py-tgb rr/rank (hd_blend / pi_blend outputs) unless marked harness (exact harness ranks; identity
with official verified on the HRCOV200 arm in this script). Statistics = tois_v7_rule.json: paired Poisson(1) query bootstrap and
timestamp-block bootstrap, B = 2000, seed 0, shared draws across all comparisons."""
import argparse, json, time, numpy as np, sys

sys.path.insert(0, ".")
import v7_common as VC  # noqa: E402

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:8.1f}s]", *a, flush=True)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()


def arm_off(z, pre):
    ks = sorted({k[3:] for k in z.files if k.startswith(f"rr_{pre}")})
    ks = [k for k in ks if k == pre or k.startswith(pre + "_s")]
    assert ks, (pre, z.files[:10])
    rr = np.mean([z[f"rr_{k}"].astype(np.float64) for k in ks], 0)
    H = {h: np.mean([(z[f"rank_{k}"].astype(np.float64) <= h) for k in ks], 0) for h in (1, 3, 10)}
    per = [float(z[f"rr_{k}"].astype(np.float64).mean()) for k in ks]
    hr = np.mean([1.0 / z[f"hrank_{k}"].astype(np.float64) for k in ks], 0) if all(f"hrank_{k}" in z.files for k in ks) else None
    return dict(rr=rr, H=H, per_seed=per, n_seeds=len(ks), src="official", harness_MRR=None if hr is None else float(hr.mean()))


def arm_harness(z, pre):
    ks = [k for k in z.files if k.startswith(pre + "_s")] or [pre]
    rk = [z[k].astype(np.float64) for k in ks]
    return dict(rr=np.mean([1.0 / r for r in rk], 0), H={h: np.mean([(r <= h) for r in rk], 0) for h in (1, 3, 10)},
                per_seed=[float(np.mean(1.0 / r)) for r in rk], n_seeds=len(ks), src="harness", harness_MRR=None)


def met(a):
    return dict(MRR=float(a["rr"].mean()), H1=float(a["H"][1].mean()), H3=float(a["H"][3].mean()), H10=float(a["H"][10].mean()),
                per_seed_MRR=a["per_seed"], n_seeds=a["n_seeds"], scoring=a["src"], harness_MRR=a["harness_MRR"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageB", required=True); ap.add_argument("--g1", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--g1_prefix", default="G1"); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(); S = a.stageB
    q = np.load(f"{S}/feat_regcn_orig/test_q.npz"); ts = q["sro"][:, 3].astype(np.int64); n = len(ts)
    zr = np.load(f"{S}/blend/official_test_icews_regcn.npz"); zl = np.load(f"{S}/blend/official_test_icews_logcl.npz")
    zls = np.load(f"{S}/blend/official_test_icews_logcl_sys.npz"); zrn = np.load(f"{S}/blend/official_test_icews_regcnrn.npz")
    zg = np.load(a.g1); hr = np.load(f"{S}/train_regcn/ranks_test.npz")
    for z in (zr, zl, zls, zrn, zg): assert np.array_equal(z["rows"], np.arange(n)), "row order"
    log("loaded", n, "queries", len(np.unique(ts)), "timestamps")
    g1n = sorted({k[3:].rsplit("_s", 1)[0] for k in zg.files if k.startswith("rr_" + a.g1_prefix)}); assert len(g1n) == 1, g1n
    A = {
        "RE-GCN base": arm_off(zr, "base"), "RE-GCN BLEND": arm_off(zr, "BLEND"), "RE-GCN POOL": arm_off(zr, "POOL"),
        "HRCOV200 (RE-GCN)": arm_off(zr, "HRCOV200"),
        "LogCL base": arm_off(zl, "base"), "LogCL BLEND": arm_off(zl, "BLEND"), "LogCL POOL": arm_off(zl, "POOL"),
        "HRCOV150 (LogCL)": arm_off(zls, "HRCOV150"),
        "G1": arm_off(zg, g1n[0]),
        "RE-GCN-RN base": arm_off(zrn, "base"),
        "HRCOV200 (RE-GCN) [harness]": arm_harness(hr, "HRCOV200"),
        "HRCOV200_RN (RE-GCN base, RN features) [harness]": arm_harness(hr, "HRCOV200_RN"),
    }
    dlb = np.abs(zls["rr_base"].astype(np.float64) - zl["rr_base"].astype(np.float64)); assert dlb.mean() < 1e-6, dlb.mean()
    logcl_base_replay = dict(frac_rr_differs=float((dlb > 1e-6).mean()), mean_abs=float(dlb.mean()))
    log("arms built")
    F = A["HRCOV200 (RE-GCN)"]; Fh = A["HRCOV200 (RE-GCN) [harness]"]
    fid = dict(max_abs_rr_diff=float(np.abs(F["rr"] - Fh["rr"]).max()), mean_rr_diff=float((F["rr"] - Fh["rr"]).mean()),
               frac_rr_differs=float(np.mean(np.abs(F["rr"] - Fh["rr"]) > 1e-6)))
    PR = [("B4|HRCOV200 (RE-GCN) - G1", "HRCOV200 (RE-GCN)", "G1", "primary"),
          ("B4|HRCOV150 (LogCL) - G1", "HRCOV150 (LogCL)", "G1", "primary"),
          ("B4|G1 - RE-GCN base", "G1", "RE-GCN base", "secondary"), ("B4|G1 - LogCL base", "G1", "LogCL base", "secondary"),
          ("B4|G1 - RE-GCN BLEND", "G1", "RE-GCN BLEND", "secondary"), ("B4|G1 - LogCL BLEND", "G1", "LogCL BLEND", "secondary"),
          ("B4|HRCOV200 (RE-GCN) - RE-GCN BLEND", "HRCOV200 (RE-GCN)", "RE-GCN BLEND", "context"),
          ("B4|HRCOV150 (LogCL) - LogCL BLEND", "HRCOV150 (LogCL)", "LogCL BLEND", "context"),
          ("C3|RE-GCN-RN base - RE-GCN base [RN effect on the base]", "RE-GCN-RN base", "RE-GCN base", "primary-partial"),
          ("C3|HRCOV200_RN - HRCOV200 [RN effect on the re-ranker features, harness]", "HRCOV200_RN (RE-GCN base, RN features) [harness]", "HRCOV200 (RE-GCN) [harness]", "primary-partial"),
          ("C3|HRCOV200 - RE-GCN base [gain without RN]", "HRCOV200 (RE-GCN)", "RE-GCN base", "primary-partial"),
          ("C3|HRCOV200_RN - RE-GCN-RN base [HRCOV with RN features vs the RN base]", "HRCOV200_RN (RE-GCN base, RN features) [harness]", "RE-GCN-RN base", "primary-partial"),
          ("C3|HRCOV200_RN - RE-GCN base", "HRCOV200_RN (RE-GCN base, RN features) [harness]", "RE-GCN base", "secondary"),
          ("C3|HRCOV200_RN - G1", "HRCOV200_RN (RE-GCN base, RN features) [harness]", "G1", "secondary"),
          ("C3|HRCOV200_RN - RE-GCN BLEND", "HRCOV200_RN (RE-GCN base, RN features) [harness]", "RE-GCN BLEND", "secondary"),
          ("C3|HRCOV200 - RE-GCN BLEND [without RN]", "HRCOV200 (RE-GCN)", "RE-GCN BLEND", "secondary"),
          ("C3|RE-GCN-RN base - RE-GCN BLEND", "RE-GCN-RN base", "RE-GCN BLEND", "secondary")]
    cols = {}
    for lab, x, y, _ in PR:
        cols[f"{lab}|MRR"] = (A[x]["rr"] - A[y]["rr"]).astype(np.float32)
        for h in (1, 3, 10): cols[f"{lab}|H{h}"] = (A[x]["H"][h] - A[y]["H"][h]).astype(np.float32)
    names = list(cols); Dm = np.stack([cols[k] for k in names], 1); del cols
    log("bootstrap", Dm.shape)
    bq = VC.boot_query_multi(Dm, None, B=a.B, chunk=10); log("query bootstrap done")
    bt = VC.boot_cluster_multi(Dm.astype(np.float64), None, ts, B=a.B); log("ts bootstrap done")
    dl = Dm.astype(np.float64).mean(0)
    S_ = {k: VC.summ_from(dl[j], bq[:, j], bt[:, j]) for j, k in enumerate(names)}
    out = dict(name="v7 B4 + C3 (tkgl-icews Stage B)", label="pre-specified later analysis (tois_v7_rule.json)", n_queries=int(n),
               n_ts=int(len(np.unique(ts))), g1_arm=g1n[0], arms={k: met(v) for k, v in A.items()}, harness_official_identity_HRCOV200=fid, logcl_base_replay_between_files=logcl_base_replay,
               items={"B4_G1_tkgl_icews": [], "C3_relnorm_tkgl_icews": []})
    for it, key in (("B4_G1_tkgl_icews", "B4"), ("C3_relnorm_tkgl_icews", "C3")):
        rows = []
        for lab, x, y, role in PR:
            if not lab.startswith(key + "|"): continue
            rows.append(dict(comparison=lab.split("|", 1)[1], a=x, b=y, role=role, **{m: S_[f"{lab}|{m}"] for m in ("MRR", "H1", "H3", "H10")}))
        prim = [r for r in rows if r["role"] == "primary"]
        if len(prim) > 1:
            for kk in ("query", "ts"):
                adj = holm([r["MRR"][kk]["p_one_sided_le0"] for r in prim])
                for r, v in zip(prim, adj): r["MRR"][f"holm_p_{kk}"] = v
        for r in rows:
            m = r["MRR"]; ok = all(m.get(f"holm_p_{kk}", m[kk]["p_one_sided_le0"]) < 0.05 for kk in ("query", "ts"))
            r["claim"] = ("present (+)" if m["present_pos"] and ok else "present (-)" if m["present_neg"] else
                          "|delta| >= bar excluded" if m["excluded_gain_ge_bar"] and m["excluded_loss_ge_bar"] else "not established")
            r["ambiguous_for_expansion"] = bool(m["ts"]["CI95"][0] <= 0.005 <= m["ts"]["CI95"][1])
        out["items"][it] = rows
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
