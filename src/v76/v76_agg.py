"""v76 / v76_agg.py -- aggregation of v76_rule.json items M2b, M3a, M4, M6 from the official per-query arrays.
Statistics exactly as tois_v7_rule.json: paired Poisson query bootstrap + timestamp-block bootstrap, B = 2000, seed 0, shared
draws within an item and dataset (v7_common), present iff L > 0 under both (Holm across an item's primary contrasts on a base),
excluded iff ts U < 0.005.  Usage (inside a Python session):  exec(open('v76/code/v76_agg.py').read()); res = run_all(CFG)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "v7/code")
import v7_common as VC

BAR = 0.005


def load(files):
    files = [files] if isinstance(files, str) else files
    Z = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k not in Z: Z[k] = z[k]
    return Z


def seedavg(Z, p, seeds=(0, 1, 2)):
    ks = [f"rr_{p}_s{s}" for s in seeds]
    if all(k in Z for k in ks): return np.mean([Z[k].astype(np.float64) for k in ks], 0)
    if f"rr_{p}" in Z: return Z[f"rr_{p}"].astype(np.float64)
    raise KeyError(p)


def hits(Z, p, seeds=(0, 1, 2)):
    ks = [f"rank_{p}_s{s}" for s in seeds]
    if all(k in Z for k in ks):
        return {f"H{k}": float(np.mean([np.mean(Z[q] <= k) for q in ks])) for k in (1, 3, 10)}
    if f"rank_{p}" in Z: return {f"H{k}": float(np.mean(Z[f"rank_{p}"] <= k)) for k in (1, 3, 10)}
    return {}


def per_seed(Z, p, seeds=(0, 1, 2)):
    return [float(Z[f"rr_{p}_s{s}"].astype(np.float64).mean()) for s in seeds if f"rr_{p}_s{s}" in Z]


def ts_of(ds):
    X = VC.loadx(f"{ds}_tirgn", "test"); return (X["F_rows"] - X["F_rows"][0]).astype(np.int64), X["F_sro"][:, 3].astype(np.int64)


def holm_adj(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def contrasts(cols, ts, primary=()):
    """cols: name -> per-query delta; returns v7 summaries + Holm over `primary` (one-sided, both directions, both bootstraps)."""
    names = list(cols); n = len(next(iter(cols.values())))
    Dm = np.stack([np.asarray(cols[k], np.float64) for k in names], 1); Mm = np.ones_like(Dm)
    bq = VC.boot_query_multi(Dm, Mm); bt = VC.boot_cluster_multi(Dm, Mm, ts)
    res = {}
    for j, k in enumerate(names):
        r = VC.summ_from(Dm[:, j].mean(), bq[:, j], bt[:, j], BAR); r["n"] = int(n)
        r["query"]["p_one_sided_ge0"] = float((bq[:, j] >= 0).mean()); r["ts"]["p_one_sided_ge0"] = float((bt[:, j] >= 0).mean())
        res[k] = r
    prim = [k for k in primary if k in res]
    if len(prim) > 1:
        for direction, key in (("pos", "p_one_sided_le0"), ("neg", "p_one_sided_ge0")):
            for b in ("query", "ts"):
                adj = holm_adj([res[k][b][key] for k in prim])
                for k, a_ in zip(prim, adj): res[k][b][f"holm_{direction}"] = a_
        for k in prim:
            r = res[k]
            r["present_pos_holm"] = bool(r["query"]["holm_pos"] < 0.05 and r["ts"]["holm_pos"] < 0.05)
            r["present_neg_holm"] = bool(r["query"]["holm_neg"] < 0.05 and r["ts"]["holm_neg"] < 0.05)
    for k in prim: res[k]["primary"] = True
    return res


def better_ctrl(C):
    mb, mp = float(C["rr_BLEND"].astype(np.float64).mean()), float(C["rr_POOL"].astype(np.float64).mean())
    return ("BLEND" if mb >= mp else "POOL"), dict(BLEND=mb, POOL=mp)


def base_identity(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)); return dict(mean_abs=float(d.mean()), frac_differs=float((d > 1e-6).mean()), max_abs=float(d.max()))


def interp_mlp_lmart(r):
    if r.get("present_pos_holm", r["present_pos"]): return "MLP better than LambdaMART"
    if r.get("present_neg_holm", r["present_neg"]): return "LambdaMART better than MLP"
    if r["excluded_gain_ge_bar"] and r["excluded_loss_ge_bar"]: return "equivalent within 0.005"
    return "not resolved"


# ------------------------------------------------------------------------------------------------ M2b
def m2b_base(key, ds, off_files, ctrl_file, stored_hcpat, train_lmart, train_bt, ranks_bt=None, ranks_lm=None):
    rows, ts = ts_of(ds); Z = load(off_files); C = load(ctrl_file)
    assert np.array_equal(Z["rows"], rows) and np.array_equal(C["rows"], rows) if "rows" in C else True
    bid = base_identity(Z["rr_base"], C["rr_base"]); assert bid["mean_abs"] < 1e-6, (key, bid)
    better, cm = better_ctrl(C)
    A = dict(MLP=seedavg(Z, "MLP64_PAT_fused"), LIN=seedavg(Z, "LIN_PAT_fused"), LMART=seedavg(Z, "LMART_PAT_fused"),
             BLEND=C["rr_BLEND"].astype(np.float64), POOL=C["rr_POOL"].astype(np.float64), base=Z["rr_base"].astype(np.float64))
    if stored_hcpat:
        S = load(stored_hcpat); A["HRCOV_PAT_stored"] = seedavg(S, "HRCOV_PAT")
    cols = {"MLP64_PAT_fused - LMART_PAT_fused": A["MLP"] - A["LMART"], "LMART_PAT_fused - LIN_PAT_fused": A["LMART"] - A["LIN"],
            "LMART_PAT_fused - better control": A["LMART"] - A[better], "MLP64_PAT_fused - LIN_PAT_fused": A["MLP"] - A["LIN"],
            "MLP64_PAT_fused - better control": A["MLP"] - A[better]}
    if "HRCOV_PAT_stored" in A: cols["MLP64_PAT re-fit - stored HRCOV_PAT"] = A["MLP"] - A["HRCOV_PAT_stored"]
    alone = {}
    if ranks_bt and ranks_lm:      # exact fused-rank statistic (labelled), alone variants
        Rb = np.load(ranks_bt); Rl = np.load(ranks_lm)
        a_m = np.mean([1.0 / Rb[f"MLP64_PAT_alone_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)
        a_l = np.mean([1.0 / Rl[f"LMART_PAT_alone_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)
        cols["[exact] MLP64_PAT_alone - LMART_PAT_alone"] = a_m - a_l
        alone = dict(MLP64_PAT_alone=float(a_m.mean()), LMART_PAT_alone=float(a_l.mean()))
        if "LIN_PAT_alone_s0" in Rb:
            a_n = np.mean([1.0 / Rb[f"LIN_PAT_alone_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)
            cols["[exact] LMART_PAT_alone - LIN_PAT_alone"] = a_l - a_n; alone["LIN_PAT_alone"] = float(a_n.mean())
    prim = ["MLP64_PAT_fused - LMART_PAT_fused", "LMART_PAT_fused - LIN_PAT_fused", "LMART_PAT_fused - better control"]
    R = contrasts(cols, ts, prim)
    TL = json.load(open(train_lmart)); TB = json.load(open(train_bt))
    out = dict(base=key, dataset=ds, n_test=int(len(rows)), better_control=better, control_MRR=cm, base_identity=bid,
               mrr={k: float(v.mean()) for k, v in A.items()}, alone_exact=alone,
               hits=dict(MLP=hits(Z, "MLP64_PAT_fused"), LIN=hits(Z, "LIN_PAT_fused"), LMART=hits(Z, "LMART_PAT_fused"), base=hits(Z, "base")),
               per_seed=dict(MLP=per_seed(Z, "MLP64_PAT_fused"), LIN=per_seed(Z, "LIN_PAT_fused"), LMART=per_seed(Z, "LMART_PAT_fused")),
               lmart=dict(selected=TL.get("selected"), grid=[{k: g[k] for k in ("num_leaves", "learning_rate", "best_round", "best_sel_MRR", "wall_s", "eval_s")} for g in TL["grid_results"]],
                          seeds={s: {k: v[k] for k in ("best_round", "best_sel_MRR", "wall_s", "eval_s", "train_s_excl_eval", "cal_alpha", "cal_beta", "test_predict_s_cpu") if k in v} for s, v in TL["seeds"].items()},
                          latency=TL.get("latency"), n_fit_queries=TL["n_fit_queries"], n_fit_rows=TL["n_fit_rows"], lightgbm=TL.get("lightgbm")),
               mlp_fit=dict(n_fit_queries=TB["arms"]["MLP64_PAT"]["n_fit_queries"], n_params=TB["arms"]["MLP64_PAT"].get("n_params"),
                            best_epochs=[TB["arms"]["MLP64_PAT"]["seeds"][s]["best_epoch"] for s in sorted(TB["arms"]["MLP64_PAT"]["seeds"])]),
               same_fit_rows=bool(TL["n_fit_queries"] == TB["arms"]["MLP64_PAT"]["n_fit_queries"] and TL["valid_cut_ts"] == TB["valid_cut_ts"]),
               contrasts=R, interpretation=interp_mlp_lmart(R["MLP64_PAT_fused - LMART_PAT_fused"]))
    return out


# ------------------------------------------------------------------------------------------------ M4
def g1_arm(train_json):
    T = json.load(open(train_json))
    sel = {arm: float(np.mean([v["select"]["MRR"] if isinstance(v["select"], dict) else v["select"] for v in T["arms"][arm]["seeds"].values()]))
           for arm in ("G1_K50", "G1_K150") if arm in T["arms"]}
    return max(sel, key=lambda k: (sel[k], k == "G1_K50")), sel


def m4_dataset(ds, g1xf_off, g1xf_train, g1_off, g1_train, hcpat_by_base, feat_off=None):
    rows, ts = ts_of(ds)
    Gx = load(g1xf_off); Go = load(g1_off)
    ax, selx = g1_arm(g1xf_train); ao, selo = g1_arm(g1_train)
    gx = seedavg(Gx, ax); go = seedavg(Go, ao)
    out = dict(dataset=ds, G1_xf_arm=ax, G1_xf_valid_select=selx, G1_arm=ao, G1_valid_select=selo,
               mrr=dict(G1_xf=float(gx.mean()), G1=float(go.mean()), RecB_xf_row=float(Gx["rr_base"].mean()), RecB_tr_row=float(Go["rr_base"].mean())),
               hits=dict(G1_xf=hits(Gx, ax), G1=hits(Go, ao)), per_seed=dict(G1_xf=per_seed(Gx, ax), G1=per_seed(Go, ao)))
    cols = {"G1_xf - G1": gx - go, "RecB_xf - RecB_tr (generator rows)": Gx["rr_base"].astype(np.float64) - Go["rr_base"].astype(np.float64)}
    prim = ["G1_xf - G1"]
    for b, f in hcpat_by_base.items():
        Hh = load(f); h = seedavg(Hh, "HRCOV_PAT"); out["mrr"][f"HRCOV[{b}]"] = float(h.mean())
        cols[f"HRCOV[{b}] - G1_xf"] = h - gx; cols[f"HRCOV[{b}] - G1"] = h - go; prim.append(f"HRCOV[{b}] - G1_xf")
    if feat_off:
        Z = load(feat_off)
        A = {nm: seedavg(Z, f"{nm}_fused") for nm in ("MLP64_PAT", "PAT_RBtr", "PAT_RBxf", "HR19", "HR19_RBtr", "HR19_RBxf")}
        for k, v in A.items(): out["mrr"][k] = float(v.mean())
        out["per_seed_feat"] = {k: per_seed(Z, f"{k}_fused") for k in A}
        cols.update({"PAT_RBxf - PAT": A["PAT_RBxf"] - A["MLP64_PAT"], "HR19_RBxf - HR19": A["HR19_RBxf"] - A["HR19"],
                     "PAT_RBtr - PAT": A["PAT_RBtr"] - A["MLP64_PAT"], "HR19_RBtr - HR19": A["HR19_RBtr"] - A["HR19"],
                     "PAT_RBxf - PAT_RBtr": A["PAT_RBxf"] - A["PAT_RBtr"], "HR19_RBxf - HR19_RBtr": A["HR19_RBxf"] - A["HR19_RBtr"]})
        prim += ["PAT_RBxf - PAT", "HR19_RBxf - HR19"]
    out["contrasts"] = contrasts(cols, ts, ())   # primaries are separate questions (no Holm across them)
    for k in prim: out["contrasts"][k]["primary"] = True
    return out


# ------------------------------------------------------------------------------------------------ M3a
def m3a(valbase_off, orig_hcpat, orig_ctrl, valbase_hcpat_off=None):
    rows, ts = ts_of("smallpedia")
    Vb = load(valbase_off); Hh = load(orig_hcpat); C = load(orig_ctrl)
    bv, cmv = better_ctrl(Vb); bo, cmo = better_ctrl(C)
    h = seedavg(Hh, "HRCOV_PAT"); bvv = Vb[f"rr_{bv}"].astype(np.float64); boo = C[f"rr_{bo}"].astype(np.float64)
    cols = {"HRCOV_PAT(original base) - better control(validation-trained base)": h - bvv,
            "validation-trained base - original base": Vb["rr_base"].astype(np.float64) - C["rr_base"].astype(np.float64),
            "better control(validation-trained) - better control(original)": bvv - boo,
            "HRCOV_PAT(original base) - better control(original base)": h - boo}
    out = dict(better_control_valbase=bv, control_MRR_valbase=cmv, better_control_orig=bo, control_MRR_orig=cmo,
               mrr=dict(valbase=float(Vb["rr_base"].mean()), orig_base=float(C["rr_base"].mean()), HRCOV_PAT_orig=float(h.mean()),
                        better_ctrl_valbase=float(bvv.mean()), better_ctrl_orig=float(boo.mean())),
               hits=dict(valbase=hits(Vb, "base"), BLEND_valbase=hits(Vb, "BLEND"), POOL_valbase=hits(Vb, "POOL")))
    if valbase_hcpat_off and os.path.exists(valbase_hcpat_off if isinstance(valbase_hcpat_off, str) else valbase_hcpat_off[0]):
        Q = load(valbase_hcpat_off); hv = seedavg(Q, "HRCOV_PAT_valbase")
        assert base_identity(Q["rr_base"], Vb["rr_base"])["mean_abs"] < 1e-6
        cols["[secondary] HRCOV_PAT(validation-trained base) - better control(validation-trained base)"] = hv - bvv
        out["mrr"]["HRCOV_PAT_valbase"] = float(hv.mean()); out["per_seed_HRCOV_PAT_valbase"] = per_seed(Q, "HRCOV_PAT_valbase")
    out["contrasts"] = contrasts(cols, ts, ())
    out["contrasts"]["HRCOV_PAT(original base) - better control(validation-trained base)"]["primary"] = True
    return out


# ------------------------------------------------------------------------------------------------ M6
def m6_family(name, seeds):
    """seeds: {seed: (ctrl_or_off_file(s) with rr_base/BLEND/POOL, hcpat file(s) with rr_HRCOV_PAT_s*)}"""
    rows, ts = ts_of("smallpedia"); res = {}
    for s, (cf, hf) in seeds.items():
        C = load(cf); Hh = load(hf); bc, cm = better_ctrl(C); h = seedavg(Hh, "HRCOV_PAT")
        R = contrasts({"HRCOV_PAT - better control": h - C[f"rr_{bc}"].astype(np.float64)}, ts)
        res[s] = dict(base=float(C["rr_base"].mean()), BLEND=cm["BLEND"], POOL=cm["POOL"], better=bc, HRCOV_PAT=float(h.mean()),
                      margin=R["HRCOV_PAT - better control"])
    m = np.array([v["margin"]["delta"] for v in res.values()]); b = np.array([v["base"] for v in res.values()])
    summ = dict(n_seeds=len(res), margin_mean=float(m.mean()), margin_sd=float(m.std(ddof=1)) if len(m) > 1 else None,
                margin_min=float(m.min()), margin_max=float(m.max()), base_mean=float(b.mean()), base_sd=float(b.std(ddof=1)) if len(b) > 1 else None)
    return dict(family=name, seeds=res, summary=summ)
