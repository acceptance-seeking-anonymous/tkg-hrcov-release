"""v78 / v78_agg.py -- aggregation and decision of v78_framing_rule + v78_rule_amendment_1 from the official per-query arrays.
Seed-mean per-query RR (official evaluator), paired deltas with query Poisson + timestamp-block bootstraps (B = 2000, shared
draws, seed 0; v7_common.summ_multi), L = 5th / U = 95th percentile.  BEATS / MATCHES / SMALL per the rule.
Usage: exec(open(os.path.join(os.environ['SRC'], 'v78', 'v78_agg.py')).read()); res = run_all(CONF, EXT)
"""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.environ.get("SRC", "src"), "v7"))
import v7_common as VC
BAR = 0.005


def arm(Z, name, seeds=(0, 1, 2)):
    ks = [f"{name}_s{s}" for s in seeds if f"rr_{name}_s{s}" in Z] or ([name] if f"rr_{name}" in Z else [])
    assert ks, name
    rr = np.mean([np.asarray(Z[f"rr_{k}"], np.float64) for k in ks], 0)
    H = {h: np.mean([(np.asarray(Z[f"rank_{k}"], np.float64) <= h).astype(np.float64) for k in ks], 0) for h in (1, 3, 10)}
    hr = [float(np.mean(1.0 / np.asarray(Z[f"hrank_{k}"], np.float64))) for k in ks if f"hrank_{k}" in Z]
    per = [float(np.mean(np.asarray(Z[f"rr_{k}"], np.float64))) for k in ks]
    return dict(rr=rr, H=H, per_seed=per, n_seeds=len(ks), harness_per_seed=hr)


def mets(a):
    return dict(MRR=float(a["rr"].mean()), H1=float(a["H"][1].mean()), H3=float(a["H"][3].mean()), H10=float(a["H"][10].mean()),
                per_seed_MRR=a["per_seed"], n_seeds=a["n_seeds"],
                harness_minus_official_per_seed=[h - o for h, o in zip(a["harness_per_seed"], a["per_seed"])] if a["harness_per_seed"] else None)


def compare(arms, pairs, ts):
    cols = {}
    for lab, x, y in pairs:
        cols[f"{lab}|MRR"] = arms[x]["rr"] - arms[y]["rr"]
        for h in (1, 3, 10): cols[f"{lab}|H{h}"] = arms[x]["H"][h] - arms[y]["H"][h]
    S = VC.summ_multi(cols, ts, bar=BAR)
    return {lab: dict(a=x, b=y, **{m: S[f"{lab}|{m}"] for m in ("MRR", "H1", "H3", "H10")}) for lab, x, y in pairs}


def criterion(r):
    """r = compare() row for HRcov - X (MRR)."""
    m = r["MRR"]
    beats = m["query"]["L"] > 0 and m["ts"]["L"] > 0 and m["delta"] >= BAR
    matches = m["query"]["U"] < BAR and m["ts"]["U"] < BAR
    return "BEATS" if beats else "MATCHES" if matches else "SMALL"


def sel_mrr(train, arm_):
    if arm_ not in train["arms"]: return None
    return float(np.mean([v["select"]["MRR"] for v in train["arms"][arm_]["seeds"].values()]))


def run_config(c, cf, ext=None):
    Hs = np.load(cf["hrcov"]); Zh = np.load(cf["v78_hist"]); Zb = np.load(cf["v78_base"]); tr = json.load(open(cf["train"]))
    n = len(Zh["rows"]); ts = Zh["ts"].astype(np.int64)
    for Z in (Hs, Zb): assert np.array_equal(np.asarray(Z["rows"]), np.arange(n)), "row order"
    Z = {k: Zh[k] for k in Zh.files}; Z.update({k: Zb[k] for k in Zb.files if k not in Z})
    b = cf["base"]; A = {}
    A["HRcov"] = arm({k: Hs[k] for k in Hs.files}, "HRCOV_PAT")
    A["base(stored)"] = arm({k: Hs[k] for k in Hs.files}, "base"); A["base"] = arm(Z, "base")
    names = {"HIST": "HIST", "SA": "SA_W100", "SA_GEN": "SA_GEN", "SA_W50": "SA_W50", "SA_W150": "SA_W150",
             "+C": f"LAD_C_{b}", "+S": f"LAD_S_{b}", "+T": f"LAD_T_{b}", "HR_W50": f"HR_W50_{b}", "HR_W150": f"HR_W150_{b}",
             "REL_LIN": f"REL_LIN_{b}", "REL_LIN_global": f"REL_LIN_global_{b}"}
    missing = []
    for k, v in names.items():
        try: A[k] = arm(Z, v)
        except AssertionError: missing.append(k)
    if ext:
        for k, (path, sysname) in ext.items():
            E = np.load(path); assert np.array_equal(np.asarray(E["rows"]), np.arange(n)); A[k] = arm({x: E[x] for x in E.files}, sysname)
        bx = arm({x: E[x] for x in E.files}, "base"); ident_ext = float(np.abs(bx["rr"] - A["base"]["rr"]).mean())
    if cf.get("sa_erm"):
        Ez = np.load(cf["sa_erm"][0]); assert np.array_equal(np.asarray(Ez["rows"]), np.arange(n))
        A["SA_ERM"] = arm({x: Ez[x] for x in Ez.files}, "SA_ERM"); A["ERM_ORDER"] = arm({x: Ez[x] for x in Ez.files}, "HIST")
    ident = dict(base_rr_mean_abs_diff=float(np.abs(A["base(stored)"]["rr"] - A["base"]["rr"]).mean()),
                 frac_base_rr_differs=float((np.abs(A["base(stored)"]["rr"] - A["base"]["rr"]) > 1e-6).mean()))
    ident["ok"] = ident["base_rr_mean_abs_diff"] < 1e-6
    # validation-selection MRRs (seed-mean, harness exact) for every arm
    vsel = {k: sel_mrr(tr, v) for k, v in names.items() if k not in ("HIST",)}
    if cf.get("sa_erm"): vsel["SA_ERM"] = sel_mrr(json.load(open(cf["sa_erm"][1])), "SA_ERM"); cf["sa_erm_select"] = vsel["SA_ERM"]
    ext_sel = None
    if cf.get("ext_select"):
        per = {}
        for m_, d_ in cf["ext_select"].items():
            v_ = max(d_, key=lambda k_: d_[k_]); per[m_] = dict(variant=v_, select_MRR=d_[v_], candidates=d_)
        mbest = max(per, key=lambda k_: per[k_]["select_MRR"]); ext_sel = dict(per_method=per, method=mbest, X=f"EXT_{per[mbest]['variant']}_{mbest}")
        cf["external_comparator"] = ext_sel["X"]
    # standalone comparator: SA (SA_ERM only if built)
    stand = "SA"
    if "SA_ERM" in A and (cf.get("sa_erm_select") or -1) > (vsel.get("SA") or -1): stand = "SA_ERM"
    stand_sens = "SA_GEN" if (vsel.get("SA_GEN") or -1) > (vsel.get("SA") or -1) else "SA"
    pairs = [("HRcov - SA", "HRcov", "SA"), ("HRcov - SA_GEN", "HRcov", "SA_GEN"), ("HRcov - HIST", "HRcov", "HIST"),
             ("HRcov - base", "HRcov", "base"), ("SA - HIST", "SA", "HIST"), ("SA - base", "SA", "base"),
             ("ladder: +C - SA", "+C", "SA"), ("ladder: +S - +C", "+S", "+C"), ("ladder: +T - +S", "+T", "+S"),
             ("ladder: HRcov - +T (+P)", "HRcov", "+T"),
             ("width: SA_W50 - SA", "SA_W50", "SA"), ("width: SA_W150 - SA", "SA_W150", "SA"),
             ("width: HR_W50 - HRcov", "HR_W50", "HRcov"), ("width: HR_W150 - HRcov", "HR_W150", "HRcov"),
             ("width: HR_W50 - SA_W50", "HR_W50", "SA_W50"), ("width: HR_W150 - SA_W150", "HR_W150", "SA_W150"),
             ("REL_LIN: HRcov - REL_LIN", "HRcov", "REL_LIN"), ("REL_LIN: REL_LIN - REL_LIN_global", "REL_LIN", "REL_LIN_global"),
             ("REL_LIN: HRcov - REL_LIN_global", "HRcov", "REL_LIN_global")]
    if stand != "SA": pairs.append((f"HRcov - {stand}", "HRcov", stand))
    if "SA_ERM" in A: pairs += [("HRcov - SA_ERM", "HRcov", "SA_ERM"), ("SA_ERM - SA", "SA_ERM", "SA"), ("HRcov - ERM_ORDER", "HRcov", "ERM_ORDER")]
    if ext:
        for k in ext: pairs += [(f"HRcov - {k}", "HRcov", k), (f"{stand} - {k}", stand, k)]
    pairs = [p for p in pairs if p[1] in A and p[2] in A]
    S = compare(A, pairs, ts)
    out = dict(config=c, ds=cf["ds"], base=b, n_queries=int(n), n_ts=int(len(np.unique(ts))), missing=missing, base_identity=ident,
               systems={k: mets(v) for k, v in A.items()}, validation_select_MRR=vsel, standalone_comparator=stand,
               standalone_comparator_sensitivity=stand_sens, comparisons=S, external_selection=ext_sel,
               ext_base_identity=(ident_ext if ext else None))
    out["decision"] = {"standalone": dict(X=stand, call=criterion(S[f"HRcov - {stand}"]))}
    out["decision_sensitivity"] = {"standalone": dict(X=stand_sens, call=criterion(S[f"HRcov - {stand_sens}"]))}
    if ext and cf.get("external_comparator") in A:
        X = cf["external_comparator"]; out["decision"]["external"] = dict(X=X, call=criterion(S[f"HRcov - {X}"]),
                                                                        standalone_beats_external=criterion(S[f"{stand} - {X}"]) == "BEATS")
    else:
        out["decision"]["external"] = dict(X=None, call="not_assessed")
    lt = out["systems"].get("+T", {}).get("MRR"); out["ladder_reproduction"] = dict(HRcov=out["systems"]["HRcov"]["MRR"], plusT=lt,
                                                                               abs_diff=(abs(lt - out["systems"]["HRcov"]["MRR"]) if lt is not None else None),
                                                                               within_0002=(abs(lt - out["systems"]["HRcov"]["MRR"]) <= 0.002) if lt is not None else None)
    flags = {}
    for k, m in out["systems"].items():
        d_ = m.get("harness_minus_official_per_seed")
        if d_: flags[k] = float(max(abs(x) for x in d_))
    out["harness_vs_official_max_abs"] = flags; out["harness_vs_official_flagged"] = {k: v for k, v in flags.items() if v > 1e-4}
    return out


def outcome(results):
    cfgs = list(results)
    st = {c: results[c]["decision"]["standalone"]["call"] for c in cfgs}
    ex = {c: results[c]["decision"]["external"] for c in cfgs}
    assessed_ext = [c for c in cfgs if ex[c]["call"] != "not_assessed"]
    f3 = sum(1 for c in assessed_ext if ex[c]["call"] == "MATCHES" and not ex[c].get("standalone_beats_external")) >= 2
    f2 = sum(1 for c in cfgs if st[c] == "MATCHES") >= 2
    f1 = all(st[c] == "BEATS" for c in cfgs) and all(ex[c]["call"] == "BEATS" for c in assessed_ext)
    lab = "F3_EXTERNAL_MATCHES" if f3 else "F2_STANDALONE_MATCHES" if f2 else "F1_RETAIN_FROZEN_BASE" if f1 else "F4_CONDITIONAL"
    return dict(label=lab, standalone_calls=st, external_calls={c: ex[c]["call"] for c in cfgs}, external_assessed=assessed_ext,
                F3=f3, F2=f2, F1=f1, note=("F1 evaluated over assessed configurations; external comparator not assessed in " +
                                           ", ".join(c for c in cfgs if c not in assessed_ext)) if len(assessed_ext) < len(cfgs) else "")


def outcome_sensitivity(results):
    st = {c: results[c]["decision_sensitivity"]["standalone"]["call"] for c in results}
    r2 = {c: dict(results[c], decision=dict(standalone=dict(call=st[c]), external=results[c]["decision"]["external"])) for c in results}
    return outcome(r2)
