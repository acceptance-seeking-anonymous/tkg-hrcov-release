#!/usr/bin/env python3
"""pilot_icews / pi_agg.py -- Stage A aggregation: test-sample rows, paired bootstrap, pre-registered GO / NO-GO, strata.

  python3 pi_agg.py --root aggroot --rule stageA_decision_rule.json --out icews_stageA_verdict.json
aggroot/: bundle_<base>/ (pi_chain.sh collect) for regcn, logcl; recb_test_sample_rank.npy (harness RecB-lite ranks,
pi_prep), recblite_official.json + recblite_official_test_sample_rank.npy, recb_official_w100.json +
recb_official_w100_test_sample_rr.npy, prep_summary.json, relnorm_summary.json, tgb_verify.json.
Harness bootstrap = oih_common.paired_bootstrap (Poisson(1), B = 2000 decision / 1000 strata, seed 0); one-sided
p = 1 - p_delta_gt0.  Re-ranker per-query RR = mean over seeds 0,1,2.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def paired_bootstrap(rr_a, rr_b, B=2000, seed=0, chunk=200):
    """verbatim copy of harness oih_common.paired_bootstrap (Poisson(1) paired bootstrap over queries)."""
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64)
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    nz = np.nonzero(d)[0]
    d0 = float(d.mean())
    if len(nz) == 0:
        return {"delta": 0.0, "CI95": [0.0, 0.0], "p_delta_gt0": 0.0, "excludes_zero": False, "n": n, "n_changed": 0}
    boots = np.empty(B)
    dn = d[nz]
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        w = rng.poisson(1.0, size=(bb, len(nz)))
        boots[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": round(d0, 6), "CI95": [round(float(lo), 6), round(float(hi), 6)],
            "p_delta_gt0": round(float((boots > 0).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(n), "n_changed": int(len(nz)),
            "method": "Poisson(1) paired bootstrap over queries, B=%d" % B}


def mets_rr(rrs):
    """rrs: list of per-seed reciprocal-rank arrays (or one).  MRR of the seed-mean RR; H@k mean over seeds."""
    if not isinstance(rrs, list):
        rrs = [rrs]
    rk = [1.0 / r for r in rrs]
    out = dict(MRR=float(np.mean(np.mean(rrs, 0))), n=int(len(rrs[0])))
    for k in (1, 3, 10):
        out[f"H{k}"] = float(np.mean([np.mean(x <= k + 1e-9) for x in rk]))
    if len(rrs) > 1:
        out["seed_sd_MRR"] = float(np.std([np.mean(r) for r in rrs]))
    return out


def one_sided_p(bs):
    return round(1.0 - bs["p_delta_gt0"], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", default="regcn,logcl")
    a = ap.parse_args()
    R = a.root
    rule = json.load(open(a.rule))
    J = lambda p: json.load(open(f"{R}/{p}"))
    prep = J("prep_summary.json"); reln = J("relnorm_summary.json"); tv = J("tgb_verify.json")
    recb_rk = np.load(f"{R}/recb_test_sample_rank.npy").astype(np.float64)
    recb_rr = 1.0 / recb_rk
    rlo = J("recblite_official.json"); rlo_rk = np.load(f"{R}/recblite_official_test_sample_rank.npy")
    rbo = J("recb_official_w100.json") if os.path.exists(f"{R}/recb_official_w100.json") else None
    rbo_rr = np.load(f"{R}/recb_official_w100_test_sample_rr.npy") if rbo else None
    V = dict(stage="A", pilot="3 of 4 (tkgl-icews)", rule_file=os.path.basename(a.rule), rule_registered_utc=rule["registered_utc"],
             hosts=dict(compute="serverA (2x RTX PRO 6000 Blackwell Max-Q, docker tkg-gpu:latest, torch 2.8.0+cu128, py-tgb 2.3.0)",
                        aggregation="local laptop (CPU)"),
             n_test_sample=int(len(recb_rr)), prep=prep, relnorm=reln, tgb_verify=tv, rows=[], decision={}, strata={}, secondary={})
    V["rows"].append(dict(base="-", system="RecB-lite (full history, lambda=%.2f, valid-selected)" % prep["recb_lite"]["lambda_selected"],
                          evaluator="harness", **mets_rr(recb_rr), host="serverA CPU"))
    V["rows"].append(dict(base="-", system="RecB-lite (full history)", evaluator="official py-tgb", MRR=rlo["official"]["MRR"],
                          H10=rlo["official"]["H10"], n=rlo["n"], host="serverA CPU",
                          note=f"per-query rank equal to harness for {rlo['frac_rank_equal']:.6f} of queries"))
    if rbo:
        V["rows"].append(dict(base="-", system="RecB official TGB predictor, default mode (lambda 0.1, alpha 0.99), window 100",
                              evaluator="official py-tgb", MRR=float(np.mean(rbo_rr)), H10=rbo["official"]["H10"], n=int(len(rbo_rr)), host="serverA CPU"))
    decision = {}
    for b in a.bases.split(","):
        bd = f"{R}/bundle_{b}"
        q = dict(np.load(f"{bd}/test_q_orig.npz"))
        qrn = dict(np.load(f"{bd}/test_q_rn.npz"))
        rk = dict(np.load(f"{bd}/ranks_test.npz"))
        tr = json.load(open(f"{bd}/train_{b}.json"))
        lk = json.load(open(f"{bd}/lookup_{b}.json"))
        sysj = json.load(open(f"{bd}/systems_{b}.json")) if os.path.exists(f"{bd}/systems_{b}.json") else None
        basej = json.load(open(f"{bd}/base_{b}_stageA.json"))
        assert len(q["r_base"]) == len(recb_rr)
        sel = tr["selection"]; A = sel["A_star"]; B_ = sel["B_star"]
        seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in rk if "_s" in k})
        rr = lambda nm: [1.0 / rk[f"{nm}_s{s}"].astype(np.float64) for s in seeds]
        base_rr = 1.0 / q["r_base"].astype(np.float64)
        arms = ["HR"] + [f"HRCOV{x}" for x in (50, 100, 150)] + [f"{A}_RN"]
        RR = {nm: np.mean(rr(nm), 0) for nm in arms}
        RR["base"] = base_rr; RR["RecB-lite"] = recb_rr
        V["rows"].append(dict(base=b, system=f"{b} (window-trained frozen base, epoch {basej['selected_epoch']})", evaluator="harness",
                              **mets_rr(base_rr), host="serverA GPU"))
        if "official" in lk:
            V["rows"].append(dict(base=b, system=f"{b} (window-trained frozen base)", evaluator="official py-tgb", MRR=lk["official"]["base"]["MRR"],
                                  H10=lk["official"]["base"]["H10"], n=lk["official"]["base"]["n"], host="serverA GPU+CPU"))
        for nm in arms:
            role = ("A* (valid-chosen)" if nm == A else ("RELNORM variant of A*" if nm.endswith("_RN") else ""))
            V["rows"].append(dict(base=b, system=nm, role=role, evaluator="harness", seeds=len(seeds), **mets_rr(rr(nm)),
                                  select_MRR_mean=tr["arms"][nm]["select_MRR_mean"], host="serverA GPU"))
        if sysj:
            for nm, m in sysj["official"].items():
                V["rows"].append(dict(base=b, system=nm, evaluator="official py-tgb (seed 0)", MRR=m["MRR"], H10=m["H10"], n=m["n"],
                                      harness_seed0_MRR=float(np.mean(1.0 / rk[nm].astype(np.float64))), host="serverA GPU+CPU"))
        # ---------------- decision
        head = q["freq_stratum"] == 2
        d_base = paired_bootstrap(RR["base"], RR[A], B=2000); d_recb = paired_bootstrap(RR["RecB-lite"], RR[A], B=2000)
        h_base = paired_bootstrap(RR["base"][head], RR[A][head], B=1000); h_recb = paired_bootstrap(RR["RecB-lite"][head], RR[A][head], B=1000)
        dmg = lambda h: bool(h["CI95"][1] < 0 and abs(h["delta"]) >= 0.001)
        ga = d_base["delta"] >= 0.005 and one_sided_p(d_base) < 0.05
        gb = d_recb["delta"] >= 0.005 and one_sided_p(d_recb) < 0.05
        gc = (not dmg(h_base)) and (not dmg(h_recb))
        off_chk = {}
        if sysj and "official" in lk:
            off_chk["base_abs_diff"] = abs(lk["official"]["base"]["MRR"] - float(np.mean(base_rr)))
            k0 = f"{A}_s{seeds[0]}"
            if k0 in sysj["official"]:
                off_chk["A_star_seed0_abs_diff"] = abs(sysj["official"][k0]["MRR"] - float(np.mean(1.0 / rk[k0].astype(np.float64))))
            for nm, m in sysj["official"].items():
                off_chk[f"{nm}_abs_diff"] = abs(m["MRR"] - float(np.mean(1.0 / rk[nm].astype(np.float64))))
        off_chk["recblite_abs_diff"] = abs(rlo["official"]["MRR"] - float(np.mean(recb_rr)))
        off_ok = all(v < 1e-4 for v in off_chk.values())
        decision[b] = dict(A_star=A, B_star=B_, selection=sel, a_vs_base=dict(**d_base, p_one_sided=one_sided_p(d_base), pass_=bool(ga)),
                           b_vs_recb_lite=dict(**d_recb, p_one_sided=one_sided_p(d_recb), pass_=bool(gb)),
                           c_head_guardrail=dict(head_share=float(head.mean()), vs_base=dict(**h_base, DAMAGE=dmg(h_base)),
                                                 vs_recb_lite=dict(**h_recb, DAMAGE=dmg(h_recb)), pass_=bool(gc)),
                           official_check=dict(abs_diffs=off_chk, all_below_1em4=bool(off_ok)),
                           PASS=bool(ga and gb and gc))
        # ---------------- secondary: RELNORM, HR vs HRCOV
        un = q["rel_unseen_train"].astype(bool)
        rn = f"{A}_RN"
        sec = dict(relnorm=dict(overall=paired_bootstrap(RR[A], RR[rn], B=1000),
                                rel_unseen_in_train=paired_bootstrap(RR[A][un], RR[rn][un], B=1000),
                                rel_seen_in_train=paired_bootstrap(RR[A][~un], RR[rn][~un], B=1000),
                                share_rel_unseen=float(un.mean()),
                                MRR_by_rel_seen={nm: dict(unseen=float(RR[nm][un].mean()), seen=float(RR[nm][~un].mean()))
                                                 for nm in ("base", "RecB-lite", "HR", f"HRCOV{B_}", A, rn)}),
                   HRCOV_vs_HR=paired_bootstrap(RR["HR"], RR[f"HRCOV{B_}"], B=1000),
                   A_star_minus_RecB_official_default=(paired_bootstrap(rbo_rr, RR[A], B=1000) if rbo else None),
                   base_vs_recb_lite=paired_bootstrap(RR["RecB-lite"], RR["base"], B=1000))
        if sysj and f"{rn}_s{seeds[0]}" in sysj["official"] and f"{A}_s{seeds[0]}" in sysj["official"]:
            sec["relnorm"]["official_seed0_MRR"] = dict(A_star=sysj["official"][f"{A}_s{seeds[0]}"]["MRR"], A_star_RN=sysj["official"][f"{rn}_s{seeds[0]}"]["MRR"])
        V["secondary"][b] = sec
        # ---------------- strata
        rbase = q["r_base"]
        strata = {"all": np.ones(len(rbase), bool), "exact recurrence (s,r',o) before t": q["trip"].astype(bool),
                  "non-recurrent": ~q["trip"].astype(bool), "pair history (any (s,o) event before t)": q["dyad"].astype(bool),
                  "no pair history": ~q["dyad"].astype(bool), "pool-absent (gold outside base top-50)": rbase > 50,
                  "gold in base top-50": rbase <= 50, "relation unseen in train (post-2020 zero-stripped code)": un,
                  "relation seen in train": ~un, "cold gold (train freq < 50)": q["freq_stratum"] == 0,
                  "mid gold": q["freq_stratum"] == 1, "head gold (train freq >= 500)": head,
                  "first-seen entity": q["first_seen"].astype(bool), "hist=no (no train+valid co-occurrence)": ~q["in_hist"].astype(bool)}
        st = {}
        for nm, m in strata.items():
            if m.sum() == 0:
                continue
            row = dict(share=float(m.mean()), n=int(m.sum()))
            for sysn in ("base", "RecB-lite", "HR", f"HRCOV{B_}", A, rn):
                row[f"MRR_{sysn}"] = float(RR[sysn][m].mean())
            if rbo:
                row["MRR_RecB_official_w100"] = float(rbo_rr[m].mean())
            row["d_Astar_vs_base"] = paired_bootstrap(RR["base"][m], RR[A][m], B=1000)
            row["d_Astar_vs_recb_lite"] = paired_bootstrap(RR["RecB-lite"][m], RR[A][m], B=1000)
            row["d_RN_vs_Astar"] = paired_bootstrap(RR[A][m], RR[rn][m], B=1000)
            st[nm] = row
        V["strata"][b] = st
        gp = q["gold_pos"]
        V.setdefault("ceilings", {})[b] = dict(gold_in_top50=float((rbase <= 50).mean()), gold_in_Ccov200=float((gp >= 0).mean()),
                                               gold_in_Ccov_Bstar=float(((gp >= 0) & (gp < 50 + B_)).mean()),
                                               MRR_top50_at1=float(np.mean(np.where(rbase <= 50, 1.0, base_rr))),
                                               MRR_Ccov_Bstar_at1=float(np.mean(np.where((gp >= 0) & (gp < 50 + B_), 1.0, base_rr))),
                                               realised_Astar=float(RR[A].mean()))
        V.setdefault("base_training", {})[b] = dict(window=basej["window"], grid=basej["grid"], selected_epoch=basej["selected_epoch"],
                                                    selected_valid_MRR=basej["selected_valid_MRR"], replay_vs_dump=lk.get("replay_vs_dump"),
                                                    official_vs_harness_base=lk.get("official_vs_harness_base"), train_s=basej.get("train_s"))
    # ---------------- POST-HOC (not pre-registered): RecB-lite on relation-normalised data (pi_relnorm_data.py)
    if os.path.exists(f"{R}/recbRN_test_sample_rank.npy"):
        rs = J("recbRN_summary.json"); rn = 1.0 / np.load(f"{R}/recbRN_test_sample_rank.npy").astype(np.float64)
        post = dict(note="POST-HOC (not pre-registered; computed after the GO decision was read): full-history RecB-lite on the relation-"
                         "normalised data dir (pi_relnorm_data.py; lambda re-selected on full valid). Tests whether the re-ranker's margin over the "
                         "recurrency rule survives when BOTH read relation-normalised history.",
                    recb_lite_rn=dict(lambda_selected=rs["lambda_selected"], valid_full_MRR=rs["valid_full"]["MRR"], test_sample_MRR=float(rn.mean())),
                    recb_lite_rn_minus_recb_lite=paired_bootstrap(recb_rr, rn, B=1000))
        for b in a.bases.split(","):
            q = dict(np.load(f"{R}/bundle_{b}/test_q_orig.npz")); un = q["rel_unseen_train"].astype(bool)
            rk = dict(np.load(f"{R}/bundle_{b}/ranks_test.npz")); A = decision[b]["A_star"]
            seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in rk if "_s" in k})
            arn = np.mean([1.0 / rk[f"{A}_RN_s{s}"].astype(np.float64) for s in seeds], 0)
            a0 = np.mean([1.0 / rk[f"{A}_s{s}"].astype(np.float64) for s in seeds], 0)
            post[b] = dict(A_star_RN_minus_recb_lite_rn=paired_bootstrap(rn, arn, B=1000),
                           A_star_RN_minus_recb_lite_rn_unseen_rel=paired_bootstrap(rn[un], arn[un], B=1000),
                           A_star_minus_recb_lite_rn=paired_bootstrap(rn, a0, B=1000),
                           MRR_unseen_rel=dict(recb_lite=float(recb_rr[un].mean()), recb_lite_rn=float(rn[un].mean()), A_star=float(a0[un].mean()), A_star_RN=float(arn[un].mean())),
                           MRR_seen_rel=dict(recb_lite=float(recb_rr[~un].mean()), recb_lite_rn=float(rn[~un].mean()), A_star=float(a0[~un].mean()), A_star_RN=float(arn[~un].mean())))
        V["post_hoc"] = post
        V["rows"].append(dict(base="-", system="RecB-lite RELNORM (full history, relation-normalised data; POST-HOC)", evaluator="harness",
                              **mets_rr(rn), host="serverA CPU"))
    V["decision"] = dict(per_base=decision, GO=bool(all(d["PASS"] for d in decision.values())),
                         rule=rule["GO_rule"], official_evaluator_reproduces_harness=bool(all(d["official_check"]["all_below_1em4"] for d in decision.values())))
    V["verdict"] = "GO" if V["decision"]["GO"] else "NO-GO"
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    json.dump(V, open(a.out, "w"), indent=1, default=conv)
    print(V["verdict"], json.dumps({b: dict(A=d["A_star"], a=d["a_vs_base"]["delta"], b=d["b_vs_recb_lite"]["delta"], c=d["c_head_guardrail"]["pass_"],
                                            PASS=d["PASS"]) for b, d in decision.items()}))


if __name__ == "__main__":
    main()
