#!/usr/bin/env python3
"""pilot_polecat / pc_agg.py -- full-test table, strata, ceilings, bootstrap and the pre-registered decision.

Reads the compact per-query arrays harvested from serverA (collect bundle) and writes polecat_pilot_verdict.json.
  python3 pc_agg.py --root <dir with bundle_<base>/ and recb_test_rank.npy> --rule polecat_decision_rule.json
"""
import os, sys, json, argparse
import numpy as np

BASES = ["regcn", "logcl"]
SEEDS = [0, 1, 2]
TRAINED = ["HR", "HR_R", "HRCOV", "HRCOV_R", "HRCOVFEAT"]
KC = 50
PUBLISHED = {  # TGB 2.0 leaderboard / Gastinger et al. 2025 (arXiv:2509.09474) tkgl-polecat test MRR, as quoted in dataset_shortlist.md
    "CountTRuCoLa (best published)": 0.256, "TGB RecB": 0.198, "CEN": 0.184, "RE-GCN (TGB, DGL, history 3)": 0.175}


def mets(rk):
    r = np.asarray(rk, np.float64)
    return dict(MRR=round(float(np.mean(1 / r)), 6), H1=round(float(np.mean(r <= 1)), 6), H3=round(float(np.mean(r <= 3)), 6),
                H10=round(float(np.mean(r <= 10)), 6), n=int(len(r)))


def boot(rr_a, rr_b, B=2000, seed=0):
    """harness oih_common.paired_bootstrap (Poisson(1) weights over changed queries), one-sided p = P(delta<=0)."""
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64); n = len(d)
    nz = np.nonzero(d)[0]
    if len(nz) == 0:
        return dict(delta=0.0, CI95=[0.0, 0.0], p_one_sided=1.0, n=n, n_changed=0)
    rng = np.random.default_rng(seed); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 200):
        bb = min(200, B - b0); w = rng.poisson(1.0, size=(bb, len(nz))); bs[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=round(float(d.mean()), 6), CI95=[round(float(lo), 6), round(float(hi), 6)],
                p_one_sided=round(float((bs <= 0).mean()), 4), n=int(n), n_changed=int(len(nz)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", default="polecat_pilot_verdict.json")
    a = ap.parse_args()
    rule = json.load(open(a.rule))
    V = dict(rule_file=a.rule, registered=rule["registered"], bases={}, published_tgb=PUBLISHED)
    recb = np.load(f"{a.root}/recb_test_rank.npy").astype(np.float64)
    prep = json.load(open(f"{a.root}/prep_summary.json"))
    V["recb_lite"] = dict(test=mets(recb), lambda_selected=prep["recb_lite"]["lambda_selected"], valid=prep["recb_lite"]["valid"],
                          host="serverA (CPU)")
    fulltest_rows = []
    fulltest_rows.append(dict(system="RecB-lite (lambda=%s, valid-selected)" % prep["recb_lite"]["lambda_selected"], base="-", seed="-",
                              host="serverA (CPU)", **mets(recb)))
    decisions = {}
    if os.path.exists(f"{a.root}/recb_official_default.json"):
        ro = json.load(open(f"{a.root}/recb_official_default.json"))
        V["recb_official_default"] = ro
        fulltest_rows.append(dict(system="RecB official TGB script (default lambda=0.1, alpha=0.99; train-mode search not run)", base="-", seed="-",
                                  host="serverA (CPU)", MRR=round(ro["mrr"], 6), H10=round(ro["hits10"], 6), valid_MRR=round(ro["val_mrr"], 6)))
    for base in BASES:
        bd = f"{a.root}/bundle_{base}"
        if not os.path.exists(bd):
            continue
        q = dict(np.load(f"{bd}/test_q.npz"))
        R = dict(np.load(f"{bd}/ranks_test.npz"))
        refs = dict(np.load(f"{bd}/refs_ranks_test.npz"))
        tr = json.load(open(f"{bd}/train_{base}.json")); rf = json.load(open(f"{bd}/refs_{base}.json"))
        rb = q["r_base"].astype(np.float64); n = len(rb)
        assert (R["base"] == q["r_base"]).all()
        # refs use the full-row lookup's re-derived base rank (equal to the dump rank except GPU-nondeterministic near-ties)
        B_ref_agree = float((refs["base"] == q["r_base"]).mean())
        B = dict(n_test=n, refs_base_rank_agreement_with_dump=None, host_bases="serverA GPU (train + dump), CPU (features / re-rankers / refs)")
        rr = {"base": 1 / rb, "RecB-lite": 1 / recb, "base_lookup": 1.0 / refs["base"].astype(np.float64)}
        rows_b = [dict(system=f"{base} (frozen base)", base=base, seed="-", host="serverA (GPU dump)", **mets(rb))]
        if os.path.exists(f"{bd}/lookup_{base}.json"):
            lj = json.load(open(f"{bd}/lookup_{base}.json"))
            if "official" in lj:
                rows_b.append(dict(system=f"{base} (frozen base) [official py-tgb Evaluator]", base=base, seed="-", host=lj["host"],
                                   MRR=round(lj["official"]["base"]["MRR"], 6), H10=round(lj["official"]["base"]["hits10"], 6), n=lj["official"]["base"]["n"]))
        for arm in TRAINED:
            ks = [f"{arm}_s{s}" for s in SEEDS if f"{arm}_s{s}" in R]
            if not ks:
                continue
            for k in ks:
                rows_b.append(dict(system=arm, base=base, seed=int(k.split("_s")[-1]), host=tr["host"], **mets(R[k])))
            rr[arm] = np.mean([1.0 / R[k].astype(np.float64) for k in ks], 0)
            m = dict(MRR=round(float(rr[arm].mean()), 6))
            for kk in ("H1", "H3", "H10"):
                m[kk] = round(float(np.mean([mets(R[k])[kk] for k in ks])), 6)
            sd = float(np.std([mets(R[k])["MRR"] for k in ks]))
            rows_b.append(dict(system=arm, base=base, seed="mean(%d)" % len(ks), MRR_sd_over_seeds=round(sd, 6), host=tr["host"], n=n, **m))
        for ref in ("PAT_BUMP", "BR_CS"):
            if ref in refs:
                rr[ref] = 1.0 / refs[ref].astype(np.float64)
                rows_b.append(dict(system=ref, base=base, seed="-", host=rf["host"], selected=rf[ref]["selected"], **mets(refs[ref])))
        # ---------------- post-hoc diagnostics (block attribution) and 6.5 gated augmentation arms
        if os.path.exists(f"{bd}/ranks_test_diag_{base}.npz"):
            RD = dict(np.load(f"{bd}/ranks_test_diag_{base}.npz")); trd = json.load(open(f"{bd}/train_diag_{base}.json"))
            for arm in sorted(set(k.rsplit("_s", 1)[0] for k in RD)):
                ks = [k for k in RD if k.rsplit("_s", 1)[0] == arm]
                rr[arm] = np.mean([1.0 / RD[k].astype(np.float64) for k in ks], 0)
                rows_b.append(dict(system=arm + " (post-hoc diagnostic)", base=base, seed="mean(%d)" % len(ks), host=trd["host"], n=n,
                                   MRR=round(float(rr[arm].mean()), 6),
                                   **{kk: round(float(np.mean([mets(RD[k])[kk] for k in ks])), 6) for kk in ("H1", "H3", "H10")}))
        aug_rr = {}
        if os.path.exists(f"{bd}/augarms_ranks_test_{base}.npz"):
            RA = dict(np.load(f"{bd}/augarms_ranks_test_{base}.npz")); ja = json.load(open(f"{bd}/augarms_{base}.json"))
            assert (RA["base"] == q["r_base"]).all()
            groups = {}
            for k in RA:
                if k in ("base", "tpool"): continue
                g_ = k.rsplit("_s", 1)[0] if k.startswith("util_") else k
                groups.setdefault(g_, []).append(k)
            for g_, ks in groups.items():
                aug_rr["AUG_" + g_] = np.mean([1.0 / RA[k].astype(np.float64) for k in ks], 0)
                rows_b.append(dict(system="AUG " + g_, base=base, seed=("mean(%d)" % len(ks)) if len(ks) > 1 else "-", host=ja["host"], n=n,
                                   MRR=round(float(aug_rr["AUG_" + g_].mean()), 6),
                                   **{kk: round(float(np.mean([mets(RA[k])[kk] for k in ks])), 6) for kk in ("H1", "H3", "H10")}))
            rr.update(aug_rr)
            B["augmentation_arms"] = dict(pools=ja["pools"], S1=ja["S1"], S2=ja["S2"], S3_selected=ja["S3_selected"],
                                          S4_arms={k: {kk: v.get(kk) for kk in ("n_pairs", "n_queries", "m", "gold_frac_admitted", "dMRR_vs_base")} for k, v in ja["S4_arms"].items()},
                                          tau=ja["tau"])
            cold = q["stratum"] == 0; head = q["stratum"] == 2
            prim = aug_rr.get("AUG_util_ll")
            if prim is not None:
                dec = dict(full=boot(rr["base"], prim), cold=boot(rr["base"][cold], prim[cold]), head=boot(rr["base"][head], prim[head]))
                for comp in ("AUG_random_matched_s0", "AUG_aa_matched_s0", "AUG_unc_matched_s0", "AUG_ungated_sameBm", "AUG_perturb_s0"):
                    if comp in aug_rr:
                        dec[f"cold_vs_{comp}"] = boot(aug_rr[comp][cold], prim[cold]); dec[f"full_vs_{comp}"] = boot(aug_rr[comp], prim)
                comps = [k for k in dec if k.startswith("cold_vs_") and "perturb" not in k]
                dec["PASS_6.5_rule"] = bool(dec["cold"]["p_one_sided"] < 0.05 and dec["cold"]["delta"] > 0 and
                                            all(dec[k]["p_one_sided"] < 0.05 for k in comps) and
                                            not (dec["head"]["CI95"][1] < 0 and abs(dec["head"]["delta"]) >= 0.001))
                B["augmentation_arms"]["tests"] = dec
        fulltest_rows.extend(rows_b)
        B["refs_base_rank_agreement_with_dump"] = B_ref_agree
        # ---------------- comparisons
        cmp_ = {}
        pairs = [("HRCOV", "HR"), ("HRCOV_R", "HR_R"), ("HRCOVFEAT", "HR"), ("HR", "base"), ("HR_R", "base"), ("HRCOV", "base"),
                 ("HRCOV_R", "base"), ("PAT_BUMP", "base_lookup"), ("BR_CS", "base_lookup"), ("base_lookup", "base"), ("HR", "RecB-lite"), ("HRCOV", "RecB-lite"),
                 ("base", "RecB-lite"), ("HRCOV", "HRCOVFEAT"), ("HRCOV_PAT", "HR"), ("HRCOV_BR", "HR"), ("HRCOV_DYAD", "HR"),
                 ("HRCOV_NODYAD", "HR"), ("HRCOV", "HRCOV_PAT"), ("HRCOV", "HRCOV_DYAD")]
        for b_, a_ in pairs:
            if b_ in rr and a_ in rr:
                cmp_[f"{b_} - {a_}"] = boot(rr[a_], rr[b_])
        B["comparisons"] = cmp_
        # ---------------- strata
        st = {"cold (gold train freq<50)": q["stratum"] == 0, "mid": q["stratum"] == 1, "head (>=500)": q["stratum"] == 2,
              "no pair history (no (s,o) event before t)": ~q["dyad"], "pair history": q["dyad"],
              "first-seen entity (s or o has no event before t)": q["first_seen"],
              "pool-absent (gold outside base top-50)": ~q["in_top50"], "gold in base top-50": q["in_top50"],
              "hist=no (harness: no train+valid co-occurrence)": ~q["in_hist"],
              "exact recurrence": q["trip"], "non-recurrent": ~q["trip"],
              "no pair & 2-hop reachable": ~q["dyad"] & q["two_hop"], "no pair & 2-hop unreachable": ~q["dyad"] & ~q["two_hop"],
              "self-loop (s==o)": q["self_loop"]}
        cat = np.full(n, "", object)
        E = q["first_seen"]; Rr = ~E & q["trip"]; Dd = ~E & ~q["trip"] & q["dyad"]; Pp = ~E & ~q["dyad"] & q["pat_any"]; Uu = ~E & ~q["dyad"] & ~q["pat_any"]
        inp = q["in_top50"]
        tax = {"E_emerging": E, "R_rec_in": Rr & inp, "R_rec_out": Rr & ~inp, "D_dyad_in": Dd & inp, "D_dyad_out": Dd & ~inp,
               "P_pat_in": Pp & inp, "P_pat_out": Pp & ~inp, "U_deep_in": Uu & inp, "U_deep_out": Uu & ~inp}
        tot = sum(int(v.sum()) for v in tax.values()); assert tot == n, (tot, n)
        best_sys = "HRCOV" if "HRCOV" in rr else "base"
        strata = {}
        for grp, D_ in (("strata", st), ("taxonomy_9", tax)):
            strata[grp] = {}
            for nm, msk in D_.items():
                if msk.sum() == 0:
                    continue
                e = dict(n=int(msk.sum()), share=round(float(msk.mean()), 4))
                for s_ in rr:
                    if s_ != "base_lookup":
                        e[f"MRR_{s_}"] = round(float(rr[s_][msk].mean()), 6)
                e["lost_MRR_share_HR"] = round(float((1 - rr.get("HR", rr["base"])[msk]).sum() / (1 - rr.get("HR", rr["base"])).sum()), 4)
                if "HRCOV" in rr and "HR" in rr:
                    e["HRCOV_minus_HR"] = boot(rr["HR"][msk], rr["HRCOV"][msk], B=1000)
                for s_ in ("HRCOV_PAT", "HRCOV_BR", "HRCOV_DYAD", "HRCOV_NODYAD"):
                    if s_ in rr and "HR" in rr:
                        e[f"d_{s_}_vs_HR"] = round(float((rr[s_] - rr["HR"])[msk].mean()), 6)
                if "AUG_util_ll" in rr:
                    e["d_AUG_util_ll_vs_base"] = round(float((rr["AUG_util_ll"] - rr["base"])[msk].mean()), 6)
                for s_ in ("HR", "PAT_BUMP", "BR_CS", "HRCOV"):
                    if s_ in rr:
                        ref_ = "base_lookup" if s_ in ("PAT_BUMP", "BR_CS") else "base"
                        e[f"d_{s_}_vs_base"] = round(float((rr[s_] - rr[ref_])[msk].mean()), 6)
                strata[grp][nm] = e
        B["strata"] = strata
        # ---------------- head guardrail (primary comparison)
        hm = q["stratum"] == 2
        if "HRCOV" in rr and "HR" in rr:
            g = boot(rr["HR"][hm], rr["HRCOV"][hm])
            damage = (g["CI95"][1] < 0) and (abs(g["delta"]) >= rule["materiality"])
            B["head_guardrail"] = dict(**g, DAMAGE=bool(damage), strict_CI_below_zero=bool(g["CI95"][1] < 0))
            pc = cmp_["HRCOV - HR"]
            passed = (pc["p_one_sided"] < 0.05) and (pc["delta"] >= rule["materiality"]) and not damage
            decisions[base] = dict(dMRR=pc["delta"], CI95=pc["CI95"], p_one_sided=pc["p_one_sided"], head_DAMAGE=bool(damage), PASS=bool(passed))
            if "HRCOV_R - HR_R" in cmp_:
                s2 = cmp_["HRCOV_R - HR_R"]
                decisions[base]["secondary_R"] = dict(dMRR=s2["delta"], CI95=s2["CI95"], p_one_sided=s2["p_one_sided"],
                                                      pass_like_primary=bool(s2["p_one_sided"] < 0.05 and s2["delta"] >= rule["materiality"]))
        # ---------------- oracle ceilings
        gp = q["gold_pos"].astype(np.int64); intop = (gp >= 0) & (gp < KC); inC = gp >= 0
        def at1(mask):
            return round(float(np.mean(np.where(mask, 1.0, 1.0 / rb))), 6)
        pa = ~q["in_top50"]
        ceil = dict(base_MRR=round(float((1 / rb).mean()), 6), top50_at1=at1(intop), Ccov_at1=at1(inC), coverage_block_only_at1=at1(inC & ~intop),
                    HR_realised=round(float(rr["HR"].mean()), 6) if "HR" in rr else None,
                    HRCOV_realised=round(float(rr["HRCOV"].mean()), 6) if "HRCOV" in rr else None,
                    gold_in_top50=round(float(intop.mean()), 4), gold_in_Ccov=round(float(inC.mean()), 4),
                    gold_in_cov_block_given_pool_absent=round(float((inC & ~intop)[pa].mean()), 4))
        reach = {}
        for nm, msk in (("all", np.ones(n, bool)), ("pool_absent", pa), ("no_pair", ~q["dyad"]), ("no_pair_pool_absent", ~q["dyad"] & pa)):
            e = dict(n=int(msk.sum()),
                     pattern_space_any=round(float(q["pat_any"][msk].mean()), 4), pattern_space_hop1=round(float(q["pat_hop1"][msk].mean()), 4),
                     uncapped_2hop_common_neighbour=round(float(q["two_hop"][msk].mean()), 4),
                     bridge_natural_set=round(float(q["g_in_bridge"][msk].mean()), 4),
                     bridge_natural_median_size=float(np.median(q["br_size"][msk])))
            for Bk in (10, 50, 200):
                e[f"bridge_list_at{Bk}"] = round(float(((q["gpos_br"] >= 0) & (q["gpos_br"] < Bk))[msk].mean()), 4)
                e[f"bridge_random_at{Bk}_matched"] = round(float((q["g_in_bridge"] * np.minimum(1.0, Bk / np.maximum(q["br_size"], 1)))[msk].mean()), 4)
                e[f"random_entities_at{Bk}"] = round(Bk / 150931, 6)
            for Bk in (10, 50, 300):
                e[f"pattern_list_at{Bk}"] = round(float(((q["gpos_pat"] >= 0) & (q["gpos_pat"] < Bk))[msk].mean()), 4)
                e[f"pattern_random_at{Bk}_matched"] = round(float((q["g_in_union"] * np.minimum(1.0, Bk / np.maximum(q["n_union"], 1)))[msk].mean()), 4)
            reach[nm] = e
        ceil["reach"] = reach
        for Bk in (10, 50):
            ceil[f"pattern_list_at{Bk}_placed_at1_MRR"] = at1((q["gpos_pat"] >= 0) & (q["gpos_pat"] < Bk))
            ceil[f"bridge_list_at{Bk}_placed_at1_MRR"] = at1((q["gpos_br"] >= 0) & (q["gpos_br"] < Bk))
        if os.path.exists(f"{bd}/aug_oracle_{base}.json"):
            ceil["augmentation_oracle"] = json.load(open(f"{bd}/aug_oracle_{base}.json"))
        B["oracle_ceilings"] = ceil
        # ---------------- training records (selection on valid)
        B["train_selection"] = {arm: {str(s): dict(best_epoch=v["best_epoch"], select_MRR=v["select"]["MRR"], test_MRR=v["test"]["MRR"],
                                                    cal_alpha=v.get("cal_alpha"), cal_beta=v.get("cal_beta"),
                                                    promoted_cov_per_query=v.get("test_promoted_cov_per_query"),
                                                    frac_queries_with_promotion=v.get("test_frac_queries_with_promotion"))
                                     for s, v in rec.get("seeds", {}).items()}
                                for arm, rec in tr["arms"].items()}
        B["refs_selection"] = {k: dict(selected=rf[k]["selected"], valid=rf[k]["valid"]) for k in ("PAT_BUMP", "BR_CS") if k in rf}
        if os.path.exists(f"{bd}/density_{base}.json"):
            dz = json.load(open(f"{bd}/density_{base}.json"))
            B["density_controls"] = {k: dz[k] for k in ("cohort_train_coverage_hard", "test_uniform_sample", "degree_bins", "n_entities")}
            B["dump_vs_lookup_rank_agreement"] = dz["dump_vs_lookup_rank_agreement"]
        if os.path.exists(f"{bd}/systems_{base}.json"):
            js_ = json.load(open(f"{bd}/systems_{base}.json"))
            B["official_tgb_systems"] = js_.get("official")
            for nm, v in (js_.get("official") or {}).items():
                fulltest_rows.append(dict(system=f"{nm} [official py-tgb Evaluator]", base=base, seed=0, host=js_["host"], MRR=round(v["MRR"], 6),
                                          H10=round(v["hits10"], 6), n=v["n"]))
        for fn in ("lookup_%s.json" % base, "official_%s.json" % base, "lists_summary_%s.json" % base, "feat_summary_%s.json" % base,
                   "dump_log_%s.json" % base):
            if os.path.exists(f"{bd}/{fn}"):
                B[fn.replace(".json", "")] = json.load(open(f"{bd}/{fn}"))
        V["bases"][base] = B
    V["fulltest_rows"] = fulltest_rows
    V["decision"] = dict(per_base=decisions,
                         useful_on_polecat=bool(len(decisions) == 2 and all(d["PASS"] for d in decisions.values())),
                         rule=rule["primary_rule"]["statement"])
    json.dump(V, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(V["decision"], indent=1))


if __name__ == "__main__":
    main()
