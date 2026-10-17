#!/usr/bin/env python3
"""pilot_smallpedia / sp_agg.py -- full-test table, strata, reach, ceilings, bootstrap and the pre-registered decision.
Reads the compact per-query bundles harvested from serverA and writes smallpedia_pilot_verdict.json (local, CPU).
  python3 sp_agg.py --root aggroot --rule smallpedia_decision_rule.json --out smallpedia_pilot_verdict.json
"""
import os, sys, json, argparse
import numpy as np

BASES = ["regcn", "logcl", "tirgn"]
SEEDS = [0, 1, 2]
TRAINED = ["HR", "HRCOV", "HRCOV_ST", "HRCOV_DM", "HRCOV_STFEAT", "HRCOV_PAT", "HR_ST", "STONLY"]
KC = 50
N_ENT = 47433
PUBLISHED = {"CountTRuCoLa (best published)": 0.644, "CEN": 0.612, "RE-GCN (TGB)": 0.594, "TGB RecB (train mode)": 0.604}
SHAPES = ["S", "SS", "ST", "TS"]
HOST_GPU = "serverA (GPU, tkg-gpu:latest)"


def mets(rk):
    r = np.asarray(rk, np.float64)
    return dict(MRR=round(float(np.mean(1 / r)), 6), H1=round(float(np.mean(r <= 1)), 6), H3=round(float(np.mean(r <= 3)), 6),
                H10=round(float(np.mean(r <= 10)), 6), n=int(len(r)))


def boot(rr_a, rr_b, B=2000, seed=0):
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64); n = len(d)
    nz = np.nonzero(d)[0]
    if n == 0:
        return None
    if len(nz) == 0:
        return dict(delta=0.0, CI95=[0.0, 0.0], p_one_sided=1.0, n=n, n_changed=0)
    rng = np.random.default_rng(seed); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 200):
        bb = min(200, B - b0); w = rng.poisson(1.0, size=(bb, len(nz))); bs[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=round(float(d.mean()), 6), CI95=[round(float(lo), 6), round(float(hi), 6)],
                p_one_sided=round(float((bs <= 0).mean()), 4), n=int(n), n_changed=int(len(nz)))


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--rule", required=True)
    ap.add_argument("--out", default="smallpedia_pilot_verdict.json")
    a = ap.parse_args()
    rule = json.load(open(a.rule)); mat = rule["protocol"]["materiality"]
    V = dict(rule_file=os.path.basename(a.rule), registered=rule["registered"], bases={}, published_tgb=PUBLISHED)
    recb = np.load(f"{a.root}/recb_test_rank.npy").astype(np.float64)
    prep = jl(f"{a.root}/prep_summary.json"); tv = jl(f"{a.root}/tgb_verify.json"); sel = jl(f"{a.root}/base_selection.json")
    names = jl(f"{a.root}/static_names.json"); rel_t = [ln.split("\t")[0] for ln in open(f"{a.root}/relation2id.txt")]
    V["tgb_verification"] = tv; V["base_selection"] = sel
    V["recb_lite"] = dict(test=mets(recb), lambda_selected=prep["recb_lite"]["lambda_selected"], valid=prep["recb_lite"]["valid"], host="serverA (CPU)")
    V["prep"] = {k: prep[k] for k in ("static_index", "static_te", "reach_sample", "strata_test", "te") if k in prep}
    fulltest_rows = [dict(system="RecB-lite (lambda=%s, valid-selected)" % prep["recb_lite"]["lambda_selected"], base="-", seed="-",
                          host="serverA (CPU)", **mets(recb))]
    decisions = {}
    for base in BASES:
        bd = f"{a.root}/bundle_{base}"
        if not os.path.exists(f"{bd}/ranks_test.npz"):
            continue
        q = dict(np.load(f"{bd}/test_q.npz")); R = dict(np.load(f"{bd}/ranks_test.npz"))
        tr = json.load(open(f"{bd}/train_{base}.json"))
        rb = q["r_base"].astype(np.float64); n = len(rb)
        assert (R["base"] == q["r_base"]).all()
        B = dict(n_test=n)
        rr = {"base": 1 / rb, "RecB-lite": 1 / recb, "base_dump": 1 / q["r_dump"].astype(np.float64)}
        dl = jl(f"{bd}/dump_log_{base}.json")
        rows_b = [dict(system=f"{base} (frozen base, full-row exact ranks)", base=base, seed="-", host=HOST_GPU, **mets(rb)),
                  dict(system=f"{base} (frozen base, harness dump gold_rank_filt)", base=base, seed="-", host=HOST_GPU, **mets(q["r_dump"]))]
        ob = jl(f"{bd}/official_base_{base}.json"); osy = jl(f"{bd}/official_systems_{base}.json")
        if ob:
            rows_b.append(dict(system=f"{base} (frozen base) [official py-tgb Evaluator]", base=base, seed="-", host=ob["host"],
                               MRR=round(ob["official"]["base"]["MRR"], 6), H10=round(ob["official"]["base"]["hits10"], 6), n=ob["n"]))
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
        if osy:
            for nm, v in osy["official"].items():
                rows_b.append(dict(system=f"{nm} [official py-tgb Evaluator]", base=base, seed=0, host=osy["host"], MRR=round(v["MRR"], 6),
                                   H10=round(v["hits10"], 6), n=v["n"]))
        fulltest_rows.extend(rows_b)
        # ---------------- verification
        fr = jl(f"{bd}/fullrows_{base}.json"); fs = jl(f"{bd}/feat_summary_{base}.json")
        B["verification"] = dict(dump_log=dict(epochs=(dl or {}).get("epochs"), best_epoch=(dl or {}).get("best_epoch"),
                                               valid_filt=(dl or {}).get("valid_filt"), test_filt=(dl or {}).get("test_filt")),
                                 fullrow_replay=(fr or {}).get("stats"),
                                 fullrow_vs_dump_test=dict(frac_rank_equal=fs["test"]["rank_agree_fullrow_vs_dump"] if fs else None,
                                                           max_abs_rank_diff=fs["test"]["max_abs_rank_diff_fullrow_vs_dump"] if fs else None),
                                 identity_f0={k: fs["test"][k] for k in fs["test"] if k.startswith("identity")} if fs else None,
                                 official_vs_harness=None)
        if ob:
            otr = dict(np.load(f"{bd}/official_base_{base}_ranks.npz"))["base"].astype(np.float64)
            B["verification"]["official_vs_harness"] = dict(harness_MRR=round(float((1 / rb).mean()), 6), official_MRR=round(ob["official"]["base"]["MRR"], 6),
                                                           frac_rank_equal=round(float(np.mean(np.abs(otr - rb) < 1e-6)), 6),
                                                           max_abs_rank_diff=float(np.abs(otr - rb).max()),
                                                           frac_differ_only_by_ties=round(float(np.mean((np.abs(otr - rb) > 1e-6) & (np.abs(otr - rb) < 1e4))), 6))
        if osy:
            orr = dict(np.load(f"{bd}/official_systems_{base}_ranks.npz"))
            B["verification"]["official_systems_vs_harness_seed0"] = {
                nm: dict(official_MRR=round(float((1 / orr[nm].astype(np.float64)).mean()), 6),
                         harness_MRR=round(float((1 / R[nm].astype(np.float64)).mean()), 6) if nm in R else None,
                         frac_rank_equal=round(float(np.mean(np.abs(orr[nm] - R[nm]) < 1e-6)), 6) if nm in R else None)
                for nm in orr}
        # ---------------- comparisons
        pairs = [("HRCOV_ST", "HRCOV"), ("HR_ST", "HR"), ("HRCOV", "HR"), ("HR", "base"), ("HRCOV", "base"), ("HRCOV_ST", "base"),
                 ("HRCOV_ST", "HRCOV_DM"), ("HRCOV_DM", "HRCOV"), ("HRCOV_STFEAT", "HRCOV"), ("HRCOV_ST", "HRCOV_STFEAT"),
                 ("STONLY", "HRCOV"), ("STONLY", "HR"), ("HRCOV_PAT", "HR"), ("HRCOV", "HRCOV_PAT"), ("HR", "RecB-lite"),
                 ("HRCOV", "RecB-lite"), ("HRCOV_ST", "RecB-lite"), ("base", "RecB-lite"), ("base", "base_dump")]
        cmp_ = {f"{b_} - {a_}": boot(rr[a_], rr[b_]) for b_, a_ in pairs if b_ in rr and a_ in rr}
        B["comparisons"] = cmp_
        # ---------------- strata
        gp = q["gold_pos_A"].astype(np.int64); gpb = q["gold_pos_B"].astype(np.int64)
        in_stA = (gp >= 150) & (gp < 200); in_dmA = gp >= 200; in_cov = (gp >= 0) & (gp < 150)
        st = {"cold (gold train freq<50)": q["stratum"] == 0, "mid": q["stratum"] == 1, "head (>=500)": q["stratum"] == 2,
              "no pair history (no (s,o) event before t)": ~q["dyad"], "pair history": q["dyad"],
              "first-seen entity (s or o has no event before t)": q["first_seen"],
              "first-seen object (o has no event before t)": q["nev_o"] == 0, "first-seen subject": q["nev_s"] == 0,
              "pool-absent (gold outside base top-50)": ~q["in_top50"], "gold in base top-50": q["in_top50"],
              "exact recurrence": q["trip"], "non-recurrent": ~q["trip"],
              "no pair & static-reachable (union 2-hop, uncapped)": ~q["dyad"] & q["union2"],
              "no pair & union-unreachable": ~q["dyad"] & ~q["union2"],
              "no pair & gold in enumerated static set": ~q["dyad"] & q["st_enum"],
              "no pair & gold outside enumerated static set": ~q["dyad"] & ~q["st_enum"],
              "gold in static block (layout A cols 150-199)": in_stA, "gold in C_cov (cols 0-149)": in_cov,
              "gold outside layout A": gp < 0,
              "first-seen & gold in static block": q["first_seen"] & in_stA}
        strata = {}
        for nm, msk in st.items():
            if msk.sum() == 0:
                continue
            e = dict(n=int(msk.sum()), share=round(float(msk.mean()), 4))
            for s_ in rr:
                if s_ not in ("base_dump",):
                    e[f"MRR_{s_}"] = round(float(rr[s_][msk].mean()), 6)
            for b_, a_ in (("HRCOV_ST", "HRCOV"), ("HR_ST", "HR"), ("HRCOV_ST", "HRCOV_DM"), ("HRCOV", "HR"), ("STONLY", "HRCOV")):
                if b_ in rr and a_ in rr:
                    e[f"{b_}_minus_{a_}"] = boot(rr[a_][msk], rr[b_][msk], B=1000)
            strata[nm] = e
        B["strata"] = strata
        # ---------------- decision (pre-registered)
        if "HRCOV_ST" in rr and "HRCOV" in rr:
            hm = q["stratum"] == 2; npm = ~q["dyad"]
            g = boot(rr["HRCOV"][hm], rr["HRCOV_ST"][hm])
            damage = (g["CI95"][1] < 0) and (abs(g["delta"]) >= mat)
            npb = boot(rr["HRCOV"][npm], rr["HRCOV_ST"][npm], B=2000)
            pc = cmp_["HRCOV_ST - HRCOV"]
            a_ok = (pc["p_one_sided"] < 0.05) and (pc["delta"] >= mat)
            c_ok = (npb["delta"] > 0) and (npb["p_one_sided"] < 0.05)
            decisions[base] = dict(full=dict(dMRR=pc["delta"], CI95=pc["CI95"], p_one_sided=pc["p_one_sided"], pass_a=bool(a_ok)),
                                   head_guardrail=dict(**g, DAMAGE=bool(damage), pass_b=bool(not damage)),
                                   no_pair=dict(**npb, pass_c=bool(c_ok)), PASS=bool(a_ok and (not damage) and c_ok))
            if "HR_ST" in rr and "HR" in rr:
                decisions[base]["secondary_HR_ST_vs_HR"] = cmp_["HR_ST - HR"]
            fsm = q["first_seen"]
            decisions[base]["secondary_first_seen"] = boot(rr["HRCOV"][fsm], rr["HRCOV_ST"][fsm], B=2000)
        # ---------------- reach (base-independent flags + base-dependent pool-absent)
        reach = {}
        masks = {"all": np.ones(n, bool), "no_pair": ~q["dyad"], "first_seen": q["first_seen"], "cold": q["stratum"] == 0,
                 "pool_absent": ~q["in_top50"], "no_pair_pool_absent": ~q["dyad"] & ~q["in_top50"]}
        for nm, msk in masks.items():
            e = dict(n=int(msk.sum()), share=round(float(msk.mean()), 4))
            e["temporal_1hop_pair_history"] = round(float(q["dyad"][msk].mean()), 4)
            e["temporal_2hop_common_nbr"] = round(float(q["two_hop"][msk].mean()), 4)
            e["temporal_union_1_2hop"] = round(float((q["dyad"] | q["two_hop"])[msk].mean()), 4)
            e["static_1hop"] = round(float(q["st_S"][msk].mean()), 4)
            e["static_2hop_uncapped"] = round(float(q["st_SS"][msk].mean()), 4)
            e["static_2hop_capped_deg_le_1000"] = round(float(q["st_SSc"][msk].mean()), 4)
            e["static_union_1_2hop_uncapped"] = round(float((q["st_S"] | q["st_SS"])[msk].mean()), 4)
            e["mixed_static_temporal_2hop"] = round(float(q["st_MX"][msk].mean()), 4)
            e["union_1_2hop_uncapped"] = round(float(q["union2"][msk].mean()), 4)
            e["enumerated_static_set (system, capped)"] = round(float(q["st_enum"][msk].mean()), 4)
            e["enumerated_static_set_median_size"] = float(np.median(q["st_enum_size"][msk]))
            e["enumerated_static_set_mean_size_frac_of_N"] = round(float((q["st_enum_size"][msk] / N_ENT).mean()), 5)
            e["enum_ctrl_uniform_random_same_size"] = round(float((q["st_enum_size"][msk] / N_ENT).mean()), 5)
            e["enum_ctrl_degree_matched_same_size"] = round(float(q["st_enum_dm"][msk].mean()), 5)
            e["enum_ctrl_popularity_same_size"] = round(float(q["st_enum_pop"][msk].mean()), 4)
            e["enum_by_shape"] = {sh: round(float(q[f"st_enum_{sh}"][msk].mean()), 4) for sh in SHAPES}
            e["static_list_at50"] = round(float(((q["gpos_st"] >= 0) & (q["gpos_st"] < 50))[msk].mean()), 4)
            e["static_list_at200"] = round(float((q["gpos_st"] >= 0)[msk].mean()), 4)
            e["static_block_A (admitted, outside C_cov)"] = round(float(in_stA[msk].mean()), 4)
            e["dm_block_A (degree-matched random)"] = round(float(in_dmA[msk].mean()), 4)
            e["C_cov (pattern + bridge + top-50)"] = round(float(in_cov[msk].mean()), 4)
            e["layout_B static block"] = round(float((gpb >= 50)[msk].mean()), 4)
            reach[nm] = e
        B["reach"] = reach
        lc = jl(f"{bd}/listctl_{base}.json")
        if lc:
            B["list_density_controls"] = {k: v for k, v in lc.items() if k not in ("base", "host")}
        # ---------------- oracle ceilings
        def at1(mask):
            return round(float(np.mean(np.where(mask, 1.0, 1.0 / rb))), 6)
        intop = (gp >= 0) & (gp < KC)
        ceil = dict(base_MRR=round(float((1 / rb).mean()), 6), top50_at1=at1(intop), Ccov_at1=at1(in_cov), Ccov_plus_static_at1=at1((gp >= 0) & (gp < 200)),
                    layoutA_all_at1=at1(gp >= 0), layoutB_at1=at1(gpb >= 0), static_block_only_at1=at1(in_stA),
                    gold_in_top50=round(float(intop.mean()), 4), gold_in_Ccov=round(float(in_cov.mean()), 4),
                    gold_in_static_block=round(float(in_stA.mean()), 4), gold_in_dm_block=round(float(in_dmA.mean()), 4),
                    gold_in_layoutB=round(float((gpb >= 0).mean()), 4))
        for arm in TRAINED:
            if arm in rr:
                ceil[f"{arm}_realised"] = round(float(rr[arm].mean()), 6)
        B["oracle_ceilings"] = ceil
        # ---------------- post-hoc attribution: where does HRCOV_ST - HRCOV come from
        if "HRCOV_ST" in rr and "HRCOV" in rr:
            d = rr["HRCOV_ST"] - rr["HRCOV"]
            att = dict(total=round(float(d.mean()), 6),
                       contribution_gold_in_static_block=round(float((d * in_stA).sum() / n), 6),
                       contribution_gold_in_C_cov=round(float((d * in_cov).sum() / n), 6),
                       contribution_gold_elsewhere=round(float((d * (gp < 0)).sum() / n), 6),
                       contribution_no_pair=round(float((d * ~q["dyad"]).sum() / n), 6),
                       contribution_first_seen=round(float((d * q["first_seen"]).sum() / n), 6))
            code = q["g_st_code"].astype(np.int64) if "g_st_code" in q else np.full(n, -1)
            if names:
                NP = len(names["prel_names"]); D1 = 2 * NP
                shp = np.where(code >= 0, code // (D1 * D1), -1); aa = (code // D1) % D1; bb = code % D1
                def pdn(x):
                    return names["prel_names"][x] if x < NP else names["prel_names"][x - NP] + "^-1"
                by_shape = {}
                for k_, sh in enumerate(SHAPES):
                    m = in_stA & (shp == k_)
                    by_shape[sh] = dict(n=int(m.sum()), contribution=round(float((d * m).sum() / n), 6), mean_delta=round(float(d[m].mean()), 6) if m.any() else None,
                                        MRR_HRCOV=round(float(rr["HRCOV"][m].mean()), 6) if m.any() else None,
                                        MRR_HRCOV_ST=round(float(rr["HRCOV_ST"][m].mean()), 6) if m.any() else None)
                att["gold_in_static_block_by_best_path_shape"] = by_shape
                labs = np.array([("-" if c_ < 0 else (f"S:{pdn(a_)}" if s_ == 0 else f"SS:{pdn(a_)}/{pdn(b_)}" if s_ == 1 else
                                  f"ST:{pdn(a_)}/{rel_t[b_ % len(rel_t)]}{'^-1' if b_ >= len(rel_t) else ''}" if s_ == 2 else
                                  f"TS:{rel_t[a_ % len(rel_t)]}{'^-1' if a_ >= len(rel_t) else ''}/{pdn(b_)}"))
                                 for c_, s_, a_, b_ in zip(code.tolist(), shp.tolist(), aa.tolist(), bb.tolist())], dtype=object)
                m = in_stA
                ul, inv = np.unique(labs[m], return_inverse=True)
                contrib = np.bincount(inv, weights=d[m]) / n; cnt = np.bincount(inv)
                o_ = np.argsort(-np.abs(contrib))[:25]
                att["top_best_path_types_gold_in_static_block"] = [dict(type=str(ul[i]), n=int(cnt[i]), contribution=round(float(contrib[i]), 6),
                                                                       mean_delta=round(float(contrib[i] * n / cnt[i]), 5)) for i in o_]
                # static relation (first static hop) aggregated
                rel1 = np.array([("-" if c_ < 0 else (pdn(a_) if s_ in (0, 1, 2) else pdn(b_))).replace("^-1", "") for c_, s_, a_, b_ in
                                 zip(code.tolist(), shp.tolist(), aa.tolist(), bb.tolist())], dtype=object)
                ul, inv = np.unique(rel1[m], return_inverse=True)
                contrib = np.bincount(inv, weights=d[m]) / n; cnt = np.bincount(inv)
                o_ = np.argsort(-np.abs(contrib))[:20]
                att["top_static_relations_gold_in_static_block"] = [dict(relation=str(ul[i]), n=int(cnt[i]), contribution=round(float(contrib[i]), 6)) for i in o_]
            # post-hoc: queries whose (s, o) pair is ALREADY a static edge (any property / direction) vs not
            sS = q["st_S"].astype(bool)
            att["direct_static_edge_split"] = dict(share_queries_with_static_s_o_edge=round(float(sS.mean()), 5),
                                                   contribution_with_edge=round(float((d * sS).sum() / n), 6),
                                                   contribution_without_edge=round(float((d * ~sS).sum() / n), 6),
                                                   delta_masked_to_queries_without_edge=boot(np.zeros(n), d * ~sS),
                                                   mean_delta_with_edge=round(float(d[sS].mean()), 5),
                                                   note="static KG is a time-less 2024 snapshot; an s-o static edge can encode the test pair under another property")
            if os.path.exists(f"{a.root}/static_edges.npz") and names:
                z = np.load(f"{a.root}/static_edges.npz"); NT = int(z["n_nodes"])
                kf = z["h"].astype(np.int64) * NT + z["t"].astype(np.int64); o_ = np.argsort(kf); kf = kf[o_]; pf = z["p"].astype(np.int64)[o_]
                same = np.zeros(n, bool); sro = q["sro"]
                for i in np.flatnonzero(sS):
                    s_, r_, ob, _ = (int(x) for x in sro[i]); rel = rel_t[r_ % len(rel_t)]; props = set()
                    for k_ in (s_ * NT + ob, ob * NT + s_):
                        lo, hi = np.searchsorted(kf, k_, "left"), np.searchsorted(kf, k_, "right")
                        props |= {names["prel_names"][x] for x in pf[lo:hi]}
                    same[i] = rel in props
                att["direct_static_edge_split"]["share_queries_same_property_as_query_relation"] = round(float(same.mean()), 5)
                att["direct_static_edge_split"]["contribution_same_property"] = round(float((d * same).sum() / n), 6)
            if "HRCOV_DM" in rr:
                dd_ = rr["HRCOV_DM"] - rr["HRCOV"]
                att["density_HRCOV_DM_minus_HRCOV"] = dict(total=round(float(dd_.mean()), 6),
                                                          contribution_gold_in_dm_block=round(float((dd_ * in_dmA).sum() / n), 6),
                                                          contribution_gold_in_C_cov=round(float((dd_ * in_cov).sum() / n), 6))
            B["attribution_post_hoc"] = att
        # ---------------- training records
        B["train_selection"] = {arm: {str(s): dict(best_epoch=v["best_epoch"], select_MRR=v["select"]["MRR"], test_MRR=v["test"]["MRR"],
                                                    cal_alpha=v.get("cal_alpha"), cal_beta=v.get("cal_beta"),
                                                    promoted_extra_per_query=v.get("test_promoted_extra_per_query"),
                                                    promoted_block4_per_query=v.get("test_promoted_block4_per_query"))
                                     for s, v in rec.get("seeds", {}).items()} for arm, rec in tr["arms"].items()}
        for fn in (f"lists_summary_{base}.json", f"feat_summary_{base}.json"):
            if os.path.exists(f"{bd}/{fn}"):
                B[fn.replace(".json", "")] = json.load(open(f"{bd}/{fn}"))
        V["bases"][base] = B
    V["fulltest_rows"] = fulltest_rows
    npass = sum(d["PASS"] for d in decisions.values())
    V["decision"] = dict(per_base=decisions, n_bases_evaluated=len(decisions), n_pass=int(npass),
                         useful_on_smallpedia=bool(npass >= 2), rule=rule["primary_rule"]["statement"],
                         comparator_note=rule["primary_rule"]["comparator_note"])
    json.dump(V, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({b: dict(PASS=d["PASS"], dMRR=d["full"]["dMRR"], p=d["full"]["p_one_sided"], nopair=d["no_pair"]["delta"],
                              nopair_p=d["no_pair"]["p_one_sided"], head=d["head_guardrail"]["delta"]) for b, d in decisions.items()}, indent=1))
    print("useful_on_smallpedia", V["decision"]["useful_on_smallpedia"])


if __name__ == "__main__":
    main()
