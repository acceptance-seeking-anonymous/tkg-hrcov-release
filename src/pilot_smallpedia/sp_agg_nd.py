#!/usr/bin/env python3
"""pilot_smallpedia / sp_agg_nd.py -- direct-static-edge leakage control (addendum): statistics, pre-registered decision,
leakage audit merge.  Adds verdict["addendum_nodirect"] to the pilot-2 verdict and appends full-test rows (local, CPU).
  python3 sp_agg_nd.py --root aggroot --verdict smallpedia_pilot_verdict.json --addendum smallpedia_decision_rule_addendum_nodirect.json
                       --audit leakage_audit.json --out smallpedia_pilot_verdict.json
Bundles: bundle_<base> (pilot 2: HRCOV, HRCOV_ST ranks), bundle_<base>_nd (HRCOV_ST_NODIRECT), bundle_<base>_nds (STRICT).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_agg import boot, mets, jl, BASES, SEEDS

MODES = {"nd": "HRCOV_ST_NODIRECT", "nds": "HRCOV_ST_NODIRECT_STRICT"}
HOST = "serverA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--verdict", required=True); ap.add_argument("--addendum", required=True)
    ap.add_argument("--audit", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = json.load(open(a.verdict)); AD = json.load(open(a.addendum)); AU = json.load(open(a.audit))
    mat = 0.001
    out = dict(addendum_file=os.path.basename(a.addendum), registered=AD["registered"], arms=AD["arms"], primary_rule=AD["primary_rule"],
               per_base={}, leakage_audit=AU)
    new_rows = []
    for b in BASES:
        q = dict(np.load(f"{a.root}/bundle_{b}/test_q.npz")); R0 = dict(np.load(f"{a.root}/bundle_{b}/ranks_test.npz")); n = len(q["r_base"])
        rr = {}
        for arm in ("HRCOV", "HRCOV_ST"):
            rr[arm] = np.mean([1.0 / R0[f"{arm}_s{s}"].astype(np.float64) for s in SEEDS], 0)
        E = dict(train={}, lists={}, official={})
        for md, nm in MODES.items():
            bd = f"{a.root}/bundle_{b}_{md}"
            if not os.path.exists(f"{bd}/ranks_test.npz"):
                continue
            R = dict(np.load(f"{bd}/ranks_test.npz")); qn = dict(np.load(f"{bd}/test_q.npz"))
            assert (qn["sro"] == q["sro"]).all() and (R["base"] == R0["base"]).all()
            ks = [f"HRCOV_ST_s{s}" for s in SEEDS if f"HRCOV_ST_s{s}" in R]
            assert len(ks) == 3, (b, md, ks)
            rr[nm] = np.mean([1.0 / R[k].astype(np.float64) for k in ks], 0)
            tr = json.load(open(f"{bd}/train_{b}.json")); arm = tr["arms"]["HRCOV_ST"]
            for k in ks:
                new_rows.append(dict(system=nm, base=b, seed=int(k.split("_s")[-1]), host=tr["host"], **mets(R[k])))
            m = dict(MRR=round(float(rr[nm].mean()), 6))
            for kk in ("H1", "H3", "H10"):
                m[kk] = round(float(np.mean([mets(R[k])[kk] for k in ks])), 6)
            new_rows.append(dict(system=nm, base=b, seed="mean(3)", MRR_sd_over_seeds=round(float(np.std([mets(R[k])["MRR"] for k in ks])), 6), host=tr["host"], n=n, **m))
            E["train"][nm] = dict(static_mode=tr.get("static_mode"), best_epochs={s: v["best_epoch"] for s, v in arm["seeds"].items()},
                                  select_MRR={s: v["select"]["MRR"] for s, v in arm["seeds"].items()}, test_MRR={s: v["test"]["MRR"] for s, v in arm["seeds"].items()},
                                  promoted_static_block_per_query={s: v.get("test_promoted_block4_per_query") for s, v in arm["seeds"].items()},
                                  identity_test=tr.get("identity_test", {}).get("HRCOV_ST"))
            ls = jl(f"{bd}/lists_summary_{b}_{md}.json")
            if ls:
                E["lists"][nm] = dict(static_mode=ls.get("static_mode"), C_cov_identical_to_pilot2=dict(valid=ls["valid"].get("ccov_identical_to_pilot2"), test=ls["test"].get("ccov_identical_to_pilot2")),
                                      test_gold_in_static=ls["test"]["gold_in_static"], test_gold_in_Lst200=ls["test"]["gold_in_Lst200"],
                                      test_median_st_size=ls["test"]["median_st_size"], test_mean_static_block=ls["test"]["mean_blocks"]["n_st_block"],
                                      bridge_ranker=ls.get("bridge_ranker", {}).get("loaded_from"))
            gpn = qn["gold_pos_A"].astype(np.int64)
            E["lists"].setdefault(nm, {})["test_gold_in_static_block_cols150_199"] = round(float(((gpn >= 150) & (gpn < 200)).mean()), 6)
            osy = jl(f"{bd}/official_systems_{md}_{b}.json")
            if osy:
                orr = dict(np.load(f"{bd}/official_systems_{md}_{b}_ranks.npz"))
                for snm, v in osy["official"].items():
                    new_rows.append(dict(system=f"{snm} [official py-tgb Evaluator]", base=b, seed=0, host=osy["host"], MRR=round(v["MRR"], 6), H10=round(v["hits10"], 6), n=v["n"]))
                    E["official"][snm] = dict(official_MRR=round(v["MRR"], 6), harness_MRR_seed0=round(float((1 / R["HRCOV_ST_s0"].astype(np.float64)).mean()), 6),
                                              frac_rank_equal=round(float(np.mean(np.abs(orr[snm] - R["HRCOV_ST_s0"]) < 1e-6)), 6), py_tgb=osy.get("py_tgb"))
        g0 = q["gold_pos_A"].astype(np.int64)
        direct = q["st_S"].astype(bool)
        strata = {"all": np.ones(n, bool), "head (>=500)": q["stratum"] == 2, "mid": q["stratum"] == 1, "cold (<50)": q["stratum"] == 0,
                  "no pair history": ~q["dyad"], "pair history": q["dyad"],
                  "first-seen object (o has no event before t)": q["nev_o"] == 0, "first-seen entity (s or o)": q["first_seen"],
                  "direct static s-o edge": direct, "no direct static s-o edge": ~direct,
                  "no pair & no direct edge": ~q["dyad"] & ~direct, "first-seen object & no direct edge": (q["nev_o"] == 0) & ~direct,
                  "gold in pilot-2 static block": (g0 >= 150) & (g0 < 200)}
        pairs = [(nm, "HRCOV") for nm in MODES.values()] + [(nm, "HRCOV_ST") for nm in MODES.values()] + [("HRCOV_ST", "HRCOV"),
                 (MODES["nds"], MODES["nd"])]
        E["comparisons_fulltest"] = {f"{x} - {y}": boot(rr[y], rr[x], B=2000) for x, y in pairs if x in rr and y in rr}
        st_out = {}
        for snm, mk in strata.items():
            e = dict(n=int(mk.sum()), share=round(float(mk.mean()), 4))
            for s_ in rr:
                e[f"MRR_{s_}"] = round(float(rr[s_][mk].mean()), 6)
            for x, y in pairs:
                if x in rr and y in rr and snm != "all":
                    e[f"{x} - {y}"] = boot(rr[y][mk], rr[x][mk], B=2000 if snm.startswith(("no pair", "first-seen", "direct", "head")) else 1000)
            st_out[snm] = e
        E["strata"] = st_out
        # direct-edge share of the remaining gain
        for nm in MODES.values():
            if nm in rr:
                d = rr[nm] - rr["HRCOV"]
                E.setdefault("contribution", {})[nm] = dict(total=round(float(d.mean()), 6), direct_edge_queries=round(float(d[direct].sum() / n), 6),
                                                            no_direct_edge_queries=round(float(d[~direct].sum() / n), 6))
        d0 = rr["HRCOV_ST"] - rr["HRCOV"]
        E.setdefault("contribution", {})["HRCOV_ST"] = dict(total=round(float(d0.mean()), 6), direct_edge_queries=round(float(d0[direct].sum() / n), 6),
                                                            no_direct_edge_queries=round(float(d0[~direct].sum() / n), 6))
        # pre-registered decision
        dec = {}
        hm = q["stratum"] == 2; npm = ~q["dyad"]
        for nm in MODES.values():
            if nm not in rr:
                continue
            pc = E["comparisons_fulltest"][f"{nm} - HRCOV"]
            g = boot(rr["HRCOV"][hm], rr[nm][hm], B=2000); npb = boot(rr["HRCOV"][npm], rr[nm][npm], B=2000)
            a_ok = (pc["p_one_sided"] < 0.05) and (pc["delta"] >= mat)
            damage = (g["CI95"][1] < 0) and (abs(g["delta"]) >= mat)
            c_ok = (npb["delta"] > 0) and (npb["p_one_sided"] < 0.05)
            fso = boot(rr["HRCOV"][q["nev_o"] == 0], rr[nm][q["nev_o"] == 0], B=2000)
            fse = boot(rr["HRCOV"][q["first_seen"]], rr[nm][q["first_seen"]], B=2000)
            dec[nm] = dict(full=dict(dMRR=pc["delta"], CI95=pc["CI95"], p_one_sided=pc["p_one_sided"], pass_a=bool(a_ok)),
                           head_guardrail=dict(**g, DAMAGE=bool(damage), pass_b=bool(not damage)), no_pair=dict(**npb, pass_c=bool(c_ok)),
                           PASS=bool(a_ok and (not damage) and c_ok), reported_first_seen_object=fso, reported_first_seen_entity=fse,
                           role="primary" if nm == MODES["nd"] else "sensitivity")
        E["decision"] = dec
        out["per_base"][b] = E
    summ = {}
    for nm in MODES.values():
        npass = sum(1 for b in BASES if nm in out["per_base"][b]["decision"] and out["per_base"][b]["decision"][nm]["PASS"])
        nev = sum(1 for b in BASES if nm in out["per_base"][b]["decision"])
        summ[nm] = dict(n_bases_evaluated=nev, n_pass=npass, survives=bool(npass >= 2))
    out["summary"] = summ
    out["PASS_survives_without_direct_static_edges"] = summ[MODES["nd"]]["survives"]
    sv, ss = summ[MODES["nd"]]["survives"], summ[MODES["nds"]]["survives"]
    out["pre_stated_interpretation_applied"] = (AD["pre_stated_interpretation"]["survives_and_strict_survives"] if sv and ss else
                                                AD["pre_stated_interpretation"]["survives_strict_fails"] if sv else AD["pre_stated_interpretation"]["fails"])
    out["framing"] = AD["pre_stated_interpretation"]["framing"]
    out["base_strength_note"] = AD["notes"][0]
    V["addendum_nodirect"] = out
    V["fulltest_rows"] = [r for r in V["fulltest_rows"] if not str(r.get("system", "")).startswith("HRCOV_ST_NODIRECT")] + new_rows
    json.dump(V, open(a.out, "w"), indent=1)
    print(json.dumps({nm: {b: (out["per_base"][b]["decision"][nm]["full"]["dMRR"], out["per_base"][b]["decision"][nm]["full"]["CI95"],
                               out["per_base"][b]["decision"][nm]["no_pair"]["delta"], out["per_base"][b]["decision"][nm]["head_guardrail"]["delta"],
                               out["per_base"][b]["decision"][nm]["PASS"]) for b in BASES if nm in out["per_base"][b]["decision"]} for nm in MODES.values()}, indent=0))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
