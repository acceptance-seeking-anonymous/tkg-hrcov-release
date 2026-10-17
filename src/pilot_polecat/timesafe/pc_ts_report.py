#!/usr/bin/env python3
"""pilot_polecat / pc_ts_report.py -- assemble bridge_timesafe_verdict.json from ts_agg_<base>.json (both bases) and
render the numeric tables of bridge_timesafe_report.md (every number copied from the JSON).
  python3 pc_ts_report.py <ts_agg_regcn.json> <ts_agg_logcl.json> <report_template.md> <out_verdict.json> <out_report.md>
"""
import sys, json

PUBLISHED_BEST = 0.256      # CountTRuCoLa, TGB 2.0 tkgl-polecat test MRR
OLD_HEADLINE = {"regcn": dict(dMRR=0.037451, CI95=[0.036891, 0.037997], HRCOV=0.271797),
                "logcl": dict(dMRR=0.035956, CI95=[0.03549, 0.036475], HRCOV=0.272057)}
MATERIAL = 0.001            # pilot materiality threshold (pre-registered rule: dMRR >= 0.001)


def f4(x):
    return f"{x:.4f}"


def ci(d):
    return f"{d['delta']:+.4f} [{d['CI95'][0]:+.4f}, {d['CI95'][1]:+.4f}]"


def main():
    A = {b: json.load(open(p)) for b, p in zip(("regcn", "logcl"), sys.argv[1:3])}
    tpl = open(sys.argv[3]).read()
    V = dict(title="Time-respecting bridge cohort index: polecat re-measurement (pilot 1 follow-up)",
             question="Does training the stage-2 bridge ranker on snapshot-before-t cohort features (instead of the static all-train "
                      "index that contains each training query's own gold fact and later facts) change HRCOV, HRCOV-with-bridge, "
                      "or the polecat headline (HRCOV-HR +0.0375/+0.0360; HRCOV 0.272 vs published 0.256)?",
             materiality=MATERIAL, published_best=dict(system="CountTRuCoLa", MRR=PUBLISHED_BEST), bases={})
    rows_full, rows_cmp, rows_coh, rows_list, rows_off, rows_str = [], [], [], [], [], []
    changed = {}
    for b, a in A.items():
        R = {r["system"]: r for r in a["rows"]}
        C = a["comparisons"]
        co, cn = a["lists_summary_old"]["bridge_cohort"], a["lists_summary_new"]["bridge_cohort"]
        ts = cn.get("timesafe") or {}
        prim = C["HRCOV_new - HR_old"]; dnew_old = C["HRCOV_new - HRCOV_old"]
        hc_new = R["HRCOV_new"]["MRR_mean3"]
        headline_changes = bool((abs(dnew_old["delta"]) >= MATERIAL) or (not a["primary_rule_PASS"]) or (hc_new < PUBLISHED_BEST))
        changed[b] = headline_changes
        V["bases"][b] = dict(
            cohort=dict(old=dict(index="static all-train (transductive)", reach_rate=co["reach_rate"], median_cset=co["median_cset"],
                                 median_cset_frac_of_N=co["median_cset"] / 150931, n_sampled=co["n_sampled"], n_train=co["n_train"]),
                        new=dict(index=cn.get("index"), reach_rate=cn["reach_rate"], median_cset=cn["median_cset"],
                                 median_cset_frac_of_N=cn.get("median_cset_frac_of_N"), n_sampled=cn["n_sampled"], n_train=cn["n_train"],
                                 n_snapshots=ts.get("n_snapshots"), mode=ts.get("mode"), staleness_t_minus_cut=ts.get("staleness_t_minus_cut"),
                                 build_s_total=ts.get("build_s_total"), seconds=cn.get("seconds"))),
            input_checks=a["input_checks"], bridge_list_change=a["bridge_list_change"], rows=a["rows"], comparisons=C,
            head_guardrail_HRCOV_new_vs_HR=a["head_guardrail_HRCOV_new_vs_HR"], primary_rule_PASS_new=a["primary_rule_PASS"],
            old_headline=OLD_HEADLINE[b], new_headline=dict(dMRR=prim["delta"], CI95=prim["CI95"], HRCOV=hc_new),
            HRCOV_new_minus_old=dnew_old, headline_changes=headline_changes,
            strata=a["strata"], official_systems=a.get("official_systems"), lists_elapsed_s=a["lists_summary_new"].get("elapsed_s"),
            host=a["host"])
        for sysn in ("HR_old", "HRCOV_old", "HRCOV_new", "HRCOV_BR_old", "HRCOV_BR_new", "HRCOVFEAT_old", "HRCOVFEAT_new", "HRCOV_PAT_old", "HRCOV_PAT_new"):
            r = R[sysn]
            rows_full.append(f"| {b} | {sysn} | {f4(r['MRR_mean3'])} | {r['MRR_sd3']:.4f} | {f4(r['H1'])} | {f4(r['H3'])} | {f4(r['H10'])} |")
        for k in ("HRCOV_new - HR_old", "HRCOV_old - HR_old", "HRCOV_new - HRCOV_old", "HRCOV_BR_new - HR_old", "HRCOV_BR_old - HR_old",
                  "HRCOV_BR_new - HRCOV_BR_old", "HRCOV_BR_new - HRCOVFEAT_new", "HRCOV_BR_old - HRCOVFEAT_old",
                  "HRCOV_new - HRCOV_PAT_new", "HRCOV_old - HRCOV_PAT_old", "HRCOVFEAT_new - HRCOVFEAT_old", "HRCOV_PAT_new - HRCOV_PAT_old"):
            if k in C:
                rows_cmp.append(f"| {b} | {k} | {ci(C[k])} | {C[k]['p_one_sided']} |")
            elif k == "HRCOV_old - HR_old":
                o = OLD_HEADLINE[b]
                rows_cmp.append(f"| {b} | {k} (pilot-1 verdict) | {o['dMRR']:+.4f} [{o['CI95'][0]:+.4f}, {o['CI95'][1]:+.4f}] | 0.0 |")
            elif k == "HRCOV_BR_old - HR_old":
                pass
        rows_coh.append(f"| {b} | static all-train (old) | {co['reach_rate']:.4f} | {co['median_cset']:.0f} ({co['median_cset']/150931:.1%}) | {co['n_train']} | — |")
        st = ts.get("staleness_t_minus_cut") or {}
        rows_coh.append(f"| {b} | snapshot-before-t, exact (new) | {cn['reach_rate']:.4f} | {cn['median_cset']:.0f} ({cn['median_cset']/150931:.1%}) | {cn['n_train']} | "
                        f"{ts.get('n_snapshots')} snapshots, staleness median {st.get('median')} |")
        L = a["bridge_list_change"]
        rows_list.append(f"| {b} | {L['Lbr_top10_overlap']:.3f} | {L['Lbr_top50_overlap']:.3f} | {L['Lbr_top200_overlap']:.3f} | {L['Ccov_bridge_block_overlap']:.3f} | "
                         f"{L['old_gold_in_Lbr_at10']:.4f} → {L['new_gold_in_Lbr_at10']:.4f} | {L['old_gold_in_Lbr_at50']:.4f} → {L['new_gold_in_Lbr_at50']:.4f} | "
                         f"{L['old_gold_in_Lbr_at50_pool_absent']:.4f} → {L['new_gold_in_Lbr_at50_pool_absent']:.4f} | "
                         f"{L['old_gold_in_bridge_block']:.4f} → {L['new_gold_in_bridge_block']:.4f} | {L['old_gold_in_Ccov']:.4f} → {L['new_gold_in_Ccov']:.4f} |")
        for k, v in (a.get("official_systems") or {}).items():
            o = v["official"]
            rows_off.append(f"| {b} | {k} | {o['MRR']:.6f} | {o.get('hits10', float('nan')):.4f} | {v['harness_MRR_same_ranks']:.6f} |")
        for nm in ("pool-absent", "cold (gold train freq<50)", "cold & pair history", "cold & no pair history", "no pair history",
                   "no pair & 2-hop unreachable", "first-seen entity", "head (>=500)", "gold in NEW bridge block"):
            e = a["strata"][nm]
            rows_str.append(f"| {b} | {nm} | {e['share']:.3f} | {e['MRR_HR_old']:.4f} | {e['MRR_HRCOV_old']:.4f} | {e['MRR_HRCOV_new']:.4f} | "
                            f"{ci(e['HRCOV_new_minus_old'])} | {ci(e['HRCOV_BR_new_minus_old'])} |")
    V["headline_changes_any_base"] = bool(any(changed.values()))
    hd = "| base | system | MRR (3-seed mean) | seed s.d. | H@1 | H@3 | H@10 |\n|---|---|---|---|---|---|---|\n"
    tpl = tpl.replace("TABLE_FULL", hd + "\n".join(rows_full))
    tpl = tpl.replace("TABLE_CMP", "| base | contrast | Δ MRR [95% CI] | p (one-sided) |\n|---|---|---|---|\n" + "\n".join(rows_cmp))
    tpl = tpl.replace("TABLE_COHORT", "| base | cohort index | gold reached by natural bridge set | median |C| (share of N) | ranker training queries | snapshots |\n|---|---|---|---|---|---|\n" + "\n".join(rows_coh))
    tpl = tpl.replace("TABLE_LISTS", "| base | Lbr top-10 overlap old/new | top-50 | top-200 | C_cov bridge block overlap | gold in Lbr@10 old → new | gold in Lbr@50 | gold in Lbr@50 (pool-absent) | gold in bridge block | gold in C_cov |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_list))
    tpl = tpl.replace("TABLE_OFFICIAL", "| base | system (new, seed 0) | official py-tgb MRR | official H@10 | harness MRR, same ranks |\n|---|---|---|---|---|\n" + "\n".join(rows_off))
    tpl = tpl.replace("TABLE_STRATA", "| base | stratum | share | HR | HRCOV old | HRCOV new | HRCOV new − old [95% CI] | HRCOV_BR new − old [95% CI] |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_str))
    json.dump(V, open(sys.argv[4], "w"), indent=1)
    open(sys.argv[5], "w").write(tpl)
    print(json.dumps({b: dict(new=V["bases"][b]["new_headline"], d_new_old=V["bases"][b]["HRCOV_new_minus_old"]["delta"],
                              changes=V["bases"][b]["headline_changes"]) for b in V["bases"]}))


if __name__ == "__main__":
    main()
