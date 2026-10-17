"""v7 / v7_build_report.py -- consolidated tois_v7_experiments.md + tois_v7_experiments.csv from the per-item JSONs in v7/out.
python3 v7_build_report.py builds the consolidated v7 report from the result files (plain file names)"""
import json, os
import v7_report as VR

OUT = "v7/out"
J = lambda f: json.load(open(f"{OUT}/{f}"))
DSN = {"tkgl-polecat": "polecat", "tkgl-smallpedia": "smallpedia", "ICEWS18": "ICEWS18", "GDELT": "GDELT"}


def link(art, fn):
    v = art.get(fn)
    return fn


def a_sections(art):
    L = []
    A1 = J("a1_membership_decomposition.json"); A2 = J("a2_exact_oracle.json"); A3 = J("a3_hits.json"); A4 = J("a4_training_selection.json")
    A6 = J("a6_temporal_sensitivity.json"); A7 = J("a7_selective_gain_loss.json")
    L.append(f"### A1 Membership and loss decomposition ({link(art, 'a1_membership_decomposition.json')})\n")
    L.append("| dataset | base MRR | disagreements stored list vs rank ≤ 50 | in-pool share v1 (stored) / v2 (rank ≤ 50) | out-of-pool share of loss v1 / v2 |\n|---|---|---|---|---|")
    for ds, D in A1["datasets"].items():
        m = D["membership"]; v1 = D["decomposition"]["v1_stored_list"]; v2 = D["decomposition"]["v2_filtered_rank_le_50"]
        L.append(f"| {DSN[ds]} | {D['MRR_base']:.5f} | {m['disagree_total']:,} ({100 * m['disagree_frac']:.3f}%; in list & rank > 50: {m['in_list_not_rank_le_50']:,}) | "
                 f"{v1['in']['share']:.4f} / {v2['in']['share']:.4f} | {v1['out']['loss_share']:.3f} / {v2['out']['loss_share']:.3f} |")
    L.append("\nInterpretation: the stored unfiltered top-50 never contains a gold with filtered rank > 50; the two pool definitions differ by ≤ 0.81% of queries, and the out-of-pool loss share is 0.39-0.63 under either definition.\n")
    L.append(f"### A2 Exact oracle ({link(art, 'a2_exact_oracle.json')})\n")
    L.append("| dataset | T50 recall / oracle | T100 | U50 | POOL_sel | HRCOV_PAT pool (mean size) | RB pool (mean size) | realised HRCOV_PAT / PT oracle |\n|---|---|---|---|---|---|---|---|")
    for ds, D in A2["datasets"].items():
        p = D["pools"]; f = lambda k: f"{p[k]['recall']:.4f} / {p[k]['oracle_MRR_exact']:.4f}"
        L.append(f"| {DSN[ds]} | {f('T50')} | {f('T100')} | {f('U50')} | {f('POOLSEL')} | {f('PT')} ({p['PT']['mean_size']:.1f}) | {f('RB')} ({p['RB']['mean_size']:.1f}) | "
                 f"{D['HRCOV_PAT_realised_MRR']:.5f} ({100 * D['HRCOV_PAT_share_of_PT_oracle']:.0f}%) |")
    tails = [p["tail_contribution"] for D in A2["datasets"].values() for p in D["pools"].values() if "tail_contribution" in p]
    shares = [D["HRCOV_PAT_share_of_PT_oracle"] for D in A2["datasets"].values()]
    L.append(f"\nInterpretation: the exact oracle adds only {min(tails):.4f}-{max(tails):.4f} over recall; HRCOV_PAT realises {100 * min(shares):.0f}-{100 * max(shares):.0f}% of its own pool's oracle.\n")
    L.append(f"### A3 Hits@k ({link(art, 'a3_hits.json')})\n")
    L.append("| run | system | MRR | H@1 | H@3 | H@10 |\n|---|---|---|---|---|---|")
    for run, R in A3["runs"].items():
        for s in ("base", "BLEND", "POOL", "HR", "HRCOV_PAT"):
            if s in R: L.append(f"| {run} | {s} | {R[s]['MRR']:.5f} | {R[s]['H1']:.4f} | {R[s]['H3']:.4f} | {R[s]['H10']:.4f} |")
    L.append("\n| run | comparison | ΔMRR [ts CI] | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] |\n|---|---|---|---|---|---|")
    for run, Dd in A3["deltas"].items():
        for c, r in Dd.items():
            if not c.endswith(("BLEND", "POOL")): continue
            L.append(f"| {run} | {c.replace('_minus_', ' − ')} | " + " | ".join(f"{r[m]['delta']:+.4f} {VR.ci4(r[m]['ts']['CI95'])}" for m in ("MRR", "H1", "H3", "H10")) + " |")
    L.append("\nInterpretation: HRCOV_PAT's gain over the controls is present at H@1, H@3 and H@10 in every run; on smallpedia it is concentrated at H@1.\n")
    L.append(f"### A4 Training selection ({link(art, 'a4_training_selection.json')})\n")
    L.append("| dataset | HR retained / fit population | HRCOV_PAT retained / fit population |\n|---|---|---|")
    for ds, D in A4["datasets"].items():
        h, p = D["HR"], D["HRCOV_PAT"]
        L.append(f"| {DSN[ds]} | {h['n_retained']:,} / {h['n_fit_population']:,} ({h['frac_retained']:.3f}) | {p['n_retained']:,} / {p['n_fit_population']:,} ({p['frac_retained']:.3f}) |")
    L.append("\nInterpretation: gold-in-pool retention drops most for no-pair and first-seen queries (polecat HRCOV_PAT: recurrent 0.936 vs first-seen 0.044); breakdowns by stratum and frequency are in the JSON.\n")
    L.append("### A5 Multi-positive masking\n\nCode reading: other true same-timestamp answers are masked in every trainer (fit loss, calibration and filtered evaluation). The pre-specified sensitivity variants are reported under B/C below.\n")
    L.append(f"### A6 Temporal sensitivity ({link(art, 'a6_temporal_sensitivity.json')})\n")
    L.append("| dataset | comparison | Δ | leave-one-period-out range | moving-block 95% CI | timestamps with Δ < 0 |\n|---|---|---|---|---|---|")
    for ds, D in A6["datasets"].items():
        for c, r in D.items():
            mb = r["moving_block"]
            L.append(f"| {DSN[ds]} | {c.replace('_minus_', ' − ')} | {r['delta']:+.5f} | [{r['lopo_min']:+.5f}, {r['lopo_max']:+.5f}] | {VR.ci(mb['CI95'])} (l = {mb['block_len']}) | {r['n_ts_negative']} / {r['n_ts']} |")
    L.append("\nInterpretation: no single period drives the margin; leaving out any period moves the polecat margin by < 0.001.\n")
    L.append(f"### A7 Selective gain and loss by base confidence ({link(art, 'a7_selective_gain_loss.json')})\n")
    L.append("| dataset | base p1 bin | share | MRR base → HRCOV_PAT | Δ vs base [ts CI] | P(override of correct base top-1) |\n|---|---|---|---|---|---|")
    for ds, rows in A7["datasets"].items():
        for r in rows:
            d = r["PAT_minus_base"]
            L.append(f"| {DSN[ds]} | {r['bin']} | {r['share']:.3f} | {r['MRR_base']:.4f} → {r['MRR_HRCOV_PAT']:.4f} | {d['delta']:+.4f} {VR.ci4(d['ts']['CI95'])} | {r['P_override_given_base_top1']:.4f} |")
    hi = [r["P_override_given_base_top1"] for rows in A7["datasets"].values() for r in rows if r["bin"].startswith(("[0.7", "[0.9"))]
    lo = [r["P_override_given_base_top1"] for rows in A7["datasets"].values() for r in rows if r["bin"].startswith("[0,")]
    L.append(f"\nInterpretation: the gain is largest when the base is unsure (p1 < 0.1) and shrinks towards zero for confident queries; overrides of a correct base top-1 occur in {100 * min(lo):.0f}-{100 * max(lo):.0f}% of such queries at p1 < 0.1 but in at most {100 * max(hi):.1f}% at p1 ≥ 0.7.\n")
    return L


def bc_item(it_json, art, fn, title, arm_keys=None):
    L = [f"### {title} ({link(art, fn)})\n", f"**Interpretation.** {it_json['interpretation']}\n"]
    for ds, D in it_json["datasets"].items():
        L.append(f"**{DSN.get(ds, ds)}** (TiRGN; {D['n_queries']:,} queries, {D['n_timestamps']} timestamps)\n")
        L.append(VR.arm_table(D["arms"], arm_keys)); L.append("")
        L.append(VR.cmp_table(D["comparisons"])); L.append("")
    if it_json.get("expansion_check"):
        e = it_json["expansion_check"]; L.append(f"Expansion rule: second base added = {e['second_base_added']} (" + "; ".join(
            f"{p['comparison']}: ts CI contains 0.005 = {any(p['ts_CI_contains_bar'].values())}, opposite signs = {p['opposite_signs_both_excluding_0']}" for p in e["primaries"]) + ").\n")
    if it_json.get("notes"):
        L.append("Notes: " + " ".join(f"({i + 1}) {n}" for i, n in enumerate(it_json["notes"][4:] if len(it_json["notes"]) > 4 else it_json["notes"])) + "\n")
    return L


def build(art, c2=None, extra_notes=()):
    B1, B2, B3, C1, A5 = (J(f) for f in ("b1_ranker_form.json", "b2_matched_pool.json", "b3_common_pool_base_ablation.json", "c1_no_dyad_lr.json", "a5_multi_positive_sensitivity.json"))
    B4, C3, C4 = J("b4_g1_tkgl_icews.json"), J("c3_relnorm_tkgl_icews.json"), J("c4_base_seed.json")
    C2 = c2 if c2 is not None else (J("c2_pattern_design.json") if os.path.exists(f"{OUT}/c2_pattern_design.json") else None)
    rule = B1["rule"]
    L = ["# TOIS v7 experiments: consolidated results (Sections 2-4)\n",
         f"All test numbers are **pre-specified later analyses**: rule {link(art, 'tois_v7_rule.json')} (registered {rule['registered_utc']}, sha256 {rule['sha256'][:12]}...) was registered before any v7 test number was read. "
         "Default bases are TiRGN on tkgl-polecat and on tkgl-smallpedia. B4 and C3 use the tkgl-icews Stage B setting (RE-GCN and LogCL bases, full test set of 4,651,378 queries). "
         "Evaluator: official py-tgb for the TGB datasets; harness ranks for ICEWS18/GDELT (A items only). Statistics: paired Poisson(1) query bootstrap and timestamp-block bootstrap, B = 2000, seed 0, shared draws within an item. "
         "A difference is called present only if the one-sided lower bound L > 0 under both bootstraps (Holm within items with several primaries). A difference ≥ 0.005 is called excluded only if the timestamp-block one-sided bound is inside ±0.005. MDE_80 = 2.486 × SE_ts (sensitivity only). "
         "Learned arms: seeds 0-2, per-query RR averaged over seeds.\n",
         "## Summary\n", "| item | result (polecat / smallpedia unless stated) | call |", "|---|---|---|"]
    def prim(itj, ds, k=0):
        r = [x for x in itj["datasets"][ds]["comparisons"] if x["role"] == "primary"][k]; return r
    pc, sp = "tkgl-polecat", "tkgl-smallpedia"
    for nm, itj, ks in (("B1 MLP64x2 − LIN", B1, 0), ("B1 MLP64x2 − QBLEND", B1, 1), ("B2 PT_common − RB_common", B2, 0), ("B3 FULL − NB_NF", B3, 0), ("C1 FULL − noDyadLR", C1, 0)):
        a, b = prim(itj, pc, ks), prim(itj, sp, ks)
        L.append(f"| {nm} | {a['MRR']['delta']:+.5f} {VR.ci(a['MRR']['ts']['CI95'])} / {b['MRR']['delta']:+.5f} {VR.ci(b['MRR']['ts']['CI95'])} (ts CI) | {a['call']} / {b['call']} |")
    for r in A5["datasets"][pc]["comparisons"]:
        L.append(f"| A5 {r['comparison']} (polecat) | {r['MRR']['delta']:+.5f} {VR.ci(r['MRR']['ts']['CI95'])} | {r['call']} |")
    if C2:
        for k in range(2):
            a, b = prim(C2, pc, k) if pc in C2["datasets"] else None, prim(C2, sp, k)
            sa = f"{a['MRR']['delta']:+.5f} {VR.ci(a['MRR']['ts']['CI95'])}" if a else "pending"
            L.append(f"| C2 {b['comparison']} | {sa} / {b['MRR']['delta']:+.5f} {VR.ci(b['MRR']['ts']['CI95'])} | {(a or {}).get('call', 'pending')} / {b['call']} |")
    for r in B4["comparisons"]:
        if r["role"] == "primary": L.append(f"| B4 {r['comparison']} (tkgl-icews) | {r['MRR']['delta']:+.5f} {VR.ci(r['MRR']['ts']['CI95'])} | {r['call']} |")
    for r in C3["comparisons"]:
        if r["role"].startswith("primary"): L.append(f"| C3 {r['comparison']} (tkgl-icews, partial) | {r['MRR']['delta']:+.5f} {VR.ci(r['MRR']['ts']['CI95'])} | {r['call']} |")
    L.append(f"| C4 second TiRGN polecat seed | not run | declared cut |")
    L.append("\n## A items (reanalysis of stored outputs)\n")
    L += a_sections(art)
    L.append("## B and C items (new fits; official scoring)\n")
    L += bc_item(B1, art, "b1_ranker_form.json", "B1 Ranker form")
    L += bc_item(B2, art, "b2_matched_pool.json", "B2 Matched-width pool (pattern prior vs RecB)")
    L += bc_item(B3, art, "b3_common_pool_base_ablation.json", "B3 Common-pool base ablation and HRCOV_PAT − G1 decomposition")
    L.append(f"### B4 G1 on tkgl-icews Stage B ({link(art, 'b4_g1_tkgl_icews.json')})\n\n**Interpretation.** {B4['interpretation']}\n")
    L.append(VR.arm_table(B4["arms"])); L.append(""); L.append(VR.cmp_table(B4["comparisons"])); L.append("")
    L.append("Notes: " + " ".join(f"({i + 1}) {n}" for i, n in enumerate(B4["notes"])) + "\n")
    L += bc_item(C1, art, "c1_no_dyad_lr.json", "C1 Without ev_dyad_lr")
    if C2:
        L += bc_item(C2, art, "c2_pattern_design.json", "C2 Pattern design (utility vs confidence ordering, 1-hop vs 2-hop)")
    L.append(f"### C3 Relation normalisation on tkgl-icews, partial ({link(art, 'c3_relnorm_tkgl_icews.json')})\n\n**Interpretation.** {C3['interpretation']}\n")
    L.append(VR.arm_table(C3["arms"])); L.append(""); L.append(VR.cmp_table(C3["comparisons"])); L.append("")
    L.append("Declared cuts: " + "; ".join(C3["declared_cuts"]) + ".\n\nNotes: " + " ".join(f"({i + 1}) {n}" for i, n in enumerate(C3["notes"])) + "\n")
    L.append(f"### C4 Second base seed ({link(art, 'c4_base_seed.json')})\n\n**Declared cut.** {C4['reason']} What it would take: {C4['what_it_would_take']}\n")
    L += bc_item(A5, art, "a5_multi_positive_sensitivity.json", "A5 sensitivity: UNMASKED and MULTIPOS (polecat TiRGN)")
    L.append("## Deviations, cuts and process notes\n")
    common = B1["notes"][:4]
    for i, n in enumerate(list(common) + list(extra_notes)): L.append(f"{i + 1}. {n}")
    L.append("")
    md = "\n".join(L)
    open(f"{OUT}/tois_v7_experiments.md", "w").write(md)
    rows = []
    for itn, itj in (("B1", B1), ("B2", B2), ("B3", B3), ("C1", C1), ("A5", A5)) + ((("C2", C2),) if C2 else ()):
        for ds, D in itj["datasets"].items(): rows += VR.csv_rows(itn, DSN[ds], "TiRGN", D["comparisons"])
    rows += VR.csv_rows("B4", "tkgl-icews", "Stage B", B4["comparisons"]); rows += VR.csv_rows("C3", "tkgl-icews", "Stage B", C3["comparisons"])
    A3 = J("a3_hits.json")
    for run, Dd in A3["deltas"].items():
        ds, base = run.split(" | ")
        rr = [dict(comparison=c.replace("_minus_", " - "), role="descriptive", **{m: r[m] for m in ("MRR", "H1", "H3", "H10")}) for c, r in Dd.items()]
        rows += VR.csv_rows("A3", DSN.get(ds, ds), base, rr)
    VR.write_csv(rows, f"{OUT}/tois_v7_experiments.csv")
    return md, rows
