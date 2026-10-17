#!/usr/bin/env python3
"""hardening / hd_tables.py -- markdown tables for hardening_report.md, generated from hardening_verdict.json only
(so every number in the report tables has a single machine-readable source)."""
import json, sys

BLAB = {"regcn": "RE-GCN (pilot ckpt)", "logcl": "LogCL (pilot ckpt)", "tirgn": "TiRGN", "regcn_tuned": "RE-GCN (re-tuned)",
        "logcl_tuned": "LogCL (re-tuned)"}
ARMLAB = {"base": "base alone", "BLEND": "base (+) RecB blend", "POOL": "RecB-POOL", "HR": "HR", "HRCOV": "HRCOV (claim)",
          "HRCOV_ST": "HRCOV_ST", "HRCOV_ST_ND": "HRCOV_ST_NODIRECT", "HR_RB": "HR_RB", "HRCOV_RB": "HRCOV_RB"}


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def p(x):
    return "<0.0005" if x == 0 else f"{x:.4f}"


def tables(V):
    out = []
    for ds, D in V["datasets"].items():
        out.append(f"\n### {D['name']} (n = {D['n_test']:,} test queries)\n")
        out.append("| base | arm | evaluator | seeds | MRR | sd | H@1 | H@3 | H@10 | harness MRR | host |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in D["rows"]:
            out.append("| " + " | ".join([BLAB.get(r["base"], r["base"]), ARMLAB.get(r["arm"], r["arm"]), r.get("evaluator", "-"),
                                           str(r.get("seeds", "-")), f4(r.get("MRR")), f4(r.get("sd", {}).get("MRR")) if isinstance(r.get("sd"), dict) else "-",
                                           f4(r.get("H1")), f4(r.get("H3")), f4(r.get("H10")), f4(r.get("harness_MRR")), r.get("host", "-")]) + " |")
        out.append(f"\nArchitecture selection (valid MRR): " + "; ".join(f"{a}: {BLAB.get(v['selected'], v['selected'])} {v['valid_MRR']:.4f}"
                                                                          for a, v in D["architecture_selection"].items()))
        out.append(f"\nTwo strongest bases: {', '.join(BLAB.get(b, b) for b in D['two_strongest'])}. Strongest blend overall (C2 comparator): "
                   f"{BLAB.get(D['strongest_blend_overall']['base'], D['strongest_blend_overall']['base'])} (+) RecB, MRR {f4(D['strongest_blend_overall']['MRR'])}.\n")
        out.append("| base | contrast | delta MRR | 95% CI | one-sided p | p(delta <= 0.005) | criterion |")
        out.append("|---|---|---|---|---|---|---|")
        for b, d in D["decision"]["per_base"].items():
            for c in ("BLEND", "POOL"):
                v = d["C1"][c]
                out.append(f"| {BLAB.get(b, b)} | HRCOV - {ARMLAB[c]} | {v['delta']:+.4f} | {ci(v['CI95'])} | {p(v['p_one_sided'])} | {p(v['p_le_margin'])} | C1 ({'pass' if d['C1_pass'] else 'FAIL'} jointly) |")
            v = d["C2"]
            out.append(f"| {BLAB.get(b, b)} | HRCOV - {d['C2_vs']} | {v['delta']:+.4f} | {ci(v['CI95'])} | {p(v['p_one_sided'])} | {p(v['p_le_margin'])} | C2 ({'pass' if d['C2_pass'] else 'FAIL'}) |")
            v = d["C3_head"]
            out.append(f"| {BLAB.get(b, b)} | head stratum HRCOV - {ARMLAB[d['better_comparator']]} | {v['delta']:+.4f} | {ci(v['CI95'])} | {p(v['p_one_sided'])} | - | C3 ({'DAMAGE' if d['C3_damage'] else 'no damage'}) |")
            for k, v in d["secondary"].items():
                out.append(f"| {BLAB.get(b, b)} | {k} (secondary) | {v['delta']:+.4f} | {ci(v['CI95'])} | {p(v['p_one_sided'])} | {p(v['p_le_margin'])} | - |")
        out.append(f"\n**Outcome ({D['name']}): {D['decision']['outcome']}**\n")
        for b, d in D["decision"]["per_base"].items():
            out.append(f"\nStrata, {BLAB.get(b, b)}: HRCOV minus {ARMLAB[d['better_comparator']]} (B = 1000)\n")
            out.append("| stratum | share | MRR base | MRR RecB-tr | MRR blend | MRR pool | MRR HR | MRR HRCOV | delta | 95% CI | contribution to full delta |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for nm, s in d["strata"].items():
                out.append(f"| {nm} | {s['share']:.3f} | {f4(s['MRR_base'])} | {f4(s['MRR_RecB_tr'])} | {f4(s['MRR_BLEND'])} | {f4(s['MRR_POOL'])} | {f4(s['MRR_HR'])} | "
                           f"{f4(s['MRR_HRCOV'])} | {s['delta_HRCOV_minus_better']:+.4f} | {ci(s['CI95'])} | {s['contribution_to_full_delta']:+.4f} |")
            oc = d.get("oracle_ceilings")
            if oc:
                out.append("\nOracle ceilings (gold-in-set fraction): " + "; ".join(f"{k} {v:.3f}" for k, v in oc.items() if k != "note"))
    return "\n".join(out)


if __name__ == "__main__":
    V = json.load(open(sys.argv[1]))
    open(sys.argv[2], "w").write(tables(V))
