"""v78 / v78_report.py -- v78_report.md / v78_results.json / v78_verdict.json from run_config() outputs.
Usage: exec(open(os.path.join(os.environ['SRC'], 'v78', 'v78_report.py')).read()); build(results, outcome_reg, outcome_sens, meta)"""
import json, numpy as np

ORDER = ["base", "HIST", "ERM_ORDER", "SA", "SA_ERM", "SA_GEN", "+C", "+S", "+T", "HRcov", "SA_W50", "SA_W150", "HR_W50", "HR_W150", "REL_LIN", "REL_LIN_global", "EXT_alone_edgeremind", "EXT_blend_edgeremind", "EXT_alone_counttrucola", "EXT_blend_counttrucola"]
LABEL = {"base": "frozen base", "HIST": "HIST (RecB_xf history order alone)", "SA": "SA (W100, 16 base-free features)",
         "SA_GEN": "SA_GEN (sensitivity: + RecB generator slots)", "+C": "+C (base top-50 candidates)", "+S": "+S (+ base-score features, fusion)",
         "+T": "+T (+ base tail order)", "HRcov": "HRcov (stored v7.7, W100)", "SA_W50": "SA W50", "SA_W150": "SA W150",
         "HR_W50": "HRcov-recipe W50 (refit)", "HR_W150": "HRcov-recipe W150 (refit)", "REL_LIN": "REL_LIN (relation offsets)",
         "REL_LIN_global": "global linear (refit)", "ERM_ORDER": "EdgeReMIND order alone (history tie-break; SA_ERM tail)",
         "SA_ERM": "SA_ERM (EdgeReMIND top-50 + base-free block, 16 features)", "EXT_alone_edgeremind": "EXT_alone EdgeReMIND (faithful, s1337)",
         "EXT_blend_edgeremind": "EXT_blend EdgeReMIND + base", "EXT_alone_counttrucola": "EXT_alone CountTRuCoLa (faithful)",
         "EXT_blend_counttrucola": "EXT_blend CountTRuCoLa + base"}
VSEL = {"SA_ERM": "SA_ERM", "SA": "SA", "SA_GEN": "SA_GEN", "+C": "+C", "+S": "+S", "+T": "+T", "SA_W50": "SA_W50", "SA_W150": "SA_W150", "HR_W50": "HR_W50",
        "HR_W150": "HR_W150", "REL_LIN": "REL_LIN", "REL_LIN_global": "REL_LIN_global"}


def f4(x): return "–" if x is None else f"{x:.4f}"
def s4(x): return "–" if x is None else f"{x:+.4f}"


def ci(m):
    return f"{s4(m['delta'])} | [{s4(m['query']['L'])}, {s4(m['query']['U'])}] | [{s4(m['ts']['L'])}, {s4(m['ts']['U'])}]"


def sys_table(r):
    L = ["| system | MRR | H@1 | H@3 | H@10 | per-seed MRR | valid-select MRR |", "|---|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in r["systems"]: continue
        m = r["systems"][k]; ps = ", ".join(f"{x:.4f}" for x in m["per_seed_MRR"]) if m["n_seeds"] > 1 else ""
        L.append(f"| {LABEL[k]} | {f4(m['MRR'])} | {f4(m['H1'])} | {f4(m['H3'])} | {f4(m['H10'])} | {ps} | {f4(r['validation_select_MRR'].get(VSEL.get(k, ''), None))} |")
    for k in r["systems"]:
        if k not in ORDER and k not in ("base(stored)",):
            m = r["systems"][k]; L.append(f"| {k} | {f4(m['MRR'])} | {f4(m['H1'])} | {f4(m['H3'])} | {f4(m['H10'])} | | |")
    return "\n".join(L)


def cmp_rows(r, labels):
    L = ["| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |", "|---|---|---|---|---|---|"]
    for lab in labels:
        if lab not in r["comparisons"]: continue
        c = r["comparisons"][lab]; L.append(f"| {lab} | {ci(c['MRR'])} | {s4(c['H1']['delta'])} | {s4(c['H10']['delta'])} |")
    return "\n".join(L)


def ext_section(r, meta):
    es = r.get("external_selection")
    if not es: return "not assessed"
    L = ["| method | gate | variant | selection-part MRR | selected (a, tau) |", "|---|---|---|---|---|"]
    for m, d in es["per_method"].items():
        cs = d["candidates"]; at = meta["ext_cfg"][r["config"]][m]
        L.append(f"| {m} | {meta['gate'][m]['call']} | alone | {f4(cs['alone'])} | – |")
        L.append(f"| {m} | {meta['gate'][m]['call']} | blend | {f4(cs['blend'])} | ({at[0]}, {at[1]}) |")
    L += ["", f"Selected external comparator: **{es['X']}** (variant with the higher selection-part MRR per method, then the better method). "
          f"Standalone comparator: **{r['standalone_comparator']}** (selection-part MRR SA {f4(r['validation_select_MRR'].get('SA'))}, SA_ERM {f4(r['validation_select_MRR'].get('SA_ERM'))}).", "",
          cmp_rows(r, [f"HRcov - {es['X']}", f"{r['standalone_comparator']} - {es['X']}", "HRcov - SA_ERM", "SA_ERM - SA", "HRcov - ERM_ORDER"] +
                   [f"HRcov - EXT_{v}_{m}" for m in ("edgeremind", "counttrucola") for v in ("alone", "blend") if f"EXT_{v}_{m}" != es["X"]])]
    return "\n".join(L)


def build(results, oreg, osens, meta, path_md="v78/v78_report.md"):
    md = [f"# v7.8 standalone ladder — report", "",
          f"Rule `v78_framing_rule.json` (sha256 {meta['rule_sha'][:12]}…, registered {meta['rule_utc']}) with `v78_rule_amendment_1.json` "
          f"(sha256 {meta['amend_sha'][:12]}…, registered {meta['amend_utc']}, before any v7.8 test number). Official py-tgb 2.3.0 evaluator with shipped negatives; "
          "learned systems = seed-mean per-query RR over ranker seeds 0–2; paired query Poisson and timestamp-block bootstraps, B = 2000, shared draws, "
          "one-sided 95% bounds (L = 5th, U = 95th percentile). BEATS: L > 0 under both and Δ ≥ 0.005; MATCHES: U < 0.005 under both; otherwise SMALL.", "",
          "## Verdict", "",
          f"**Registered outcome label: {oreg['label']}**" + (" (provisional: " + oreg["note"] + ")" if oreg.get("note") else ""), "",
          "| configuration | standalone comparator | HRcov − standalone: call | external comparator | HRcov − external: call |", "|---|---|---|---|---|"]
    for c, r in results.items():
        d = r["decision"]
        md.append(f"| {c} ({r['ds']}, {r['base']}) | {d['standalone']['X']} | {d['standalone']['call']} | {d['external']['X'] or '—'} | {d['external']['call']} |")
    md += ["", f"Sensitivity (registered, not decision-relevant): with the standalone comparator chosen from {{SA, SA_GEN}} on the selection part, the label is **{osens['label']}** "
           f"(calls: {', '.join(f'{c}: {v}' for c, v in osens['standalone_calls'].items())}).", ""]
    for c, r in results.items():
        md += [f"## {c}: {r['ds']}, {r['base']} (n = {r['n_queries']:,} test queries, {r['n_ts']} timestamps)", "", sys_table(r), "",
               "Decision and context comparisons:", "",
               cmp_rows(r, [f"HRcov - {r['decision']['standalone']['X']}", "HRcov - SA_GEN", "HRcov - HIST", "HRcov - base", "SA - HIST", "SA - base"] +
                        [k for k in r["comparisons"] if k.startswith("HRcov - EXT") or k.startswith("SA - EXT")]), "",
               "External comparators and SA_ERM:", "",
               ext_section(r, meta), "",
               "Ladder (descriptive; each rung adds one base component):", "",
               cmp_rows(r, ["ladder: +C - SA", "ladder: +S - +C", "ladder: +T - +S", "ladder: HRcov - +T (+P)", "HRcov - SA"]), "",
               f"+T reproduces HRcov: |Δ| = {f4(r['ladder_reproduction']['abs_diff'])} (≤ 0.002: {r['ladder_reproduction']['within_0002']}).", "",
               "Width sweep (descriptive):", "",
               cmp_rows(r, ["width: SA_W50 - SA", "width: SA_W150 - SA", "width: HR_W50 - HRcov", "width: HR_W150 - HRcov",
                            "width: HR_W50 - SA_W50", "HRcov - SA", "width: HR_W150 - SA_W150"]), "",
               "REL_LIN diagnostic (not decision-relevant):", "",
               cmp_rows(r, ["REL_LIN: HRcov - REL_LIN", "REL_LIN: REL_LIN - REL_LIN_global", "REL_LIN: HRcov - REL_LIN_global"]), "",
               f"Checks: base-rank identity with the stored HRcov file: mean |Δ RR| = {r['base_identity']['base_rr_mean_abs_diff']:.2e} (ok: {r['base_identity']['ok']}); "
               f"harness − official MRR flagged (> 1e-4): {json.dumps({k: round(v, 5) for k, v in r['harness_vs_official_flagged'].items()}) or 'none'}.", ""]
    md += ["## Implementation checks", ""] + [f"- {x}" for x in meta["checks"]] + ["", "## Deviations", ""] + [f"- {x}" for x in meta["deviations"]] + ["",
           "## Files", ""] + [f"- {x}" for x in meta.get("files", [])]
    open(path_md, "w").write("\n".join(md) + "\n")
    return "\n".join(md)
