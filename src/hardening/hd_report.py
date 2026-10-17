#!/usr/bin/env python3
"""hardening / hd_report.py -- builds the data sections of hardening_report.md from hardening_verdict.json and the
run logs (every number in the generated sections is read from a file; narrative sections are passed in as text files).
  python3 hd_report.py --verdict V --cfg C --extra extra.json --intro intro.md --discussion disc.md --static recb.md --out report.md
"""
import json, argparse, os
import numpy as np
import importlib.util

ap = argparse.ArgumentParser()
for k in ("verdict", "extra", "intro", "discussion", "static", "out", "tables_py"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
V = json.load(open(a.verdict)); X = json.load(open(a.extra))
spec = importlib.util.spec_from_file_location("hd_tables", a.tables_py); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
BL = T.BLAB
f4 = lambda x: "-" if x is None else f"{x:.4f}"
ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
out = [open(a.intro).read().rstrip(), ""]

# ---------------- RecB section (static, data from recb meta files)
out.append("\n" + open(a.static).read().rstrip() + "\n")
# ---------------- base verification
out.append("## 3. Bases: verification, selected epochs, and strength relative to published rows\n")
out.append("| dataset | base | architecture config | selected epoch | selection valid MRR | full-valid MRR | test MRR (official) | pilot test MRR | TGB 2.0 row (same architecture) | host |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for ds, D in V["datasets"].items():
    for b, rec in D["bases"].items():
        if "missing" in rec: continue
        bx = X["bases"][ds][b]
        out.append(f"| {D['name']} | {BL.get(b, b)} | {bx['config']} | {rec['selected_epoch']} | {rec['valid_MRR_base']:.4f} ({bx['sel_set']}) | "
                   f"{f4(bx.get('full_valid'))} | {rec['arms']['base']['official']['MRR']:.4f} | {f4(bx.get('pilot_test'))} | {bx.get('tgb_row', '-')} | {rec['host']} |")
out.append("")
out.append(X["base_notes"])
for nm, g in X["grids"].items():
    out.append(f"\n{nm} (full-valid filtered MRR, valid-best epoch per configuration; host {g['host']}):\n")
    out.append("| history L | lr | snapshot order | best epoch | best valid MRR | train s |")
    out.append("|---|---|---|---|---|---|")
    for r in g["runs"]:
        sel = " (selected)" if (r["L"], r["lr"], r["shuffle"]) == (g["selected"]["L"], g["selected"]["lr"], g["selected"]["shuffle"]) else ""
        out.append(f"| {r['L']} | {r['lr']:g} | {'shuffled' if r['shuffle'] else 'time-ordered'} | {r['best_epoch']} | {r['best_valid_MRR']:.4f}{sel} | {r['train_s']} |")
if X.get("tirgn_polecat"):
    tp = X["tirgn_polecat"]
    out.append(f"\nTiRGN on tkgl-polecat (history L = {tp['L']}, authors' hyper-parameters, {tp['gpu']}; full-valid filtered MRR per epoch, epochs indexed from 0):\n")
    out.append("| epoch | train loss | valid MRR | train s |"); out.append("|---|---|---|---|")
    for e in tp["epoch_log"]:
        out.append(f"| {e['epoch']}{' (selected)' if e['epoch'] == tp['best_epoch'] else ''} | {e['loss']:.4f} | {e['valid_filt_MRR']:.4f} | {e['train_epoch_s']} |")
    out.append(f"\nStop reason: {tp['stop_reason']}; selected epoch {tp['best_epoch']} (valid MRR {tp['best_valid']:.4f}, H@10 {tp['valid_H10']:.4f}).")
if X.get("curve_note"):
    out.append("\n" + X["curve_note"] + "\n")


# ---------------- controls: selected configurations
out.append("## 4. Recurrency controls: valid-selected configurations\n")
out.append("| dataset | base | BASE(+)RecB selected (variant, form, weight) | valid MRR | RecB-POOL selected (K; variant, form, weight) | valid MRR | valid MRR base | max batch deviation official vs per-query formula |")
out.append("|---|---|---|---|---|---|---|---|")
for ds, D in V["datasets"].items():
    for b, rec in D["bases"].items():
        if "missing" in rec: continue
        s = rec["selected"]
        out.append(f"| {D['name']} | {BL.get(b, b)} | {', '.join(s['blend'])} | {rec['valid_MRR_blend_selected']:.4f} | {s['pool'][0]}; {', '.join(s['pool'][1])} | "
                   f"{rec['valid_MRR_pool_selected']:.4f} | {rec['valid_MRR_base']:.4f} | {rec['official_vs_formula_max_batch_dev']} |")
out.append("\nVariant 'def' = RecB default (λ = 0.1, α = 0.99), 'tr' = RecB trained (per-relation λ_r, α_r); MIX weight = β; LOGLIN weight = (w, ε).\n")

# ---------------- full-test rows + decision tables
out.append("## 5. Full-test results and the pre-registered decision\n")
out.append("All rows: every test query of the dataset, both directions, time-aware filtered, official py-tgb 2.3.0 Evaluator with the shipped "
           "negatives (per-query reciprocal rank; re-rankers averaged over seeds 0-2, `sd` = across-seed SD of the seed MRRs). "
           "`harness MRR` = the harness exact filtered evaluator on the same score rows. Published rows are copied from the cited sources, not re-run.\n")
out.append(T.tables(V))

# ---------------- pool widening (all bases)
out.append("\n## 6. What the gain is made of\n")
out.append("### 6.1 Pool-widening gain (HRCOV minus HR, same base, official evaluator, B = 2000) and margin over the better recurrency control\n")
out.append("| dataset | base | valid MRR base | HR | HRCOV | HRCOV - HR | 95% CI | HRCOV - better control | 95% CI |")
out.append("|---|---|---|---|---|---|---|---|---|")
for ds, D in V["datasets"].items():
    for b, rec in sorted(((b, r) for b, r in D["bases"].items() if "missing" not in r), key=lambda t: -t[1]["valid_MRR_base"]):
        if "HRCOV_minus_HR_official" not in rec: continue
        p, c = rec["HRCOV_minus_HR_official"], rec["HRCOV_minus_better_control_official"]
        out.append(f"| {D['name']} | {BL.get(b, b)} | {rec['valid_MRR_base']:.4f} | {rec['arms']['HR']['official']['MRR']:.4f} | {rec['arms']['HRCOV']['official']['MRR']:.4f} | "
                   f"{p['delta']:+.4f} | {ci(p['CI95'])} | {c['delta']:+.4f} | {ci(c['CI95'])} |")
out.append("")
out.append("### 6.2 RecB score as an extra re-ranker feature (secondary arms HR_RB / HRCOV_RB; harness evaluator, B = 2000)\n")
out.append("| dataset | base | HR | HR_RB | HR_RB - HR | 95% CI | HRCOV | HRCOV_RB | HRCOV_RB - HRCOV | 95% CI |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for ds, D in V["datasets"].items():
    for b, rec in D["bases"].items():
        if "missing" in rec or "HR_RB" not in rec["arms"]: continue
        A = rec["arms"]; d1 = rec["HR_RB_minus_HR_harness"]; d2 = rec["HRCOV_RB_minus_HRCOV_harness"]
        out.append(f"| {D['name']} | {BL.get(b, b)} | {A['HR']['harness']['MRR']:.4f} | {A['HR_RB']['harness']['MRR']:.4f} | {d1['delta']:+.4f} | {ci(d1['CI95'])} | "
                   f"{A['HRCOV']['harness']['MRR']:.4f} | {A['HRCOV_RB']['harness']['MRR']:.4f} | {d2['delta']:+.4f} | {ci(d2['CI95'])} |")
out.append("")
out.append("### 6.3 Oracle ceilings (fraction of test queries whose gold is inside the candidate set = MRR of a perfect re-ranker over that set)\n")
out.append("| dataset | base | base top-50 | RecB-trained top-50 | base top-50 ∪ RecB top-50 | base top-100 ∪ RecB top-100 | HRCOV pool C_cov (pilot report; TiRGN polecat: this study's feature summary) | HRCOV MRR |")
out.append("|---|---|---|---|---|---|---|---|")
for ds, D in V["datasets"].items():
    for b, d in D["decision"]["per_base"].items():
        oc = d["oracle_ceilings"]
        out.append(f"| {D['name']} | {BL.get(b, b)} | {oc['base_top50']:.3f} | {oc['RecB_trained_top50']:.3f} | {oc['base_top50_union_RecB_top50']:.3f} | "
                   f"{oc['base_top100_union_RecB_top100']:.3f} | {X['ccov'].get(ds, {}).get(b, '-')} | {D['bases'][b]['arms']['HRCOV']['official']['MRR']:.4f} |")
out.append("")
out.append(open(a.discussion).read().rstrip())
open(a.out, "w").write("\n".join(out) + "\n")
print("written", a.out, sum(len(x) for x in out))
