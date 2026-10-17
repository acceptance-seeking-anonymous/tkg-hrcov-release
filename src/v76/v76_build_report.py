"""v76 / v76_build.py -- builds v76_results.json, v76_report.md and v76_manuscript_markers.csv from v76/agg/v76_agg_state.json.
exec(open('v76/code/v76_build.py').read())"""
import json, csv, os
S = json.load(open("v76/agg/v76_agg_state.json"))
HASH_CONVENTION = ("write_prereg: registered sha256 = sha256 of json.dumps(body, indent=1, sort_keys=True, default=str), where body = rule + registered_utc and WITHOUT the sha256 field; "
                   "the saved file is json.dump(body + sha256, indent=1) in insertion order, so its file-bytes sha256 differs. "
                   "The exact hashed bytes are stored as *_registered.json; v76_rule_hash_check.py re-derives both from the saved files.")
RULE = dict(file="v76_rule.json", registered_utc="2026-10-15T12:07:43Z",
            sha256="edcfca14128e22f6aaec415eca54da083d2678354bd4cdc95f475638863e0a04", sha256_kind="registered (content) hash, as recorded inside the file",
            file_sha256="4daf8b0f99271ff8f3018762dd12f03695a406e89c3837af44d4aa7370b8471d",
            registered_bytes=dict(file="v76_rule_registered.json", sha256="edcfca14128e22f6aaec415eca54da083d2678354bd4cdc95f475638863e0a04"))
AM1 = dict(file="v76_rule_amendment_1.json", registered_utc="2026-10-15T12:49:21Z",
           sha256="bc8cd8f87192ee980dc0fdde8a3a8e8bd3770bdf01339bb133be16df260697ff", sha256_kind="registered (content) hash, as recorded inside the file",
           file_sha256="f02c5f6b85dc726be51392a9ef54ba28722ceb73321532f7aab01527b3b458d1", amends_sha256="edcfca14128e22f6aaec415eca54da083d2678354bd4cdc95f475638863e0a04",
           registered_bytes=dict(file="v76_rule_amendment_1_registered.json", sha256="bc8cd8f87192ee980dc0fdde8a3a8e8bd3770bdf01339bb133be16df260697ff"))
HASH_CHECK = dict(file="v76_rule_hash_check.py")
DEV = json.load(open("v76/rule/v76_deviations.json"))
NAMES = {"polecat_tirgn": "tkgl-polecat TiRGN (s42)", "polecat_cogn": "tkgl-polecat CognTKE", "smallpedia_tirgn": "tkgl-smallpedia TiRGN", "smallpedia_regcn_tgb": "tkgl-smallpedia RE-GCN (TGB)"}
f4 = lambda x: f"{x:.4f}"; s4 = lambda x: f"{x:+.4f}"


def ci(r, b="ts"): return f"[{r[b]['CI95'][0]:+.4f}, {r[b]['CI95'][1]:+.4f}]"


def verdict(r):
    pos = r.get("present_pos_holm", r["present_pos"]); neg = r.get("present_neg_holm", r["present_neg"])
    eq = r["excluded_gain_ge_bar"] and r["excluded_loss_ge_bar"]
    s = "present (+)" if pos else ("present (-)" if neg else "not present")
    return s + ("; within +-0.005 (both directions excluded)" if eq else ("; gain >= 0.005 excluded" if r["excluded_gain_ge_bar"] else ""))


def row(name, r): return f"| {name} | {s4(r['delta'])} | {ci(r, 'query')} | {ci(r, 'ts')} | {verdict(r)} |"


M2B, M3, M4, M6R, COST = S["M2B"], S["M3"], S["M4"], S["M6R"], S["COST"]
M6T = S.get("M6T")
L = []
L.append("# Manuscript A v7.6 - reviewer-requested experiments (M2b, M3a, M4, M6)\n")
L.append("**Rule provenance.**\n\n| File | Registered UTC | Registered sha256 (recorded in file; = sha256 of *_registered.json) | sha256 of the saved file bytes |\n|---|---|---|---|\n"
         f"| v76_rule.json | {RULE['registered_utc']} | {RULE['sha256']} | {RULE['file_sha256']} |\n| v76_rule_amendment_1.json | {AM1['registered_utc']} | {AM1['sha256']} | {AM1['file_sha256']} |\n")
L.append("The two hashes per file differ by construction: " + HASH_CONVENTION + " The amendment is a separate file registered 42 min after the rule; it cites the rule's registered hash and changes only the M2b LambdaMART selection statistic. Each file has one stored version.\n")
L.append(f"Rule: v76_rule.json (registered {RULE['registered_utc']}, registered sha256 {RULE['sha256'][:12]}..., before any test number of these arms), amendment 1 ({AM1['registered_utc']}, registered sha256 {AM1['sha256'][:12]}...: tie-exact LambdaMART selection statistic). "
         "Statistics as tois_v7_rule.json: official py-tgb 2.3.0 per-query RR; paired Poisson query bootstrap and timestamp-block bootstrap (B = 2000, seed 0, shared draws); "
         "present iff the one-sided lower bound > 0 under both (Holm across an item's primary contrasts on a base); 'excluded' iff the timestamp-block upper bound of |d| < 0.005. "
         "All test sets were inspected in earlier rounds: every number below is a pre-specified later analysis, not an untouched confirmation.\n")
L.append("Finiteness: every new array was scanned before aggregation (dumps, full rows, features, RecB rows, fitted models, promoted lists, official outputs): 0 NaN and 0 +inf; the only -inf values sit at padded candidate slots of feature log-prob tables (0 at real candidates).\n")
# ------------------------------------------------------------------ M2b
L.append("## M2b - LambdaMART on the same features, pool, rows and fusion\n")
L.append("**Card.** LightGBM 4.6.0 lambdarank (CPU, 16 threads) on the 25 HRCOV_PAT features of the W = 100 pool; one group per fit query (valid, unfiltered candidates; label 1 = gold); fit rows = validation ts < 70% cut with the gold in the pool (cap 120,000); "
         "grid num_leaves {15, 31, 63} x learning_rate {0.03, 0.1}, <= 3,000 rounds, early stopping (patience 100) and grid choice on the selection-row MRR of the exact fused-rank statistic of the LambdaMART score (ties resolved as the promoted list resolves them, amendment 1); grid on seed 0, seeds 1-2 refit; "
         "calibrated alpha/beta fusion with the base log-prob as for the MLP. MLP64_PAT (= HRCOV_PAT ranker) and LIN_PAT re-fit in the same run on the same rows.\n")
L.append("| Base | Better control | LIN | LambdaMART | MLP (HRCOV_PAT) | MLP stored | LMART grid point |\n|---|---|---|---|---|---|---|")
for B, r in M2B.items():
    m = r["mrr"]; bc = r["better_control"]
    L.append(f"| {NAMES[B]} | {f4(m[bc])} ({bc}) | {f4(m['LIN'])} | {f4(m['LMART'])} | {f4(m['MLP'])} | {f4(m['HRCOV_PAT_stored'])} | {r['lmart']['selected']['num_leaves']} leaves, lr {r['lmart']['selected']['learning_rate']} |")
L.append("\nMRR, official evaluator, test; learned arms = 3-seed means. H@10 in v76_results.json.\n")
L.append("| Base | Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |\n|---|---|---|---|---|---|")
for B, r in M2B.items():
    for k in ("MLP64_PAT_fused - LMART_PAT_fused", "LMART_PAT_fused - LIN_PAT_fused", "LMART_PAT_fused - better control"):
        L.append(row(f"{NAMES[B]} | {k}", r["contrasts"][k]).replace("| ", "| ", 1))
L.append("\nSecondary (alone scores, exact fused-rank statistic): " + "; ".join(f"{NAMES[B]}: MLP - LMART {s4(r['contrasts']['[exact] MLP64_PAT_alone - LMART_PAT_alone']['delta'])}" for B, r in M2B.items()) + ".\n")
L.append("**Cost.**\n\n| Base | LMART fit per seed (s, incl. per-round selection) | best rounds | LMART grid total (s) | MLP fit per seed (s, GPU) | Scoring latency, ms / 1,000 queries: LMART CPU / MLP CPU / MLP GPU |\n|---|---|---|---|---|---|")
for B, c in COST.items():
    la = c["latency"]
    L.append(f"| {NAMES[B]} | {', '.join(str(v) for v in c['LMART_seed_wall_s'].values())} | {', '.join(str(v) for v in c['LMART_best_rounds'].values())} | {c['LMART_grid_wall_s']} | "
             f"{', '.join(str(v) for k, v in c['MLP_seed_wall_s_from_log'].items() if k != '0')} (seeds 1-2) | {la['LMART_cpu_ms_per_1000q']:.1f} / {la['MLP_cpu_ms_per_1000q']:.1f} / {la['MLP_gpu_ms_per_1000q']:.1f} |")
_d = {B: {k: M2B[B]["contrasts"][k]["delta"] for k in ("MLP64_PAT_fused - LMART_PAT_fused", "LMART_PAT_fused - LIN_PAT_fused", "LMART_PAT_fused - better control")} for B in M2B}
_ml = [v["MLP64_PAT_fused - LMART_PAT_fused"] for v in _d.values()]; _ll = [v["LMART_PAT_fused - LIN_PAT_fused"] for v in _d.values()]; _lc = [v["LMART_PAT_fused - better control"] for v in _d.values()]
_ratio = {B: COST[B]["latency"]["LMART_cpu_ms_per_1000q"] / COST[B]["latency"]["MLP_cpu_ms_per_1000q"] for B in COST}
_trees = [t for B in COST for t in COST[B]["LMART_best_rounds"].values()]
L.append(f"\n**Reading.** LambdaMART matches the MLP within the 0.005 bar on all four bases (MLP - LambdaMART from {min(_ml):+.4f} to {max(_ml):+.4f}); the small differences are present in both directions (MLP ahead on the two tkgl-polecat bases, LambdaMART ahead on the two tkgl-smallpedia bases). "
         f"Like the MLP, LambdaMART exceeds the linear scorer ({min(_ll):+.4f} to {max(_ll):+.4f}; on CognTKE the margin is at the bar) and the better recurrence control ({min(_lc):+.4f} to {max(_lc):+.4f}) on every base. The added value is therefore a non-linear candidate-level ranker over these features, not the particular 2x64 network. "
         f"Cost: the MLP (5,889 parameters) scores at 16-18 ms per 1,000 queries on CPU (2-4 ms on GPU); LambdaMART ({min(_trees)}-{max(_trees)} trees) is {_ratio['polecat_tirgn']:.1f}x and {_ratio['polecat_cogn']:.1f}x slower on the tkgl-polecat bases, {_ratio['smallpedia_tirgn']:.1f}x slower on tkgl-smallpedia TiRGN, and {1 / _ratio['smallpedia_regcn_tgb']:.1f}x faster on tkgl-smallpedia RE-GCN, whose selected model has 67-123 trees. Fitting cost is minutes for either learner.\n")
L.append("**Caveats.** Amendment 1: the first LambdaMART run selected round 1 everywhere because the v7 statistic counts tied scores optimistically; it was discarded (its log printed round-1 test MRRs; nothing was chosen from them). LambdaMART fit times include the per-round selection evaluation on 56-61k queries; MLP times are GPU. LogCL (optional base) was not run.\n")
# ------------------------------------------------------------------ M3a
c3 = M3["contrasts"]; m3 = M3["mrr"]; t3 = S["m3a_train"]
L.append("## M3a - base trained with the validation-fit labels (tkgl-smallpedia TiRGN)\n")
L.append(f"**Card.** sp_tirgn.py recipe unchanged except that the training snapshots are train plus the {t3['n_valid_fit_snapshots']} validation snapshots with ts < 105 ({t3['valid_fit_facts']:,} facts); epoch selection on the {t3['n_valid_select_snapshots']} validation-select snapshots only ({t3['valid_select_facts']:,} facts). "
         f"Best epoch {t3['best_epoch']} of 30 (no early stop), validation-select MRR {t3['best_valid_filt_MRR']:.4f}; training {t3['train_s']} s on serverA GPU 0 (no memory issue). BLEND / POOL grids as hardening, selected on the 56,460 validation-select queries "
         f"(selected: BLEND {' '.join(str(x) for x in S['m3a_sel']['blend_cfg'])}, POOL K = {S['m3a_sel']['pool_cfg'][0]}).\n")
L.append(f"| System | MRR |\n|---|---|\n| Original TiRGN base | {f4(m3['orig_base'])} |\n| Validation-trained base | {f4(m3['valbase'])} |\n| Better control, original base ({M3['better_control_orig']}) | {f4(m3['better_ctrl_orig'])} |\n"
         f"| Better control, validation-trained base ({M3['better_control_valbase']}) | {f4(m3['better_ctrl_valbase'])} |\n| HRCOV_PAT, original base (stored v7.4 run) | {f4(m3['HRCOV_PAT_orig'])} |\n| HRCOV_PAT, validation-trained base (secondary) | {f4(m3['HRCOV_PAT_valbase'])} |\n")
L.append("| Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |\n|---|---|---|---|---|")
for k, r in c3.items(): L.append(row(("**primary** " if r.get("primary") else "") + k, r))
L.append(f"\n**Reading.** Giving the base the validation-fit labels raises it by {s4(c3['validation-trained base - original base']['delta'])} and the better recurrence control by {s4(c3['better control(validation-trained) - better control(original)']['delta'])}. "
         f"HRCOV_PAT on the original base still exceeds the better control on the validation-trained base by {s4(c3['HRCOV_PAT(original base) - better control(validation-trained base)']['delta'])} {ci(c3['HRCOV_PAT(original base) - better control(validation-trained base)'])} (vs {s4(c3['HRCOV_PAT(original base) - better control(original base)']['delta'])} on the original base): validation supervision of the base closes about "
         f"{100 * c3['better control(validation-trained) - better control(original)']['delta'] / c3['HRCOV_PAT(original base) - better control(original base)']['delta']:.0f}% of the margin, and the supervision asymmetry does not explain it. On the validation-trained base, HRCOV_PAT keeps {s4(c3[[k for k in c3 if k.startswith('[secondary]')][0]]['delta'])} over that base's own better control (secondary; its fit rows are in-sample for this base).\n")
L.append("**Caveats.** One dataset and one base (as specified); single training seed; the validation-fit facts are a superset of the ranker's fit rows (all facts of those timestamps); RecB's trained parameters inside the controls are the v7.4 ones (selected on all validation queries).\n")
# ------------------------------------------------------------------ M4
L.append("## M4 - cross-fitted RecB (parameters selected without the ranker's fit labels)\n")
xs = S["xfsel"]
L.append(f"**Card.** RecB_xf: TGB train() selection of (lambda_r, alpha_r) from the stored exact validation ranks of all 182 configurations, restricted to validation-select queries (ts >= cut). "
         f"tkgl-polecat: {xs['polecat']['n_select_queries']:,} queries, parameters changed for {xs['polecat']['n_rel_changed_vs_original']} of {xs['polecat']['n_rel_with_valid']} directed relations; "
         f"tkgl-smallpedia: {xs['smallpedia']['n_select_queries']:,} queries, changed for {xs['smallpedia']['n_rel_changed_vs_original']} of {xs['smallpedia']['n_rel_with_valid']} ({xs['smallpedia']['n_rel_valid_but_no_select_default']} without validation-select queries fall back to the TGB defaults). "
         f"Optimism of the original parameters on the fit rows (validation MRR, original - xf): polecat {xs['polecat']['valid_MRR_orig_fit_rows'] - xs['polecat']['valid_MRR_xf_fit_rows']:+.4f}, smallpedia {xs['smallpedia']['valid_MRR_orig_fit_rows'] - xs['smallpedia']['valid_MRR_xf_fit_rows']:+.4f}. "
         "G1_xf = G1 recipe unchanged on RecB_xf candidates/features (K selected on validation-select: K = 150 on both); RecB-feature arms re-fit in one run per base with no RecB features, with trained-RecB features (tr) and with RecB_xf features (xf).\n")
for ds in ("polecat", "smallpedia"):
    r = M4[ds]; m = r["mrr"]
    L.append(f"**{ds}** - G1 {f4(m['G1'])}, G1_xf {f4(m['G1_xf'])}; RecB generator row tr {f4(m['RecB_tr_row'])}, xf {f4(m['RecB_xf_row'])}; RecB-feature arms (TiRGN): PAT {f4(m['MLP64_PAT'])}, PAT_RBtr {f4(m['PAT_RBtr'])}, PAT_RBxf {f4(m['PAT_RBxf'])}, HR19 {f4(m['HR19'])}, HR19_RBtr {f4(m['HR19_RBtr'])}, HR19_RBxf {f4(m['HR19_RBxf'])}.\n")
    L.append("| Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |\n|---|---|---|---|---|")
    for k, v in r["contrasts"].items(): L.append(row(("**" + k + "**") if v.get("primary") else k, v))
    L.append("")
rng_old = [M4[d]["contrasts"][k]["delta"] for d in M4 for k in M4[d]["contrasts"] if k.startswith("HRCOV[") and k.endswith("- G1")]
rng_new = [M4[d]["contrasts"][k]["delta"] for d in M4 for k in M4[d]["contrasts"] if k.startswith("HRCOV[") and k.endswith("- G1_xf")]
L.append(f"**Reading.** Cross-fitting changes G1 by {s4(M4['polecat']['contrasts']['G1_xf - G1']['delta'])} on tkgl-polecat (present, below the bar) and {s4(M4['smallpedia']['contrasts']['G1_xf - G1']['delta'])} on tkgl-smallpedia (not present under the timestamp-block bootstrap; a gain >= 0.005 is excluded). "
         f"G1 was therefore not materially understated: the base-contribution range HRCOV - G1 moves from {min(rng_old):+.4f}..{max(rng_old):+.4f} to {min(rng_new):+.4f}..{max(rng_new):+.4f}. "
         f"The RecB-score-as-feature loss on tkgl-smallpedia TiRGN persists with cross-fitted parameters (HR19 {s4(M4['smallpedia']['contrasts']['HR19_RBxf - HR19']['delta'])}, HRCOV_PAT {s4(M4['smallpedia']['contrasts']['PAT_RBxf - PAT']['delta'])}; with trained parameters in the same run {s4(M4['smallpedia']['contrasts']['HR19_RBtr - HR19']['delta'])} / {s4(M4['smallpedia']['contrasts']['PAT_RBtr - PAT']['delta'])}; v7.4, harness, bridge runs: -0.0269 / -0.0119). "
         "Cross-fitting recovers only 0.0015-0.0038, so the symptom is not explained by selecting RecB's parameters on the ranker's fit labels; on tkgl-polecat all RecB-feature effects stay within +-0.0003.\n")
L.append("**Caveats.** Validation-select has fewer queries per relation than full validation (smallpedia: 70 relations fall back to defaults). The v7.4 RecB-feature values come from different (bridge) runs scored with the harness evaluator and are not paired with these. The default-mode RecB sensitivity for G1 was not run.\n")
# ------------------------------------------------------------------ M6
L.append("## M6 - base-seed variation on tkgl-smallpedia (lowest priority, descriptive)\n")
fams = [M6R] + ([M6T] if M6T else [])
for F in fams:
    sm = F["summary"]
    L.append(f"**{F['family']}** ({sm['n_seeds']} base seeds)\n\n| Base seed | Base | BLEND | POOL | HRCOV_PAT | Margin (HRCOV_PAT - better control) | 95% CI (ts-block) |\n|---|---|---|---|---|---|---|")
    for s, v in F["seeds"].items():
        L.append(f"| {s} | {f4(v['base'])} | {f4(v['BLEND'])} | {f4(v['POOL'])} | {f4(v['HRCOV_PAT'])} | {s4(v['margin']['delta'])} | {ci(v['margin'])} |")
    L.append(f"\nMargin mean {sm['margin_mean']:.4f} +- {sm['margin_sd']:.4f} (SD, n-1), range {sm['margin_min']:.4f}..{sm['margin_max']:.4f}; base MRR {sm['base_mean']:.4f} +- {sm['base_sd']:.4f}; "
             f"HRCOV_PAT MRR {min(v['HRCOV_PAT'] for v in F['seeds'].values()):.4f}..{max(v['HRCOV_PAT'] for v in F['seeds'].values()):.4f}.\n" if sm.get("margin_sd") is not None else "\n")
    if F.get("note"): L.append(F["note"] + "\n")
if not M6T:
    L.append("TiRGN seeds 43/44: running on serverA GPU 0 after items 1-3 released it (started 2026-10-15 14:07 UTC); added when complete or reported as partial at the window's end.\n")
L.append("Existing runs counted as one seed each (TiRGN pilot-2 seed 42; RE-GCN G3 seed 1, the same TGB script and defaults). The intervals are conditional on each trained base; the SD over base seeds is the descriptive between-seed spread.\n")
if M6T:
    L.append(f"**Reading.** Every base seed keeps a present margin over its better recurrence control (lowest timestamp-block lower bound {min(v['margin']['ts']['CI95'][0] for F in fams for v in F['seeds'].values()):+.4f}). "
             f"The between-seed SD of the margin is {M6T['summary']['margin_sd']:.4f} for TiRGN and {M6R['summary']['margin_sd']:.4f} for RE-GCN. For TiRGN the variation comes from the base and its control (base SD {M6T['summary']['base_sd']:.4f}; BLEND "
             f"{min(v['BLEND'] for v in M6T['seeds'].values()):.4f}..{max(v['BLEND'] for v in M6T['seeds'].values()):.4f}), while HRCOV_PAT stays within a range of "
             f"{max(v['HRCOV_PAT'] for v in M6T['seeds'].values()) - min(v['HRCOV_PAT'] for v in M6T['seeds'].values()):.4f}. The single-seed TiRGN margin reported earlier (+0.0310, seed 42) lies within the three-seed range. "
             "Completed 2026-10-15, well inside the window (stop rule: no new seed after 2026-10-19T10:00Z).\n")
# ------------------------------------------------------------------ deviations
L.append("## Deviations and clarifications\n")
for d in DEV["entries"]: L.append(f"- **{d['id']} ({d['item']}, {d['type']}).** {d['text']}" + (f" Re-check: {d['recheck']}." if d.get("recheck") else ""))
open("v76/out/v76_report.md", "w").write("\n".join(L) + "\n")
# ------------------------------------------------------------------ results json
res = dict(name="v76_results", rule=RULE, amendments=[AM1], hash_convention=HASH_CONVENTION, hash_check=HASH_CHECK, deviations=DEV["entries"], finiteness=S["finite"],
           M2b=dict(bases=M2B, cost=COST), M3a=dict(results=M3, training=S["m3a_train"], selection=S["m3a_sel"]),
           M4=dict(results=M4, recb_xf_selection=S["xfsel"]), M6=dict(regcn_tgb=M6R, tirgn=M6T if M6T else "running"))
json.dump(res, open("v76/out/v76_results.json", "w"), indent=1)
# ------------------------------------------------------------------ markers
mk = []
lm_d = [M2B[B]["contrasts"]["MLP64_PAT_fused - LMART_PAT_fused"]["delta"] for B in M2B]
ll_d = [M2B[B]["contrasts"]["LMART_PAT_fused - LIN_PAT_fused"]["delta"] for B in M2B]
lc_d = [M2B[B]["contrasts"]["LMART_PAT_fused - better control"]["delta"] for B in M2B]
mk.append(("\\pend{V76-LMART}", "MLP - LambdaMART range over 4 bases", f"{min(lm_d):+.4f} to {max(lm_d):+.4f} (all within 0.005)",
           f"A LambdaMART ranker (LightGBM lambdarank) trained on the same 25 features, candidate pool, fit/selection rows and fusion matches the 2x64 MLP within 0.005 MRR on all four bases (MLP - LambdaMART between {min(lm_d):+.4f} and {max(lm_d):+.4f}) and, like the MLP, exceeds the linear scorer ({min(ll_d):+.4f} to {max(ll_d):+.4f}) and the better recurrence control ({min(lc_d):+.4f} to {max(lc_d):+.4f}); the gain comes from a non-linear candidate-level ranker over these features rather than from the specific network."))
for B in M2B:
    r = M2B[B]["contrasts"]
    mk.append(("\\pend{V76-LMART}", f"{NAMES[B]}: LambdaMART MRR / MLP - LMART / LMART - LIN / LMART - better control",
               f"{M2B[B]['mrr']['LMART']:.4f} / {r['MLP64_PAT_fused - LMART_PAT_fused']['delta']:+.4f} {ci(r['MLP64_PAT_fused - LMART_PAT_fused'])} / {r['LMART_PAT_fused - LIN_PAT_fused']['delta']:+.4f} / {r['LMART_PAT_fused - better control']['delta']:+.4f}", ""))
p = c3["HRCOV_PAT(original base) - better control(validation-trained base)"]
mk.append(("\\pend{V76-VALBASE}", "HRCOV_PAT(original) - better control(validation-trained base), tkgl-smallpedia TiRGN", f"{p['delta']:+.4f} {ci(p)}",
           f"Retraining TiRGN on tkgl-smallpedia with the validation-fit labels (selection on validation-select only) raises the base by {c3['validation-trained base - original base']['delta']:+.4f} and the better recurrence control by {c3['better control(validation-trained) - better control(original)']['delta']:+.4f} MRR; HRCOV_PAT on the original base still exceeds the better control on the validation-trained base by {p['delta']:+.4f} {ci(p)} (vs {c3['HRCOV_PAT(original base) - better control(original base)']['delta']:+.4f} on the original base), so validation supervision of the base closes about {100 * c3['better control(validation-trained) - better control(original)']['delta'] / c3['HRCOV_PAT(original base) - better control(original base)']['delta']:.0f}% of the margin."))
mk.append(("\\pend{V76-VALBASE}", "validation-trained base MRR / better control (validation-trained)", f"{m3['valbase']:.4f} / {m3['better_ctrl_valbase']:.4f}", ""))
gp, gs = M4["polecat"]["contrasts"]["G1_xf - G1"], M4["smallpedia"]["contrasts"]["G1_xf - G1"]
mk.append(("\\pend{V76-G1XF}", "G1_xf MRR polecat / smallpedia; G1_xf - G1", f"{M4['polecat']['mrr']['G1_xf']:.4f} / {M4['smallpedia']['mrr']['G1_xf']:.4f}; {gp['delta']:+.4f} {ci(gp)} / {gs['delta']:+.4f} {ci(gs)}",
           f"With RecB's per-relation parameters cross-fitted (selected on validation-select only), G1 changes by {gp['delta']:+.4f} (tkgl-polecat) and {gs['delta']:+.4f} (tkgl-smallpedia), and the frozen base adds {min(rng_new):+.3f} to {max(rng_new):+.3f} MRR over the cross-fitted G1 (previously {min(rng_old):+.3f} to {max(rng_old):+.3f}). The loss from adding the RecB score as a feature on tkgl-smallpedia TiRGN persists with cross-fitted parameters (HR {M4['smallpedia']['contrasts']['HR19_RBxf - HR19']['delta']:+.4f}, HRcov {M4['smallpedia']['contrasts']['PAT_RBxf - PAT']['delta']:+.4f}), so it is not caused by selecting those parameters on the ranker's fit labels."))
mk.append(("\\pend{V76-G1XF}", "HRCOV - G1_xf range (5 decision bases incl. CognTKE)", f"{min(rng_new):+.4f} to {max(rng_new):+.4f} (was {min(rng_old):+.4f} to {max(rng_old):+.4f})", ""))
for F in fams:
    sm = F["summary"]
    if sm.get("margin_sd") is not None:
        mk.append(("\\pend{V76-SEEDS}", f"{F['family']}: margin mean +- SD over {sm['n_seeds']} base seeds", f"{sm['margin_mean']:.4f} +- {sm['margin_sd']:.4f} (range {sm['margin_min']:.4f}..{sm['margin_max']:.4f})", ""))
if len([F for F in fams if F["summary"].get("margin_sd") is not None]) == 2:
    a, b = fams[0]["summary"], fams[1]["summary"]
    mk.append(("\\pend{V76-SEEDS}", "suggested wording", "", f"Across three base seeds on tkgl-smallpedia the margin of HRCOV_PAT over the better recurrence control is {b['margin_mean']:.4f} +- {b['margin_sd']:.4f} for TiRGN and {a['margin_mean']:.4f} +- {a['margin_sd']:.4f} for RE-GCN (TGB configuration) (mean +- SD)."))
elif fams:
    a = fams[0]["summary"]
    mk.append(("\\pend{V76-SEEDS}", "suggested wording (partial)", "", f"Across three base seeds of RE-GCN (TGB configuration) on tkgl-smallpedia the margin of HRCOV_PAT over the better recurrence control is {a['margin_mean']:.4f} +- {a['margin_sd']:.4f} (mean +- SD; TiRGN seeds pending)."))
with open("v76/out/v76_manuscript_markers.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["hook", "quantity", "value", "suggested_wording"]); w.writerows(mk)
print("built", len(L), "report lines,", len(mk), "markers")
