"""Verbatim local notebook cell(s), session-archive index 650, 652, 654, 656 (local python kernel, workspace root as cwd).
Read-back of verdict/results; rank cross-checks (pipeline ranks vs scorer strict ranks);
checks + latency table merged into fresh_window_results.json. Cell 652 raised (numpy not yet imported); 654 is the same code with the import.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 650
V_ = json.load(open("fw/out/fresh_window_verdict.json")); R_ = json.load(open("fw/out/fresh_window_results.json"))
for k, m in V_["margins"].items(): print(k, round(m["point"], 5), "q", [round(x, 5) for x in (m["query"]["L"], m["query"]["U"])], m["query"]["CI95"] and [round(x,5) for x in m["query"]["CI95"]], "ts", [round(x, 5) for x in (m["ts_block"]["L"], m["ts_block"]["U"])], [round(x,5) for x in m["ts_block"]["CI95"]])
for k, m in R_["metrics"].items(): print(k, {kk: round(v, 4) for kk, v in m["official_ties_averaged"].items() if kk != "per_seed_MRR"}, m["official_ties_averaged"].get("per_seed_MRR"), "strict MRR", round(m["harness_strict"]["MRR"], 4))
print(V_["checks"]); 
d = R_["deltas_official"]
for k in ("SA-external_comparator", "BLEND-base"): print(k, round(d[k]["delta"], 5), [round(x, 5) for x in d[k]["query"]["CI95"]], [round(x, 5) for x in d[k]["ts"]["CI95"]])
for k, v in R_["deltas_harness_strict"].items(): print("strict", k, round(v["delta"], 5), round(v["query"]["L"], 5), round(v["ts"]["L"], 5), round(v["query"]["U"], 5), round(v["ts"]["U"], 5))
for w, v in R_["weekly_deltas"].items(): print(w, v["days"], v["n"], round(v["HRcov-SA"], 4), round(v["HRcov-external_comparator"], 4), round(v["HRcov-BLEND"], 4))
# ===== cell 652
d2 = "hpc/7349980c-57f6-49ec-a9d5-65fb82f265fb/h"
T = np.load(f"{d2}/official_test_gdeltfw_tirgn.npz"); S_ = np.load(f"{d2}/official_test_gdeltfw_sa.npz"); E_ = np.load(f"{d2}/official_test_gdeltfw_ext.npz")
RH = np.load(f"{d2}/ranks_test.npz"); RS = np.load(f"{d2}/ranks_test_SA_W100.npz")
chk2 = {}
for s in (0, 1, 2):
    a_ = RH[f"HRCOV_PAT_s{s}"].astype(np.float64); b_ = T[f"hrank_HRCOV_PAT_s{s}"].astype(np.float64); c_ = T[f"rank_HRCOV_PAT_s{s}"].astype(np.float64)
    chk2[f"HRcov_s{s}"] = dict(pipeline_vs_scorer_strict_max_abs=float(np.abs(a_ - b_).max()), frac_equal=float((np.abs(a_ - b_) < 1e-3).mean()),
                               official_minus_strict_mean=float((c_ - b_).mean()))
    a_ = RS[f"SA_W100_s{s}"].astype(np.float64); b_ = S_[f"hrank_SA_W100_s{s}"].astype(np.float64); c_ = S_[f"rank_SA_W100_s{s}"].astype(np.float64)
    chk2[f"SA_s{s}"] = dict(pipeline_vs_scorer_strict_max_abs=float(np.abs(a_ - b_).max()), frac_equal=float((np.abs(a_ - b_) < 1e-3).mean()),
                            official_minus_strict_mean=float((c_ - b_).mean()))
b0 = RH["base"].astype(np.float64); chk2["base_fh_apply_vs_scorer_strict_max_abs"] = float(np.abs(b0 - T["hrank_base"]).max())
chk2["base_official_minus_strict_mean"] = float((T["rank_base"].astype(np.float64) - T["hrank_base"]).mean())
chk2["EXT_official_minus_strict_mean"] = float((E_["rank_EXT"].astype(np.float64) - E_["hrank_EXT"]).mean())
chk2["EXT_frac_queries_with_ties_at_gold"] = float((E_["rank_EXT"] != E_["hrank_EXT"]).mean())
apj = json.load(open(f"{d2}/apply_fh.json")); saj = json.load(open(f"{d2}/train_fw_sa.json"))
chk2["HRcov_select_rank_agreement"] = {s: r["select_rank_agreement"] for s, r in apj["arms"]["HRCOV_PAT"].items()}
chk2["SA_identity_tail_order_test"] = saj["arms"]["SA_W100"].get("identity_tail_order_test")
chk2["HRcov_test_frac_queries_with_promotion"] = {s: r.get("test_frac_queries_with_promotion") for s, r in apj["arms"]["HRCOV_PAT"].items()}
chk2["SA_test_frac_with_promotion"] = {s: r.get("test_frac_with_promotion") for s, r in saj["arms"]["SA_W100"]["seeds"].items()}
print(json.dumps(chk2, indent=0)[:2500])
# ===== cell 654
import numpy as np
T = np.load(f"{d2}/official_test_gdeltfw_tirgn.npz"); S_ = np.load(f"{d2}/official_test_gdeltfw_sa.npz"); E_ = np.load(f"{d2}/official_test_gdeltfw_ext.npz")
RH = np.load(f"{d2}/ranks_test.npz"); RS = np.load(f"{d2}/ranks_test_SA_W100.npz")
chk2 = {}
for s in (0, 1, 2):
    a_ = RH[f"HRCOV_PAT_s{s}"].astype(np.float64); b_ = T[f"hrank_HRCOV_PAT_s{s}"].astype(np.float64); c_ = T[f"rank_HRCOV_PAT_s{s}"].astype(np.float64)
    chk2[f"HRcov_s{s}"] = dict(pipeline_vs_scorer_strict_max_abs=float(np.abs(a_ - b_).max()), frac_equal=float((np.abs(a_ - b_) < 1e-3).mean()),
                               official_minus_strict_mean=float((c_ - b_).mean()))
    a_ = RS[f"SA_W100_s{s}"].astype(np.float64); b_ = S_[f"hrank_SA_W100_s{s}"].astype(np.float64); c_ = S_[f"rank_SA_W100_s{s}"].astype(np.float64)
    chk2[f"SA_s{s}"] = dict(pipeline_vs_scorer_strict_max_abs=float(np.abs(a_ - b_).max()), frac_equal=float((np.abs(a_ - b_) < 1e-3).mean()),
                            official_minus_strict_mean=float((c_ - b_).mean()))
b0 = RH["base"].astype(np.float64); chk2["base_fh_apply_vs_scorer_strict_max_abs"] = float(np.abs(b0 - T["hrank_base"]).max())
chk2["base_official_minus_strict_mean"] = float((T["rank_base"].astype(np.float64) - T["hrank_base"]).mean())
chk2["EXT_official_minus_strict_mean"] = float((E_["rank_EXT"].astype(np.float64) - E_["hrank_EXT"]).mean())
chk2["EXT_frac_queries_with_ties_at_gold"] = float((E_["rank_EXT"] != E_["hrank_EXT"]).mean())
apj = json.load(open(f"{d2}/apply_fh.json")); saj = json.load(open(f"{d2}/train_fw_sa.json"))
chk2["HRcov_select_rank_agreement"] = {s: r["select_rank_agreement"] for s, r in apj["arms"]["HRCOV_PAT"].items()}
chk2["SA_identity_tail_order_test"] = saj["arms"]["SA_W100"].get("identity_tail_order_test")
chk2["HRcov_test_frac_queries_with_promotion"] = {s: r.get("test_frac_queries_with_promotion") for s, r in apj["arms"]["HRCOV_PAT"].items()}
chk2["SA_test_frac_with_promotion"] = {s: r.get("test_frac_with_promotion") for s, r in saj["arms"]["SA_W100"]["seeds"].items()}
print(json.dumps(chk2, indent=0)[:2500])
# ===== cell 656
s2log = open(f"{d2}/stage2_test.log").read()
R_["checks"].update(chk2)
R_["checks"].update(dict(freeze_verification="stage2_verify: 201 files, 0 problems (rc 0, 2026-10-16T16:34:01Z)",
    recb_valid_csr_identical_to_frozen=True, recb_xf_valid_csr_identical_to_frozen=True,
    countrucola_validation_reapplication_max_abs_rr_dev=0.0, countrucola_test_queries_without_ranking_line=0,
    countrucola_stage2_main_rc=1, countrucola_stage2_main_rc_note="released main.py raised in utils.write_results_csv (pandas fillna dtype) AFTER apply finished; rankings complete; no rerun",
    evaluator_formula_vs_vendored_tgb_Evaluator_max_abs=9.934107481068821e-09))
R_["latency_table_s_per_1000_queries"] = LT.to_dict(orient="records")
R_["latency_raw"] = dict(systems=L, countrucola=CL)
R_["stage2_chronology_utc"] = [["2026-10-16T16:33:56Z", "Stage 2 job 7349980c starts"], ["2026-10-16T16:34:01Z", "freeze verified"],
    ["2026-10-16T16:34:15Z", "TiRGN test rows (GPU 0, 14 s)"], ["2026-10-16T16:39:51Z", "HRcov frozen re-ranker applied to test"],
    ["2026-10-16T16:40:09Z", "SA frozen ranker applied + scored"], ["2026-10-16T16:40:26Z", "base / BLEND / HRcov scored (fh_blend)"],
    ["2026-10-16T17:09:43Z", "CountTRuCoLa frozen rules applied to test + validation"], ["2026-10-16T17:10:22Z", "comparator scored (fw_ext); Stage 2 ends"],
    [V_["decided_utc"], "decision computed (fw_agg.py, once)"]]
R_["systems"] = {"base": "TiRGN (forward-holdout recipe, best epoch 11)", "BLEND": "TiRGN (+) RecB, MIX beta 0.3 default-lambda (validation-selected; reference row)",
                 "HRcov": "HRCOV_PAT, seeds 0-2 (per-query RR averaged)", "SA": "SA_W100, seeds 0-2 (per-query RR averaged)",
                 "external_comparator": "CountTRuCoLa C0 (released default config) blended with TiRGN, a 0.1, tau 0.5 (selection-part MRR 0.40772 vs alone 0.36841)"}
json.dump(R_, open("fw/out/fresh_window_results.json", "w"), indent=1)
V_["checks"] = dict(V_["checks"], pipeline_ranks_equal_scorer=True, freeze_verified=True, countrucola_validation_reproduced=True)
V_["interpretation"] = ("R4_OTHER (amendment 1): HRcov BEATS the CountTRuCoLa comparator; versus SA the call is SMALL - the gain is positive under both "
                        "bootstraps but below the 0.005 bar, and MATCHES is not established because the timestamp-block upper bound (0.0053) exceeds 0.005. "
                        "Reported as mixed in the main text with all calls and intervals.")
V_["deviations"] = [
 "CountTRuCoLa configuration: no released entry for GDELTFW; the released gdelt/polecat entries fail the code's own window check; candidates C1/C2 (window 50) stopped for memory/time before any rule or score existed (addendum 1); comparator from the released default configuration C0.",
 "CountTRuCoLa built-in evaluation disabled (no TGB negative files); dataset object supplied by a harness-backed adapter; otherwise released code unchanged.",
 "CountTRuCoLa Stage-2 main.py exited rc=1 in the released results-CSV writer after rule application had finished; rankings complete and validation re-application exact; not rerun.",
 "Only the selected comparator (C0 blend) was applied to test; CountTRuCoLa alone is reported on validation only.",
 "Two launches failed before computing anything (CountTRuCoLa dataset adapter; SA chain cut parse) and were relaunched with the fix; no data of any split had been scored by them.",
 "GPU 0 was shared with another track's process (1.7-10 GB) during Stage 1 and latency; this track's processes were capped at 30 GB; GPU 1 not used."]
json.dump(V_, open("fw/out/fresh_window_verdict.json", "w"), indent=1)
print(V_["outcome"], V_["calls"])
