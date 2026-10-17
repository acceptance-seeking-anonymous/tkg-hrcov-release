#!/usr/bin/env python3
"""ablations_A / abl_agg.py -- POST-HOC ABLATION (not a decision; ablation_rule.json): per base and arm, test MRR / H@10
(seed-averaged per query), delta vs FULL (HRCOV_PAT, 25 features) with the paired Poisson(1) bootstrap (B = 2000, seed 0).
Smallpedia: official py-tgb evaluator per-query RR / rank (CPU run on serverA); polecat: exact fused rank (= the evaluator's statistic)
from abl_train.py, validated against the official base ranks and the stored official HRCOV_PAT lists."""
import json, sys, csv, argparse, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g7_cluster_bootstrap import boot_query
ARMS = ["FULL", "A10_noBase", "A11_noEvid", "A11_noCov"]
NAMES = {"FULL": "FULL (HRCOV, bridge-free, 25 features)", "A10_noBase": "A10 = A11 minus base (evidence + coverage, 17)",
         "A11_noEvid": "A11 minus evidence (base + coverage, 14)", "A11_noCov": "A11 minus coverage (base + evidence, 19)"}
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True); ap.add_argument("--csv", required=True)
a = ap.parse_args(); C = json.load(open(a.cfg))
OUT = dict(name="ablations_A", label="POST-HOC ABLATION (not a decision)", rule=C["rule"], B=2000, seed=0, bases={})
rows = []
for key, B in C["bases"].items():
    Rk = np.load(B["ranks"]); rec = dict(dataset=B["dataset"], base=B["base"], evaluator=B["evaluator"], arms={}, validation={})
    Of = np.load(B["official"]) if B.get("official") else None
    def rr_hit(arm):
        if Of is not None:
            rr = np.mean([Of[f"rr_{arm}_s{s}"] for s in (0, 1, 2)], 0); h = np.mean([Of[f"rank_{arm}_s{s}"] <= 10 for s in (0, 1, 2)], 0)
        else:
            rr = np.mean([1.0 / Rk[f"{arm}_s{s}"] for s in (0, 1, 2)], 0); h = np.mean([Rk[f"{arm}_s{s}"] <= 10 for s in (0, 1, 2)], 0)
        return rr.astype(np.float64), h.astype(np.float64)
    R = {arm: rr_hit(arm) for arm in ARMS}
    full_rr, full_h = R["FULL"]
    for arm in ARMS:
        rr, h = R[arm]
        r_ = dict(MRR=float(rr.mean()), H10=float(h.mean()), per_seed_MRR=[float(np.mean(Of[f"rr_{arm}_s{s}"]) if Of is not None else np.mean(1.0 / Rk[f"{arm}_s{s}"])) for s in (0, 1, 2)])
        if arm != "FULL":
            bq = boot_query(rr - full_rr); r_["delta_MRR_vs_FULL"] = bq["delta"]; r_["CI95"] = bq["CI95"]; r_["p_one_sided_le0"] = bq["p_one_sided"]
            r_["delta_H10_vs_FULL"] = float(h.mean() - full_h.mean())
        if Of is not None:
            hr = np.mean([1.0 / Rk[f"{arm}_s{s}"] for s in (0, 1, 2)], 0); r_["harness_minus_official_MRR"] = float(hr.mean() - rr.mean())
            r_["max_abs_per_query_rr_diff"] = float(np.abs(hr - rr).max())
        rec["arms"][arm] = r_
        rows.append([B["dataset"], B["base"], NAMES[arm], f"{r_['MRR']:.5f}", f"{r_['H10']:.5f}",
                     "" if arm == "FULL" else f"{r_['delta_MRR_vs_FULL']:+.5f}", "" if arm == "FULL" else f"{r_['CI95'][0]:+.5f}",
                     "" if arm == "FULL" else f"{r_['CI95'][1]:+.5f}", "" if arm == "FULL" else f"{r_['delta_H10_vs_FULL']:+.5f}", B["evaluator"]])
    rec["arms"]["A11_minus_base"] = dict(same_as="A10_noBase")
    # validation
    if B.get("official_base"):
        Ob = np.load(B["official_base"]); rb = Rk["base"].astype(np.float64); ob = Ob["rank_base"].astype(np.float64)
        rec["validation"]["base_rank_vs_official"] = dict(frac_equal=float(np.mean(np.abs(rb - ob) < 1e-6)), max_abs=float(np.abs(rb - ob).max()),
                                                          MRR_harness=float(np.mean(1 / rb)), MRR_official=float(np.mean(Ob["rr_base"])))
    if B.get("official_full_prev"):
        Op = np.load(B["official_full_prev"]); prev = np.mean([Op[f"rr_HRCOV_PAT_s{s}"] for s in (0, 1, 2)], 0)
        hr = np.mean([1.0 / Rk[f"FULL_s{s}"] for s in (0, 1, 2)], 0)
        rec["validation"]["FULL_vs_stored_official_HRCOV_PAT"] = dict(MRR_this=float(hr.mean()), MRR_stored_official=float(prev.mean()),
                                                                      frac_queries_equal=float(np.mean(np.abs(hr - prev) < 1e-6)), max_abs=float(np.abs(hr - prev).max()))
    OUT["bases"][key] = rec
json.dump(OUT, open(a.out, "w"), indent=1)
with open(a.csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "base", "arm", "test_MRR", "test_H10", "delta_MRR_vs_FULL", "CI95_lo", "CI95_hi", "delta_H10_vs_FULL", "evaluator"])
    w.writerows(rows)
print(json.dumps({k: {a_: (round(v["MRR"], 5), round(v.get("delta_MRR_vs_FULL", 0), 5)) for a_, v in b["arms"].items() if "MRR" in v} for k, b in OUT["bases"].items()}, indent=1))
