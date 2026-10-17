#!/usr/bin/env python3
"""G8 / g8_agg.py -- decision of g8_rule.json (+ g8_deviation_B.json) from the official-evaluator per-query reciprocal ranks.
  python3 g8_agg.py --cfg g8_agg_cfg.json --out g8_verdict.json --csv g8_table.csv
Decision: HRCOV = HRCOV_PAT (bridge-free, seeds 0-2, per-query RR averaged over seeds). C1 vs each control (BLEND, POOL):
delta >= 0.005 and one-sided paired Poisson bootstrap p < 0.05 (B = 2000, seed 0). C2 head guardrail vs the better control on
test (gold train freq >= 500; B = 1000): DAMAGE iff CI95 entirely < 0 and |delta| >= 0.001. Fidelity: |MRR(base) - 0.287| <= 0.010.
Secondary (not in the decision): HRCOV - CognTKE; HRCOV - HRCOV_TiRGN (HRCOV_PAT and with bridge); HR rows; timestamp-block
cluster bootstrap for every decision delta; unreached-gold stratum; strata; -inf base sensitivity; evaluator agreement."""
import os, sys, json, argparse, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g7_cluster_bootstrap import boot_query, boot_cluster, split_order, load_quads
MARGIN = 0.005; PUB = 0.287; FID_TOL = 0.010

def mrr(x): return float(np.mean(x))
def seedavg(Z, arm): return np.mean([Z[f"rr_{arm}_s{s}"] for s in (0, 1, 2)], 0)
def hits(Z, arm, k, seeds=False):
    if seeds: return float(np.mean([np.mean(Z[f"rank_{arm}_s{s}"] <= k) for s in (0, 1, 2)]))
    return float(np.mean(Z[f"rank_{arm}"] <= k))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True); ap.add_argument("--csv", required=True)
    a = ap.parse_args(); C = json.load(open(a.cfg))
    Z = np.load(C["official"]); n = len(Z["rr_base"])
    q = {sp: load_quads(f"{C['dd']}/{sp}.txt") for sp in ("train", "test")}
    ts, fid, gold = split_order(q["test"]); assert len(ts) == n
    N = int(open(f"{C['dd']}/stat.txt").read().split()[0])
    freq = np.bincount(q["train"][:, 0], minlength=N) + np.bincount(q["train"][:, 2], minlength=N); gf = freq[gold]
    S = np.load(C["strata"]); assert (S["sro"][:, 2] == gold).all() and (S["sro"][:, 3] == ts).all(), "strata order"
    reach = np.load(C["reach_test"]).astype(bool); assert len(reach) == n
    arms = dict(base=Z["rr_base"], BLEND=Z["rr_BLEND"], POOL=Z["rr_POOL"], HR=seedavg(Z, "HR"), HRCOV=seedavg(Z, "HRCOV_PAT"),
                HRCOV_BRIDGE=seedavg(Z, "HRCOV"))
    M = {k: mrr(v) for k, v in arms.items()}
    better = "BLEND" if M["BLEND"] >= M["POOL"] else "POOL"
    out = dict(name="g8_verdict", rule=C["rule"], deviation=C["deviation"], dataset="tkgl-polecat", n_test=int(n),
               evaluator="official py-tgb 2.3.0 Evaluator per query (ties 0.5); HRCOV = HRCOV_PAT seeds 0-2 averaged per query",
               mrr=M, per_seed={k: [mrr(Z[f"rr_{k}_s{s}"]) for s in (0, 1, 2)] for k in ("HR", "HRCOV_PAT", "HRCOV")},
               hits={"base": {f"H{k}": hits(Z, "base", k) for k in (1, 3, 10)}, "BLEND": {f"H{k}": hits(Z, "BLEND", k) for k in (1, 3, 10)},
                     "POOL": {f"H{k}": hits(Z, "POOL", k) for k in (1, 3, 10)},
                     "HR": {f"H{k}": hits(Z, "HR", k, True) for k in (1, 3, 10)}, "HRCOV": {f"H{k}": hits(Z, "HRCOV_PAT", k, True) for k in (1, 3, 10)}},
               harness_mrr={k: float(np.mean(1.0 / Z[f"hrank_{k}"])) for k in ("base", "BLEND", "POOL")},
               better_control=better)
    # fidelity
    out["fidelity"] = dict(published=PUB, ours=M["base"], abs_diff=abs(M["base"] - PUB), tol=FID_TOL,
                           flag="BASE_FIDELITY_OK" if abs(M["base"] - PUB) <= FID_TOL else "BASE_FIDELITY_LOW",
                           published_H10=0.455, ours_H10=out["hits"]["base"]["H10"])
    # C1
    c1 = {}
    for c in ("BLEND", "POOL"):
        d = arms["HRCOV"] - arms[c]; bq = boot_query(d); bt = boot_cluster(d, ts)
        c1[c] = dict(query=bq, ts_block=bt, PASS=bool(bq["delta"] >= MARGIN and bq["p_one_sided"] < 0.05),
                     PASS_ts_block=bool(bt["delta"] >= MARGIN and bt["p_one_sided"] < 0.05))
    head = gf >= 500
    dh = (arms["HRCOV"] - arms[better])[head]; bh = boot_query(dh, B=1000)
    damage = bool(bh["CI95"][1] < 0 and abs(bh["delta"]) >= 0.001)
    c1_pass = all(v["PASS"] for v in c1.values())
    label = "PRACTICAL_ADVANCE_ON_COGNTKE" if (c1_pass and not damage) else ("HEAD_DAMAGE" if c1_pass else "RECURRENCY_DONE_WELL")
    out["decision"] = dict(C1=c1, C1_PASS=c1_pass, C2_head=dict(vs=better, n_head=int(head.sum()), **bh, DAMAGE=damage),
                           label=label, label_ts_block_sensitivity=("PRACTICAL_ADVANCE_ON_COGNTKE" if all(v["PASS_ts_block"] for v in c1.values()) and not damage else "differs"))
    # secondary
    sec = {}
    for nm, other in (("HRCOV_minus_CognTKE", arms["base"]), ("HR_minus_CognTKE", arms["base"]), ("HR_minus_better_control", arms[better]),
                      ("HRCOV_minus_HR", arms["HR"]), ("HRCOV_BRIDGE_minus_HRCOV", arms["HRCOV"])):
        lhs = arms["HR"] if nm.startswith("HR_") else (arms["HRCOV_BRIDGE"] if nm.startswith("HRCOV_BRIDGE") else arms["HRCOV"])
        d = lhs - other; sec[nm] = dict(query=boot_query(d), ts_block=boot_cluster(d, ts))
    T1 = np.load(C["tirgn_hcpat_official"]); T2 = np.load(C["tirgn_round1_official"])
    hrcov_tirgn_pat = np.mean([T1[f"rr_HRCOV_PAT_s{s}"] for s in (0, 1, 2)], 0); hrcov_tirgn_br = np.mean([T2[f"rr_HRCOV_s{s}"] for s in (0, 1, 2)], 0)
    for nm, oth in (("HRCOV_minus_HRCOV_TiRGN_PAT", hrcov_tirgn_pat), ("HRCOV_minus_HRCOV_TiRGN_bridge", hrcov_tirgn_br),
                    ("CognTKE_minus_TiRGN_base", T2["rr_base"])):
        lhs = arms["base"] if nm.startswith("CognTKE") else arms["HRCOV"]
        d = lhs - oth; sec[nm] = dict(other_MRR=mrr(oth), query=boot_query(d), ts_block=boot_cluster(d, ts))
    # unreached stratum (gold outside CognTKE's reasoning subgraph: logit exactly 0)
    un = ~reach
    sec["unreached_gold"] = dict(share_test=float(un.mean()), n=int(un.sum()),
                                 MRR={k: float(v[un].mean()) for k, v in arms.items()},
                                 HRCOV_minus_better_control=boot_query((arms["HRCOV"] - arms[better])[un], B=1000),
                                 HRCOV_minus_CognTKE=boot_query((arms["HRCOV"] - arms["base"])[un], B=1000),
                                 reached_HRCOV_minus_better_control=boot_query((arms["HRCOV"] - arms[better])[~un], B=1000),
                                 reached_MRR={k: float(v[~un].mean()) for k, v in arms.items()})
    # strata
    rk_base = Z["rank_base"]
    strata = {"head (gold train freq >= 500)": head, "mid (50-499)": (gf >= 50) & (gf < 500), "cold (< 50)": gf < 50,
              "pool-absent (gold outside CognTKE top-50)": rk_base > 50, "first-seen": S["first_seen"].astype(bool),
              "no-pair-history": ~S["dyad"].astype(bool), "unreached": un}
    sec["strata"] = {k: dict(share=float(m.mean()), MRR={a_: float(arms[a_][m].mean()) for a_ in ("base", better, "HR", "HRCOV")},
                             HRCOV_minus_better_control=boot_query((arms["HRCOV"] - arms[better])[m], B=1000)) for k, m in strata.items()}
    if os.path.exists(C.get("ninf_official", "")):
        Zn = np.load(C["ninf_official"]); sec["base_ninf_unreached"] = dict(MRR=mrr(Zn["rr_base"]), H10=float(np.mean(Zn["rank_base"] <= 10)),
                                                                          minus_primary=mrr(Zn["rr_base"]) - M["base"])
    out["secondary"] = sec
    meta = json.load(open(C["blend_test_meta"])); out["evaluator_max_dev"] = meta.get("max_dev", meta.get("evaluator_max_dev", meta.get("official_vs_formula_max_batch_dev")))
    out["references"] = C.get("references", {})
    json.dump(out, open(a.out, "w"), indent=1)
    rows = [("CognTKE (ours, README cfg, valid-best epoch)", M["base"], out["hits"]["base"]["H10"]),
            (f"CognTKE + RecB blend (valid-selected)", M["BLEND"], out["hits"]["BLEND"]["H10"]),
            (f"CognTKE + RecB-POOL (valid-selected)", M["POOL"], out["hits"]["POOL"]["H10"]),
            ("CognTKE + HR (seeds 0-2)", M["HR"], out["hits"]["HR"]["H10"]),
            ("CognTKE + HRCOV (bridge-free, seeds 0-2)", M["HRCOV"], out["hits"]["HRCOV"]["H10"]),
            ("CognTKE + HRCOV with bridge (secondary)", M["HRCOV_BRIDGE"], float("nan"))]
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "test_MRR_official", "test_H10_official"])
        for r in rows: w.writerow([r[0], f"{r[1]:.5f}", "" if np.isnan(r[2]) else f"{r[2]:.5f}"])
    print(json.dumps(dict(label=label, mrr=M, fidelity=out["fidelity"]["flag"]), indent=1))

if __name__ == "__main__":
    main()
