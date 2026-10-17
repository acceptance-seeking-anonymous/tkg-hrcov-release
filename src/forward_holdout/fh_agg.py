#!/usr/bin/env python3
"""phase2_forward_holdout / fh_agg.py -- STAGE 2 decision of fh_decision_rule.json (GDELT25X forward holdout of HRCOV_PAT).

Written and hashed BEFORE any test number of this track existed (see freeze_manifest.json).  Implements the rule exactly:
  arms (harness exact time-aware filtered rank, strict '>', both directions, every test query; fh_blend.py --stage test):
      base (TiRGN, GDELT25X-trained), BLEND = TiRGN(+)RecB and POOL = RecB-POOL (valid-selected), HR, HRCOV_PAT
      (per-query RR averaged over seeds 0-2); RecB default / trained standalone (fh_recb.py formula, average ties)
  primary statistic: paired Poisson(1) bootstrap over TIMESTAMP BLOCKS (all queries of one test day share one weight),
      B = 2000, seed 0; delta* = sum_d w_d S_d / sum_d w_d M_d; percentile CI95, one-sided p = P*(delta* <= 0),
      one-sided 95% upper bound U = 95th percentile (strata: B = 1000)
  alongside (not decision): query-level Poisson(1) bootstrap (G2 primary, hd_agg.boot) and the G7 multinomial
      timestamp-block bootstrap (g7_cluster_bootstrap.boot_cluster)
  C1: for each control c in {BLEND, POOL}: delta_c = MRR(HRCOV_PAT) - MRR(c) >= 0.005 and block p_c < 0.05
  C2: (G2: vs the strongest blend over all bases) -- one base, so C2 == the BLEND part of C1 (recorded, not re-tested)
  C3: head stratum (gold train-window frequency >= 500): DAMAGE iff block CI95 of HRCOV_PAT - better control lies
      entirely below 0 and |delta| >= 0.001
  labels: PRACTICAL_ADVANCE | HEAD_DAMAGE | CONTROL_BETTER | BELOW_MARGIN_BOUNDED (U < 0.005) | INCONCLUSIVE (U >= 0.005)
  validity checks V1-V5 (any failure -> outcome BLOCKED, numbers reported as diagnostics only)
  python3 fh_agg.py --cfg fh_agg_cfg.json --rule fh_decision_rule.json --out fh_verdict.json
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_agg import boot, mets, load_quads, split_queries, system_block   # noqa: E402
from g7_cluster_bootstrap import boot_cluster                              # noqa: E402

CLAIM = "HRCOV_PAT"
MARGIN = 0.005
CONTROLS = ("BLEND", "POOL")


def boot_block_poisson(d, cid, B=2000, seed=0, chunk=100):
    """paired Poisson(1) bootstrap over clusters (timestamp blocks): each block gets one Poisson(1) weight per draw."""
    d = np.asarray(d, np.float64)
    u, inv = np.unique(cid, return_inverse=True); nC = len(u)
    S = np.bincount(inv, weights=d, minlength=nC); M = np.bincount(inv, minlength=nC).astype(np.float64)
    rng = np.random.default_rng(seed); bs = np.empty(B)
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        w = rng.poisson(1.0, size=(bb, nC)).astype(np.float64)
        den = w @ M
        bs[b0:b0 + bb] = np.where(den > 0, (w @ S) / np.maximum(den, 1.0), 0.0)
    lo, hi = np.percentile(bs, [2.5, 97.5]); L5, U95 = np.percentile(bs, [5, 95])
    return dict(delta=float(d.mean()), CI95=[float(lo), float(hi)], p_one_sided=float((bs <= 0).mean()),
                p_le_margin=float((bs <= MARGIN).mean()), U_one_sided_95=float(U95), L_one_sided_95=float(L5),
                B=B, n=int(len(d)), n_blocks=int(nC), method="paired Poisson(1) bootstrap over timestamp blocks")


def stats(d, ts, B=2000):
    return dict(block_poisson=boot_block_poisson(d, ts, B=B), query_poisson=boot(d, B=B), block_multinomial=boot_cluster(d, ts, B=B))


def hk(rank, k):
    return (np.asarray(rank, np.float64) <= k).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--rule", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.load(open(a.cfg)); rule = json.load(open(a.rule))
    Q = {sp: load_quads(f"{C['dd']}/{sp}.txt") for sp in ("train", "test")}
    st_ = open(f"{C['dd']}/stat.txt").read().split(); N, R = int(st_[0]), int(st_[1])
    sro = split_queries(Q["test"], R); n = len(sro); ts = sro[:, 3]
    freq = np.bincount(Q["train"][:, 0], minlength=N) + np.bincount(Q["train"][:, 2], minlength=N)
    S = np.load(C["strata"]); assert (S["sro"] == sro).all(), "strata row order"
    gf = freq[sro[:, 2]]
    V = dict(name="fh_verdict", rule_file=os.path.basename(a.rule), rule_registered_utc=rule["registered_utc"],
             rule_sha256_expected=C.get("rule_sha256"), claim_system=CLAIM, dataset="GDELT25X", base="tirgn",
             evaluator="harness exact time-aware filtered rank (all arms); RecB standalone: hd_recb formula (average ties)",
             n_test=int(n), n_timestamps=int(len(np.unique(ts))))
    # ---------------- validity checks
    Z = {k: v for k, v in np.load(C["official"]).items()}
    ap_ = json.load(open(C["apply_json"])); fs = json.load(open(C["feat_summary_test"]))
    vchk = dict(V1_hashes=C.get("hash_check_passed"),
                V2_frozen_models_reproduce_select=all(r["select_rank_agreement"] >= 0.9999 for arm in ap_["arms"].values() for r in arm.values()),
                V3_fullrow_vs_dump_test=bool(fs["verify"]["test"]["rank_agree_ok"]),
                V4_all_test_queries_scored=bool(len(Z["rr_base"]) == n and (Z["rows"] == np.arange(n)).all() and all((Z[k] > 0).all() for k in Z if k.startswith("rank_"))),
                V5_identity=all(abs(x - 1.0) < 1e-12 for x in ap_["identity_test"].values()))
    V["validity"] = vchk
    valid_ok = all(bool(x) for x in vchk.values())
    # ---------------- arms
    arms = {}; rr = {}; rk = {}
    for arm in ("base", "BLEND", "POOL"):
        rr[arm] = Z[f"rr_{arm}"].astype(np.float64); rk[arm] = Z[f"rank_{arm}"].astype(np.float64)
        arms[arm] = mets(rr[arm], rk[arm])
    for pre in ("HR", CLAIM):
        sb = system_block(Z, pre)
        names = [f"{pre}_s{s}" for s in (0, 1, 2) if f"rr_{pre}_s{s}" in Z]
        assert len(names) == 3, (pre, names)
        arms[pre] = dict(sb[0]["official"], sd=sb[0]["official_sd"], per_seed_MRR=sb[0]["per_seed_official_MRR"], seeds=3)
        rr[pre] = sb[1]
        rk[pre] = None
        for k in (1, 3, 10):
            rr[f"{pre}_H{k}"] = np.mean([hk(Z[f"rank_{nm}"], k) for nm in names], 0)
    RB = np.load(C["recb_ranks"])
    for k_, lab in (("clean_tr", "RecB_trained_clean"), ("clean_def", "RecB_default_clean"), ("faith_tr", "RecB_trained_faithful"), ("faith_def", "RecB_default_faithful")):
        r_ = RB[k_].astype(np.float64); rr[lab] = 1.0 / r_; rk[lab] = r_
        arms[lab] = mets(rr[lab], r_)
    V["arms"] = arms
    tj = json.load(open(C["blend_test_json"])); V["selected_controls"] = tj.get("selected")
    # ---------------- decision
    better = "BLEND" if rr["BLEND"].mean() >= rr["POOL"].mean() else "POOL"
    c1 = {c: stats(rr[CLAIM] - rr[c], ts) for c in CONTROLS}
    c1_pass = all(c1[c]["block_poisson"]["delta"] >= MARGIN and c1[c]["block_poisson"]["p_one_sided"] < 0.05 for c in CONTROLS)
    hm = gf >= 500
    if hm.sum() > 0:
        c3 = dict(block_poisson=boot_block_poisson((rr[CLAIM] - rr[better])[hm], ts[hm], B=1000), query_poisson=boot((rr[CLAIM] - rr[better])[hm], B=1000),
                  share=float(hm.mean()), n=int(hm.sum()))
        damage = (c3["block_poisson"]["CI95"][1] < 0) and (abs(c3["block_poisson"]["delta"]) >= 0.001)
    else:                                          # no head query: guardrail not applicable (reported)
        c3 = dict(share=0.0, n=0, not_applicable=True); damage = False
    pb = c1[better]["block_poisson"]
    if not valid_ok:
        outcome = "BLOCKED"
    elif c1_pass and not damage:
        outcome = "PRACTICAL_ADVANCE"
    elif c1_pass and damage:
        outcome = "HEAD_DAMAGE"
    elif pb["CI95"][1] < 0:
        outcome = "CONTROL_BETTER"
    elif pb["U_one_sided_95"] < MARGIN:
        outcome = "BELOW_MARGIN_BOUNDED"
    else:
        outcome = "INCONCLUSIVE"
    g2_map = {"PRACTICAL_ADVANCE": "PRACTICAL_ADVANCE", "HEAD_DAMAGE": "HEAD_DAMAGE"}.get(outcome, "RECURRENCY_DONE_WELL" if outcome != "BLOCKED" else "BLOCKED")
    hits = {}
    for k in (1, 3, 10):
        hits[f"H{k}"] = {c: boot_block_poisson(rr[f"{CLAIM}_H{k}"] - hk(rk[c], k), ts) for c in CONTROLS}
    V["decision"] = dict(better_control=better, C1=c1, C1_pass=bool(c1_pass), C2="identical to the BLEND component of C1 (single base)",
                         C3_head=c3, C3_damage=bool(damage), outcome=outcome, outcome_in_G2_vocabulary=g2_map,
                         U_vs_better_control=pb["U_one_sided_95"], hits_deltas_vs_controls=hits,
                         reading=("the forward holdout excludes an advance >= 0.005 over the better control at one-sided 95%"
                                  if outcome == "BELOW_MARGIN_BOUNDED" else None))
    # ---------------- secondary (reported, not decision)
    sec = {f"{CLAIM}_minus_HR (pool widening)": stats(rr[CLAIM] - rr["HR"], ts),
           f"HR_minus_{better}": stats(rr["HR"] - rr[better], ts),
           f"{CLAIM}_minus_RecB_trained_clean": stats(rr[CLAIM] - rr["RecB_trained_clean"], ts),
           "BLEND_minus_base": stats(rr["BLEND"] - rr["base"], ts),
           f"{CLAIM}_minus_base": stats(rr[CLAIM] - rr["base"], ts)}
    V["secondary"] = sec
    # ---------------- strata (B = 1000)
    strata = {"head (gold train freq >= 500)": hm, "mid (50-499)": (gf >= 50) & (gf < 500), "low-frequency (< 50)": gf < 50,
              "no-pair-history": ~S["dyad"], "pair-history": S["dyad"], "first-seen (s or o)": S["first_seen"],
              "exact recurrence (s,r,o) before t": S["trip"], "gold outside base top-50": Z["hrank_base"] > 50,
              "gold in base top-50": Z["hrank_base"] <= 50}
    srows = {}
    for nm, m in strata.items():
        m = np.asarray(m, bool)
        if m.sum() == 0: continue
        bs = boot_block_poisson((rr[CLAIM] - rr[better])[m], ts[m], B=1000)
        srows[nm] = dict(share=float(m.mean()), n=int(m.sum()), **{f"MRR_{k}": float(rr[k][m].mean()) for k in ("base", "BLEND", "POOL", "HR", CLAIM, "RecB_trained_clean")},
                         delta_claim_minus_better=bs["delta"], CI95=bs["CI95"], U=bs["U_one_sided_95"])
    V["strata"] = srows
    # ---------------- temporal stability (reported): 7-day blocks of consecutive test timestamps
    uts = np.unique(ts); wk = np.searchsorted(uts, ts) // 7
    d = rr[CLAIM] - rr[better]
    V["temporal"] = dict(per_week={int(w): dict(n=int((wk == w).sum()), delta=float(d[wk == w].mean())) for w in np.unique(wk)},
                         leave_one_week_out={int(w): float(d[wk != w].mean()) for w in np.unique(wk)})
    V["deviations"] = C.get("deviations", [])
    json.dump(V, open(a.out, "w"), indent=1)
    print(json.dumps(dict(outcome=outcome, better=better, delta=pb["delta"], CI95=pb["CI95"], U=pb["U_one_sided_95"], validity=vchk)))


if __name__ == "__main__":
    main()
