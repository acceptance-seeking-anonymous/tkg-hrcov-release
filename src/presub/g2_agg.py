#!/usr/bin/env python3
"""presub / g2_agg.py -- G2 decision of a_presub_rule.json (HRCOV_PAT vs recurrency used well, ICEWS18 / GDELT).

The hardening decision family (hd_agg.py) with the G2 claim system and evaluator:
  * every arm's per-query RR = 1 / harness exact filtered rank (hd_blend.py, official=False datasets), re-rankers averaged
    over seeds 0-2 per query; RecB reference rows from hd_recb.py (its formula: average ties, harness filter)
  * two strongest bases by valid MRR (full-row harness ranks on the valid rows of hd_blend --stage valid)
  * C1: HRCOV_PAT_b - each of {b(+)RecB, RecB-POOL_b} >= 0.005 with one-sided p < 0.05
    C2: HRCOV_PAT_b - max over all bases b'(+)RecB (by test MRR) > 0 with p < 0.05
    C3: head stratum (gold train frequency >= 500): no DAMAGE (95% CI entirely below 0 and |delta| >= 0.001)
    labels PRACTICAL_ADVANCE / HEAD_DAMAGE / RECURRENCY_DONE_WELL
  * timestamp-block cluster bootstrap (G7 method) reported alongside every decision delta (not part of the label)
  * secondary: HRCOV_PAT - HR (pool widening), HR - better control, HRCOV_PAT - RecB trained (clean);
    HR(TiRGN) vs residual arm B (must be within 0.002, else the discrepancy is reported); full-row verification
    (feat summaries) carried through.
  python3 g2_agg.py --cfg g2_agg_cfg.json --rule a_presub_rule.json --out g2_verdict.json [--smoke n]
cfg: {DS: {name, dd, strata, recb_ranks, recb_sel, residual_hr_tirgn, bases: {b: {official, test_json, valid_json,
      feat_summary}}}}
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_agg import boot, mets, hmets, load_quads, split_queries, system_block, MARGIN
from g7_cluster_bootstrap import boot_cluster

CLAIM = "HRCOV_PAT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--rule", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="code check: first n test rows of every array")
    a = ap.parse_args()
    cut = (lambda x: np.asarray(x)[:a.smoke]) if a.smoke else (lambda x: np.asarray(x))
    C = json.load(open(a.cfg)); rule = json.load(open(a.rule))
    V = dict(name="g2_verdict", rule_file=os.path.basename(a.rule), rule_registered_utc=rule["registered_utc"],
             evaluator="harness exact time-aware filtered rank (all arms)", claim_system=CLAIM, smoke=int(a.smoke), datasets={})
    for ds, D in C.items():
        Q = {sp: load_quads(f"{D['dd']}/{sp}.txt") for sp in ("train", "test")}
        st = open(f"{D['dd']}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
        sro = cut(split_queries(Q["test"], R)); n = len(sro)
        ts = sro[:, 3]
        freq = np.bincount(Q["train"][:, 0], minlength=N) + np.bincount(Q["train"][:, 2], minlength=N)
        S = np.load(D["strata"]); assert (cut(S["sro"]) == sro).all(), "strata row order"
        gf = freq[sro[:, 2]]
        strata = {"head (gold train freq >= 500)": gf >= 500, "mid (50-499)": (gf >= 50) & (gf < 500), "cold (< 50)": gf < 50,
                  "no-pair-history": ~cut(S["dyad"]), "pair-history": cut(S["dyad"]), "first-seen (s or o)": cut(S["first_seen"]),
                  "exact recurrence (s,r,o) before t": cut(S["trip"])}
        out = dict(name=D["name"], n_test=int(n), n_timestamps=int(len(np.unique(ts))), bases={}, rows=[])
        RB = np.load(D["recb_ranks"]); rsel = json.load(open(D["recb_sel"]))
        recb_rows = {}
        for k, lab in (("faith_def", "RecB default (TGB script, faithful)"), ("faith_tr", "RecB trained (TGB train mode, faithful)"),
                       ("clean_def", "RecB default (clean psi reset)"), ("clean_tr", "RecB trained (clean psi reset)")):
            rk = cut(RB[k]).astype(np.float64)
            recb_rows[k] = dict(label=lab, **mets(1.0 / rk, rk))
            out["rows"].append(dict(base="-", arm=lab, evaluator="hd_recb formula (average ties, harness filter)", seeds="-",
                                    **{k2: recb_rows[k][k2] for k2 in ("MRR", "H1", "H3", "H10")}))
        out["recb"] = dict(rows=recb_rows, valid=dict(faith_tr=rsel["faith"]["valid_MRR_trained"], clean_tr=rsel["clean"]["valid_MRR_trained"],
                                                      faith_def=rsel["faith"]["valid_MRR_default"], clean_def=rsel["clean"]["valid_MRR_default"]))
        rr_recb_tr = 1.0 / cut(RB["clean_tr"]).astype(np.float64)
        blends = {}
        for b, B in D["bases"].items():
            if not os.path.exists(B["official"]):
                out["bases"][b] = dict(missing=B["official"]); continue
            Z = {k: cut(v) for k, v in np.load(B["official"]).items()}
            tj = json.load(open(B["test_json"])); vj = json.load(open(B["valid_json"]))
            assert len(Z["rr_base"]) == n and (Z["rows"] == np.arange(n)).all()
            rec = dict(host=tj.get("host"), valid_MRR_base=vj["valid_MRR_base"], n_valid=vj.get("n"), selected=tj.get("selected"),
                       evaluator=tj.get("evaluator"), valid_MRR_blend_selected=max(vj["valid_MRR_blend"]),
                       valid_MRR_pool_selected=max(vj["valid_MRR_pool"]), arms={})
            if B.get("feat_summary") and os.path.exists(B["feat_summary"]):
                rec["fullrow_verification"] = json.load(open(B["feat_summary"])).get("verify")
            rr = {}
            for arm in ("base", "BLEND", "POOL"):
                rec["arms"][arm] = dict(harness=mets(Z[f"rr_{arm}"], Z[f"rank_{arm}"]))
                rr[arm] = np.asarray(Z[f"rr_{arm}"], np.float64)
            for pre in ("HR", CLAIM):
                sb = system_block(Z, pre)
                if sb:
                    rec["arms"][pre] = dict(harness=sb[0]["official"], harness_sd=sb[0]["official_sd"], seeds=sb[0]["seeds"],
                                            per_seed_MRR=sb[0]["per_seed_official_MRR"])
                    rr[pre] = sb[1]
            rec["_rr"] = rr; rec["_hrank_base"] = Z["hrank_base"]
            blends[b] = rr["BLEND"]
            out["bases"][b] = rec
        ok_b = [b for b, r in out["bases"].items() if "missing" not in r and CLAIM in r["arms"]]
        two = sorted(ok_b, key=lambda b: -out["bases"][b]["valid_MRR_base"])[:2]
        out["valid_MRR_ranking"] = {b: out["bases"][b]["valid_MRR_base"] for b in sorted(ok_b, key=lambda b: -out["bases"][b]["valid_MRR_base"])}
        out["two_strongest"] = two
        bb = max(blends, key=lambda b: blends[b].mean()) if blends else None
        out["strongest_blend_overall"] = dict(base=bb, MRR=float(blends[bb].mean()) if bb else None)
        dec = {}
        for b in two:
            rec = out["bases"][b]; rr = rec["_rr"]
            better = "BLEND" if rr["BLEND"].mean() >= rr["POOL"].mean() else "POOL"
            dd_ = {f"C1 {CLAIM} - BASE(+)RecB": rr[CLAIM] - rr["BLEND"], f"C1 {CLAIM} - RecB-POOL": rr[CLAIM] - rr["POOL"],
                   f"C2 {CLAIM} - {bb}(+)RecB": rr[CLAIM] - blends[bb]}
            c1 = {c: boot(rr[CLAIM] - rr[c]) for c in ("BLEND", "POOL")}
            c1_pass = all(v["delta"] >= MARGIN and v["p_one_sided"] < 0.05 for v in c1.values())
            c2 = boot(rr[CLAIM] - blends[bb])
            c2_pass = c2["delta"] > 0 and c2["p_one_sided"] < 0.05
            hm = strata["head (gold train freq >= 500)"]
            c3 = boot(rr[CLAIM][hm] - rr[better][hm], B=1000)
            damage = (c3["CI95"][1] < 0) and (abs(c3["delta"]) >= 0.001)
            tsb = {k: boot_cluster(v, ts) for k, v in dd_.items()}
            tsb[f"C3 head {CLAIM} - {better}"] = boot_cluster((rr[CLAIM] - rr[better])[hm], ts[hm])
            dec[b] = dict(better_comparator=better, C1=c1, C1_pass=bool(c1_pass), C2_vs=f"{bb}(+)RecB", C2=c2, C2_pass=bool(c2_pass),
                          C3_head=c3, C3_damage=bool(damage), PASS=bool(c1_pass and c2_pass and not damage), ts_block_bootstrap=tsb)
            sec = {}
            if "HR" in rr:
                sec[f"{CLAIM}_minus_HR (pool widening)"] = boot(rr[CLAIM] - rr["HR"])
                sec[f"HR_minus_{better}"] = boot(rr["HR"] - rr[better])
            sec[f"{CLAIM}_minus_RecB_trained_clean"] = boot(rr[CLAIM] - rr_recb_tr)
            sec["BLEND_minus_base"] = boot(rr["BLEND"] - rr["base"])
            dec[b]["secondary"] = sec
            pa = rec["_hrank_base"] > 50
            strat = dict(strata); strat["pool-absent (gold outside base top-50)"] = pa; strat["gold in base top-50"] = ~pa
            srows = {}
            for nm, m in strat.items():
                if m.sum() == 0: continue
                bs = boot(rr[CLAIM][m] - rr[better][m], B=1000)
                srows[nm] = dict(share=float(m.mean()), n=int(m.sum()), MRR_base=float(rr["base"][m].mean()), MRR_RecB_tr=float(rr_recb_tr[m].mean()),
                                 MRR_BLEND=float(rr["BLEND"][m].mean()), MRR_POOL=float(rr["POOL"][m].mean()),
                                 MRR_HR=float(rr["HR"][m].mean()) if "HR" in rr else None, MRR_claim=float(rr[CLAIM][m].mean()),
                                 delta_claim_minus_better=bs["delta"], CI95=bs["CI95"])
            dec[b]["strata"] = srows
        outcome = "PRACTICAL_ADVANCE" if len(two) == 2 and all(dec[b]["PASS"] for b in two) else (
            "HEAD_DAMAGE" if two and all(dec[b]["C1_pass"] and dec[b]["C2_pass"] for b in two) else "RECURRENCY_DONE_WELL")
        if len(two) < 2: outcome += " (PROVISIONAL: fewer than two bases with the claim system)"
        out["decision"] = dict(per_base=dec, outcome=outcome, claim_survives=outcome == "PRACTICAL_ADVANCE")
        if "tirgn" in out["bases"] and "HR" in out["bases"]["tirgn"].get("arms", {}):
            hr_t = out["bases"]["tirgn"]["arms"]["HR"]["harness"]["MRR"]
            out["HR_tirgn_vs_residual_armB"] = dict(HR=hr_t, residual=D.get("residual_hr_tirgn"),
                                                   absdiff=abs(hr_t - D["residual_hr_tirgn"]) if D.get("residual_hr_tirgn") else None,
                                                   within_0p002=bool(D.get("residual_hr_tirgn") and abs(hr_t - D["residual_hr_tirgn"]) <= 0.002))
        for b, rec in out["bases"].items():
            if "missing" in rec: continue
            for arm, v in rec["arms"].items():
                out["rows"].append(dict(base=b, arm=arm, evaluator="harness exact filtered", seeds=v.get("seeds", "-"), **v["harness"],
                                        sd=v.get("harness_sd")))
            rec.pop("_rr"); rec.pop("_hrank_base")
        V["datasets"][ds] = out
    ok = [V["datasets"][d]["decision"]["claim_survives"] for d in V["datasets"]]
    V["overall"] = "survives on both datasets" if all(ok) else ("survives on one dataset" if any(ok) else "does not survive")
    json.dump(V, open(a.out, "w"), indent=1)
    print(json.dumps({d: V["datasets"][d]["decision"]["outcome"] for d in V["datasets"]}), V["overall"])


if __name__ == "__main__":
    main()
