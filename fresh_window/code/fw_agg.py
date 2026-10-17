#!/usr/bin/env python3
"""phase2_fresh_window / fw_agg.py -- v79 decision (v79_fresh_window_rule + amendment_1 + implementation note), run once on the
Stage-2 rank files.  Primary metric: all-entity time-aware filtered MRR with ties averaged; harness strict ranks alongside.
HRcov and SA per-query RR = mean over seeds 0-2.  Paired per-query RR differences; query bootstrap (Poisson(1)) and
timestamp-block bootstrap (multinomial over the 61 test days), B = 2000, seed 0, draws shared across comparisons
(v7_common.summ_multi); one-sided 95% bounds = 5th / 95th percentiles.
  BEATS(X): L > 0 under both AND point >= 0.005;  MATCHES(X): U < 0.005 under both;  SMALL otherwise.
Outcome: R1_BEATS_BOTH | R2_SA_MATCHES | R3_EXTERNAL_MATCHES | R4_OTHER.
  python3 fw_agg.py --tirgn official_test_gdeltfw_tirgn.npz --sa official_test_gdeltfw_sa.npz --ext official_test_gdeltfw_ext.npz \
                    --ts2id timestamp2id.txt --comparator ext_comparator.json --out fresh_window_verdict.json --results fresh_window_results.json
"""
import os, sys, json, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v7_common as V

BAR = 0.005


def mets(rank_list, rr_list):
    per = [dict(MRR=float(np.mean(rr)), H1=float(np.mean(rk <= 1)), H3=float(np.mean(rk <= 3)), H10=float(np.mean(rk <= 10))) for rk, rr in zip(rank_list, rr_list)]
    out = {k: float(np.mean([p[k] for p in per])) for k in ("MRR", "H1", "H3", "H10")}
    if len(per) > 1: out["per_seed_MRR"] = [p["MRR"] for p in per]
    return out


def call(s):
    q, t = s["query"], s["ts"]
    if q["L"] > 0 and t["L"] > 0 and s["delta"] >= BAR: return "BEATS"
    if q["U"] < BAR and t["U"] < BAR: return "MATCHES"
    return "SMALL"


def main():
    ap = argparse.ArgumentParser()
    for k in ("tirgn", "sa", "ext", "ts2id", "comparator", "out", "results"): ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    T = np.load(a.tirgn); S = np.load(a.sa); E = np.load(a.ext); comp = json.load(open(a.comparator))
    n = len(T["rr_base"]); assert len(S["rr_HIST"]) == n and len(E["rr_EXT"]) == n
    assert (T["rows"] == np.arange(n)).all() and (S["rows"] == np.arange(n)).all() and (E["rows"] == np.arange(n)).all()
    ts = S["ts"].astype(np.int64); assert (E["ts"] == ts).all()
    chk = dict(base_rr_equal_tirgn_vs_ext=float(np.abs(T["rr_base"] - E["rr_base"]).max()), n_test_queries=int(n), n_test_days=int(len(np.unique(ts))))
    seedsH = [k[3:] for k in T.files if k.startswith("rr_HRCOV_PAT_s")]; seedsS = [k[3:] for k in S.files if k.startswith("rr_SA_W100_s")]
    assert len(seedsH) == 3 and len(seedsS) == 3, (seedsH, seedsS)
    sysd = {"base": (T, ["base"]), "BLEND": (T, ["BLEND"]), "HRcov": (T, seedsH), "SA": (S, seedsS), "external_comparator": (E, ["EXT"])}
    metrics, rr_mean = {}, {}
    for nm, (Z, keys) in sysd.items():
        rr = [Z[f"rr_{k}"].astype(np.float64) for k in keys]; rk = [Z[f"rank_{k}"].astype(np.float64) for k in keys]
        hk = [Z[f"hrank_{k}"].astype(np.float64) for k in keys]
        metrics[nm] = dict(official_ties_averaged=mets(rk, rr), harness_strict=mets(hk, [1.0 / h for h in hk]), n_seeds=len(keys))
        rr_mean[nm] = np.mean(rr, 0); rr_mean[nm + "__strict"] = np.mean([1.0 / h for h in hk], 0)
    cols = {"HRcov-SA": rr_mean["HRcov"] - rr_mean["SA"], "HRcov-external_comparator": rr_mean["HRcov"] - rr_mean["external_comparator"],
            "HRcov-BLEND": rr_mean["HRcov"] - rr_mean["BLEND"], "HRcov-base": rr_mean["HRcov"] - rr_mean["base"],
            "SA-external_comparator": rr_mean["SA"] - rr_mean["external_comparator"], "BLEND-base": rr_mean["BLEND"] - rr_mean["base"]}
    summ = V.summ_multi(cols, ts, bar=BAR)
    cols_s = {"HRcov-SA": rr_mean["HRcov__strict"] - rr_mean["SA__strict"], "HRcov-external_comparator": rr_mean["HRcov__strict"] - rr_mean["external_comparator__strict"]}
    summ_strict = V.summ_multi(cols_s, ts, bar=BAR)
    calls = {"SA": call(summ["HRcov-SA"]), "external_comparator": call(summ["HRcov-external_comparator"])}
    if calls["SA"] == "BEATS" and calls["external_comparator"] == "BEATS": outcome = "R1_BEATS_BOTH"
    elif calls["SA"] == "MATCHES": outcome = "R2_SA_MATCHES"
    elif calls["external_comparator"] == "MATCHES": outcome = "R3_EXTERNAL_MATCHES"
    else: outcome = "R4_OTHER"
    id2day = {}
    for ln in open(a.ts2id):
        k, v = ln.split(); id2day[int(v)] = k
    days = np.array([id2day[int(x)] for x in ts]); wk = np.array([datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])).isocalendar()[1] for d in days])
    weekly = {}
    for w in np.unique(wk):
        m = wk == w
        weekly[str(int(w))] = dict(n=int(m.sum()), days=f"{min(days[m])}..{max(days[m])}", **{k: float(cols[k][m].mean()) for k in ("HRcov-SA", "HRcov-external_comparator", "HRcov-BLEND")})
    verdict = dict(rule="v79_fresh_window_rule.json (sha256 209bb1b5...) + amendment_1 (sha256 207689b8...) + fresh_window_implementation_note.json",
                   dataset="GDELTFW", base="TiRGN", test_window="2026-03-01..2026-04-30", n_test_queries=int(n), bar=BAR, bootstrap=dict(B=V.B, seed=0, shared_draws=True),
                   comparator=comp, calls=calls, outcome=outcome,
                   margins={k: dict(point=summ[k]["delta"], query=dict(L=summ[k]["query"]["L"], U=summ[k]["query"]["U"], CI95=summ[k]["query"]["CI95"]),
                                    ts_block=dict(L=summ[k]["ts"]["L"], U=summ[k]["ts"]["U"], CI95=summ[k]["ts"]["CI95"])) for k in ("HRcov-SA", "HRcov-external_comparator", "HRcov-BLEND", "HRcov-base")},
                   metrics_official={k: v["official_ties_averaged"] for k, v in metrics.items()}, checks=chk,
                   decided_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    results = dict(metrics=metrics, deltas_official=summ, deltas_harness_strict=summ_strict, weekly_deltas=weekly, checks=chk, calls=calls, outcome=outcome)
    json.dump(verdict, open(a.out, "w"), indent=1); json.dump(results, open(a.results, "w"), indent=1)
    print(json.dumps(dict(outcome=outcome, calls=calls), indent=1))


if __name__ == "__main__":
    main()
