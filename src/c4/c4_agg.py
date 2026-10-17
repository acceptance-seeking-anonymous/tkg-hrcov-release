"""C4 / c4_agg.py -- base-seed variation on tkgl-polecat TiRGN (registered c4_protocol.json, parent tois_v7_rule.json).
Usage: python3 v7/code/c4_agg.py --s43 <official_test_polecat_tirgn2.npz> --out c4_results.json [--sel sel_polecat_tirgn2.json]
All numbers: official py-tgb evaluator per-query RR (rr_*), re-ranker arms seed-mean per query (seeds 0-2);
paired Poisson query bootstrap + timestamp-block bootstrap (B = 2000, seed 0, shared draws) via v7_common.summ_multi.
Label: pre-specified later analysis (C4); descriptive, no claim beyond the bounds."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v7_common as VC

S42 = dict(ctrl="v7/data/f_serverB/off_polecat_tirgn_ctrl.npz", hcpat="v7/data/f_serverB/off_polecat_tirgn_hcpat.npz",
           full=os.path.expandvars("$RESULTS_DIR/runs/v7/data/off/official_test_polecat_v7_polecat_tirgn_00.npz"),
           ev=os.path.expandvars("$RESULTS_DIR/runs/v7/data/off/official_test_polecat_v7_polecat_tirgn_01.npz"))
FULL, EV = "MLP64_PAT_fused", "EV_PT_alone"   # HRCOV_PAT (= FULL, v7 MLP64_PAT arm) and EV_NF_PT (v7 EV_PT arm, score alone)


def seedavg(Z, p, key="rr"): return np.mean([Z[f"{key}_{p}_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)
def hk(Z, p, k): return float(np.mean([np.mean(Z[f"rank_{p}_s{s}"] <= k) for s in (0, 1, 2)])) if f"rank_{p}_s0" in Z else None
def hk1(Z, k, arm): return float(np.mean(Z[f"rank_{arm}"] <= k)) if f"rank_{arm}" in Z else None


def arms_of(Z, ev_Z=None, ctrl_Z=None):
    c = ctrl_Z if ctrl_Z is not None else Z; e = ev_Z if ev_Z is not None else Z
    A = dict(base=c["rr_base"].astype(np.float64), BLEND=c["rr_BLEND"].astype(np.float64), POOL=c["rr_POOL"].astype(np.float64),
             HRCOV_PAT=seedavg(Z, FULL), EV_NF_PT=seedavg(e, EV))
    H = dict(base={k: hk1(c, k, "base") for k in (1, 3, 10)}, BLEND={k: hk1(c, k, "BLEND") for k in (1, 3, 10)},
             POOL={k: hk1(c, k, "POOL") for k in (1, 3, 10)}, HRCOV_PAT={k: hk(Z, FULL, k) for k in (1, 3, 10)},
             EV_NF_PT={k: hk(e, EV, k) for k in (1, 3, 10)})
    return A, H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--s43", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sel", default=None); ap.add_argument("--meta", default=None); ap.add_argument("--train", default=None)
    a = ap.parse_args()
    Z3 = np.load(a.s43); Zc = np.load(S42["ctrl"]); Zf = np.load(S42["full"]); Ze = np.load(S42["ev"]); Zh = np.load(S42["hcpat"])
    X = VC.loadx("polecat_tirgn", "test"); rows = X["F_rows"] - X["F_rows"][0]; ts = X["F_sro"][:, 3].astype(np.int64)
    n = len(rows); assert n == 532636, n
    for nm, Z in (("s43", Z3), ("s42_full", Zf), ("s42_ev", Ze)):
        assert np.array_equal(Z["rows"], rows), f"row order differs: {nm}"
    for nm, Z in (("s42_ctrl", Zc), ("s42_hcpat", Zh)):
        assert len(Z["rr_base"]) == n, nm
    # seed-42 base identity across the v7 files (same base checkpoint and official pass)
    base_id = {}   # live TiRGN replay: deep near-tie blocks may differ (as in v7_bagg_run); require mean |d| < 1e-6
    for nm, Z in (("full", Zf), ("ev", Ze), ("hcpat", Zh)):
        d_ = np.abs(Z["rr_base"].astype(np.float64) - Zc["rr_base"]); assert d_.mean() < 1e-6, f"seed-42 base differs in {nm}"
        base_id[nm] = dict(frac_base_rr_differs=float((d_ > 1e-6).mean()), mean_abs=float(d_.mean()), max_abs=float(d_.max()))
    for Z in (Z3,):
        for k in Z.files:
            if k.startswith(("rr_", "rank_")): assert np.isfinite(Z[k]).all(), f"non-finite {k}"
    A3, H3 = arms_of(Z3); A2, H2 = arms_of(Zf, Ze, Zc)
    A2_stored = Zh  # sensitivity: stored seed-42 HRCOV_PAT (claim system) instead of the v7 re-fit
    hp42s = seedavg(Zh, "HRCOV_PAT")
    M3 = {k: float(v.mean()) for k, v in A3.items()}; M2 = {k: float(v.mean()) for k, v in A2.items()}
    better43 = "BLEND" if M3["BLEND"] >= M3["POOL"] else "POOL"; better42 = "BLEND" if M2["BLEND"] >= M2["POOL"] else "POOL"
    cols = {
        # primary + B3 on the seed-43 base
        "s43: HRCOV_PAT - better control": A3["HRCOV_PAT"] - A3[better43],
        "s43: HRCOV_PAT - BLEND": A3["HRCOV_PAT"] - A3["BLEND"],
        "s43: HRCOV_PAT - POOL": A3["HRCOV_PAT"] - A3["POOL"],
        "s43: FULL - EV_NF_PT [B3 base contribution, common PT pool]": A3["HRCOV_PAT"] - A3["EV_NF_PT"],
        "s43: HRCOV_PAT - base": A3["HRCOV_PAT"] - A3["base"],
        # the same contrasts on seed 42 (v7 official files) for side-by-side
        "s42: HRCOV_PAT - better control": A2["HRCOV_PAT"] - A2[better42],
        "s42: HRCOV_PAT - BLEND": A2["HRCOV_PAT"] - A2["BLEND"],
        "s42: HRCOV_PAT - POOL": A2["HRCOV_PAT"] - A2["POOL"],
        "s42: FULL - EV_NF_PT [B3 base contribution, common PT pool]": A2["HRCOV_PAT"] - A2["EV_NF_PT"],
        "s42: HRCOV_PAT - base": A2["HRCOV_PAT"] - A2["base"],
        # seed variation: seed 43 - seed 42, per-query paired
        "seed43-seed42: base MRR": A3["base"] - A2["base"],
        "seed43-seed42: BLEND MRR": A3["BLEND"] - A2["BLEND"],
        "seed43-seed42: POOL MRR": A3["POOL"] - A2["POOL"],
        "seed43-seed42: HRCOV_PAT MRR": A3["HRCOV_PAT"] - A2["HRCOV_PAT"],
        "seed43-seed42: (HRCOV_PAT - BLEND)": (A3["HRCOV_PAT"] - A3["BLEND"]) - (A2["HRCOV_PAT"] - A2["BLEND"]),
        "seed43-seed42: (HRCOV_PAT - POOL)": (A3["HRCOV_PAT"] - A3["POOL"]) - (A2["HRCOV_PAT"] - A2["POOL"]),
        "seed43-seed42: (FULL - EV_NF_PT)": (A3["HRCOV_PAT"] - A3["EV_NF_PT"]) - (A2["HRCOV_PAT"] - A2["EV_NF_PT"]),
        # sensitivity: seed-42 stored HRCOV_PAT (claim system) instead of the v7 re-fit
        "sens seed43-seed42(stored HRCOV_PAT): HRCOV_PAT MRR": A3["HRCOV_PAT"] - hp42s,
        "sens seed43-seed42(stored HRCOV_PAT): (HRCOV_PAT - BLEND)": (A3["HRCOV_PAT"] - A3["BLEND"]) - (hp42s - A2["BLEND"]),
    }
    R = VC.summ_multi(cols, ts, bar=0.005)
    rr3, rr2 = A3["base"], A2["base"]
    base_var = dict(MRR_s43=M3["base"], MRR_s42=M2["base"], abs_diff=abs(M3["base"] - M2["base"]),
                    frac_queries_base_rr_differs=float((np.abs(rr3 - rr2) > 1e-6).mean()),
                    pearson_rr=float(np.corrcoef(rr3, rr2)[0, 1]),
                    frac_gold_hit1_indicator_agree=float(np.mean((Z3["rank_base"] <= 1) == (Zc["rank_base"] <= 1))) if "rank_base" in Zc else None)
    per_seed = {f"s43_{p}": [float(Z3[f"rr_{p}_s{s}"].mean()) for s in (0, 1, 2)] for p in (FULL, EV)}
    per_seed.update({f"s42_{FULL}": [float(Zf[f"rr_{FULL}_s{s}"].mean()) for s in (0, 1, 2)],
                     f"s42_{EV}": [float(Ze[f"rr_{EV}_s{s}"].mean()) for s in (0, 1, 2)],
                     "s42_HRCOV_PAT_stored": [float(Zh[f"rr_HRCOV_PAT_s{s}"].mean()) for s in (0, 1, 2)]})
    out = dict(name="c4_results", label="pre-specified later analysis (C4)",
               protocol="c4_protocol.json (registered 2026-10-14T18:54:49Z; sha256 171774ea...)",
               parent_rule="tois_v7_rule.json", dataset="tkgl-polecat", base="TiRGN", seeds=[42, 43], n_test=n,
               evaluator="official py-tgb Evaluator per-query RR; re-ranker arms averaged over re-ranker seeds 0-2 per query",
               seed42_sources=S42, seed43_source=os.path.basename(a.s43),
               systems={"HRCOV_PAT (= FULL)": f"{FULL}_s0-2 (v7 MLP64_PAT arm)", "EV_NF_PT": f"{EV}_s0-2 (v7 EV_PT arm)",
                        "BLEND/POOL": "valid-selected on each seed's own valid grid"},
               mrr=dict(seed43=M3, seed42=M2, seed42_HRCOV_PAT_stored=float(hp42s.mean())), hits=dict(seed43=H3, seed42=H2), per_seed=per_seed,
               better_control=dict(seed43=better43, seed42=better42), base_seed_variation=base_var, seed42_base_identity_across_files=base_id, contrasts=R,
               selection=json.load(open(a.sel)) if a.sel and os.path.exists(a.sel) else None,
               official_meta={k: v for k, v in json.load(open(a.meta)).items() if k in ("official_vs_formula_max_batch_dev", "py_tgb", "selected", "gpu", "elapsed_s")} if a.meta and os.path.exists(a.meta) else None,
               base_training=json.load(open(a.train)) if a.train and os.path.exists(a.train) else None,
               notes=["seed-42 HRCOV_PAT here is the v7 re-fit (MLP64_PAT_fused, same code path as seed 43); the stored claim-system HRCOV_PAT is a sensitivity row",
                      "present iff one-sided L > 0 under both query and ts-block bootstraps; excluded_gain_ge_bar iff ts U < 0.005"])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(mrr=out["mrr"], better=out["better_control"], base_var=base_var,
                          key={k: (round(v["delta"], 5), [round(x, 5) for x in v["ts"]["CI95"]], v["present_pos"], v["present_neg"]) for k, v in R.items()}), indent=1))


if __name__ == "__main__":
    main()
