"""v7 / v7_bagg_run.py -- B1/B2/B3/C1/A5 aggregation for one default base from the official per-system arrays.
Usage (inside a Python session): exec(open('v7/code/v7_bagg_run.py').read()); res = run_ds(ds, official_files, ctrl_file, hcpat_file, g1_file, g1_arm, x_key)
All numbers: official TGB evaluator (rr_/rank_), seed-mean per-query RR, query Poisson + timestamp-block bootstrap (shared draws, B=2000)."""
import numpy as np, json
import v7_common as VC
import v7_bagg as BA

ARMS = {  # display name -> system prefix in the official npz (seeds _s0.._s2)
    "MLP64x2 (FULL, re-fit)": "MLP64_PAT_fused", "LIN": "LIN_PAT_fused", "MLP16": "MLP16_PAT_fused", "QBLEND": "QBLEND_alone",
    "PT_common": "PT_common_fused", "RB_common": "RB_common_fused", "PT_gen (= FULL)": "MLP64_PAT_fused", "RB_gen": "RB_gen_fused",
    "NB_F": "NB_PAT_fused", "B_NF": "MLP64_PAT_alone", "NB_NF": "NB_PAT_alone", "EV_NF_PT": "EV_PT_alone", "EV_NF_RB": "EV_RB_alone",
    "noDyadLR": "noDyadLR_fused", "UNMASKED": "UNMASKED_fused", "MULTIPOS": "MULTIPOS_fused",
}
F = "MLP64x2 (FULL, re-fit)"
ITEMS = {
    "B1_ranker_form": dict(primary=[("MLP64x2 - LIN", F, "LIN"), ("MLP64x2 - QBLEND", F, "QBLEND")],
                           secondary=[("MLP16 - LIN", "MLP16", "LIN"), ("MLP64x2 - MLP16", F, "MLP16")]),
    "B2_matched_pool": dict(primary=[("PT_common - RB_common", "PT_common", "RB_common")],
                            secondary=[("PT_gen - RB_gen", "PT_gen (= FULL)", "RB_gen")]),
    "B3_common_pool_base_ablation": dict(primary=[("FULL - NB_NF", F, "NB_NF")],
                                         secondary=[("FULL - NB_F", F, "NB_F"), ("FULL - B_NF", F, "B_NF"),
                                                    ("FULL - EV_NF_PT [base contribution, common pool]", F, "EV_NF_PT"),
                                                    ("EV_NF_PT - EV_NF_RB [candidate source, common inputs]", "EV_NF_PT", "EV_NF_RB"),
                                                    ("EV_NF_RB - G1 [pool width, generator features, tail order]", "EV_NF_RB", "G1"),
                                                    ("HRCOV_PAT(stored) - G1 [total]", "HRCOV_PAT (stored)", "G1"),
                                                    ("FULL - G1 [total, re-fit]", F, "G1")]),
    "C1_no_dyad_lr": dict(primary=[("FULL - noDyadLR", F, "noDyadLR")], secondary=[]),
    "A5_multi_positive_sensitivity": dict(primary=[], secondary=[("UNMASKED - FULL", "UNMASKED", F), ("MULTIPOS - FULL", "MULTIPOS", F)]),
}


def merged(files):
    Z = {}
    rows = None
    for f in files:
        z = np.load(f)
        if rows is None: rows = z["rows"]
        assert np.array_equal(z["rows"], rows), f
        for k in z.files:
            if k.startswith(("rr_", "rank_", "hrank_")) and k not in Z: Z[k] = z[k]
    Z["rows"] = rows
    return Z


def run_ds(ds, official_files, ctrl_file, hcpat_file, g1_file, g1_arm, x_key, better_ctrl, bar=0.005):
    Z = merged(official_files)
    C = np.load(ctrl_file); Hc = np.load(hcpat_file); G = np.load(g1_file)
    for Q in (C, Hc, G): assert np.array_equal(Q["rows"], Z["rows"])
    rep = {}
    for nm_, Q in (("ctrl", C), ("hcpat", Hc)):
        d_ = np.abs(Q["rr_base"].astype(np.float64) - Z["rr_base"].astype(np.float64)); assert d_.mean() < 1e-6, (nm_, d_.mean())
        rep[nm_] = dict(frac_base_rr_differs=float((d_ > 1e-6).mean()), mean_abs=float(d_.mean()))
    X = VC.loadx(x_key, "test"); assert np.array_equal(X["F_rows"] - X["F_rows"][0], Z["rows"]); ts = X["F_sro"][:, 3].astype(np.int64)
    arms, missing = {}, []
    for nm, pre in ARMS.items():
        try: arms[nm] = BA.arm_from(Z, pre)
        except AssertionError: missing.append(nm)
    arms["HRCOV_PAT (stored)"] = BA.arm_from({k: Hc[k] for k in Hc.files}, "HRCOV_PAT")
    arms["G1"] = BA.arm_from({k: G[k] for k in G.files}, g1_arm)
    for c in ("base", "BLEND", "POOL"): arms[c] = BA.arm_from({k: C[k] for k in C.files}, c)
    out = dict(dataset=ds, base="TiRGN", n_queries=int(len(ts)), n_ts=int(len(np.unique(ts))), missing_arms=missing,
               better_control=better_ctrl, base_replay_identity=rep, arms={k: BA.metrics(v) for k, v in arms.items()}, items={})
    # re-fit fidelity
    pairs_all = [("FULL(re-fit) - HRCOV_PAT(stored)", F, "HRCOV_PAT (stored)")] if F in arms else []
    item_pairs = {}
    for it, D in ITEMS.items():
        P = [(l, a, b, "primary") for l, a, b in D["primary"]] + [(l, a, b, "secondary") for l, a, b in D["secondary"]]
        item_pairs[it] = [p for p in P if p[1] in arms and p[2] in arms]
        pairs_all += [(f"{it}|{l}", a, b) for l, a, b, _ in item_pairs[it]]
    # context rows vs controls for every arm (descriptive)
    for nm in list(ARMS) :
        if nm in arms and not nm.startswith("PT_gen"):
            pairs_all.append((f"ctx|{nm} - {better_ctrl}", nm, better_ctrl))
    S = BA.compare(arms, pairs_all, ts, bar=bar)
    out["refit_fidelity"] = S.get("FULL(re-fit) - HRCOV_PAT(stored)")
    for it, P in item_pairs.items():
        rows = []
        for l, a, b, role in P:
            r = S[f"{it}|{l}"]; rows.append(dict(comparison=l, role=role, **r))
        prim = [r for r in rows if r["role"] == "primary"]
        if len(prim) > 1:
            for key in ("query", "ts"):
                adj = BA.holm([r["MRR"][key]["p_one_sided_le0"] for r in prim])
                for r, a_ in zip(prim, adj): r["MRR"][f"holm_p_{key}"] = a_
        for r in prim:
            m = r["MRR"]; ci = m["ts"]["CI95"]
            r["ambiguous_for_expansion"] = bool(ci[0] <= bar <= ci[1])
            r["holm_ok"] = bool(all(m.get(f"holm_p_{k}", m[k]["p_one_sided_le0"]) < 0.05 for k in ("query", "ts")))
            r["claim"] = ("present (+)" if m["present_pos"] and r["holm_ok"] else "present (-)" if m["present_neg"] else
                          "|delta| >= bar excluded" if m["excluded_gain_ge_bar"] and m["excluded_loss_ge_bar"] else
                          "gain >= bar excluded" if m["excluded_gain_ge_bar"] else "not established")
        out["items"][it] = rows
    out["context_vs_control"] = {k.split("|", 1)[1]: v for k, v in S.items() if k.startswith("ctx|")}
    return out, arms, ts
