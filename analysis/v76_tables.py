#!/usr/bin/env python3
"""analysis/v76_tables.py -- the v7.6 numbers of the manuscript (LambdaMART, validation-trained base, cross-fitted RecB, base seeds),
typeset exactly as in the paper, from results/v76/v76_results.json.

Items: two rows of the mechanism table (tab:mechanism), two rows of the base-free-control table (tab:g1), the outcome cell of the
reviewer-requested row of the rules table (tab:rules), and the numeric fragments of the corresponding text (LambdaMART ranges,
validation-trained base, base seeds).  Writes results/tables/v76_fragments.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, write_json, mt, mci


def build():
    R = load("v76_results.json")
    B = R["M2b"]["bases"]; BK = ["polecat_tirgn", "polecat_cogn", "smallpedia_tirgn", "smallpedia_regcn_tgb"]
    ML = {k: B[k]["contrasts"]["MLP64_PAT_fused - LMART_PAT_fused"] for k in BK}
    LL = {k: B[k]["contrasts"]["LMART_PAT_fused - LIN_PAT_fused"] for k in BK}
    LC = {k: B[k]["contrasts"]["LMART_PAT_fused - better control"] for k in BK}
    rng = lambda D: (min(D[k]["delta"] for k in BK), max(D[k]["delta"] for k in BK))
    (ml_lo, ml_hi), (ll_lo, ll_hi), (lc_lo, lc_hi) = rng(ML), rng(LL), rng(LC)
    within = all(ML[k]["excluded_gain_ge_bar"] and ML[k]["excluded_loss_ge_bar"] for k in BK)
    out = {}
    out["tab:mechanism MLP - LambdaMART"] = (r"2$\times$64 MLP $-$ LambdaMART on the same features & " + mt(ML["polecat_tirgn"]["delta"]) + " "
                                            + mci(ML["polecat_tirgn"]["ts"]["CI95"]) + " & " + mt(ML["smallpedia_tirgn"]["delta"]) + " "
                                            + mci(ML["smallpedia_tirgn"]["ts"]["CI95"]) + r" \\")
    out["tab:mechanism LambdaMART - linear"] = (r"LambdaMART $-$ linear scorer & " + mt(LL["polecat_tirgn"]["delta"]) + " " + mci(LL["polecat_tirgn"]["ts"]["CI95"])
                                               + " & " + mt(LL["smallpedia_tirgn"]["delta"]) + " " + mci(LL["smallpedia_tirgn"]["ts"]["CI95"]) + r" \\")
    out["text LambdaMART ranges"] = (r"MLP $-$ LambdaMART from " + mt(ml_lo) + " to " + mt(ml_hi) + r", in both directions) and also exceeds the linear scorer ("
                                     + mt(ll_lo) + " to " + mt(ll_hi) + ") and the better control (" + mt(lc_lo) + " to " + mt(lc_hi) + ")")
    M3 = R["M3a"]["results"]["contrasts"]
    va = M3["HRCOV_PAT(original base) - better control(validation-trained base)"]; vb = M3["validation-trained base - original base"]
    vc = M3["better control(validation-trained) - better control(original)"]; vo = M3["HRCOV_PAT(original base) - better control(original base)"]
    share = (vo["delta"] - va["delta"]) / vo["delta"]
    out["text validation-trained base"] = ("raises the base by " + mt(vb["delta"]) + " and the better control by " + mt(vc["delta"])
                                           + r", and \HRCOV{} on the original base still exceeds the retrained base's better control by " + mt(va["delta"]) + " "
                                           + mci(va["ts"]["CI95"]) + " (against " + mt(vo["delta"]) + f"): base supervision closes about {round(100 * share)}\\% of the margin")
    P4, S4 = R["M4"]["results"]["polecat"], R["M4"]["results"]["smallpedia"]
    gp, gs = P4["contrasts"]["G1_xf - G1"], S4["contrasts"]["G1_xf - G1"]
    lab = lambda c: ("within the bar" if c["excluded_gain_ge_bar"] and c["excluded_loss_ge_bar"] and (c["present_pos"] or c["present_neg"])
                     else r"not present$^{\ddagger}$")
    out["tab:g1 G1xf polecat"] = (r"G1$_{\mathrm{xf}}$ $-$ G1 (cross-fitted RecB; G1$_{\mathrm{xf}}$ " + f"{P4['mrr']['G1_xf']:.4f}" + r")$^{\dagger}$ & "
                                  + mt(gp["delta"]) + " & " + mci(gp["query"]["CI95"]) + " & " + mci(gp["ts"]["CI95"]) + " & " + lab(gp) + r" \\")
    out["tab:g1 G1xf smallpedia"] = (r"G1$_{\mathrm{xf}}$ $-$ G1 (cross-fitted RecB; G1$_{\mathrm{xf}}$ " + f"{S4['mrr']['G1_xf']:.4f}" + r")$^{\dagger}$ & "
                                     + mt(gs["delta"]) + " & " + mci(gs["query"]["CI95"]) + " & " + mci(gs["ts"]["CI95"]) + " & " + lab(gs) + r" \\")
    T6, R6 = R["M6"]["tirgn"]["summary"], R["M6"]["regcn_tgb"]["summary"]
    pm = lambda s: f"{s['margin_mean']:.4f} $\\pm$ {s['margin_sd']:.4f}"
    out["text base seeds"] = ("three training seeds give a margin of $+$" + pm(T6) + r" for TiRGN (mean $\pm$ SD; " + mt(T6["margin_min"]) + " to "
                              + mt(T6["margin_max"]) + ") and $+$" + pm(R6) + " for the TGB-configuration RE-GCN (" + mt(R6["margin_min"]) + " to "
                              + mt(R6["margin_max"]) + ")")
    out["tab:rules outcome"] = (("LambdaMART within the bar of the MLP on all four bases" if within else "LambdaMART NOT within the bar")
                                + f"; validation-trained base closes {round(100 * share)}\\% of the margin; G1 changes by " + mt(gp["delta"]) + " and "
                                + mt(gs["delta"]) + "; base-seed margins $+$" + pm(T6) + " (TiRGN) and $+$" + pm(R6) + r" (RE-GCN) \\")
    return out


def main():
    out = build()
    write_json(out, "v76_fragments.json")
    for k, v in out.items():
        print(f"[{k}]\n  {v}")


if __name__ == "__main__":
    main()
