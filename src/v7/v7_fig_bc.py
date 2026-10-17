"""v7 / v7_fig_bc.py -- summary figure for the B/C items (forest of primary/secondary deltas, B3 decomposition, tkgl-icews B4/C3).
Run inside a Python session after apply_figure_style(): exec(open('v7/code/v7_fig_bc.py').read()); fig = make_fig()"""
import json, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt

OUT = "v7/out"
COL = {"tkgl-polecat": "#1f5fa8", "tkgl-smallpedia": "#d07a1c"}
LAB = {"tkgl-polecat": "polecat (TiRGN)", "tkgl-smallpedia": "smallpedia (TiRGN)"}


def _find(itj, ds, comp):
    for r in itj["datasets"].get(ds, {}).get("comparisons", []):
        if r["comparison"] == comp: return r
    return None


def make_fig():
    B1, B2, B3, C1, A5 = (json.load(open(f"{OUT}/{f}")) for f in ("b1_ranker_form.json", "b2_matched_pool.json", "b3_common_pool_base_ablation.json", "c1_no_dyad_lr.json", "a5_multi_positive_sensitivity.json"))
    B4 = json.load(open(f"{OUT}/b4_g1_tkgl_icews.json"))
    try: C2 = json.load(open(f"{OUT}/c2_pattern_design.json"))
    except FileNotFoundError: C2 = None
    rows = [("B1", "MLP 2×64 − linear", B1, "MLP64x2 - LIN"), ("B1", "MLP 2×64 − query-conditioned blend", B1, "MLP64x2 - QBLEND"),
            ("B1", "MLP 16 − linear", B1, "MLP16 - LIN"),
            ("B2", "pattern pool − RecB pool (common inputs)", B2, "PT_common - RB_common"), ("B2", "pattern pool − RecB pool (own features)", B2, "PT_gen - RB_gen"),
            ("B3", "full − no base features, no fusion", B3, "FULL - NB_NF"), ("C1", "full − without dyad likelihood ratio", C1, "FULL - noDyadLR")]
    if C2:
        rows += [("C2", "utility − confidence ordering", C2, "UTIL12 - CONF12"), ("C2", "1+2-hop − 1-hop patterns", C2, "UTIL12 - UTIL1")]
    rows += [("A5", "other answers as negatives − masked", A5, "UNMASKED - FULL"), ("A5", "multi-positive loss − masked", A5, "MULTIPOS - FULL")]
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.55, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0]); XMAX = 0.0175; XMIN = -0.0025
    ax.axvspan(-0.005, 0.005, color="0.92", zorder=0, lw=0); ax.axvline(0, color="0.35", lw=0.7, zorder=1)
    y = 0; yt, yl = [], []; prev = None
    for it, lab, itj, comp in rows:
        if prev is not None and it != prev: y += 0.6
        prev = it
        for k, ds in enumerate(("tkgl-polecat", "tkgl-smallpedia")):
            r = _find(itj, ds, comp)
            if r is None: continue
            yy = y + (-0.17 if k == 0 else 0.17); m = r["MRR"]; d = m["delta"]; lo, hi = m["ts"]["CI95"]; qlo, qhi = m["query"]["CI95"]
            if d > XMAX:
                ax.annotate("", xy=(XMAX, yy), xytext=(XMAX - 0.0022, yy), arrowprops=dict(arrowstyle="-|>", color=COL[ds], lw=1.0))
                ax.text(XMAX - 0.0025, yy, f"{d:+.4f}", ha="right", va="center", fontsize=6, color=COL[ds])
                continue
            ax.plot([lo, hi], [yy, yy], color=COL[ds], lw=2.2, solid_capstyle="butt", zorder=3)
            ax.plot([qlo, qhi], [yy, yy], color="white", lw=0.6, zorder=4)
            ax.plot(d, yy, "o", ms=3.4, color=COL[ds], mec="white", mew=0.4, zorder=5)
        yt.append(y); yl.append(f"{it}  {lab}"); y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.invert_yaxis(); ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("ΔMRR (official evaluator)"); ax.set_title("Each design choice: effect on MRR", loc="left")
    ax.set_xticks([0.0, 0.005, 0.010, 0.015]); ax.set_xticklabels(["0", "0.005", "0.010", "0.015"])
    for k, ds in enumerate(COL): ax.plot([], [], "o-", color=COL[ds], lw=2.2, ms=3.4, label=LAB[ds])
    ax.add_patch(mpl.patches.Rectangle((0, 0), 0, 0, color="0.92", label="±0.005 practical margin"))
    ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.6)
    # (b) B3 decomposition
    bx = fig.add_subplot(gs[0, 1])
    comps = [("FULL - EV_NF_PT [base contribution, common pool]", "base +\npattern"), ("EV_NF_PT - EV_NF_RB [candidate source, common inputs]", "candidate\nsource"),
             ("EV_NF_RB - G1 [pool width, generator features, tail order]", "remainder"), ("FULL - G1 [total, re-fit]", "total")]
    xs = np.arange(len(comps))
    for k, ds in enumerate(COL):
        for j, (c, _) in enumerate(comps):
            r = _find(B3, ds, c); m = r["MRR"]; lo, hi = m["ts"]["CI95"]
            bx.bar(j + (k - 0.5) * 0.36, m["delta"], 0.34, color=COL[ds], alpha=0.9 if j < 3 else 1.0, edgecolor="none")
            bx.plot([j + (k - 0.5) * 0.36] * 2, [lo, hi], color="0.15", lw=0.8)
            if abs(m["delta"]) < 0.0004: bx.plot(j + (k - 0.5) * 0.36, m["delta"], "_", ms=6, mew=1.2, color=COL[ds])
    bx.axhline(0, color="0.35", lw=0.7); bx.set_xticks(xs); bx.set_xticklabels([c[1] for c in comps], fontsize=6)
    bx.set_ylabel("ΔMRR"); bx.set_title("HRCOV_PAT − G1, decomposed", loc="left")
    # (c) tkgl-icews
    cx = fig.add_subplot(gs[1, 1])
    systems = [("RE-GCN base", "RE-GCN"), ("RE-GCN BLEND", "RE-GCN + RecB blend"), ("LogCL base", "LogCL"), ("LogCL BLEND", "LogCL + RecB blend"),
               ("G1", "G1 (no neural base)"), ("HRCOV150 (LogCL)", "HRCOV, LogCL base"), ("HRCOV200 (RE-GCN)", "HRCOV, RE-GCN base")]
    vals = [B4["arms"][s]["MRR"] for s, _ in systems]
    yy = np.arange(len(systems))
    for i, ((s, lab), v) in enumerate(zip(systems, vals)):
        focal = s.startswith("HRCOV"); c = "#1f5fa8" if focal else ("#6a3d9a" if s == "G1" else "0.55")
        cx.plot([0.25, v], [i, i], color=c, lw=0.8); cx.plot(v, i, "o", ms=4, color=c)
        cx.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=6, color=c)
    cx.set_yticks(yy); cx.set_yticklabels([l for _, l in systems], fontsize=6); cx.set_xlim(0.25, 0.40)
    cx.set_xlabel("MRR, tkgl-icews test (4.65M queries)"); cx.set_title("G1 trails HRCOV on tkgl-icews", loc="left")
    for a_, L_ in ((ax, "a"), (bx, "b"), (cx, "c")): panel_letter(a_, L_)
    return fig
