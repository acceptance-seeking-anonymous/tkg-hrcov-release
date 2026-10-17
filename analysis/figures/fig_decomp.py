"""Figure fig:decomp: where the MRR gain comes from (decomposition), from results/."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import matplotlib
matplotlib.use("Agg")
from common import res, text, OUT
from _style import apply_figure_style, panel_letter
FIGOUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(OUT), "figures")); os.makedirs(FIGOUT, exist_ok=True)
apply_figure_style(sizes=(8, 7, 7))
# figdecomp_code.py -- Manuscript A v7.5 (CognTKE row added; 5.0 in width, fonts >= 7 pt, panels stacked). v7.2: Figure fig:decomp (panel a split by the B3 common-pool intervention on the TiRGN bases). Run inside the v5_build state after apply_figure_style(sizes=(8,7,6)).
# Inputs: g1_verdict.json (RecB trained clean, G1 test MRR), hcpat_all_decision_bases.json (bridge-free HRCOV, better control),
# hardening_verdict.json and g3_verdict.json (HR, official evaluator). Writes fig_decomp.pdf/.png and fig_decomp_values.json.
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
HPj = json.loads(text("hcpat_all_decision_bases.json")); G1j = json.loads(text("g1_verdict.json")); hvj = json.loads(text("hardening_verdict.json")); g3j = json.loads(text("g3_verdict.json"))
ROWS = [("polecat", "polecat", "tirgn", "tkgl-polecat, TiRGN"), ("polecat", "polecat", "logcl", "tkgl-polecat, LogCL (pilot)"),
        ("smallpedia_G3_pool", "smallpedia", "tirgn", "tkgl-smallpedia, TiRGN"), ("smallpedia_G3_pool", "smallpedia", "regcn_tgb", "tkgl-smallpedia, RE-GCN (TGB config.)")]
V = []
for dsk, ds, b, lab in ROWS:
    pb = HPj["datasets"][dsk]["per_base"][b]
    hr = g3j["regcn_tgb_arms"]["HR"]["official"]["MRR"] if b == "regcn_tgb" else hvj["details"]["datasets"][ds]["bases"][b]["arms"]["HR"]["official"]["MRR"]
    V.append(dict(label=lab, RecB=G1j["datasets"][ds]["recb_row_MRR"], G1=G1j["datasets"][ds]["G1_test_MRR"], HRCOV=pb["HRCOV_PAT_MRR"], HR=hr,
                  control=pb["better_control_MRR"], control_name=pb["better_control"]))
G8j = json.load(open(res("g8_verdict.json")))  # CognTKE (separate rule): no common-pool split
V.insert(0, dict(label="tkgl-polecat, CognTKE", RecB=V[0]["RecB"], G1=V[0]["G1"], HRCOV=G8j["mrr"]["HRCOV"], HR=G8j["mrr"]["HR"],
                 control=G8j["mrr"]["BLEND"], control_name="BLEND"))
B3 = json.load(open(res("b3_common_pool_base_ablation.json")))
def _b3(ds, name):
    return [c for c in B3["datasets"][ds]["comparisons"] if c["comparison"].startswith(name)][0]["MRR"]["delta"]
SPLIT = {}
for lab, ds in [("tkgl-polecat, TiRGN", "tkgl-polecat"), ("tkgl-smallpedia, TiRGN", "tkgl-smallpedia")]:
    SPLIT[lab] = (_b3(ds, "FULL - EV_NF_PT"), _b3(ds, "EV_NF_PT - EV_NF_RB"), _b3(ds, "EV_NF_RB - G1"))
json.dump({"rows": V, "b3_split": SPLIT}, open(os.path.join(OUT, "fig_decomp_values.json"), "w"), indent=1)
C_G1, C_HR, C_COV = "#2A9D8F", "#F2A65A", "#C8501E"
C_S1, C_S2, C_S3, C_ND = "#4D4D4D", "#8C8C8C", "#C8C8C8", "#E3E3E3"
fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.5))
fig.subplots_adjust(left=0.40, right=0.93, top=0.95, bottom=0.17, hspace=0.95)
y = np.arange(len(V))[::-1]
ax = axes[0]
a1 = [v["G1"] - v["RecB"] for v in V]
ax.barh(y, a1, color=C_G1, height=0.6, label="G1 $-$ RecB (learned ranker, no base)")
done = set()
for yi, v, s1 in zip(y, V, a1):
    tot = v["HRCOV"] - v["G1"]
    if v["label"] in SPLIT:
        t1, t2, t3 = SPLIT[v["label"]]; left = s1
        for w, c, lab in [(t1, C_S1, "base info. + pattern features (common pool)"), (t2, C_S2, "candidate source (common inputs)"),
                          (t3, C_S3, "pool width, features, tail order")]:
            ax.barh(yi, w, left=left, color=c, height=0.6, label=None if lab in done else lab); done.add(lab); left += w
    else:
        ax.barh(yi, tot, left=s1, color=C_ND, height=0.6, hatch="////", edgecolor="#A0A0A0", linewidth=0.0,
                label=None if "nd" in done else "HRcov $-$ G1, not decomposed"); done.add("nd")
    ax.text(s1 + tot + 0.0015, yi, f"+{s1 + tot:.3f}", va="center", ha="left", fontsize=7)
ax.set_xlim(0, 0.118); ax.set_xlabel("MRR gain over trained RecB", labelpad=2)
ax.set_yticks(y); ax.set_yticklabels([v["label"] for v in V])
ax.set_title("(a) From recurrence baseline to HRcov", loc="left", fontsize=8)
ax.legend(loc="upper left", bbox_to_anchor=(0.01, ax.get_position().y0 - 0.075), bbox_transform=fig.transFigure, frameon=False, fontsize=7, ncol=2,
          handlelength=1.2, columnspacing=0.8)
ax = axes[1]
b1 = [v["HR"] - v["control"] for v in V]; b2 = [v["HRCOV"] - v["HR"] for v in V]
ax.barh(y, b1, color=C_HR, height=0.6, label="re-ranking the top-50 (HR $-$ control)")
ax.barh(y, b2, left=b1, color=C_COV, height=0.6, label="pattern block (HRcov $-$ HR)")
for yi, s1, s2 in zip(y, b1, b2):
    ax.text(s1 + s2 + 0.0015, yi, f"+{s1 + s2:.3f}", va="center", ha="left", fontsize=7)
ax.set_xlim(0, 0.072); ax.set_xlabel("MRR gain over the better recurrence control", labelpad=2)
ax.set_yticks(y); ax.set_yticklabels([v["label"] for v in V])
ax.set_title("(b) In-pool ordering vs added candidates", loc="left", fontsize=8)
ax.legend(loc="upper left", bbox_to_anchor=(0.01, ax.get_position().y0 - 0.075), bbox_transform=fig.transFigure, frameon=False, fontsize=7, ncol=2,
          handlelength=1.2, columnspacing=0.8)
for a in axes: a.margins(y=0.08)
mpl.rcParams["savefig.bbox"] = "standard"
fig.savefig(os.path.join(FIGOUT, "fig_decomp.pdf"), bbox_inches=None); fig.savefig(os.path.join(FIGOUT, "fig_decomp.png"), dpi=300, bbox_inches=None)
