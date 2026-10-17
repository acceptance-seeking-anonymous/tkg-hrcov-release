"""Figure fig:basestrength; run analysis/fig_basestrength_data.py first (writes results/tables/base_strength.json)."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import matplotlib
matplotlib.use("Agg")
from common import res, text, OUT
from _style import apply_figure_style, panel_letter
FIGOUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(OUT), "figures")); os.makedirs(FIGOUT, exist_ok=True)
# Figure: margin of HRCOV over the better same-base recurrence control against the strength of the frozen base.
# Input: v75_derived.json (written by v75_fill.py from the experiment JSONs; key "base_strength", 18 runs).
# Rendered at 5.0 in (12.7 cm) width; smallest font 7 pt. Greyscale-safe: one marker shape and line style per dataset;
# filled markers = decision bases, open markers = other trained bases.
import json
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "legend.frameon": False})
D = json.load(open(os.path.join(OUT, "base_strength.json"), encoding="utf-8"))
R = D["base_strength"]
STY = {  # Okabe-Ito colours; marker and line style carry the dataset in greyscale
    "tkgl-polecat": ("#D55E00", "o", "-"), "tkgl-smallpedia": ("#0072B2", "s", "--"), "tkgl-icews": ("#009E73", "^", "-."),
    "ICEWS18": ("#CC79A7", "D", ":"), "GDELT": ("#E69F00", "v", (0, (5, 1.5, 1, 1.5))), "GDELT25X": ("#000000", "*", "-")}
SHORT = {"RE-GCN (pilot)": "RE-GCN", "LogCL (pilot)": "LogCL", "TiRGN": "TiRGN", "CognTKE": "CognTKE", "RE-GCN (TGB config.)": "RE-GCN (TGB)",
         "RE-GCN (re-tuned)": "RE-GCN (re-tuned)"}
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.axhline(0.005, color="#7F7F7F", lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.text(0.655, 0.0058, "registered margin 0.005", ha="right", va="bottom", fontsize=7, color="#555555")
for ds, (c, mk, ls) in STY.items():
    rr = sorted([r for r in R if r["dataset"] == ds], key=lambda r: r["base_mrr"])
    if len(rr) > 1:
        ax.plot([r["base_mrr"] for r in rr], [r["margin"] for r in rr], color=c, lw=1.0, ls=ls, zorder=2)
    for r in rr:
        ax.scatter(r["base_mrr"], r["margin"], s=(70 if mk == "*" else 30), marker=mk, zorder=3, linewidths=1.0,
                   facecolors=(c if r["decision"] else "white"), edgecolors=c)
# direct labels for the two datasets on which several bases were trained
OFF = {("tkgl-polecat", "RE-GCN (pilot)"): (6, 2), ("tkgl-polecat", "LogCL (pilot)"): (6, 3), ("tkgl-polecat", "TiRGN"): (6, 3),
       ("tkgl-polecat", "CognTKE"): (6, -2), ("tkgl-smallpedia", "RE-GCN (pilot)"): (-6, 0), ("tkgl-smallpedia", "TiRGN"): (1, -11),
       ("tkgl-smallpedia", "RE-GCN (TGB config.)"): (-7, -9), ("tkgl-smallpedia", "LogCL (pilot)"): (0.1, 9), ("tkgl-smallpedia", "RE-GCN (re-tuned)"): (6, 4)}
for r in R:
    k = (r["dataset"], r["base"])
    if k in OFF:
        dx, dy = OFF[k]
        ax.annotate(SHORT[r["base"]], (r["base_mrr"], r["margin"]), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha=("left" if dx > 0 else "right"), va="center", color=STY[r["dataset"]][0])
ax.set_xlabel("MRR of the frozen base alone")
ax.set_ylabel("HRcov $-$ better control (MRR)")
ax.set_xlim(0.14, 0.67); ax.set_ylim(0.0, 0.072)
h = [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.0, markersize=(8 if mk == "*" else 5), markerfacecolor=c, label=ds) for ds, (c, mk, ls) in STY.items()]
h += [Line2D([], [], color="#555555", marker="o", ls="none", markersize=5, markerfacecolor="#555555", label="decision base"),
      Line2D([], [], color="#555555", marker="o", ls="none", markersize=5, markerfacecolor="white", label="other trained base")]
ax.legend(handles=h, loc="upper right", ncol=2, handlelength=2.4, columnspacing=1.0, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIGOUT, "fig_basestrength.pdf")); fig.savefig(os.path.join(FIGOUT, "fig_basestrength.png"), dpi=300)
