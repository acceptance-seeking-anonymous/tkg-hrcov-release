"""Figure fig:case: rank ladder of the case-study queries, from results/case_study/."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import matplotlib
matplotlib.use("Agg")
from common import res, text, OUT
from _style import apply_figure_style, panel_letter
FIGOUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(OUT), "figures")); os.makedirs(FIGOUT, exist_ok=True)
# figcase_code.py -- rank-ladder figure of the three case-study queries (Manuscript A v6, Figure fig:case; v7.5: 5.0 in width, fonts >= 7 pt).
# Inputs: case_study_candidates.json (file 3b7022dd, ranks of base/BLEND/RecB/HR/HRCOV+BR), case_hcpat_ranks.json
# (bridge-free HRCOV ranks, seeds 0-2), labels from the same candidates file. Run inside the v5_build state (docs, LABS) after
import json, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
CASEC_ = json.loads(text("case_study_candidates.json")); LABS = CASEC_["labels"]; CHR_ = json.loads(text("case_hcpat_ranks.json"))
ccq = {q["i"]: q for v in CASEC_["candidates"].values() for q in v}
def hrcov(qi):
    k = CHR_["query_index"].index(qi); return [int(CHR_["rank_HRCOV_PAT"][f"s{s}"][k]) for s in range(3)]
CASES = [(80139, "S1", "#1B9E77", "o"), (24954, "S2", "#7570B3", "s"), (67287, "S3", "#222222", "^")]
LADDER = {}
for qi, lab, col, mk in CASES:
    R_ = ccq[qi]["ranks"]
    LADDER[lab] = {"q": qi, "base": int(R_["base"]), "BLEND": int(R_["BLEND"]), "RecB": int(R_["RecB_tr"]),
                   "HR": [int(R_[f"HR_s{s}"]) for s in range(3)], "HRCOV": hrcov(qi), "subject": LABS.get(ccq[qi]["s"]),
                   "year": ccq[qi]["t"]}
short = {"S1": "South Korea men's national football team", "S2": "Mexico men's national football team", "S3": "LB Châteauroux"}
xl = ["TiRGN\n(base)", "TiRGN$\\oplus$RecB\n(blend)", "RecB\n(trained)", "HR\n(top-50 re-rank)", "HRcov\n(this paper)"]
dod = {"S1": -0.10, "S2": 0.0, "S3": 0.10}
fig, ax = plt.subplots(figsize=(4.85, 3.2))
ax.axhline(50, color="#9A9A9A", lw=0.8, ls=(0, (4, 3)), zorder=0)
ax.text(4.42, 50 * 1.12, "base top-50 cut-off", ha="right", va="top", fontsize=7, color="#6E6E6E")
for qi, lab, col, mk in CASES:
    d = LADDER[lab]; xs = np.arange(5) + dod[lab]
    y0 = [d["base"], d["BLEND"], d["RecB"], d["HR"][0], d["HRCOV"][0]]
    ax.plot(xs, y0, color=col, lw=1.2, zorder=2)
    ax.scatter(xs, y0, s=22, marker=mk, facecolor=col if lab != "S3" else "white", edgecolor=col, linewidth=1.0, zorder=3,
               label=f"{lab}: {d['subject']} $\\rightarrow$ {short[lab]} ({d['year']})")
    for j, key in [(3, "HR"), (4, "HRCOV")]:
        lo, hi = min(d[key]), max(d[key])
        if hi > lo:
            ax.errorbar([xs[j]], [d[key][0]], yerr=[[d[key][0] - lo], [hi - d[key][0]]], fmt="none", ecolor=col, elinewidth=1.0, capsize=2.2, zorder=2)
ax.set_yscale("log"); ax.set_ylim(1300, 0.75)
ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])); ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
ax.set_xticks(range(5)); ax.set_xticklabels(xl); ax.get_xticklabels()[4].set_fontweight("bold")
ax.set_xlim(-0.45, 4.45); ax.set_ylabel("Filtered rank of the gold (log scale)", labelpad=6)
ax.text(-0.43, 0.84, "rank 1 = best", fontsize=7, color="#6E6E6E", ha="left", va="center")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=7, handlelength=1.2, borderaxespad=0.0, frameon=False, ncol=1)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGOUT, "fig_case_ladder.pdf")); fig.savefig(os.path.join(FIGOUT, "fig_case_ladder.png"), dpi=300)
