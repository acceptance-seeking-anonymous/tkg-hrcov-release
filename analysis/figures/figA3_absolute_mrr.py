"""Supplement figure: test MRR per frozen base (absolute values), from results/."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import matplotlib
matplotlib.use("Agg")
from common import res, text, OUT
from _style import apply_figure_style, panel_letter
FIGOUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(OUT), "figures")); os.makedirs(FIGOUT, exist_ok=True)
import json
import re
import csv
import io
import pandas as _pdfig
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
with open(res("hardening_verdict.json"), encoding="utf-8") as f:
    hv = json.load(f)

with open(res("dataset_profiles.json"), encoding="utf-8") as f:
    dp = json.load(f)

# Load G3 verdict
G3J = json.loads(open(res("g3_verdict.json")).read())
G3J_arms = G3J["regcn_tgb_arms"]

# Load HRCOV_PAT (bridge-free) rows
hcp = json.loads(open(res("hcpat_definitional.json")).read())

PUBV_FIG = _pdfig.read_csv(res("published_rows_verified.csv"))


def md_tables(doc):
    L = doc.splitlines(); out=[]; i=0
    while i < len(L)-1:
        if L[i].startswith("|") and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", L[i+1]):
            hdr=[c.strip() for c in L[i].strip().strip("|").split("|")]; rows=[]; j=i+2
            while j < len(L) and L[j].startswith("|"):
                rows.append([c.strip() for c in L[j].strip().strip("|").split("|")]); j+=1
            out.append((i,hdr,rows)); i=j
        else: i+=1
    return out

with open(res("smallpedia_pilot_report.md"), encoding="utf-8") as f:
    smallpedia_report = f.read()

bridge_report = open(res("bridge_timesafe_report.md")).read()

BASE_NAME = {"regcn": "RE-GCN (pilot)", "logcl": "LogCL (pilot)", "tirgn": "TiRGN",
             "regcn_tuned": "RE-GCN (re-tuned)", "logcl_tuned": "LogCL (re-tuned)",
             "regcn_tgb": "RE-GCN (TGB config.)", "cogntke": "CognTKE"}

def _hcpat_rows():
    out = {}
    pc = hcp["datasets"]["polecat"]["per_base"]["tirgn"]; out[("tkgl-polecat", "tirgn")] = (pc["HRCOV_PAT_MRR"], None, None, None, None, "a")
    for b in ["regcn_tuned"]:
        v = hcp["datasets"]["smallpedia_hardening_pool"]["per_base"][b]; out[("tkgl-smallpedia", b)] = (v["HRCOV_PAT_MRR"], None, None, None, None, "a")
    v = hcp["datasets"]["smallpedia_G3_pool"]["per_base"]["regcn_tgb"]; out[("tkgl-smallpedia", "regcn_tgb")] = (v["HRCOV_PAT_MRR"], None, None, None, None, "a")
    _, hdr, rs = [t for t in md_tables(bridge_report) if t[1][:3] == ["base", "system", "MRR (3-seed mean)"]][0]
    for r in rs:
        if r[1] == "HRCOV_PAT_old":
            out[("tkgl-polecat", r[0])] = (float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), "b")
    _, hdr, rs = [t for t in md_tables(smallpedia_report) if t[1][:2] == ["base", "system"] and "Δ vs HRCOV [95% CI]" in t[1]][0]
    bm = {"RE-GCN": "regcn", "LogCL": "logcl", "TiRGN": "tirgn"}
    for r in rs:
        if r[1] == "`HRCOV_PAT`":
            out[("tkgl-smallpedia", bm[r[0]])] = (float(r[2]), float(r[6]), float(r[3]), float(r[4]), float(r[5]), "b")
    return out

apply_figure_style(sizes=(8,7,7))
R = hv["fulltest_rows"]
HP = _hcpat_rows()
# v5.1: bridge-free HRCOV on all four decision bases (hcpat_all_decision_bases.json 54883d49)
_H2 = json.load(open(res("hcpat_all_decision_bases.json")))
HP[("tkgl-polecat", "logcl")] = (_H2["datasets"]["polecat"]["per_base"]["logcl"]["HRCOV_PAT_MRR"], None, None, None, None, "a")
HP[("tkgl-smallpedia", "tirgn")] = (_H2["datasets"]["smallpedia_G3_pool"]["per_base"]["tirgn"]["HRCOV_PAT_MRR"], None, None, None, None, "a")
A3 = G3J_arms
# v7.4: CognTKE as a frozen base on tkgl-polecat (g8_verdict.json, file version)
G8V = json.load(open(res("g8_verdict.json")))
HP[("tkgl-polecat", "cogntke")] = (G8V["mrr"]["HRCOV"], None, None, None, None, "a")

def val(ds, b, arm):
    if b == "cogntke":
        return G8V["mrr"][{"BLEND": "BLEND", "POOL": "POOL", "base": "base", "HR": "HR", "HRCOV": "HRCOV"}[arm]]
    if b == "regcn_tgb":
        return A3[{"BLEND": "BLEND", "POOL": "POOL", "base": "base", "HR": "HR", "HRCOV": "HRCOV"}[arm]]["official"]["MRR"]
    return [r_ for r_ in R if r_["dataset"] == ds and r_["base"] == b and r_["arm"] == arm][0]["MRR"]

COL = {"base": "#9a9a9a", "ctl": "#2f6db3", "HR": "#f2a65a", "HRCOV": "#c8501e"}

SBV = json.load(open(res("icews_stageB_verdict.json"), encoding="utf-8"))
def sbrow(base, prefix):
    return [r_ for r_ in SBV["rows"] if r_["base"] == base and r_["system"].startswith(prefix)][0]
def val_ic(b, arm):
    if arm == "base": return sbrow(b, b + " base")["MRR"]
    if arm == "ctl": return max(sbrow(b, b + "(+)RecB blend")["MRR"], sbrow(b, "RecB-POOL_" + b)["MRR"])
    if arm == "HR": return [r_ for r_ in SBV["rows"] if r_["base"] == b and r_["system"] == "HR"][0]["MRR"]
    if arm == "HRCOV": return SBV["decision"][b]["HRCOV_test_MRR"]
COL = {"base": "#9a9a9a", "ctl": "#2f6db3", "HR": "#f2a65a", "HRCOV": "#c8501e"}
panels = [("tkgl-polecat", ["regcn", "logcl", "tirgn", "cogntke"]),
          ("tkgl-smallpedia", ["regcn", "logcl", "logcl_tuned", "regcn_tuned", "regcn_tgb", "tirgn"]),
          ("tkgl-icews", ["regcn", "logcl"])]
BN_IC = {"regcn": "RE-GCN (full train)", "logcl": "LogCL (full train)"}
fig = plt.figure(figsize=(5.0, 5.2))  # v7.5: 12.7 cm width, fonts >= 7 pt; base marker is a triangle (greyscale-safe)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.0, 0.62], wspace=1.15, hspace=0.75)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
refstyle = {"RecB trained (a, b: TGB script;\nc: re-implementation)": "--",
            "TGB 2.0 RE-GCN row": (0, (1, 1.5)),
            "best published row (a: CognTKE;\nb: CountTRuCoLa; c: EdgeReMIND)": (0, (4, 1, 1, 1))}
for ax, (ds, order) in zip(axes, panels):
    for yi, b in enumerate(order):
        if ds == "tkgl-icews":
            base, ctl, hr, hc, hp = val_ic(b, "base"), val_ic(b, "ctl"), val_ic(b, "HR"), val_ic(b, "HRCOV"), True
        else:
            base = val(ds, b, "base"); ctl = max(val(ds, b, "BLEND"), val(ds, b, "POOL")); hr = val(ds, b, "HR")
            hp = HP.get((ds, b)); hc = hp[0] if hp else val(ds, b, "HRCOV")
        lo, hi = min(base, ctl, hr, hc), max(base, ctl, hr, hc)
        ax.plot([lo, hi], [yi, yi], color="#dddddd", lw=1.2, zorder=1)
        ax.scatter(base, yi, s=26, marker="v", color=COL["base"], zorder=3)
        ax.scatter(ctl, yi, s=24, marker="s", color=COL["ctl"], zorder=3)
        ax.scatter(hr, yi, s=26, marker="D", color=COL["HR"], zorder=4)
        if hp:
            ax.scatter(hc, yi, s=40, marker="o", color=COL["HRCOV"], zorder=5, edgecolors="white", linewidths=0.4)
        else:
            ax.scatter(hc, yi, s=40, marker="o", facecolors="white", edgecolors=COL["HRCOV"], linewidths=1.2, zorder=5)
    if ds == "tkgl-icews":
        recb = [r_ for r_ in SBV["rows"] if r_["system"].startswith("RecB (hardening re-implementation, window 0, faith_tr)")][0]["MRR"]
        pub_regcn = SBV["leaderboard_reference"]["rows"]["RE-GCN"]
    else:
        recb = [r_ for r_ in R if r_["dataset"] == ds and r_["arm"] == "RecB trained (TGB train mode, faithful)"][0]["MRR"]
        pub_regcn = [r_ for r_ in R if r_["dataset"] == ds and r_["arm"] == "published: RE-GCN (TGB 2.0)"][0]["MRR"]
    _pv_ = PUBV_FIG[(PUBV_FIG.dataset == ds) & PUBV_FIG.MRR.notna() & ~PUBV_FIG.status.str.startswith("omitted")]
    pub_best = float(_pv_.MRR.max())
    for xv, (lab, ls) in zip([recb, pub_regcn, pub_best], refstyle.items()):
        ax.axvline(xv, color="#555555", ls=ls, lw=0.8, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([(BN_IC if ds == "tkgl-icews" else BASE_NAME)[b] for b in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("test MRR")
    ax.set_title(f"{ds}", loc="left")
axes[0].set_xlim(0.155, 0.315)
axes[1].set_xlim(0.40, 0.685)
axes[2].set_xlim(0.17, 0.39)
panel_letter(axes[0], "a", dx=-0.95)
panel_letter(axes[1], "b", dx=-0.85)
panel_letter(axes[2], "c", dx=-0.95, dy=1.06)
handles = [Line2D([], [], ls="", marker="v", ms=5, color=COL["base"], label="base alone"),
           Line2D([], [], ls="", marker="s", ms=5, color=COL["ctl"], label="better recurrence control"),
           Line2D([], [], ls="", marker="D", ms=5, color=COL["HR"], label="HR (top-50 re-ranker)"),
           Line2D([], [], ls="", marker="o", ms=6, color=COL["HRCOV"], label="HRcov (mean of 3 seeds)"),
           Line2D([], [], ls="", marker="o", ms=6, markerfacecolor="white", markeredgecolor=COL["HRCOV"], label="HRcov+BR (no bridge-free run)")] + \
          [Line2D([], [], color="#555555", ls=ls, lw=0.8, label=lab) for lab, ls in refstyle.items()]
lax = fig.add_subplot(gs[1, 1]); lax.axis("off")
fig.legend(handles=handles, loc="lower left", ncol=1, frameon=False, fontsize=7, handletextpad=0.4, borderaxespad=0.0, bbox_to_anchor=(0.50, 0.05))
fig.subplots_adjust(bottom=0.08, left=0.25, right=0.985, top=0.95)
mpl.rcParams["savefig.bbox"] = "standard"
fig.savefig(os.path.join(FIGOUT, "figA3_absolute_mrr.png"), dpi=300)
fig.savefig(os.path.join(FIGOUT, "figA3_absolute_mrr.pdf"))
