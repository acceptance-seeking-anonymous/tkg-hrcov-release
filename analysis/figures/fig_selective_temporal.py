"""Figures fig:selective and fig:temporal, from results/v7/a6_*.json and a7_*.json."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import matplotlib
matplotlib.use("Agg")
from common import res, text, OUT
from _style import apply_figure_style, panel_letter
FIGOUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(OUT), "figures")); os.makedirs(FIGOUT, exist_ok=True)
apply_figure_style(sizes=(8, 7, 7))
# figv7a_code.py -- Manuscript A v7 (later analyses). Figures fig:selective (A7) and fig:temporal (A6).
# v7.5: rendered at 5.0 in (12.7 cm) width with every font >= 7 pt; one marker and line style per dataset (greyscale-safe).
# Run after apply_figure_style(sizes=(8,7,7)). Inputs: a7_selective_gain_loss.json, a6_temporal_sensitivity.json
# (hardening-agent outputs, copied verbatim). Writes fig_selective.pdf/.png, fig_temporal.pdf/.png.
import json, numpy as np, matplotlib.pyplot as plt
A7J = json.load(open(res("a7_selective_gain_loss.json"))); A6J = json.load(open(res("a6_temporal_sensitivity.json")))
DSL = [("tkgl-polecat", "tkgl-polecat"), ("tkgl-smallpedia", "tkgl-smallpedia"), ("ICEWS18", "ICEWS18"), ("GDELT", "GDELT")]
COL = {"tkgl-polecat": "#2A6FB0", "tkgl-smallpedia": "#D9822B", "ICEWS18": "#3A9A5B", "GDELT": "#8E5BB5"}
TICK = ["<0.1", "0.1–0.2", "0.2–0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", "≥0.9"]
NMIN = 100
MK = {"tkgl-polecat": "o", "tkgl-smallpedia": "s", "ICEWS18": "D", "GDELT": "v"}
LS = {"tkgl-polecat": "-", "tkgl-smallpedia": "--", "ICEWS18": "-.", "GDELT": (0, (1, 1.2))}

# ---------------- fig:selective (A7)
fig, axes = plt.subplots(1, 3, figsize=(5.0, 2.55), gridspec_kw=dict(wspace=0.62))
for ds, lab in DSL:
    bins = A7J["datasets"][ds]
    idx = [i for i, b in enumerate(bins) if b["n"] >= NMIN]
    x = np.array(idx, float) + {"tkgl-polecat": -0.15, "tkgl-smallpedia": -0.05, "ICEWS18": 0.05, "GDELT": 0.15}[ds]
    d = np.array([bins[i]["PAT_minus_base"]["delta"] for i in idx])
    lo = np.array([bins[i]["PAT_minus_base"]["ts"]["CI95"][0] for i in idx]); hi = np.array([bins[i]["PAT_minus_base"]["ts"]["CI95"][1] for i in idx])
    bl = np.array([bins[i]["BLEND_minus_base"]["delta"] for i in idx])
    axes[0].errorbar(x, d, yerr=[d - lo, hi - d], color=COL[ds], marker=MK[ds], ms=3.2, lw=1.0, ls=LS[ds], elinewidth=0.8, capsize=1.6, label=lab)
    axes[0].plot(x, bl, color=COL[ds], ls=":", lw=0.7, marker=MK[ds], ms=2.0, mfc="white")
    net = np.array([bins[i]["frac_improved"] - bins[i]["frac_harmed"] for i in idx])
    axes[1].plot(x, net, color=COL[ds], marker=MK[ds], ms=3.2, lw=1.0, ls=LS[ds])
    ov = np.array([bins[i]["P_override_given_base_top1"] for i in idx])
    axes[2].plot(x, ov, color=COL[ds], marker=MK[ds], ms=3.2, lw=1.0, ls=LS[ds])
for a in axes:
    a.set_xticks(range(7)); a.set_xticklabels(TICK, fontsize=7, rotation=60, ha="right", rotation_mode="anchor"); a.axhline(0, color="#888888", lw=0.6); a.set_xlabel("base top-1 prob.")
axes[0].set_ylabel(r"$\Delta$MRR vs base"); axes[0].set_title("(a) gain over base", loc="left", fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=4, handlelength=2.2,
           bbox_to_anchor=(0.5, 1.10), columnspacing=0.9)
axes[1].set_ylabel("improved $-$ harmed"); axes[1].set_title("(b) net share helped", loc="left", fontsize=8)
axes[2].set_ylabel("correct top-1 displaced"); axes[2].set_title("(c) overrides", loc="left", fontsize=8)
fig.savefig(os.path.join(FIGOUT, "fig_selective.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIGOUT, "fig_selective.png"), dpi=300, bbox_inches="tight")

# ---------------- fig:temporal (A6)
PAN = [("tkgl-polecat", "HRCOV_PAT_minus_POOL", "tkgl-polecat, TiRGN: HRCOV $-$ RecB-POOL"),
       ("tkgl-smallpedia", "HRCOV_PAT_minus_BLEND", "tkgl-smallpedia, TiRGN: HRCOV $-$ BASE⊕RecB"),
       ("ICEWS18", "HRCOV_PAT_minus_BLEND", "ICEWS18, TiRGN: HRCOV $-$ BASE⊕RecB"),
       ("GDELT", "HRCOV_PAT_minus_BLEND", "GDELT, TiRGN: HRCOV $-$ BASE⊕RecB")]
fig2, ax2 = plt.subplots(2, 2, figsize=(5.0, 3.75), gridspec_kw=dict(wspace=0.36, hspace=0.95))
for a, (ds, key, title) in zip(ax2.ravel(), PAN):
    m = A6J["datasets"][ds][key]; p = m["per_ts"]
    n = np.array(p["n"], float); dlt = np.array(p["delta"]); se = np.array(p["se_normal"]); t = np.arange(len(n))
    run = np.cumsum(n * dlt) / np.cumsum(n)
    a.fill_between(t, dlt - 1.96 * se, dlt + 1.96 * se, color=COL[ds], alpha=0.18, lw=0)
    a.plot(t, dlt, color=COL[ds], lw=0.9, marker=MK[ds] if len(t) < 60 else None, ms=2.6, label="per-timestamp $\\Delta$ ($\\pm$1.96 SE)")
    a.plot(t, run, color="black", lw=0.9, ls="--", label="cumulative mean")
    a.axhline(0, color="#888888", lw=0.6); a.axhline(0.005, color="#888888", lw=0.6, ls=":", label="0.005 margin")
    a.set_title(title.split(": ")[0] + f": {m['delta']:+.4f}" + "\n" + f"leave-out [{m['lopo_min']:+.4f}, {m['lopo_max']:+.4f}]", loc="left", fontsize=7)
    a.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
for a in ax2[1]: a.set_xlabel("test timestamp (index)", labelpad=1)
for a in ax2[:, 0]: a.set_ylabel(r"$\Delta$MRR")
fig2.legend(*ax2[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=0.8)
fig2.savefig(os.path.join(FIGOUT, "fig_temporal.pdf"), bbox_inches="tight"); fig2.savefig(os.path.join(FIGOUT, "fig_temporal.png"), dpi=300, bbox_inches="tight")
