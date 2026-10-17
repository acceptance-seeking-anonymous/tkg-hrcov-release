# hardening / hd_figs.py -- two report figures from hardening_verdict.json (exec inside a Python session with
# VERDICT_PATH and OUT_PREFIX set; figure-style helpers loaded: apply_figure_style, panel_letter).
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

V = json.load(open(VERDICT_PATH))
apply_figure_style(sizes=(8, 7, 6))
DSN = {"polecat": "tkgl-polecat", "smallpedia": "tkgl-smallpedia"}
BLAB = {"regcn": "RE-GCN (pilot)", "logcl": "LogCL (pilot)", "tirgn": "TiRGN", "regcn_tuned": "RE-GCN (re-tuned)",
        "logcl_tuned": "LogCL (re-tuned)"}
C_BASE, C_CTRL, C_HR, C_HRCOV = "#9a9a9a", "#2171b5", "#fdae6b", "#d94801"

# ---------------- Figure 1: absolute official test MRR per base: base, best recurrency control, HR, HRCOV
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.55))
for ax, ds in zip(axes, ("polecat", "smallpedia")):
    D = V["datasets"][ds]
    bases = [b for b in D["bases"] if "missing" not in D["bases"][b] and "HRCOV" in D["bases"][b]["arms"]]
    bases = sorted(bases, key=lambda b: D["bases"][b]["valid_MRR_base"])     # strongest (valid) on top
    for i, b in enumerate(bases):
        A = D["bases"][b]["arms"]
        ctrl = max(A["BLEND"]["official"]["MRR"], A["POOL"]["official"]["MRR"])
        xs = [A["base"]["official"]["MRR"], ctrl, A["HR"]["official"]["MRR"], A["HRCOV"]["official"]["MRR"]]
        ax.plot([min(xs), max(xs)], [i, i], color="#d9d9d9", lw=0.8, zorder=1)
        ax.scatter([xs[0]], [i], s=22, color=C_BASE, zorder=3, label="base model alone" if i == 0 else None)
        ax.scatter([xs[1]], [i], s=22, color=C_CTRL, marker="s", zorder=3, label="best recurrency control\n(base + RecB blend or pool)" if i == 0 else None)
        ax.scatter([xs[2]], [i], s=22, color=C_HR, marker="D", zorder=3, label="history re-ranker (HR)" if i == 0 else None)
        ax.errorbar([xs[3]], [i], xerr=[A["HRCOV"]["official_sd"]["MRR"]], fmt="o", ms=5.5, color=C_HRCOV, ecolor="black",
                    elinewidth=0.6, capsize=1.5, zorder=4, label="HRCOV (claim; mean of 3 seeds)" if i == 0 else None)
    rtr = D["recb"]["rows"]["faith_tr"]["MRR"]
    ax.axvline(rtr, color="#4d4d4d", lw=0.8, ls="--", zorder=0)
    pub = {k: v for k, v in D.get("published", {}).items() if "RecB" not in k}
    k = max(pub, key=pub.get)
    ax.axvline(pub[k], color="#4d4d4d", lw=0.8, ls=":", zorder=0)
    ytop = len(bases) - 0.35
    ax.text(rtr, ytop, " RecB trained", ha="left", va="bottom", fontsize=6, color="#4d4d4d")
    ax.text(pub[k], ytop + 0.32, f" best published\n ({k.split(' (')[0]})", ha="left", va="bottom", fontsize=6, color="#4d4d4d")
    ax.set_yticks(range(len(bases))); ax.set_yticklabels([BLAB.get(b, b) for b in bases])
    ax.set_ylim(-0.6, len(bases) + 0.6)
    ax.set_xlabel("test MRR (official TGB evaluator)")
    ax.margins(x=0.06)
    ax.set_title(f"{DSN[ds]}: HRCOV above every recurrency control", loc="left")
    panel_letter(ax, "ab"[("polecat", "smallpedia").index(ds)])
axes[0].legend(loc="upper left", frameon=False, fontsize=6, bbox_to_anchor=(-0.02, -0.26), ncol=4, columnspacing=1.0, handletextpad=0.3)
fig.savefig(f"{OUT_PREFIX}fig1_absolute_mrr.png", dpi=300, bbox_inches="tight")
FIG1 = fig

# ---------------- Figure 2: per-stratum delta HRCOV - best recurrency control, two strongest bases
ORDER = ["R_rec_out (recurrence, gold outside base top-50)", "pool-absent (gold outside base top-50)", "gold in base top-50",
         "exact recurrence (s,r,o) before t", "no-pair-history", "first-seen (s or o)",
         "cold (< 50)", "mid (50-499)", "head (gold train freq >= 500)"]
SHORT = {ORDER[0]: "recurrence, gold outside base top-50", ORDER[1]: "gold outside base top-50", ORDER[2]: "gold inside base top-50",
         ORDER[3]: "exact (s,r,o) recurrence", ORDER[4]: "no (s,o) history", ORDER[5]: "first-seen entity",
         ORDER[6]: "cold gold (train freq < 50)", ORDER[7]: "mid gold (50-499)", ORDER[8]: "head gold (>= 500)"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw=dict(wspace=0.10))
COLS = ["#d94801", "#7f2704"]; MK = ["o", "s"]
for ax, ds in zip(axes, ("polecat", "smallpedia")):
    D = V["datasets"][ds]
    for k, b in enumerate(D["two_strongest"]):
        St = D["decision"]["per_base"][b]["strata"]
        ys = np.arange(len(ORDER)) + (k - 0.5) * 0.32
        first = True
        for y, nm in zip(ys, ORDER):
            if nm not in St: continue
            s = St[nm]; d = s["delta_HRCOV_minus_better"]; lo, hi = s["CI95"]
            ax.plot([lo, hi], [y, y], color=COLS[k], lw=1.0)
            ax.plot([d], [y], marker=MK[k], ms=3.5, color=COLS[k], ls="none",
                    label=f"{BLAB.get(b, b)} (vs its {D['decision']['per_base'][b]['better_comparator'].lower()})" if first else None)
            first = False
    ax.axvline(0, color="#4d4d4d", lw=0.7)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[o] for o in ORDER])
    if ds == "polecat": ax.invert_yaxis()
    ax.set_title(DSN[ds], loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax.margins(x=0.06)
    panel_letter(ax, "ab"[("polecat", "smallpedia").index(ds)])
fig.supxlabel("MRR difference, HRCOV minus best recurrency control (95% bootstrap CI)", fontsize=8, y=0.0)
fig.savefig(f"{OUT_PREFIX}fig2_strata_delta.png", dpi=300, bbox_inches="tight")
FIG2 = fig


def bbox_check(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    tl = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")) for ax in fig.axes}
    ov = [(a.get_text()[:30], b.get_text()[:30]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov += [(t.get_text()[:30], "spine") for t, bt in texts for s, bs in spines if bt.overlaps(bs) and t not in tl[s.axes]]
    return ov
