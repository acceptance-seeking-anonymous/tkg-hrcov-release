"""analysis/figures/_style.py -- matplotlib settings shared by the figure scripts (font sizes in pt: base, legend, ticks)."""
import matplotlib as mpl


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {"font.family": "sans-serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base, "legend.fontsize": secondary,
          "xtick.labelsize": tick, "ytick.labelsize": tick, "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
          "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
          "axes.spines.top": boxed, "axes.spines.right": boxed, "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
          "axes.grid": bool(grid), "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
          "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelweight": "normal", "lines.linewidth": 1.2,
          "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42}
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold", fontsize=fontsize, va="bottom", ha="left")
