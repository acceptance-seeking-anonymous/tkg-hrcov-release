#!/usr/bin/env python3
"""make_fresh_window_table.py -- rebuild the fresh-window table of Manuscript A (\\label{tab:fresh}; Table 9 in v7.9, Table 10
in the v8.0 text, where Table 9 is the ladder table tab:ladder) from the released fresh-window files and check every cell
against the manuscript TeX.  make_table9.py in this directory is an alias kept under the name used in the review.

Inputs (released with the fresh-window experiment):
  --results   fresh_window_results.json      metrics per system (official ties-averaged ranks), deltas, latency table (full precision)
  --verdict   fresh_window_verdict.json      registered margins (point, query and timestamp-block bootstrap bounds), calls, outcome
  --latency   fresh_window_latency_table.csv latency table as released (4 decimals); cross-checked against the JSON copy
  --build     build_summary.json             GDELTFW build summary (entity count for the footnote)
  --tex       manuscript TeX containing \\label{tab:fresh}
Outputs: --out CSV with one line per checked cell (panel, row, column, rebuilt, tex, match) and --tex_out with the rebuilt
LaTeX of both panels. Exit status 1 if any cell differs.  The defaults point at the copies in this repository; the manuscript
TeX is not part of the repository (the numbers of the table are also checked by analysis/check_manuscript_numbers.py against
tests/expected/).

Formatting rules (those of the manuscript): MRR / H@k and Delta-MRR / bounds with 4 decimals, explicit sign on contrasts;
latency in ms per query = s per 1,000 queries with 3 decimals, end to end. Rounding is Python's format() on the stored
double. Panel (b) uses the one-sided 95% bounds L and U of the verdict file (B = 2000 shared draws).

  python3 fresh_window/reproduce/make_fresh_window_table.py --tex manuscriptA_tois_v7.9.tex   (repository root; the input
      defaults are results/fresh_window/fresh_window_results.json, registration/fresh_window_verdict.json,
      results/fresh_window/fresh_window_latency_table.csv and fresh_window/01_window_build/build_summary.json)
"""
import argparse, csv, json, os, re, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root

ROWS_A = [  # (TeX row label, results key, latency-table system name)
    ("Frozen base (TiRGN)", "base", "TiRGN base"),
    ("BASE$\\oplus$RecB (reference)", "BLEND", "BLEND (TiRGN + RecB)"),
    ("\\HRCOV{}", "HRcov", "HRcov"),
    ("SA (matched standalone, no base)", "SA", "SA"),
    ("CountTRuCoLa, default configuration, $+$ base (blend)$^{\\dagger}$", "external_comparator", "CountTRuCoLa comparator (C0 blend)"),
]
ROWS_B = [  # (TeX row label, margin key, call source)
    ("\\HRCOV{} $-$ SA", "HRcov-SA", "SA"),
    ("\\HRCOV{} $-$ CountTRuCoLa comparator$^{\\dagger}$", "HRcov-external_comparator", "external_comparator"),
    ("\\HRCOV{} $-$ BASE$\\oplus$RecB", "HRcov-BLEND", "(reference)"),
    ("\\HRCOV{} $-$ frozen base", "HRcov-base", "--"),
]
COLS_A = ["MRR", "H@1", "H@10", "ms/query batch size 1", "ms/query batched"]
COLS_B = ["Delta MRR", "query L", "query U", "ts-block L", "ts-block U", "Call"]


def f4(x): return f"{x:.4f}"
def s4(x): return ("+" if x >= 0 else "-") + f"{abs(x):.4f}"
def f3(x): return f"{x:.3f}"


def rebuild(R, V, lat_csv):
    lat = {(d["system"], d["setting"]): d for d in R["latency_table_s_per_1000_queries"]}
    A = []
    for lab, key, lsys in ROWS_A:
        m = R["metrics"][key]["official_ties_averaged"]
        b1 = lat[(lsys, "batch size 1")]["end_to_end_s_per_1k"]; bb = lat[(lsys, "batched (production)")]["end_to_end_s_per_1k"]
        for setting, v in (("batch size 1", b1), ("batched (production)", bb)):   # released CSV (4 dp) must agree with the JSON
            c = lat_csv[(lsys, setting)]
            assert abs(c - v) <= 5.0001e-5, (lsys, setting, c, v)
        A.append((lab, [f4(m["MRR"]), f4(m["H1"]), f4(m["H10"]), f3(b1), f3(bb)]))
    B = []
    for lab, key, call in ROWS_B:
        mg = V["margins"][key]; d = R["deltas_official"][key]           # verdict margins == results deltas (same computation)
        assert abs(mg["point"] - d["delta"]) < 1e-12 and abs(mg["query"]["L"] - d["query"]["L"]) < 1e-12 \
            and abs(mg["ts_block"]["U"] - d["ts"]["U"]) < 1e-12, key
        cl = V["calls"][call] if call in V["calls"] else call
        B.append((lab, [s4(mg["point"]), s4(mg["query"]["L"]), s4(mg["query"]["U"]), s4(mg["ts_block"]["L"]), s4(mg["ts_block"]["U"]), cl]))
    return A, B


def tex_norm(c):
    c = c.strip()
    c = c.replace("$+$", "+").replace("$-$", "-").replace("{,}", ",")
    return re.sub(r"\s+", " ", c).strip()


def parse_tex(tex):
    i = tex.index("\\label{tab:fresh}"); s = tex.rfind("\\begin{table", 0, i); e = tex.index("\\end{table}", i)
    T = tex[s:e]; line0 = tex[:s].count("\n") + 1
    panels = []
    for m in re.finditer(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", T, re.S):
        body = m.group(1); mid = body.index("\\midrule"); bot = body.index("\\bottomrule")
        rows = [r.strip() for r in body[mid + len("\\midrule"):bot].split("\\\\") if r.strip()]
        panels.append([[x.strip() for x in r.split("&")] for r in rows])
    foot = re.search(r"\\begin\{minipage\}.*?\\end\{minipage\}", T, re.S).group(0)
    cap = re.search(r"\\caption\{(.*?)\}\n", T, re.S).group(1)
    return panels, foot, cap, line0


def main():
    ap = argparse.ArgumentParser()
    J = lambda *p: os.path.join(_ROOT, *p)
    ap.add_argument("--results", default=J("results", "fresh_window", "fresh_window_results.json"))
    ap.add_argument("--verdict", default=J("registration", "fresh_window_verdict.json"))
    ap.add_argument("--latency", default=J("results", "fresh_window", "fresh_window_latency_table.csv"))
    ap.add_argument("--build", default=J("fresh_window", "01_window_build", "build_summary.json"))
    ap.add_argument("--tex", required=True); ap.add_argument("--out", default="fresh_window_table_check.csv")
    ap.add_argument("--tex_out", default="fresh_window_table_rebuilt.tex")
    a = ap.parse_args()
    R = json.load(open(a.results)); V = json.load(open(a.verdict)); BS = json.load(open(a.build))
    lat_csv = {(r["system"], r["setting"]): float(r["end_to_end_s_per_1k"]) for r in csv.DictReader(open(a.latency))}
    A, B = rebuild(R, V, lat_csv)
    panels, foot, cap, line0 = parse_tex(open(a.tex).read())
    assert len(panels) == 2 and len(panels[0]) == len(A) and len(panels[1]) == len(B), [len(p) for p in panels]
    out = []
    for pnl, rebuilt, cols in (("a", A, COLS_A), ("b", B, COLS_B)):
        texrows = panels[0] if pnl == "a" else panels[1]
        for (lab, vals), tr in zip(rebuilt, texrows):
            out.append((pnl, lab, "row label", lab, tr[0], tr[0].strip() == lab))
            if pnl == "a":
                tcells = [tex_norm(x) for x in tr[1:]]
            else:   # Delta, [L, U], [L, U], call
                q = re.findall(r"[+-]?\d\.\d+", tex_norm(tr[2])); t = re.findall(r"[+-]?\d\.\d+", tex_norm(tr[3]))
                tcells = [tex_norm(tr[1])] + q + t + [tex_norm(tr[4])]
            for c, v, t_ in zip(cols, vals, tcells):
                out.append((pnl, lab, c, v, t_, v == t_))
    # footnote / caption facts
    W = V["test_window"]; d0, d1 = W.split("..")
    import datetime as _dt
    ndays = (_dt.date.fromisoformat(d1) - _dt.date.fromisoformat(d0)).days + 1
    comp = V["comparator"]; LR = R["latency_raw"]["systems"]
    lw = re.search(r"LEARN_WINDOW_SIZE=(\d+)", comp["ct_params"]).group(1)
    nb = re.search(r"\+ (\d+) EXT_blend", comp["candidates"]).group(1)
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    F = tex_norm(foot)
    facts = [
        ("test window", f"{d0} to {d1}", f"Test window {d0} to {d1}" in F),
        ("test queries", f"{V['n_test_queries']:,}", f"{V['n_test_queries']:,} queries" in F),
        ("test days", str(ndays), f"{ndays} days" in F and f"the {ndays} test days" in F),
        ("entities", f"{BS['n_entities']:,}", f"all {BS['n_entities']:,} entities" in F),
        ("ranker seeds", f"0--{R['metrics']['HRcov']['n_seeds'] - 1}", f"seeds 0--{R['metrics']['HRcov']['n_seeds'] - 1}" in F),
        ("CT learning window", lw, f"learning window {lw}" in F),
        ("blend a", str(comp["a"]), f"$a={comp['a']}$" in F),
        ("blend tau", str(comp["tau"]), f"$\\tau={comp['tau']}$" in F),
        ("number of blends", nb, f"{nb} blends" in F),
        ("bootstrap B", str(V["bootstrap"]["B"]), f"$B={V['bootstrap']['B']}$" in F),
        ("shared draws", str(V["bootstrap"]["shared_draws"]), ("shared draws" in F) == bool(V["bootstrap"]["shared_draws"])),
        ("latency slice size", f"{LR['n']:,}", f"{LR['n']:,}-query validation slice" in F),
        ("latency repeats", words.get(LR["reps"], str(LR["reps"])), f"median of {words.get(LR['reps'], LR['reps'])} repeats" in F),
        ("latency CPU threads", words.get(LR["cpu_threads"], str(LR["cpu_threads"])), f"{words.get(LR['cpu_threads'], LR['cpu_threads'])} CPU threads" in F),
        ("outcome", V["outcome"], V["outcome"].replace("_", "\\_") in foot),
    ]
    for nm, v, ok in facts:
        out.append(("footnote", nm, "value", v, "(present in TeX footnote)" if ok else "(not found in TeX footnote)", ok))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["panel", "row", "column", "rebuilt", "tex", "match"]); w.writerows(out)
    # rebuilt LaTeX (same layout as the manuscript)
    L = ["% rebuilt by make_fresh_window_table.py from fresh_window_results.json / fresh_window_verdict.json / fresh_window_latency_table.csv",
         "(a) System & MRR & H@1 & H@10 & batch size 1 & batched \\\\", "\\midrule"]
    L += [f"{lab} & " + " & ".join(v) + " \\\\" for lab, v in A]
    L += ["", "(b) Contrast & $\\Delta$MRR & Query bootstrap [$L$, $U$] & Timestamp-block [$L$, $U$] & Call \\\\", "\\midrule"]
    tx = lambda s: s.replace("+", "$+$", 1) if s.startswith("+") else s.replace("-", "$-$", 1)
    L += [f"{lab} & {tx(v[0])} & [{tx(v[1])}, {tx(v[2])}] & [{tx(v[3])}, {tx(v[4])}] & {v[5]} \\\\" for lab, v in B]
    open(a.tex_out, "w").write("\n".join(L) + "\n")
    bad = [r for r in out if not r[5]]
    print(f"fresh-window table (tab:fresh, TeX line {line0}): {len(out)} cells checked, {len(bad)} mismatches")
    for r in bad:
        print("  MISMATCH", r)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
