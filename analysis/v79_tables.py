#!/usr/bin/env python3
"""analysis/v79_tables.py -- rebuild, from results/, the numbers of the tables added in Manuscript A v7.9:
main text tab:standalone (a, b) and tab:fresh (a, b); supplement tab:v78sys, tab:v78ladder, tab:v78ctx, tab:v78ext (v7.8 ladder,
external methods), tab:fitpop, tab:fitrefit, tab:fitsize, tab:fitstrata (post-verdict fit-population audit), tab:freshweek,
tab:freshlat (2026 GDELT window).

  build()                 -> {table: [row, ...]}, row = {"nums": [float, ...], "text": [str, ...] or None}
  extract_from_tex(paths) -> the same structure read from the LaTeX sources (used once to write
                             tests/expected/v79_fragments_manuscript.json; the check itself needs no LaTeX)
  compare(expected, got)  -> list of mismatches; a printed number t with d decimals matches a value v iff
                             round(v, d) == t (so -0.0000 and 0.0000 are equal), integers must be equal

Inputs: results/v78/v78_results.json, results/v78/v78_adapted_check.json, results/fitpop/fitpop_audit.json,
results/fresh_window/fresh_window_results.json.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CF = ("S1", "S2", "S3")
MET4 = ("MRR", "H1", "H3", "H10")
# label columns per table (dropped before reading numbers)
LABEL_COLS = {"tab:standalone/a": 1, "tab:standalone/b": 1, "tab:fresh/a": 1, "tab:fresh/b": 1, "tab:v78sys": 1,
              "tab:v78ladder": 2, "tab:v78ctx": 2, "tab:v78ext": 2, "tab:fitpop": 1, "tab:fitrefit/a": 1,
              "tab:fitrefit/b": 2, "tab:fitsize": 2, "tab:fitstrata": 2, "tab:freshweek": 1, "tab:freshlat": 2,
              "tab:ladder": 2}
TEXT_RX = re.compile(r"\b(SMALL|MATCHES|BEATS|PASS|FAIL|blend|alone)\b")


def _load(rel):
    return json.load(open(os.path.join(RES, rel)))


def row(nums, text=None):
    return {"nums": [float(x) for x in nums], "text": text}


def ci5(m):
    """delta, query L, U, ts L, U from either result layout."""
    if "MRR" in m and isinstance(m["MRR"], dict):
        m = m["MRR"]
    if "query_L" in m:
        return [m["delta"], m["query_L"], m["query_U"], m["ts_L"], m["ts_U"]]
    q, t = m["query"], m["ts"]
    if isinstance(q, dict):
        return [m["delta"], q["L"], q["U"], t["L"], t["U"]]
    return [m["delta"], q[0], q[1], t[0], t[1]]


def build():
    R = _load("v78/v78_results.json"); AD = _load("v78/v78_adapted_check.json")
    FP = _load("fitpop/fitpop_audit.json"); FW = _load("fresh_window/fresh_window_results.json")
    C = R["configs"]; out = {}

    # ---- tab:standalone (a) systems, (b) decision contrasts
    sysk = ["base", "HRcov", "SA", "EXT_alone_counttrucola", "EXT_blend_counttrucola", "EXT_alone_edgeremind",
            "EXT_blend_edgeremind", "REL_LIN"]
    rows = [row([C[c]["systems"][k][m] for c in CF for m in MET4]) for k in sysk]
    for k in ("EXT_adapted_counttrucola", "EXT_adapted_blend_counttrucola"):
        rows.append(row([AD["configs"][c]["test"][k][m] for c in CF for m in MET4]))
    out["tab:standalone/a"] = rows
    rows = []
    for c in CF:
        d = C[c]["decision"]; a = AD["configs"][c]
        rows.append(row(ci5(C[c]["comparisons"]["HRcov - SA"]), [d["standalone"]["call"]]))
        rows.append(row(ci5(C[c]["comparisons"]["HRcov - EXT_blend_counttrucola"]), [d["external"]["call"]]))
        rows.append(row(ci5(C[c]["comparisons"]["SA - EXT_blend_counttrucola"]), []))
        rows.append(row(ci5(a["comparisons"]["HRcov - EXT_adapted_blend_counttrucola"]), [a["external_call_reapplied"]]))
        rows.append(row(ci5(a["comparisons"]["SA - EXT_adapted_blend_counttrucola"]), []))
    out["tab:standalone/b"] = rows

    # ---- tab:fresh (a) systems + latency, (b) contrasts
    lat = {(r["system"], r["setting"]): r for r in FW["latency_table_s_per_1000_queries"]}
    latname = {"base": "TiRGN base", "BLEND": "BLEND (TiRGN + RecB)", "HRcov": "HRcov", "SA": "SA",
               "external_comparator": "CountTRuCoLa comparator (C0 blend)"}
    rows = []
    for k in ("base", "BLEND", "HRcov", "SA", "external_comparator"):
        m = FW["metrics"][k]["official_ties_averaged"]
        rows.append(row([m["MRR"], m["H1"], m["H10"], lat[(latname[k], "batch size 1")]["end_to_end_s_per_1k"],
                         lat[(latname[k], "batched (production)")]["end_to_end_s_per_1k"]]))
    out["tab:fresh/a"] = rows
    calls = FW["calls"]
    out["tab:fresh/b"] = [row(ci5(FW["deltas_official"]["HRcov-SA"]), [calls["SA"]]),
                          row(ci5(FW["deltas_official"]["HRcov-external_comparator"]), [calls["external_comparator"]]),
                          row(ci5(FW["deltas_official"]["HRcov-BLEND"]), []),
                          row(ci5(FW["deltas_official"]["HRcov-base"]), [])]

    # ---- supplement: v7.8 systems, ladder / width / generator contrasts, context contrasts, external selection
    sysk = ["HIST", "ERM_ORDER", "SA", "SA_GEN", "SA_ERM", "+C", "+S", "+T", "HRcov", "SA_W50", "SA_W150", "HR_W50",
            "HR_W150", "REL_LIN", "REL_LIN_global"]
    out["tab:v78sys"] = [row([C[c]["systems"][k][m] for c in CF for m in MET4]) for k in sysk]
    lad = ["ladder: +C - SA", "ladder: +S - +C", "ladder: +T - +S", "ladder: HRcov - +T (+P)", "width: SA_W50 - SA",
           "width: SA_W150 - SA", "width: HR_W50 - HRcov", "width: HR_W150 - HRcov", "width: HR_W50 - SA_W50",
           "width: HR_W150 - SA_W150", "HRcov - SA_GEN", "HRcov - SA_ERM", "SA_ERM - SA"]
    out["tab:v78ladder"] = [row(ci5(C[c]["comparisons"][k])) for k in lad for c in CF]
    ctx = ["HRcov - base", "SA - base", "HRcov - HIST", "SA - HIST", "HRcov - EXT_alone_edgeremind",
           "SA - EXT_alone_edgeremind", "HRcov - EXT_blend_edgeremind", "SA - EXT_blend_edgeremind",
           "HRcov - EXT_alone_counttrucola", "SA - EXT_alone_counttrucola", "REL_LIN: HRcov - REL_LIN",
           "REL_LIN: REL_LIN - REL_LIN_global"]
    out["tab:v78ctx"] = [row(ci5(C[c]["comparisons"][k])) for k in ctx for c in CF]
    rows = []
    for c in CF:
        sc = AD["configs"][c]["selection_candidates"]
        for m in ("counttrucola", "edgeremind"):
            nums = [sc[m]["alone"], sc[m]["blend"]] + list(R["ext_selected_grid"][c][m])
            if m == "counttrucola":
                g = AD["configs"][c]["selected_grid_point_adapted_blend"]
                nums += [sc[m]["adapted"], sc[m]["adapted_blend"], g["a"], g["tau"]]
            rows.append(row(nums, [R["gates"][m]["call"], C[c]["external_selection"]["per_method"][m]["variant"]]))
    out["tab:v78ext"] = rows

    # ---- supplement: fit-population audit (post-verdict)
    rows, rows2 = [], []
    for c in CF:
        f = FP["configs"][c]; pa = f["per_arm"]; po = f["pairwise_overlap"]
        nums = [f["common_start_fit_population"]]
        for k in ("SA", "+C", "HRcov"):
            nums += [pa[k]["gold_in_pool_fit_queries_used"], 100 * pa[k]["frac_of_start"]]
        nums += [f["intersection"]["n"], 100 * f["intersection"]["frac_of_start"], po["SA vs +C"]["jaccard"],
                 po["SA vs HRcov"]["jaccard"], po["+C vs HRcov"]["jaccard"]]
        rows.append(row(nums))
        rows2.append(row([po["SA vs +C"]["only_first"], po["SA vs +C"]["only_second"], po["+C vs HRcov"]["only_first"],
                          po["+C vs HRcov"]["only_second"], f["hrcov_stored_training_n_fit_queries"]]))
    out["tab:fitpop"] = rows + rows2
    arms = ("SA", "+C", "+S", "+T", "HRcov")
    rows = []
    for c in CF:
        t = FP["configs"][c]["test_MRR"]
        rows.append(row([t["registered"][a] for a in arms])); rows.append(row([t["common_intersection"][a] for a in arms]))
    out["tab:fitrefit/a"] = rows
    steps = ("+C - SA", "+S - +C", "+T - +S", "HRcov - +T (+P)", "HRcov - SA")
    out["tab:fitrefit/b"] = [row(ci5(FP["configs"][c]["comparisons"][f"reg: {s}"]) + ci5(FP["configs"][c]["comparisons"][f"int: {s}"]))
                             for c in CF for s in steps]
    rows = []
    for c in CF:
        d = FP["configs"][c]["diagnostic_size_matched_harness"]; h = d["harness_MRR"]; S = d["comparisons"]
        for a in arms:
            rows.append(row([h[f"{a}_reg"], h[f"{a}_sm"], S[f"sm - reg: {a}"]["delta"], h[f"{a}_int"]]))
        cl = d["hrcov_minus_sa_call"]
        rows.append(row([S[f"{x}: HRcov - SA"]["delta"] for x in ("reg", "sm", "int")], [cl["reg"], cl["sm"], cl["int"]]))
    out["tab:fitsize"] = rows
    strata = ("gold in SA and HRcov pools", "gold in HRcov pool only (base / pattern candidates)", "gold in SA pool only",
              "gold in neither pool")
    rows = []
    for c in CF:
        st = FP["configs"][c]["diagnostic_test_strata_official"]["strata"]
        for s in strata:
            v = st[s]; nums = [v["share"]]
            for a in ("SA", "+C", "HRcov"):
                nums += [v["MRR"][a]["registered"], v["MRR"][a]["intersection"]]
            rows.append(row(nums + [v["contribution_to_int_minus_reg"]["HRcov"]]))
    out["tab:fitstrata"] = rows

    # ---- supplement: 2026 GDELT window, weekly contrasts and latency stages
    rows = []
    for w in sorted(FW["weekly_deltas"], key=int):
        v = FW["weekly_deltas"][w]; a, b = v["days"].split("..")
        rows.append(row([v["n"], v["HRcov-SA"], v["HRcov-external_comparator"], v["HRcov-BLEND"]],
                        [f"{a[:4]}-{a[4:6]}-{a[6:]} to {b[:4]}-{b[4:6]}-{b[6:]}"]))
    out["tab:freshweek"] = rows
    rows = []
    for setting in ("batch size 1", "batched (production)"):
        for k in ("base", "BLEND", "HRcov", "SA", "external_comparator"):
            r = lat[(latname[k], setting)]
            rows.append(row([r["base_inference_s_per_1k"], r["candidates_features_s_per_1k"], r["ranker_or_blend_s_per_1k"],
                             r["end_to_end_s_per_1k"]]))
    out["tab:freshlat"] = rows
    return out


def build_v80():
    """Table tab:ladder of the manuscript in progress (v8.0-text, Section 6.3): registered and shared-query (common
    intersection) ladder contrasts with the timestamp-block bounds, from results/fitpop/fitpop_audit.json; the other tables
    of v8.0-text that this release checks (tab:standalone, tab:fresh) are unchanged from v7.9 and come from build()."""
    FP = _load("fitpop/fitpop_audit.json"); out = {}
    rows = []
    for step in ("+C - SA", "+S - +C", "+T - +S", "HRcov - SA"):
        for proto, key, ck in (("registered", "reg", "registered"), ("shared", "int", "common_intersection")):
            nums = []; calls = []
            for c in CF:
                m = FP["configs"][c]["comparisons"][f"{key}: {step}"]
                nums += [m["delta"], m["ts_L"], m["ts_U"]]
                calls.append(FP["configs"][c]["hrcov_minus_sa_call"][ck])
            rows.append(row(nums, calls if step == "HRcov - SA" else []))
    out["tab:ladder"] = rows
    # footnote: the last rung +P (HRcov - +T) is at most 0.0009 in absolute value under both protocols
    p_max = max(abs(FP["configs"][c]["comparisons"][f"{k}: HRcov - +T (+P)"]["delta"]) for c in CF for k in ("reg", "int"))
    out["tab:ladder/footnote"] = [row([p_max])]
    return out


# ------------------------------------------------------------------------------------------------ LaTeX side
_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _block(src, label):
    i = src.find("\\label{" + label + "}")
    if i < 0:
        raise KeyError(label)
    s = src.rfind("\\begin{table", 0, i); e = src.find("\\end{table", i)
    return src[s:e]


def _cells(line):
    line = line.split("\\\\")[0]
    return [c.strip() for c in line.split("&")]


def _norm(cell):
    c = cell.replace("$+$", "+").replace("$-$", "-").replace("{,}", "").replace("$\\to$", " ")
    c = re.sub(r"\$\^\{[^}]*\}\$|\^\{[^}]*\}", "", c)
    return c


def _datarow(line):
    """a data row has cells and at least one decimal or thousands-separated number (header and call rows have none)"""
    return "&" in line and "\\\\" in line and bool(re.search(r"\d\.\d|\d\{,\}\d{3}", line))


def extract_from_tex(main_tex, supp_tex):
    ms, sp = open(main_tex).read(), open(supp_tex).read()
    spec = [("tab:standalone", ms, ("a", "b")), ("tab:fresh", ms, ("a", "b")), ("tab:v78sys", sp, None),
            ("tab:v78ladder", sp, None), ("tab:v78ctx", sp, None), ("tab:v78ext", sp, None), ("tab:fitpop", sp, None),
            ("tab:fitrefit", sp, ("a", "b")), ("tab:fitsize", sp, None), ("tab:fitstrata", sp, None),
            ("tab:freshweek", sp, None), ("tab:freshlat", sp, None)]
    out = {}
    for lab, src, parts in spec:
        blk = _block(src, lab)
        segs = blk.split("\\begin{tabular}")[1:]
        names = [f"{lab}/{p}" for p in parts] if parts else [lab]
        assert len(segs) >= len(names), (lab, len(segs))
        for name, seg in zip(names, segs):
            k = LABEL_COLS[name]; rows = []
            for line in seg.split("\\end{tabular}")[0].split("\n"):
                line = line.strip()
                if not _datarow(line):
                    continue
                cells = _cells(line)
                body = [_norm(c) for c in cells[k:]]
                nums = []
                for c in body:
                    nums += _NUM.findall(c)
                text = TEXT_RX.findall(" ".join(body))
                if name == "tab:freshweek":
                    text = [_norm(cells[0])]
                rows.append({"nums": nums, "text": text, "label": " | ".join(cells[:k])})
            out[name] = rows
    return out


def extract_v80(main_tex):
    ms = open(main_tex).read(); out = {}
    for lab, parts in (("tab:standalone", ("a", "b")), ("tab:fresh", ("a", "b")), ("tab:ladder", None)):
        blk = _block(ms, lab); segs = blk.split("\\begin{tabular}")[1:]
        names = [f"{lab}/{p}" for p in parts] if parts else [lab]
        for name, seg in zip(names, segs):
            k = LABEL_COLS[name]; rows = []
            for line in seg.split("\\end{tabular}")[0].split("\n"):
                line = line.strip()
                if not _datarow(line): continue
                cells = _cells(line); body = [_norm(c) for c in cells[k:]]
                nums = []
                for c in body: nums += _NUM.findall(c)
                rows.append({"nums": nums, "text": TEXT_RX.findall(" ".join(body)), "label": " | ".join(cells[:k])})
            out[name] = rows
        if lab == "tab:ladder":
            m = re.search(r"at most (\d\.\d+) in absolute value", blk)
            out["tab:ladder/footnote"] = [{"nums": [m.group(1)], "text": None, "label": "footnote: +P at most ... in absolute value"}]
    return out


def compare(expected, got):
    bad = []
    for name, erows in expected.items():
        grows = got.get(name)
        if grows is None or len(grows) != len(erows):
            bad.append((name, "row count", len(erows), None if grows is None else len(grows))); continue
        for i, (e, g) in enumerate(zip(erows, grows)):
            if len(e["nums"]) != len(g["nums"]):
                bad.append((name, i, e["label"], "n numbers", e["nums"], g["nums"])); continue
            for t, v in zip(e["nums"], g["nums"]):
                d = len(t.split(".")[1]) if "." in t else 0
                if (round(v, d) != float(t)) if d else (int(round(v)) != int(t)):
                    bad.append((name, i, e["label"], t, v))
            if g["text"] is not None and e["text"] != g["text"]:
                bad.append((name, i, e["label"], "text", e["text"], g["text"]))
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--extract-v80":
        json.dump(dict(source="Manuscript A v8.0-text LaTeX source (main text, in progress): numbers as typeset in the data rows of "
                              "tab:standalone, tab:fresh (unchanged from v7.9) and tab:ladder (new), label columns dropped; text = calls",
                       tables=extract_v80(sys.argv[2])), sys.stdout, indent=1)
    elif len(sys.argv) == 4 and sys.argv[1] == "--extract":
        json.dump(dict(source="Manuscript A v7.9 LaTeX sources (main text and supplement): numbers as typeset in the "
                              "data rows of each table, label columns dropped; text = calls / gate / selected variant",
                       tables=extract_from_tex(sys.argv[2], sys.argv[3])), sys.stdout, indent=1)
    else:
        exp = json.load(open(os.path.join(ROOT, "tests", "expected", "v79_fragments_manuscript.json")))["tables"]
        bad = compare(exp, build())
        n = sum(len(r["nums"]) for rows in exp.values() for r in rows)
        print(("PASS" if not bad else "FAIL"), f"v7.9 tables: {len(exp)} table parts, {n} numbers")
        for b in bad[:30]:
            print("   mismatch:", b)
        sys.exit(1 if bad else 0)
