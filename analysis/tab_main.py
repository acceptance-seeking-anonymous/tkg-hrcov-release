#!/usr/bin/env python3
"""analysis/tab_main.py -- Table "Controlled same-base comparison on the TGB 2.0 datasets" (tab:main) from the result files.

Sources (results/):
  hardening/hardening_verdict.json   fulltest_rows: RecB (train mode, clean) and the TiRGN base / BASE+RecB / RecB-POOL / HR rows
                                     on tkgl-polecat and tkgl-smallpedia (official py-tgb 2.3.0 Evaluator, every test query)
  datasets/dataset_profiles.json     RecB-lite on tkgl-polecat / tkgl-smallpedia (harness exact evaluator)
  v7/a3_hits.json                    HRCOV (bridge-free, 3-seed mean) and G1 (K = 150) MRR / H@10
  presub/g3_verdict.json             tkgl-smallpedia RE-GCN (TGB configuration): base, controls, HR
  g8/g8_verdict.json                 tkgl-polecat CognTKE: base, controls, HR, HRCOV
  icews/icews_stageB_verdict.json    tkgl-icews (Stage B, every test query)
  v7/b4_g1_tkgl_icews.json           G1 on tkgl-icews
The control row is the better of BASE+RecB and RecB-POOL by test MRR (each selected on validation).
Writes results/tables/tab_main.csv and tab_main.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, write_csv, write_json, f4

DOT = None


def build():
    hv = load("hardening_verdict.json")["fulltest_rows"]
    def row(ds, base, arm):
        r = [x for x in hv if x["dataset"] == ds and x["base"] == base and x["arm"] == arm]
        assert len(r) == 1, (ds, base, arm, len(r)); return r[0]
    dp = load("dataset_profiles.json")["datasets"]
    a3 = load("a3_hits.json")["runs"]
    g3 = {k: v["official"] for k, v in load("g3_verdict.json")["regcn_tgb_arms"].items()}   # official py-tgb scores
    g8 = load("g8_verdict.json")
    sb = load("icews_stageB_verdict.json")["rows"]
    b4 = load("b4_g1_tkgl_icews.json")["arms"]

    def sbrow(base, prefix, evaluator=None, exact=False):
        r = [x for x in sb if x["base"] == base and (x["system"] == prefix if exact else x["system"].startswith(prefix))
             and (evaluator is None or x["evaluator"] == evaluator)]
        assert len(r) == 1, (base, prefix, [x["system"] for x in r]); return r[0]
    def mh(x): return (x["MRR"], x["H10"])
    def ctl(a, b): return a if a[0] >= b[0] else b               # better control by test MRR
    pc, sp = "tkgl-polecat", "tkgl-smallpedia"
    T = []   # (block, base, candidates, ranker, polecat(MRR,H10), smallpedia(...), icews(...), icews superscript)
    T.append(("references", "RecB, train mode (clean)", "all", "rule", mh(row(pc, "-", "RecB trained (clean psi reset)")),
              mh(row(sp, "-", "RecB trained (clean psi reset)")), mh(sbrow("-", "RecB (hardening re-implementation, window 0, clean_tr)")), "h"))
    T.append(("references", "RecB-lite (h)", "all", "rule", mh(dp[pc]["recb_lite"]["test"]), mh(dp[sp]["recb_lite"]["test"]),
              mh(sbrow("-", "RecB-lite (full history", "harness")), ""))
    T.append(("references", "G1 (no base)", "RecB top-150", "learned", mh(a3[pc + " | G1 (base-free, K150)"]["G1_K150"]),
              mh(a3[sp + " | G1 (base-free, K150)"]["G1_K150"]), mh(b4["G1"]), ""))
    # CognTKE (tkgl-polecat only; better control is BASE+RecB)
    m, h = g8["mrr"], g8["hits"]
    cb = g8["better_control"]
    T.append(("bases", "CognTKE", "all", "--", (m["base"], h["base"]["H10"]), DOT, DOT, ""))
    T.append(("bases", "CognTKE", "top-K u RecB / all", "BASE+RecB or RecB-POOL", (m[cb], h[cb]["H10"]), DOT, DOT, ""))
    T.append(("bases", "CognTKE", "top-50", "HR (ours)", (m["HR"], h["HR"]["H10"]), DOT, DOT, ""))
    T.append(("bases", "CognTKE", "top-50 + B", "HRCOV (ours)", (m["HRCOV"], h["HRCOV"]["H10"]), DOT, DOT, ""))
    # TiRGN (tkgl-polecat, tkgl-smallpedia)
    T.append(("bases", "TiRGN", "all", "--", mh(row(pc, "tirgn", "base")), mh(row(sp, "tirgn", "base")), DOT, ""))
    T.append(("bases", "TiRGN", "top-K u RecB / all", "BASE+RecB or RecB-POOL", ctl(mh(row(pc, "tirgn", "BLEND")), mh(row(pc, "tirgn", "POOL"))),
              ctl(mh(row(sp, "tirgn", "BLEND")), mh(row(sp, "tirgn", "POOL"))), DOT, ""))
    T.append(("bases", "TiRGN", "top-50", "HR (ours)", mh(row(pc, "tirgn", "HR")), mh(row(sp, "tirgn", "HR")), DOT, ""))
    T.append(("bases", "TiRGN", "top-50 + B", "HRCOV (ours)", mh(a3[pc + " | TiRGN"]["HRCOV_PAT"]), mh(a3[sp + " | TiRGN"]["HRCOV_PAT"]), DOT, ""))
    # LogCL (tkgl-icews, full-train base)
    T.append(("bases", "LogCL", "all", "--", DOT, DOT, mh(sbrow("logcl", "logcl base (all train facts")), "f"))
    T.append(("bases", "LogCL", "top-K u RecB / all", "BASE+RecB or RecB-POOL", DOT, DOT,
              ctl(mh(sbrow("logcl", "logcl(+)RecB blend")), mh(sbrow("logcl", "RecB-POOL_logcl"))), ""))
    T.append(("bases", "LogCL", "top-50", "HR (ours)", DOT, DOT, mh(sbrow("logcl", "HR", exact=True)), "h"))
    T.append(("bases", "LogCL", "top-50 + B", "HRCOV (ours)", DOT, DOT, mh(sbrow("logcl", "HRCOV150 (valid-chosen B*")), ""))
    # RE-GCN (tkgl-smallpedia TGB configuration; tkgl-icews full-train base)
    T.append(("bases", "RE-GCN", "all", "--", DOT, (g3["base"]["MRR"], g3["base"]["H10"]), mh(sbrow("regcn", "regcn base (all train facts")), "f"))
    T.append(("bases", "RE-GCN", "top-K u RecB / all", "BASE+RecB or RecB-POOL", DOT, ctl(mh(g3["BLEND"]), mh(g3["POOL"])),
              ctl(mh(sbrow("regcn", "regcn(+)RecB blend")), mh(sbrow("regcn", "RecB-POOL_regcn"))), ""))
    T.append(("bases", "RE-GCN", "top-50", "HR (ours)", DOT, mh(g3["HR"]), mh(sbrow("regcn", "HR", exact=True)), "h"))
    T.append(("bases", "RE-GCN", "top-50 + B", "HRCOV (ours)", DOT, mh(a3[sp + " | RE-GCN (TGB config)"]["HRCOV_PAT"]),
              mh(sbrow("regcn", "HRCOV200 (valid-chosen B*")), ""))
    return T


def main():
    T = build()
    hdr = ["block", "base_or_system", "candidates", "ranker_or_control", "polecat_MRR", "polecat_H10", "smallpedia_MRR", "smallpedia_H10",
           "icews_MRR", "icews_H10", "icews_MRR_mark"]
    rows = []
    for blk, b, c, r, p, s, i, mark in T:
        cells = []
        for x in (p, s, i):
            cells += ["." if x is DOT else f4(x[0]), "." if x is DOT else f4(x[1])]
        rows.append([blk, b, c, r] + cells + [mark])
    write_csv(rows, hdr, "tab_main.csv"); write_json(dict(header=hdr, rows=rows), "tab_main.json")
    for r in rows:
        print(" | ".join(r))


if __name__ == "__main__":
    main()
