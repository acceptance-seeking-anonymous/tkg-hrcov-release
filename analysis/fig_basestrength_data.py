#!/usr/bin/env python3
"""analysis/fig_basestrength_data.py -- data of the base-strength figure (fig:basestrength): for every trained base, the MRR of
the frozen base alone and the margin of HRCOV over the better same-base recurrence control (18 runs).

Sources (results/): hardening/hcpat_all_decision_bases.json, hardening/hardening_verdict.json, reports/bridge_timesafe_report.md,
reports/smallpedia_pilot_report.md, g8/g8_verdict.json, presub/g3_verdict.json, icews/icews_stageB_verdict.json,
presub/g2_verdict.json, forward_holdout/fh_verdict.json.
Writes results/tables/base_strength.json (input of analysis/figures/fig_basestrength.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, text, md_tables, write_json


def hcpat_rows_md():
    """bridge-free HRCOV (HRCOV_PAT) MRR of the pilot bases, from the two pilot reports."""
    out = {}
    _, hdr, rs = [t for t in md_tables(text("bridge_timesafe_report.md")) if t[1][:3] == ["base", "system", "MRR (3-seed mean)"]][0]
    for r in rs:
        if r[1] == "HRCOV_PAT_old":
            out[("tkgl-polecat", r[0])] = float(r[2])
    _, hdr, rs = [t for t in md_tables(text("smallpedia_pilot_report.md")) if t[1][:2] == ["base", "system"] and "Δ vs HRCOV [95% CI]" in t[1]][0]
    bm = {"RE-GCN": "regcn", "LogCL": "logcl", "TiRGN": "tirgn"}
    for r in rs:
        if r[1] == "`HRCOV_PAT`":
            out[("tkgl-smallpedia", bm[r[0]])] = float(r[2])
    return out


def build():
    H2 = load("hcpat_all_decision_bases.json"); HV = load("hardening_verdict.json")["fulltest_rows"]
    G8 = load("g8_verdict.json"); G3 = load("g3_verdict.json"); ST = load("icews_stageB_verdict.json"); G2 = load("g2_verdict.json")
    FH = load("fh_verdict.json"); HP = hcpat_rows_md()
    pc = H2["datasets"]["polecat"]["per_base"]; sp = H2["datasets"]["smallpedia_G3_pool"]["per_base"]
    def hvm(ds, b, arm): return [x for x in HV if x["dataset"] == ds and x["base"] == b and x["arm"] == arm][0]["MRR"]
    BS = []   # dataset, base, base MRR, better-control MRR, HRCOV MRR, decision base?
    for b, nm in [("regcn", "RE-GCN (pilot)"), ("logcl", "LogCL (pilot)"), ("tirgn", "TiRGN")]:
        ctl = max(hvm("tkgl-polecat", b, "BLEND"), hvm("tkgl-polecat", b, "POOL"))
        hrc = pc[b]["HRCOV_PAT_MRR"] if b in pc else HP[("tkgl-polecat", b)]
        if b in pc: ctl = pc[b]["better_control_MRR"]
        BS.append(("tkgl-polecat", nm, hvm("tkgl-polecat", b, "base"), ctl, hrc, b == "tirgn"))
    BS.append(("tkgl-polecat", "CognTKE", G8["mrr"]["base"], G8["mrr"]["BLEND"], G8["mrr"]["HRCOV"], True))
    for b, nm in [("regcn", "RE-GCN (pilot)"), ("logcl", "LogCL (pilot)")]:
        ctl = max(hvm("tkgl-smallpedia", b, "BLEND"), hvm("tkgl-smallpedia", b, "POOL"))
        BS.append(("tkgl-smallpedia", nm, hvm("tkgl-smallpedia", b, "base"), ctl, HP[("tkgl-smallpedia", b)], False))
    spH = H2["datasets"]["smallpedia_hardening_pool"]["per_base"]["regcn_tuned"]
    BS.append(("tkgl-smallpedia", "RE-GCN (re-tuned)", hvm("tkgl-smallpedia", "regcn_tuned", "base"), spH["better_control_MRR"], spH["HRCOV_PAT_MRR"], False))
    a = G3["regcn_tgb_arms"]
    BS.append(("tkgl-smallpedia", "RE-GCN (TGB config.)", a["base"]["official"]["MRR"], sp["regcn_tgb"]["better_control_MRR"], sp["regcn_tgb"]["HRCOV_PAT_MRR"], True))
    BS.append(("tkgl-smallpedia", "TiRGN", hvm("tkgl-smallpedia", "tirgn", "base"), sp["tirgn"]["better_control_MRR"], sp["tirgn"]["HRCOV_PAT_MRR"], True))
    icb = {"regcn": "RE-GCN (full train)", "logcl": "LogCL (full train)"}
    for b in ("regcn", "logcl"):
        x = ST["decision"][b]
        rows = [r for r in ST["rows"] if r["base"] == b and r["system"].startswith(b + " base (all train facts")]
        assert len(rows) == 1 and rows[0]["evaluator"] == "official py-tgb"
        BS.append(("tkgl-icews", icb[b], rows[0]["MRR"], x["comparator_test_MRR"][x["best_comparator"]], x["HRCOV_test_MRR"], True))
    for ds in ("ICEWS18", "GDELT"):
        for b, nm in [("regcn", "RE-GCN"), ("logcl", "LogCL"), ("tirgn", "TiRGN")]:
            arms = G2["datasets"][ds]["bases"][b]["arms"]; ctl = max(arms["BLEND"]["harness"]["MRR"], arms["POOL"]["harness"]["MRR"])
            BS.append((ds, nm, arms["base"]["harness"]["MRR"], ctl, arms["HRCOV_PAT"]["harness"]["MRR"], b in G2["datasets"][ds]["two_strongest"]))
    fa = FH["arms"]
    BS.append(("GDELT25X", "TiRGN", fa["base"]["MRR"], max(fa["BLEND"]["MRR"], fa["POOL"]["MRR"]), fa["HRCOV_PAT"]["MRR"], True))
    assert len(BS) == 18
    return [dict(dataset=a, base=b, base_mrr=c, control_mrr=d, hrcov_mrr=e, margin=e - d, decision=f) for a, b, c, d, e, f in BS]


def main():
    R = build()
    write_json(dict(base_strength=R), "base_strength.json")
    for r in R:
        print(f"{r['dataset']:16s} {r['base']:22s} base {r['base_mrr']:.4f}  control {r['control_mrr']:.4f}  HRCOV {r['hrcov_mrr']:.4f}  margin {r['margin']:+.4f}"
              + ("  (decision base)" if r["decision"] else ""))


if __name__ == "__main__":
    main()
