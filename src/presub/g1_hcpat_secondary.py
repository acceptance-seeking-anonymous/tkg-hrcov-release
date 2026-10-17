#!/usr/bin/env python3
"""presub / g1_hcpat_secondary.py -- post-hoc secondary (labelled): G1 comparison with the manuscript's HRCOV definition
(HRCOV_PAT) on the re-run bases and with the G3 base regcn_tgb; registered G1 rule thresholds applied for reference only."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g7_cluster_bootstrap import boot_query, boot_cluster, load_quads, split_order
J1 = os.path.expandvars("$RESULTS_DIR/runs")
V = json.load(open("out/g1_verdict.json"))
items = {"polecat": [("tirgn", "HRCOV_PAT", os.path.expandvars("$RESULTS_DIR/runs/official_test_polecat_tirgn_hcpat.npz"))],
         "smallpedia": [("regcn_tuned", "HRCOV_PAT", os.path.expandvars("$RESULTS_DIR/runs/official_test_smallpedia_regcn_tuned_hcpat.npz")),
                        ("regcn_tgb", "HRCOV", os.path.expandvars("$RESULTS_DIR/runs/official_test_smallpedia_regcn_tgb.npz")),
                        ("regcn_tgb", "HRCOV_PAT", os.path.expandvars("$RESULTS_DIR/runs/official_test_smallpedia_regcn_tgb_hcpat.npz"))]}
OUT = dict(label="post-hoc secondary: G1 vs the manuscript HRCOV definition (HRCOV_PAT) and vs the G3 base; not the registered G1 decision", rows=[])
for ds, lst in items.items():
    G = np.load(f"{J1}/official_test_{ds}_recb_g1.npz"); arm = V["datasets"][ds]["G1_arm"]
    g1 = np.mean([G[f"rr_{arm}_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)
    ts, fid, _ = split_order(load_quads(f"../data/datasets/harness_data/tkgl-{ds}/test.txt"))
    for b, sysn, p in lst:
        Z = np.load(p); s = np.mean([Z[f"rr_{sysn}_s{k}"].astype(np.float64) for k in (0, 1, 2)], 0)
        d = s - g1; q = boot_query(d); t = boot_cluster(d, ts)
        OUT["rows"].append(dict(dataset=ds, base=b, system=sysn, G1_arm=arm, system_MRR=float(s.mean()), G1_MRR=float(g1.mean()), query=q, ts_block=t,
                                registered_thresholds_met=bool(q["delta"] >= 0.005 and q["p_one_sided"] < 0.05 and t["CI95"][0] > 0)))
json.dump(OUT, open("out/g1_hcpat_secondary.json", "w"), indent=1)
for r in OUT["rows"]:
    print(r["dataset"], r["base"], r["system"], round(r["system_MRR"], 5), round(r["G1_MRR"], 5), round(r["query"]["delta"], 5), [round(c, 5) for c in r["query"]["CI95"]], [round(c, 5) for c in r["ts_block"]["CI95"]], r["registered_thresholds_met"])
