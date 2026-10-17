#!/usr/bin/env python3
"""presub / g3_eval.py -- G3 decision of a_presub_rule.json (run locally from hardening/agg after the G3 job).

1. acceptance: TGB script official test MRR vs the TGB 2.0 RE-GCN row 0.594 ('reproduced' iff |diff| <= 0.02)
2. replay check: official MRR of the dumped rows (hd_blend base-only pass, label regcn_tgb_check) vs the script MRR (<= 1e-3)
3. decision: hardening rule (hd_agg.py, unchanged) re-evaluated for tkgl-smallpedia with regcn_tgb added to the pool
   (architecture selection by valid MRR among regcn, regcn_tuned, regcn_tgb; two strongest; C2 over all bases)
4. secondary: HRCOV_regcn_tgb - HRCOV_TiRGN, base-relative gains (HRCOV - base, HRCOV - better control) per base,
   G1 - regcn_tgb controls (if the G1 official npz is given); query + timestamp-block bootstraps.
  python3 g3_eval.py --g3 <G3 job dir> --cfg agg_cfg.json --rule hardening_decision_rule.json --out g3_verdict.json [--g1 <G1 official npz> --g1_train <train_g1 json>]
"""
import os, sys, json, argparse, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hd_agg import boot, load_quads, split_queries
from g7_cluster_bootstrap import boot_cluster

ap = argparse.ArgumentParser()
ap.add_argument("--g3", required=True); ap.add_argument("--cfg", required=True); ap.add_argument("--rule", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--g1", default=""); ap.add_argument("--g1_train", default="")
a = ap.parse_args()
J = a.g3
run = json.load(open(f"{J}/g3_tgb_run.json")); dl = json.load(open(f"{J}/g3_dump_log.json"))
chk = json.load(open(f"{J}/blend_test_smallpedia_regcn_tgb_check.json"))
tj = json.load(open(f"{J}/blend_test_smallpedia_regcn_tgb.json"))
OUT = dict(name="g3_verdict", rule="a_presub_rule.json (G3)")
OUT["acceptance"] = dict(tgb_script_test_MRR=run["test_mrr"], tgb_script_valid_MRR=run["val_mrr"], best_epoch_0indexed=run["best_epoch"],
                         target=0.594, absdiff=abs(run["test_mrr"] - 0.594), reproduced=bool(abs(run["test_mrr"] - 0.594) <= 0.02),
                         deviation="DGL replaced by the pure-torch dgl_shim", versions=run["versions"], gpu=run["gpu"], train_s=run["total_s"])
rep = chk["metrics"]["base"]["official"]["MRR"]
OUT["replay_check"] = dict(official_MRR_of_dumped_rows=rep, script_test_MRR=run["test_mrr"], absdiff=abs(rep - run["test_mrr"]),
                           within_1e3=bool(abs(rep - run["test_mrr"]) <= 1e-3), dump_valid_filt_MRR=dl["valid_filt"]["MRR"],
                           dump_test_filt_MRR=dl["test_filt"]["MRR"], per_query_path_check=dl["per_query_path_check"],
                           pipeline_base_arm_official_MRR=tj["metrics"]["base"]["official"]["MRR"])
# ---- decision: hd_agg on the smallpedia config + regcn_tgb
C = json.load(open(a.cfg)); sp = dict(C["smallpedia"]); sp["bases"] = dict(sp["bases"])
sp["bases"]["regcn_tgb"] = dict(official=os.path.abspath(f"{J}/official_test_smallpedia_regcn_tgb.npz"),
                                test_json=os.path.abspath(f"{J}/blend_test_smallpedia_regcn_tgb.json"),
                                valid_json=os.path.abspath(f"{J}/blend_valid_smallpedia_regcn_tgb.json"), arch="regcn",
                                selected_epoch=f"{run['best_epoch']} (TGB script valid-best, 0-indexed)", host="GPU serverB (RTX 4090)")
cfg_path = os.path.abspath(a.out.replace(".json", "_aggcfg.json")); json.dump({"smallpedia": sp}, open(cfg_path, "w"), indent=1)
agg_out = os.path.abspath(a.out.replace(".json", "_hardening_smallpedia.json"))
r = subprocess.run([sys.executable, f"{HERE}/hd_agg.py", "--cfg", cfg_path, "--rule", a.rule, "--out", agg_out], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
V = json.load(open(agg_out))["datasets"]["smallpedia"]
OUT["decision"] = dict(architecture_selection=V["architecture_selection"], two_strongest=V["two_strongest"],
                       strongest_blend_overall=V["strongest_blend_overall"], outcome=V["decision"]["outcome"],
                       round1_outcome="PRACTICAL_ADVANCE", outcome_changed=V["decision"]["outcome"] != "PRACTICAL_ADVANCE",
                       per_base={b: {k: d[k] for k in ("better_comparator", "C1", "C1_pass", "C2_vs", "C2", "C2_pass", "C3_head", "C3_damage", "PASS")}
                                 for b, d in V["decision"]["per_base"].items()},
                       full_verdict_file=os.path.basename(agg_out))
OUT["regcn_tgb_arms"] = V["bases"]["regcn_tgb"]["arms"]
OUT["regcn_tgb_valid_MRR_base"] = V["bases"]["regcn_tgb"]["valid_MRR_base"]
# ---- secondary
dd = sp["dd"]; Q = load_quads(f"{dd}/test.txt"); R = int(open(f"{dd}/stat.txt").read().split()[1])
ts = split_queries(Q, R)[:, 3]


def seedmean(Z, pre):
    return np.mean([Z[f"rr_{pre}_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0)


Zs = {b: np.load(B["official"]) for b, B in sp["bases"].items() if os.path.exists(B["official"])}
sec = {}
hr_tgb = seedmean(Zs["regcn_tgb"], "HRCOV"); hr_tir = seedmean(Zs["tirgn"], "HRCOV")
sec["HRCOV_regcn_tgb - HRCOV_tirgn"] = dict(query=boot(hr_tgb - hr_tir), ts=boot_cluster(hr_tgb - hr_tir, ts))
gains = {}
for b, Z in Zs.items():
    h = seedmean(Z, "HRCOV"); bl = Z["rr_BLEND"].astype(np.float64); po = Z["rr_POOL"].astype(np.float64); bs = Z["rr_base"].astype(np.float64)
    better = bl if bl.mean() >= po.mean() else po
    gains[b] = dict(base_MRR=float(bs.mean()), HRCOV_MRR=float(h.mean()), better_control_MRR=float(better.mean()),
                    HRCOV_minus_base=float((h - bs).mean()), HRCOV_minus_better_control=float((h - better).mean()))
sec["base_relative_gains"] = gains
if a.g1 and os.path.exists(a.g1):
    G = np.load(a.g1); T = json.load(open(a.g1_train))
    sel = {arm: float(np.mean([v["select"]["MRR"] for v in T["arms"][arm]["seeds"].values()])) for arm in ("G1_K50", "G1_K150") if arm in T["arms"]}
    arm = max(sel, key=lambda k: (sel[k], k == "G1_K50")); g1 = seedmean(G, arm)
    Z = Zs["regcn_tgb"]
    for c in ("BLEND", "POOL"):
        d = g1 - Z[f"rr_{c}"].astype(np.float64)
        sec[f"G1({arm}) - regcn_tgb {c}"] = dict(query=boot(d), ts=boot_cluster(d, ts))
OUT["secondary"] = sec
json.dump(OUT, open(a.out, "w"), indent=1)
print(json.dumps(dict(acceptance=OUT["acceptance"]["reproduced"], test=run["test_mrr"], replay=OUT["replay_check"]["within_1e3"],
                      outcome=OUT["decision"]["outcome"], two=OUT["decision"]["two_strongest"])))
