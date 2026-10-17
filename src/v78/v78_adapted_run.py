"""v78_adapted_run.py -- driver of the post-verdict label-adapted check (the analysis steps as run, in order):
v78_adapted_agg.py (selection re-applied, all comparisons) -> outcome sensitivity -> consistency check against the registered
v7.8 results -> v78_adapted_report.py (writes $V78_OUT/v78_adapted_check.json and .md).
Usage (repository root):  source configs/env.sh; bash analysis/stage_v78_inputs.sh <npz dir>; V78_OUT=out python3 src/v78/v78_adapted_run.py
"""
import os, json
SRC = os.environ["SRC"]; REPO = os.path.dirname(SRC)
exec(open(os.path.join(SRC, "v78", "v78_adapted_agg.py")).read())
OS_AD = outcome_sensitivity(R_AD)
# registered v7.8 results as they stood at the verdict (the shipped file plus the post-verdict block added later)
reg = json.load(open(os.path.join(REPO, "results", "v78", "v78_results.json")))
for _k in ("post_verdict_checks", "status_note_post_verdict"):
    reg.pop(_k, None)
chk = {}
for c in ("S1", "S2", "S3"):
    rc = reg["configs"][c]; ra = R_AD[c]
    same = []; diff = []
    for lab, v in rc["comparisons"].items():
        if lab in ra["comparisons"]:
            a = v["MRR"]; b_ = ra["comparisons"][lab]["MRR"]
            ok = abs(a["delta"] - b_["delta"]) < 1e-12 and all(abs(a[k][q] - b_[k][q]) < 1e-12 for k in ("query", "ts") for q in ("L", "U"))
            (same if ok else diff).append(lab)
    sysd = max(abs(rc["systems"][k]["MRR"] - ra["systems"][k]["MRR"]) for k in rc["systems"])
    chk[c] = dict(n_same=len(same), n_diff=len(diff), diff=diff[:5], sys_max=sysd)
print(chk)
os.makedirs(os.environ.get("V78_OUT", "."), exist_ok=True)
exec(open(os.path.join(SRC, "v78", "v78_adapted_report.py")).read())
