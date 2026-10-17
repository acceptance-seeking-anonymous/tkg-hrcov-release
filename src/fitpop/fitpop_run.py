"""fitpop_run.py -- driver of the post-verdict fit-population audit aggregation (the analysis steps as run, in order):
fitpop_agg.py (counts, common-intersection refit, official evaluator) -> fitpop_diag_agg.py (size-matched control,
test strata) -> wording of the +C = +S = +T identity from the registered trainer records -> fitpop_report.py
(writes $FITPOP_OUT/fitpop_audit.json and .md).
Usage (repository root):  source configs/env.sh; bash analysis/stage_v78_inputs.sh <npz dir>; FITPOP_OUT=out python3 src/fitpop/fitpop_run.py
"""
import os, json
SRC = os.environ["SRC"]
for _f in ("fitpop_agg.py", "fitpop_diag_agg.py"):
    exec(open(os.path.join(SRC, "fitpop", _f)).read())
# +C, +S and +T read the same pool directory and gold_pos array: state it from the registered trainer records
treg = {ds: json.load(open(os.path.join(RD, f"train_v78_{ds}.json"))) for ds in ("smallpedia", "polecat")}
for c, cf in CONFP.items():
    b = cf["b"]; ar = treg[cf["ds"]]["arms"]
    n = [ar[f"LAD_{x}_{b}"]["n_fit_retained"] for x in "CST"]; pools = [ar[f"LAD_{x}_{b}"]["pool"] for x in "CST"]
    assert len(set(n)) == 1 and len(set(pools)) == 1
    FP["counts"][c]["identical_identities"]["+C = +S = +T"] = (
        f"identical by construction, not recomputed row by row: the three rungs read the same pool directory ({pools[0]}) and the same gold_pos array "
        f"(the tail order changes only the ranking tail); the registered trainer records equal n_fit_retained = {n[0]} for all three")
exec(open(os.path.join(SRC, "fitpop", "fitpop_report.py")).read())
