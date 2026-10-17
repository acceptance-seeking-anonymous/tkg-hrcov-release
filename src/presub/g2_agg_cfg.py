#!/usr/bin/env python3
"""presub / g2_agg_cfg.py -- write the g2_agg.py config for the datasets / bases processed by g2_run_serverA.sh."""
import os
import json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--G", required=True); ap.add_argument("--dsl", required=True); ap.add_argument("--bases", required=True)
ap.add_argument("--sfx", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
G = a.G; cfg = {}
for ds, dd, lds, hr in (("ICEWS18", os.path.expandvars("$EXT_DIR/logcl_redun/data/ICEWS18"), "icews18", 0.3664),
                        ("GDELT", os.path.expandvars("$EXT_DIR/gdelt_armAD/data/GDELT"), "gdelt", 0.2656)):
    if ds not in a.dsl.split():
        continue
    cfg[ds] = dict(name=ds, dd=dd, strata=f"{G}/{ds}/work/strata_test.npz", recb_ranks=f"{G}/recb/recb_test_ranks_{ds}.npz",
                   recb_sel=f"{G}/recb/recb_selection_{ds}.json", residual_hr_tirgn=hr,
                   bases={b: dict(official=f"{G}/official_test_{lds}_{b}.npz", test_json=f"{G}/blend_test_{lds}_{b}.json",
                                  valid_json=f"{G}/blend_valid_{lds}_{b}.json", feat_summary=f"{G}/{ds}/work/feat_{b}{a.sfx}/feat_summary.json")
                          for b in a.bases.split()})
json.dump(cfg, open(a.out, "w"), indent=1)
print(json.dumps({k: list(v["bases"]) for k, v in cfg.items()}))
