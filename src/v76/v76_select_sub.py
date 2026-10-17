#!/usr/bin/env python3
"""v76 / v76_select_sub.py -- M3a: hd_select.py restricted to the validation-select queries (harness ts id >= cut).
Reads blend_valid_<ds>_<base>.npz/json (hd_blend --stage valid: exact ranks of every grid configuration for every valid
row) and writes the selection json in hd_select's format (argmax of subset MRR, first index on ties).
  python3 v76_select_sub.py --dd DATA_DIR --ds smallpedia --base tirgn_valfit --dir D --cut 105 --out sel.json
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_recb as HR

ap = argparse.ArgumentParser()
ap.add_argument("--dd", required=True); ap.add_argument("--ds", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--dir", required=True); ap.add_argument("--cut", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
st = open(f"{a.dd}/stat.txt").read().split(); R = int(st[1])
sro = HR.split_queries(HR.load_quads(f"{a.dd}/valid.txt"), R, "valid")
z = np.load(f"{a.dir}/blend_valid_{a.ds}_{a.base}.npz"); m = json.load(open(f"{a.dir}/blend_valid_{a.ds}_{a.base}.json"))
rows = z["rows"]; ts = sro[rows, 3]; msk = ts >= a.cut
RK = z["ranks"].astype(np.float64); nb = len(m["blend_cfgs"])
mrr = (1.0 / RK[msk]).mean(0); vb, vp = mrr[:nb].tolist(), mrr[nb:].tolist()
bi = max(range(len(vb)), key=lambda i: (vb[i], -i)); pi = max(range(len(vp)), key=lambda i: (vp[i], -i))
base_sub = float((1.0 / z["base"][msk].astype(np.float64)).mean())
out = {a.base: dict(blend_idx=bi, blend_cfg=m["blend_cfgs"][bi], valid_MRR_blend=vb[bi], pool_idx=pi,
                    pool_cfg=[m["pool_cfgs"][pi][0], m["blend_cfgs"][m["pool_cfgs"][pi][1]]], valid_MRR_pool=vp[pi],
                    valid_MRR_base=base_sub, n_valid=int(msk.sum()), selection_rows=f"valid ts >= {a.cut} (validation-select only)",
                    n_valid_all=int(len(rows)), valid_MRR_base_all_rows=m["valid_MRR_base"], host=m.get("host"),
                    full_valid_argmax=dict(blend_idx=int(np.argmax(m["valid_MRR_blend"])), pool_idx=int(np.argmax(m["valid_MRR_pool"]))))}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out[a.base]))
