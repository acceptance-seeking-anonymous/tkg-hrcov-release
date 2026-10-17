#!/usr/bin/env python3
"""hardening / hd_select.py -- valid-only selection of the BASE(+)RecB blend and the RecB-POOL configuration per base
(argmax of valid MRR over the pre-registered grids; first index on ties).  Reads blend_valid_<ds>_<base>.json.
  python3 hd_select.py --ds polecat --bases regcn,logcl --dir D --out sel_polecat.json
"""
import json, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--bases", required=True)
ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
for b in a.bases.split(","):
    m = json.load(open(f"{a.dir}/blend_valid_{a.ds}_{b}.json"))
    vb, vp = m["valid_MRR_blend"], m["valid_MRR_pool"]
    bi = max(range(len(vb)), key=lambda i: (vb[i], -i)); pi = max(range(len(vp)), key=lambda i: (vp[i], -i))
    out[b] = dict(blend_idx=bi, blend_cfg=m["blend_cfgs"][bi], valid_MRR_blend=vb[bi], pool_idx=pi,
                  pool_cfg=[m["pool_cfgs"][pi][0], m["blend_cfgs"][m["pool_cfgs"][pi][1]]], valid_MRR_pool=vp[pi],
                  valid_MRR_base=m["valid_MRR_base"], n_valid=m["n"], host=m.get("host"))
    print(b, json.dumps(out[b]))
json.dump(out, open(a.out, "w"), indent=1)
