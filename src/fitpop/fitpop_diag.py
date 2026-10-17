#!/usr/bin/env python3
"""fitpop_diag.py -- diagnostics for the fit-population audit.
  --mode prep    : size-matched random subsamples of each pool's own fit population (size = common intersection, rng 1)
                   and test-split gold-in-pool flags per pool (SA50, PC_<b>, stored HRcov pool)
  --mode collect : harness test ranks of the registered arms, the intersection refit, the own-population HRcov re-run and the
                   size-matched refits, extracted into one small npz per configuration
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import v78_train_fitpop as TR
from fitpop_audit import fitset, CFGS

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--v78", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); FP = f"{a.v78}/fitpop"; os.makedirs(a.out, exist_ok=True); info = {}
    for c, ds, b in CFGS:
        cfg = C.DS[ds]; P = f"{a.v78}/{ds}/pools"; bc = cfg["bases"][b]
        if a.mode == "prep":
            I = np.load(f"{FP}/fitint_{c}.npy"); flags = {}; info[c] = dict(n_int=int(len(I)))
            for tag, pool in (("sa", "SA50"), ("pc", f"PC_{b}"), ("pt", f"PT:{b}")):
                if pool.startswith("PT:"): Dv = TR.load_pt(bc, cfg["layout"], "valid", 0); Dt = TR.load_pt(bc, cfg["layout"], "test", 0); sro, gp, gpt = Dv["sro"], Dv["gp"], Dt["gp"]
                else:
                    Qv = np.load(f"{P}/{pool}/valid_q.npz"); Qt = np.load(f"{P}/{pool}/test_q.npz"); sro, gp, gpt = Qv["sro"].astype(np.int64), Qv["gold_pos"].astype(np.int64), Qt["gold_pos"].astype(np.int64)
                cut, fit_idx, ret, used = fitset(sro, gp); assert np.isin(I, used).all()
                sm = np.sort(np.random.default_rng(1).choice(used, len(I), replace=False)); np.save(f"{FP}/sizematch_{c}_{tag}.npy", sm)
                info[c][tag] = dict(own=int(len(used)), sizematch=int(len(sm)), sizematch_overlap_with_intersection=int(np.isin(sm, I).sum()))
                flags[f"test_gold_in_{tag}"] = gpt >= 0
            np.savez(f"{a.out}/test_goldflags_{c}.npz", **flags)
            info[c]["test_frac"] = {k: float(v.mean()) for k, v in flags.items()}
            print(c, json.dumps(info[c]), flush=True)
        else:
            Z = {}
            reg = np.load(f"{a.v78}/{ds}/train/ranks_test.npz")
            for k in ["SA_W100", f"LAD_C_{b}", f"LAD_S_{b}", f"LAD_T_{b}"]:
                for s in (0, 1, 2): Z[f"reg__{k}_s{s}"] = reg[f"{k}_s{s}"]
            for tag, d in (("int", f"{c}_int"), ("own", f"{c}_own"), ("smsa", f"{c}_sm_sa"), ("smpc", f"{c}_sm_pc"), ("smpt", f"{c}_sm_pt")):
                R = np.load(f"{FP}/{d}/ranks_test.npz")
                for k in R.files: Z[f"{tag}__{k}"] = R[k]
            np.savez_compressed(f"{a.out}/harness_ranks_{c}.npz", **Z); print(c, len(Z), flush=True)
    if a.mode == "prep": json.dump(info, open(f"{a.out}/diag_prep.json", "w"), indent=1)

if __name__ == "__main__":
    main()
