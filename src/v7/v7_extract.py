#!/usr/bin/env python3
"""v7 / v7_extract.py -- compact per-query extraction of STORED pipeline outputs (no model is run) for the v7 A-items.
For each (dataset, base) feature dir: every 1-D per-query array of the {test,valid}_q.npz files (features, lists, strata),
plus derived arrays from the stored candidate tensors: n_filt_top50 (other true answers inside the stored top-50) and
n_filt_above_gold (other true answers ranked above the gold inside the stored list), gold train frequency.
  python3 v7_extract.py --out DIR
"""
import os, sys, json, argparse
import numpy as np
O = os.path.expandvars("$WORK_DIR"); T = f"{O}/tracks"; H = f"{T}/hardening"; SCR = os.path.expandvars("$EXT_DIR")
DS = {
 "polecat_tirgn": dict(layout="pc", feat=f"{T}/pc_tirgn/work/feat_tirgn", lists=f"{T}/pc_tirgn/work/lists_tirgn", strata=f"{T}/pc_tirgn/work/strata_%s.npz",
                       dd=f"{T}/datasets/harness_data/tkgl-polecat"),
 "polecat_logcl": dict(layout="pc", feat=f"{T}/pilot_polecat/work/feat_logcl", lists=f"{T}/pilot_polecat/work/lists_logcl", strata=f"{T}/pilot_polecat/work/strata_%s.npz",
                       dd=f"{T}/datasets/harness_data/tkgl-polecat"),
 "smallpedia_tirgn": dict(layout="sp", feat=f"{T}/pilot_smallpedia/work/feat_tirgn", lists=None, strata=f"{T}/pilot_smallpedia/work/strata_%s.npz",
                          dd=f"{T}/datasets/harness_data/tkgl-smallpedia"),
 "smallpedia_regcn_tgb": dict(layout="sp", feat=f"{H}/ablations_A/inputs/sp_regcn_tgb_feat", lists=None, strata=f"{T}/pilot_smallpedia/work/strata_%s.npz",
                              dd=f"{T}/datasets/harness_data/tkgl-smallpedia"),
 "icews18_tirgn": dict(layout="g2", feat=f"{H}/presub_g2/ICEWS18/work/feat_tirgn", lists=f"{H}/presub_g2/ICEWS18/work/lists_tirgn",
                       strata=f"{H}/presub_g2/ICEWS18/work/strata_%s.npz", dd=f"{SCR}/logcl_redun/data/ICEWS18"),
 "gdelt_tirgn": dict(layout="g2", feat=f"{H}/presub_g2/GDELT/work/feat_tirgn", lists=f"{H}/presub_g2/GDELT/work/lists_tirgn",
                     strata=f"{H}/presub_g2/GDELT/work/strata_%s.npz", dd=f"{SCR}/gdelt_armAD/data/GDELT"),
}


def compact(v):
    v = np.asarray(v)
    if v.dtype == np.bool_: return v.astype(np.uint8)
    if v.dtype == np.int64: return v.astype(np.int32)
    if v.dtype == np.float64: return v.astype(np.float32)
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--only", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    info = {}
    for key, C in DS.items():
        if a.only and key not in a.only.split(","): continue
        info[key] = {}
        tr = np.loadtxt(f"{C['dd']}/train.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
        N = int(open(f"{C['dd']}/stat.txt").read().split()[0])
        freq = np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)
        for sp in ("test", "valid"):
            out = {}; keys = {}
            for nm, path in (("F", f"{C['feat']}/{sp}_q.npz"), ("L", f"{C['lists']}/{sp}_q.npz" if C["lists"] else None), ("S", C["strata"] % sp)):
                if path is None or not os.path.exists(path): keys[nm] = "missing"; continue
                Z = np.load(path, allow_pickle=False); keys[nm] = {}
                n0 = None
                for k in Z.files:
                    v = Z[k]; keys[nm][k] = list(v.shape)
                    if v.ndim == 1 or (k == "sro" and v.ndim == 2):
                        out[f"{nm}_{k}"] = compact(v)
            # stored candidate tensors (first 50 columns = stored base top-50)
            pre = f"{C['feat']}/{sp}_" + ("A_" if C["layout"] == "sp" else "")
            if os.path.exists(pre + "filt.npy"):
                FI = np.load(pre + "filt.npy", mmap_mode="r"); CA = np.load(pre + "cand.npy", mmap_mode="r")
                n = FI.shape[0]; nft = np.zeros(n, np.int16); nfa = np.zeros(n, np.int16); W = FI.shape[1]
                gp = out.get("F_gold_pos_A", out.get("F_gold_pos"))
                for lo in range(0, n, 50000):
                    f = np.asarray(FI[lo:lo + 50000, :50]); nft[lo:lo + 50000] = f.sum(1)
                    g = gp[lo:lo + 50000].astype(np.int64); j = np.arange(50)[None, :]
                    above = f & ((g[:, None] < 0) | (j < g[:, None]))
                    nfa[lo:lo + 50000] = above.sum(1)
                out["D_n_filt_top50"] = nft; out["D_n_filt_above_gold_top50"] = nfa; keys["W_cand"] = int(W)
            sro = out.get("F_sro")
            if sro is not None:
                out["D_gold_freq"] = freq[np.asarray(sro)[:, 2].astype(np.int64)].astype(np.int32)
                out["D_subj_freq"] = freq[np.asarray(sro)[:, 0].astype(np.int64)].astype(np.int32)
            np.savez_compressed(f"{a.out}/x_{key}_{sp}.npz", **out)
            info[key][sp] = dict(keys=keys, n=int(len(next(iter(out.values())))), size=os.path.getsize(f"{a.out}/x_{key}_{sp}.npz"))
            print(key, sp, info[key][sp]["n"], info[key][sp]["size"], flush=True)
    json.dump(info, open(f"{a.out}/extract_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
