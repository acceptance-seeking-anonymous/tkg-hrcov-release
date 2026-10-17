#!/usr/bin/env python3
"""pilot_smallpedia / sp_listctl.py -- list-level density controls (CPU): reach of the ranked candidate lists
L_st / L_pat / L_br truncated at B vs uniform random, degree-matched random (same train-degree bins as the list members)
and popularity (top-|L| by train degree) at the SAME list size; all test queries, per stratum.  pc_density.list_controls."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import DS, WORK, N_ENT, KC, Splits, split_queries, jdump


def list_controls(L, gold, dbin, nbin, rpop, N, Bs):
    out = {}
    for B in Bs:
        Lb = L[:, :B]; ok = Lb >= 0
        hit = ((Lb == gold[:, None]) & ok).any(1); size = ok.sum(1); gbin = dbin[gold]
        cnt_same = ((dbin[np.maximum(Lb, 0)] == gbin[:, None]) & ok).sum(1)
        out[f"B{B}"] = dict(n=int(len(L)), reach=round(float(hit.mean()), 5), mean_size=round(float(size.mean()), 2),
                            uniform_random_same_size=round(float((size / N).mean()), 6),
                            degree_matched_random_same_size=round(float((cnt_same / nbin[gbin]).mean()), 6),
                            popularity_top_same_size=round(float((rpop[gold] < size).mean()), 5))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
    S = Splits(DS); freq = S.freq
    dbin = np.where(freq == 0, 0, 1 + np.floor(np.log2(np.maximum(freq, 1)))).astype(np.int64); nbin = np.bincount(dbin).astype(np.float64)
    rpop = np.empty(N_ENT, np.int64); rpop[np.argsort(-freq, kind="stable")] = np.arange(N_ENT)
    ld = f"{WORK}/lists_{a.base}"; fq = dict(np.load(f"{WORK}/feat_{a.base}/test_q.npz"))
    gold = fq["sro"][:, 2]
    masks = {"all": np.ones(len(gold), bool), "no_pair": ~fq["dyad"], "first_seen": fq["first_seen"], "pool_absent": ~fq["in_top50"],
             "no_pair_pool_absent": ~fq["dyad"] & ~fq["in_top50"], "cold": fq["stratum"] == 0}
    res = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)")
    for nm, Bs in (("Lst", (10, 50, 100, 200)), ("Lpat", (10, 50, 300)), ("Lbr", (10, 50, 200))):
        L = np.load(f"{ld}/test_{nm}.npy", mmap_mode="r")
        L = np.asarray(L).astype(np.int64)
        res[nm] = {mk: list_controls(L[m], gold[m], dbin, nbin, rpop, N_ENT, Bs) for mk, m in masks.items()}
    # static BLOCK actually admitted in layout A (outside C_cov) and the degree-matched block
    CA = np.load(f"{WORK}/feat_{a.base}/test_A_cand.npy", mmap_mode="r")
    for nm, sl in (("static_block_A", slice(150, 200)), ("dm_block_A", slice(200, 250)), ("cov_blocks_A", slice(50, 150))):
        L = np.asarray(CA[:, sl]).astype(np.int64)
        res[nm] = {mk: list_controls(L[m], gold[m], dbin, nbin, rpop, N_ENT, (L.shape[1],)) for mk, m in masks.items()}
    CB = np.load(f"{WORK}/feat_{a.base}/test_B_cand.npy", mmap_mode="r")
    L = np.asarray(CB[:, 50:150]).astype(np.int64)
    res["static_block_B"] = {mk: list_controls(L[m], gold[m], dbin, nbin, rpop, N_ENT, (50, 100)) for mk, m in masks.items()}
    jdump(res, f"listctl_{a.base}.json")
    print(json.dumps(res["static_block_A"]["no_pair"]), json.dumps(res["dm_block_A"]["no_pair"]))


if __name__ == "__main__":
    main()
