#!/usr/bin/env python3
"""fitpop_audit.py -- fit-population audit of the v7.8 ladder (SA, +C, +S, +T) and HRcov (review item 2).
For each configuration: the common starting fit population (validation fit part, ts < 70% cut), the gold-in-pool fit queries
each arm keeps (v78_train.py / hcpat_train.py rule: gold_pos >= 0 in the arm's pool, cap 120,000 with rng(0) if exceeded), and
overlap of the retained query identities.  Writes the intersection (row positions) for the common-intersection refit.
  python3 fitpop_audit.py --v78 V --hcpat S1=json,S2=json,S3=json --out OUT
"""
import os, sys, json, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import v78_train_fitpop as TR

CFGS = [("S1", "smallpedia", "tirgn"), ("S2", "polecat", "tirgn"), ("S3", "polecat", "cogn")]
FIT_MAX = 120000

def fitset(sro, gp):
    ts = sro[:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; ret = fit_idx[gp[fit_idx] >= 0]; used = ret
    if len(ret) > FIT_MAX: used = np.sort(np.random.default_rng(0).choice(ret, FIT_MAX, replace=False))
    return int(cut), fit_idx, ret, used

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v78", required=True); ap.add_argument("--hcpat", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    hc = dict(kv.split("=") for kv in a.hcpat.split(","))
    res = {}
    for c, ds, b in CFGS:
        cfg = C.DS[ds]; P = f"{a.v78}/{ds}/pools"
        D = {}
        for arm, pool in (("SA", "SA50"), ("LAD", f"PC_{b}")):
            Q = np.load(f"{P}/{pool}/valid_q.npz"); D[arm] = dict(rows=Q["rows"].astype(np.int64), sro=Q["sro"].astype(np.int64), gp=Q["gold_pos"].astype(np.int64), K=int(Q["K"]))
        bc = cfg["bases"][b]; Dp = TR.load_pt(bc, cfg["layout"], "valid", 0)
        Qp = np.load(f"{bc['feat']}/valid_q.npz"); D["HRcov"] = dict(rows=Qp["rows"].astype(np.int64)[:len(Dp["gp"])], sro=Dp["sro"], gp=Dp["gp"], K=50)
        rowmode = {}
        for k in ("LAD", "HRcov"):
            assert np.array_equal(D[k]["sro"], D["SA"]["sro"]), (c, k, "query order differs")
            if not np.array_equal(D[k]["rows"], D["SA"]["rows"]):   # stored HRcov feature dirs index the sample by position
                off = np.unique(D[k]["rows"] - D["SA"]["rows"]); assert len(off) == 1, (c, k, "row ids differ", D[k]["rows"][:5], D["SA"]["rows"][:5])
                rowmode[k] = f"row ids = validation-split row ids + {int(off[0])} (global quad index); same queries, same order (sro equal row by row)"
        S = {}
        for k in D:
            cut, fit_idx, ret, used = fitset(D[k]["sro"], D[k]["gp"]); S[k] = dict(cut=cut, fit=fit_idx, ret=ret, used=used)
        assert len({S[k]["cut"] for k in S}) == 1 and all(np.array_equal(S[k]["fit"], S["SA"]["fit"]) for k in S)
        n0 = len(S["SA"]["fit"])
        arms = {"SA": "SA", "+C": "LAD", "+S": "LAD", "+T": "LAD", "HRcov": "HRcov"}
        sets = {k: set(S[v]["used"].tolist()) for k, v in arms.items()}
        inter = np.array(sorted(set.intersection(*sets.values())), dtype=np.int64)
        pair = {}
        for x, y in itertools.combinations(["SA", "+C", "HRcov"], 2):
            A_, B_ = sets[x], sets[y]; i_ = len(A_ & B_); u_ = len(A_ | B_)
            pair[f"{x} vs {y}"] = dict(intersection=i_, union=u_, jaccard=i_ / u_, only_first=len(A_ - B_), only_second=len(B_ - A_))
        hj = json.load(open(hc[c])); hn = hj["arms"]["HRCOV_PAT"]["n_fit_queries"]
        res[c] = dict(dataset=cfg["name"], base=b, valid_rows=int(len(D["SA"]["rows"])), cut_ts=S["SA"]["cut"],
                      common_start_fit_population=n0,
                      per_arm={k: dict(pool={"SA": "SA50 (history top-100)", "LAD": f"PC_{b} (base top-50 + 50 history candidates)", "HRcov": f"HRcov pool (base top-50 + <= 50 pattern candidates; stored HRCOV_PAT features)"}[v],
                                      gold_in_pool_retained=int(len(S[v]["ret"])), used=int(len(S[v]["used"])), frac_of_start=len(S[v]["used"]) / n0,
                                      capped=bool(len(S[v]["ret"]) > FIT_MAX)) for k, v in arms.items()},
                      identical_identities={"+C = +S = +T": "identical by construction (same pool directory and gold_pos array)", "SA = +C": sets["SA"] == sets["+C"], "+C = HRcov": sets["+C"] == sets["HRcov"], "SA = HRcov": sets["SA"] == sets["HRcov"]},
                      pairwise=pair, intersection_all_arms=int(len(inter)), intersection_frac_of_start=len(inter) / n0,
                      hrcov_stored_training_n_fit_queries=int(hn), hrcov_count_matches_stored=bool(int(hn) == len(S["HRcov"]["used"])), row_id_note=rowmode)
        np.save(f"{a.out}/fitint_{c}.npy", inter)
        print(c, json.dumps({k: v for k, v in res[c].items() if k not in ("pairwise", "per_arm")}), flush=True)
        print("   ", {k: (v["used"], round(v["frac_of_start"], 4)) for k, v in res[c]["per_arm"].items()}, {k: round(v["jaccard"], 4) for k, v in pair.items()}, flush=True)
    json.dump(res, open(f"{a.out}/fitpop_counts.json", "w"), indent=1)

if __name__ == "__main__":
    main()
