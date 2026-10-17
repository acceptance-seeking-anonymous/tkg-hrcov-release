#!/usr/bin/env python3
"""presub / g1_agg.py -- G1 decision of a_presub_rule.json from the official-evaluator per-query RRs.
Inputs per dataset (json --cfg): g1_official (hd_blend.py --source recb --no_controls output with systems G1_K50_s0..2,
G1_K150_s0..2), g1_train (train_g1.json: valid-select MRR per arm / seed), hardening official npz of the two strongest
bases, the hardening verdict (two strongest, better comparator), dd (data dir, for timestamps).
  python3 g1_agg.py --cfg g1_agg_cfg.json --verdict hardening_verdict_raw.json --out g1_verdict.json
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g7_cluster_bootstrap import boot_query, boot_cluster, load_quads, split_order

MARGIN = 0.005


def seed_mean(Z, pre):
    ks = [f"rr_{pre}_s{s}" for s in (0, 1, 2) if f"rr_{pre}_s{s}" in Z.files]
    assert ks, pre
    return np.mean([Z[k].astype(np.float64) for k in ks], 0), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--verdict", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="code check: first n test rows of every array")
    a = ap.parse_args()
    cut = (lambda x: x[:a.smoke]) if a.smoke else (lambda x: x)
    C = json.load(open(a.cfg)); V = json.load(open(a.verdict)); OUT = dict(name="g1_verdict", rule="a_presub_rule.json (G1)", datasets={})
    labels = {}
    for ds, D in C.items():
        VD = V["datasets"][ds]
        ts, fid, gold = map(cut, split_order(load_quads(f"{D['dd']}/test.txt")))
        G = np.load(D["g1_official"]); T = json.load(open(D["g1_train"]))
        sel = {arm: float(np.mean([v["select"]["MRR"] for v in T["arms"][arm]["seeds"].values()])) for arm in ("G1_K50", "G1_K150") if arm in T["arms"]}
        g1_arm = max(sel, key=lambda k: (sel[k], k == "G1_K50"))          # higher mean valid-select MRR (tie -> K50)
        rr_g1, ns = seed_mean(G, g1_arm); rr_g1 = cut(rr_g1)
        rr_recb = cut(G["rr_base"].astype(np.float64))
        res = dict(valid_select_MRR=sel, G1_arm=g1_arm, n_seeds=ns, G1_test_MRR=float(rr_g1.mean()),
                   other_arm={k: float(seed_mean(G, k)[0].mean()) for k in sel if k != g1_arm},
                   recb_row_MRR=float(rr_recb.mean()), recb_check=D.get("recb_expected"), per_base={})
        if D.get("recb_expected") is not None:
            res["recb_check_absdiff"] = abs(res["recb_row_MRR"] - D["recb_expected"])
        assert len(rr_g1) == len(ts)
        flags = []
        for b in VD["two_strongest"]:
            Z = np.load(D["bases"][b]); hrcov, _ = seed_mean(Z, "HRCOV"); hr, _ = seed_mean(Z, "HR")
            better = VD["decision"]["per_base"][b]["better_comparator"]; ctrl = Z[f"rr_{better}"].astype(np.float64)
            hrcov, hr, ctrl = cut(hrcov), cut(hr), cut(ctrl)
            assert ("rows" not in Z.files) or (cut(Z["rows"]) == np.arange(len(hrcov))).all(), "base npz not in test split order"
            d = hrcov - rr_g1
            q, t, f = boot_query(d), boot_cluster(d, ts), boot_cluster(d, fid)
            ok = q["delta"] >= MARGIN and q["p_one_sided"] < 0.05 and t["CI95"][0] > 0
            ci0 = (q["CI95"][0] <= 0 <= q["CI95"][1]) or (t["CI95"][0] <= 0 <= t["CI95"][1]) or q["CI95"][1] < 0 or t["CI95"][1] < 0
            lr = (q["delta"] < MARGIN) or ci0
            flags.append(dict(base=b, BASE_CONTRIBUTES=bool(ok), LEARNED_RECURRENCY=bool(lr)))
            res["per_base"][b] = dict(HRCOV_MRR=float(hrcov.mean()), G1_MRR=float(rr_g1.mean()),
                                      HRCOV_minus_G1=dict(query=q, ts=t, fact=f),
                                      secondary={"G1 - RecB trained": boot_query(rr_g1 - rr_recb),
                                                 f"G1 - {better} of {b}": boot_query(rr_g1 - ctrl),
                                                 "HR - G1_K50": boot_query(hr - cut(seed_mean(G, "G1_K50")[0])) if "G1_K50" in sel else None},
                                      flags=flags[-1])
        lab = ("LEARNED_RECURRENCY" if any(x["LEARNED_RECURRENCY"] for x in flags) else
               "BASE_CONTRIBUTES" if all(x["BASE_CONTRIBUTES"] for x in flags) else "LEARNED_RECURRENCY")
        res["outcome"] = lab; labels[ds] = lab
        OUT["datasets"][ds] = res
    tgb = [labels[d] for d in labels if d in ("polecat", "smallpedia")]
    n_lr = sum(x == "LEARNED_RECURRENCY" for x in tgb)
    OUT["manuscript_consequence"] = ("reframe as learned recurrency (both TGB datasets)" if n_lr == 2 else
                                     "restrict the base-contribution claim to the BASE_CONTRIBUTES dataset" if n_lr == 1 else
                                     "base contribution demonstrated on both TGB datasets") if len(tgb) == 2 else "incomplete"
    OUT["smoke"] = int(a.smoke)
    json.dump(OUT, open(a.out, "w"), indent=1)
    print(json.dumps(labels), OUT["manuscript_consequence"])


if __name__ == "__main__":
    main()
