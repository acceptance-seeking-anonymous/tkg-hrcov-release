#!/usr/bin/env python3
"""presub / hcpat_agg.py -- DEFINITIONAL RE-RUN (not part of the hardening decision): HRCOV_PAT (HRCOV without the bridge
block) vs HRCOV and vs the better recurrency control, official per-query RRs, query-level Poisson and timestamp-block
bootstraps; then the hardening rule evaluated with HRCOV_PAT in place of HRCOV where HRCOV_PAT exists.
cfg: {ds: {dd, strata, two_strongest: [..], strongest_blend_base, bases: {b: {round1: official npz, hcpat: official npz}}}}
  python3 hcpat_agg.py --cfg hcpat_cfg.json --out hcpat_definitional.json
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_agg import boot, load_quads, split_queries, MARGIN
from g7_cluster_bootstrap import boot_cluster

ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
C = json.load(open(a.cfg))


def sm(Z, pre):
    ks = [f"rr_{pre}_s{s}" for s in (0, 1, 2) if f"rr_{pre}_s{s}" in Z.files]
    return np.mean([Z[k].astype(np.float64) for k in ks], 0), [float(Z[k].astype(np.float64).mean()) for k in ks]


def both(d, ts):
    return dict(query=boot(d), ts_block=boot_cluster(d, ts))


OUT = dict(name="hcpat_definitional_rerun", label="definitional re-run, not part of the hardening decision",
           definition="HRCOV_PAT = HRCOV recipe without the bridge block: base top-50 ++ first <= 50 pattern-prior candidates (W 100); "
                      "25 features (8 base + 11 evidence + 6 pattern coverage; the 5 bridge features dropped); 3 seeds; official py-tgb evaluator",
           datasets={})
for ds, D in C.items():
    Q = {sp: load_quads(f"{D['dd']}/{sp}.txt") for sp in ("train", "test")}
    st = open(f"{D['dd']}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
    sro = split_queries(Q["test"], R); ts = sro[:, 3]
    freq = np.bincount(Q["train"][:, 0], minlength=N) + np.bincount(Q["train"][:, 2], minlength=N)
    S = np.load(D["strata"]); assert (S["sro"] == sro).all()
    head = freq[sro[:, 2]] >= 500
    res = dict(per_base={})
    rr = {}
    for b, B in D["bases"].items():
        Z1 = np.load(B["round1"]); hrcov, hrcov_seeds = sm(Z1, "HRCOV")
        bl = Z1["rr_BLEND"].astype(np.float64); po = Z1["rr_POOL"].astype(np.float64)
        better = "BLEND" if bl.mean() >= po.mean() else "POOL"; ctrl = bl if better == "BLEND" else po
        rec = dict(HRCOV_MRR=float(hrcov.mean()), HRCOV_per_seed=hrcov_seeds, better_control=better, better_control_MRR=float(ctrl.mean()),
                   BLEND_MRR=float(bl.mean()), POOL_MRR=float(po.mean()))
        rr[b] = dict(HRCOV=hrcov, BLEND=bl, POOL=po, better=ctrl, better_name=better)
        if B.get("hcpat") and os.path.exists(B["hcpat"]):
            Z2 = np.load(B["hcpat"]); pat, pat_seeds = sm(Z2, "HRCOV_PAT")
            assert len(pat) == len(sro) and (Z2["rows"] == np.arange(len(sro))).all()
            base_dev = float(np.abs(Z2["rr_base"].astype(np.float64) - Z1["rr_base"].astype(np.float64)).max())
            rec.update(HRCOV_PAT_MRR=float(pat.mean()), HRCOV_PAT_per_seed=pat_seeds, base_row_max_abs_rr_dev_vs_round1=base_dev,
                       HRCOV_PAT_minus_HRCOV=both(pat - hrcov, ts), HRCOV_PAT_minus_better_control=both(pat - ctrl, ts),
                       HRCOV_minus_better_control_round1=both(hrcov - ctrl, ts))
            rr[b]["HRCOV_PAT"] = pat
        res["per_base"][b] = rec
    # hardening rule with HRCOV_PAT in place of HRCOV where available
    bb = D["strongest_blend_base"]; blend_bb = rr[bb]["BLEND"] if bb in rr else np.load(D["strongest_blend_npz"])["rr_BLEND"].astype(np.float64)
    dec = {}
    for b in D["two_strongest"]:
        if b not in rr:
            continue
        sysn = "HRCOV_PAT" if "HRCOV_PAT" in rr[b] else "HRCOV"
        s = rr[b][sysn]
        c1 = {c: boot(s - rr[b][c]) for c in ("BLEND", "POOL")}
        c1p = all(v["delta"] >= MARGIN and v["p_one_sided"] < 0.05 for v in c1.values())
        c2 = boot(s - blend_bb); c2p = c2["delta"] > 0 and c2["p_one_sided"] < 0.05
        c3 = boot(s[head] - rr[b]["better"][head], B=1000); dmg = c3["CI95"][1] < 0 and abs(c3["delta"]) >= 0.001
        dec[b] = dict(system=sysn, C1={k: dict(delta=v["delta"], CI95=v["CI95"], p_one_sided=v["p_one_sided"]) for k, v in c1.items()}, C1_pass=bool(c1p),
                      C2_vs=f"{bb}(+)RecB", C2=dict(delta=c2["delta"], CI95=c2["CI95"], p_one_sided=c2["p_one_sided"]), C2_pass=bool(c2p),
                      C3_head=dict(delta=c3["delta"], CI95=c3["CI95"]), C3_damage=bool(dmg), PASS=bool(c1p and c2p and not dmg),
                      ts_block={k: boot_cluster(s - rr[b][k], ts)["CI95"] for k in ("BLEND", "POOL")})
    ok = [d["PASS"] for d in dec.values()]
    res["rule_with_HRCOV_PAT"] = dict(two_strongest=D["two_strongest"], per_base=dec,
                                      outcome=("PRACTICAL_ADVANCE" if len(ok) == 2 and all(ok) else
                                               "HEAD_DAMAGE" if all(d["C1_pass"] and d["C2_pass"] for d in dec.values()) else "RECURRENCY_DONE_WELL"),
                                      note=D.get("note", ""))
    OUT["datasets"][ds] = res
json.dump(OUT, open(a.out, "w"), indent=1)
print(json.dumps({d: v["rule_with_HRCOV_PAT"]["outcome"] for d, v in OUT["datasets"].items()}))
