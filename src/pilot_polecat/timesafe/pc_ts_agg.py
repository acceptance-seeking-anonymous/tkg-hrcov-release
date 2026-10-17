#!/usr/bin/env python3
"""pilot_polecat / pc_ts_agg.py -- old (static all-train bridge cohort index) vs new (snapshot-before-t cohort index)
comparison for one base, run on serverA after the work_ts chain.  Writes ts_agg_<base>.json in the cwd.

old : $P/work      (pilot 1 as reported: stage-2 bridge ranker trained on transductive cohort features)
new : $P/work_ts   (same pipeline; only the stage-2 TRAINING cohort index changed; valid/test bridge index unchanged)
HR is not retrained: its inputs (base top-50 + F_hist) do not depend on the bridge and are verified identical here.
Bootstrap = harness oih_common.paired_bootstrap recipe (Poisson(1) over changed queries, B = 2000), as in pc_agg.py.
  python3 pc_ts_agg.py --base regcn [--systems systems_regcn.json]
"""
import os, sys, json, argparse, time
import numpy as np

P = os.environ.get("PC_P", os.path.expandvars("$WORK_DIR/tracks/pilot_polecat"))
OLD, NEW = f"{P}/work", f"{P}/work_ts"
SEEDS = [0, 1, 2]
KC = 50


def mets(rk):
    r = np.asarray(rk, np.float64)
    return dict(MRR=round(float(np.mean(1 / r)), 6), H1=round(float(np.mean(r <= 1)), 6), H3=round(float(np.mean(r <= 3)), 6),
                H10=round(float(np.mean(r <= 10)), 6), n=int(len(r)))


def boot(rr_a, rr_b, B=2000, seed=0):
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64); n = len(d)
    nz = np.nonzero(d)[0]
    if len(nz) == 0:
        return dict(delta=0.0, CI95=[0.0, 0.0], p_one_sided=1.0, n=n, n_changed=0)
    rng = np.random.default_rng(seed); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 200):
        bb = min(200, B - b0); w = rng.poisson(1.0, size=(bb, len(nz))); bs[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=round(float(d.mean()), 6), CI95=[round(float(lo), 6), round(float(hi), 6)],
                p_one_sided=round(float((bs <= 0).mean()), 4), n=int(n), n_changed=int(len(nz)))


def topk_overlap(A, B_, k, chunk=20000):
    """mean |A[:, :k] ∩ B[:, :k]| / k per row (-1 pads ignored)."""
    out = np.zeros(len(A))
    for lo in range(0, len(A), chunk):
        a = np.asarray(A[lo:lo + chunk, :k]); b = np.asarray(B_[lo:lo + chunk, :k])
        eq = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
        out[lo:lo + chunk] = eq.any(2).sum(1) / np.maximum((a >= 0).sum(1), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--systems", default="")
    a = ap.parse_args(); b = a.base; t0 = time.time()
    qo = dict(np.load(f"{OLD}/feat_{b}/test_q.npz")); qn = dict(np.load(f"{NEW}/feat_{b}/test_q.npz"))
    assert (qo["r_base"] == qn["r_base"]).all() and (qo["sro"] == qn["sro"]).all()
    n = len(qo["r_base"])
    res = dict(base=b, host="serverA (CPU, tkg-gpu:latest)", n_test=n, old_work=OLD, new_work=NEW)
    # ---------------- input identity checks (HR inputs, pattern block)
    chk = {}
    chk["r_base_identical"] = True
    for sp in ("valid", "test"):
        Xo = np.load(f"{OLD}/feat_{b}/{sp}_X.npy", mmap_mode="r"); Xn = np.load(f"{NEW}/feat_{b}/{sp}_X.npy", mmap_mode="r")
        meta = json.load(open(f"{NEW}/feat_{b}/meta.json"))
        nh = 19   # 8 base + 11 evidence features = F_hist (FEAT_NAMES order: base, evid, cov)
        same = 0; tot = 0
        for lo in range(0, Xo.shape[0], 20000):
            xo = np.asarray(Xo[lo:lo + 20000, :KC, :nh]); xn = np.asarray(Xn[lo:lo + 20000, :KC, :nh])
            same += int((xo == xn).all(axis=(1, 2)).sum()); tot += len(xo)
        chk[f"{sp}_F_hist_top50_identical_frac"] = same / tot
        chk["feat_names"] = meta.get("feat_names", None)
    Lpo = np.load(f"{OLD}/lists_{b}/test_Lpat.npy", mmap_mode="r"); Lpn = np.load(f"{NEW}/lists_{b}/test_Lpat.npy", mmap_mode="r")
    eqp = 0
    for lo in range(0, n, 50000):
        eqp += int((np.asarray(Lpo[lo:lo + 50000]) == np.asarray(Lpn[lo:lo + 50000])).all(1).sum())
    chk["test_Lpat_identical_frac"] = eqp / n
    Co = np.load(f"{OLD}/lists_{b}/test_Ccov.npy", mmap_mode="r"); Cn = np.load(f"{NEW}/lists_{b}/test_Ccov.npy", mmap_mode="r")
    eqc = 0
    for lo in range(0, n, 50000):
        eqc += int((np.asarray(Co[lo:lo + 50000, :100]) == np.asarray(Cn[lo:lo + 50000, :100])).all(1).sum())
    chk["test_Ccov_cols0_99_identical_frac"] = eqc / n
    res["input_checks"] = chk
    # ---------------- list-level change of the bridge
    Lbo = np.load(f"{OLD}/lists_{b}/test_Lbr.npy", mmap_mode="r"); Lbn = np.load(f"{NEW}/lists_{b}/test_Lbr.npy", mmap_mode="r")
    samp = np.sort(np.random.default_rng(1).choice(n, 100000, replace=False))
    lo_, ln_ = np.asarray(Lbo[samp]), np.asarray(Lbn[samp])
    bo = np.asarray(Co[samp, 100:]); bn = np.asarray(Cn[samp, 100:])
    lch = dict(sample_n=len(samp),
               Lbr_top10_overlap=float(topk_overlap(lo_, ln_, 10).mean()), Lbr_top50_overlap=float(topk_overlap(lo_, ln_, 50).mean()),
               Lbr_top200_overlap=float(topk_overlap(lo_, ln_, 200).mean()), Ccov_bridge_block_overlap=float(topk_overlap(bo, bn, 50).mean()))
    for nm, q in (("old", qo), ("new", qn)):
        gp = q["gpos_br"].astype(np.int64); pa = ~q["in_top50"].astype(bool)
        for K in (10, 50, 200):
            lch[f"{nm}_gold_in_Lbr_at{K}"] = round(float(((gp >= 0) & (gp < K)).mean()), 5)
            lch[f"{nm}_gold_in_Lbr_at{K}_pool_absent"] = round(float(((gp >= 0) & (gp < K))[pa].mean()), 5)
        g = q["gold_pos"].astype(np.int64)
        lch[f"{nm}_gold_in_Ccov"] = round(float((g >= 0).mean()), 5)
        lch[f"{nm}_gold_in_bridge_block"] = round(float((g >= 100).mean()), 5)
        lch[f"{nm}_gold_natural_bridge_reach"] = round(float(q["g_in_bridge"].mean()), 5)
    res["bridge_list_change"] = lch
    for nm, W in (("old", OLD), ("new", NEW)):
        res[f"lists_summary_{nm}"] = json.load(open(f"{W}/lists_{b}/lists_summary.json"))
    # ---------------- ranks
    Ro = dict(np.load(f"{OLD}/train_{b}/ranks_test.npz")); Ro.update(np.load(f"{OLD}/train_{b}/ranks_test_diag.npz"))
    Rn = dict(np.load(f"{NEW}/train_{b}/ranks_test.npz")); Rn.update(np.load(f"{NEW}/train_{b}/ranks_test_diag.npz"))

    def rr(R, arm):
        ks = [f"{arm}_s{s}" for s in SEEDS]
        assert all(k in R for k in ks), (arm, sorted(R))
        return np.mean([1.0 / R[k].astype(np.float64) for k in ks], 0), [mets(R[k]) for k in ks]

    systems = {}
    for tag, R, arms in (("old", Ro, ["HR", "HRCOV", "HRCOVFEAT", "HRCOV_BR", "HRCOV_PAT"]), ("new", Rn, ["HRCOV", "HRCOVFEAT", "HRCOV_BR", "HRCOV_PAT"])):
        for arm in arms:
            m, per = rr(R, arm)
            systems[f"{arm}_{tag}"] = dict(rr=m, per_seed=per)
    rows = []
    for k, v in systems.items():
        rows.append(dict(system=k, base=b, MRR_mean3=round(float(v["rr"].mean()), 6),
                         MRR_sd3=round(float(np.std([p["MRR"] for p in v["per_seed"]])), 6),
                         **{h: round(float(np.mean([p[h] for p in v["per_seed"]])), 6) for h in ("H1", "H3", "H10")},
                         per_seed_MRR=[p["MRR"] for p in v["per_seed"]]))
    res["rows"] = rows
    S = {k: v["rr"] for k, v in systems.items()}
    cmp_ = {}
    pairs = [("HRCOV_new", "HR_old", "primary rule re-evaluated: HRCOV(time-safe) - HR"),
             ("HRCOV_new", "HRCOV_old", "time-safe minus transductive HRCOV"),
             ("HRCOV_BR_new", "HR_old", "HRCOV-with-bridge (top-50 + bridge block) - HR"),
             ("HRCOV_BR_new", "HRCOV_BR_old", "time-safe minus transductive HRCOV_BR"),
             ("HRCOVFEAT_new", "HR_old", "coverage features only - HR"),
             ("HRCOVFEAT_new", "HRCOVFEAT_old", "time-safe minus transductive HRCOVFEAT"),
             ("HRCOV_PAT_new", "HR_old", "pattern block - HR"),
             ("HRCOV_PAT_new", "HRCOV_PAT_old", "time-safe minus transductive HRCOV_PAT"),
             ("HRCOV_new", "HRCOV_PAT_new", "bridge block marginal over pattern block (time-safe)"),
             ("HRCOV_old", "HRCOV_PAT_old", "bridge block marginal over pattern block (transductive)"),
             ("HRCOV_BR_new", "HRCOVFEAT_new", "bridge candidates beyond coverage features (time-safe)"),
             ("HRCOV_BR_old", "HRCOVFEAT_old", "bridge candidates beyond coverage features (transductive)")]
    for x, y, lab in pairs:
        cmp_[f"{x} - {y}"] = dict(label=lab, **boot(S[y], S[x]))
    res["comparisons"] = cmp_
    head = qn["stratum"] == 2
    g = boot(S["HR_old"][head], S["HRCOV_new"][head])
    g["DAMAGE"] = bool(g["CI95"][1] < 0 and abs(g["delta"]) >= 0.001)
    res["head_guardrail_HRCOV_new_vs_HR"] = g
    d = cmp_["HRCOV_new - HR_old"]
    res["primary_rule_PASS"] = bool(d["p_one_sided"] < 0.05 and d["delta"] >= 0.001 and not g["DAMAGE"])
    # ---------------- strata
    dy = qn["dyad"].astype(bool); cold = qn["stratum"] == 0
    st = {"cold (gold train freq<50)": cold, "head (>=500)": head, "no pair history": ~dy, "pair history": dy,
          "first-seen entity": qn["first_seen"].astype(bool), "pool-absent": ~qn["in_top50"].astype(bool),
          "no pair & 2-hop unreachable": ~dy & ~qn["two_hop"].astype(bool), "cold & pair history": cold & dy,
          "cold & no pair history": cold & ~dy, "gold in NEW bridge block": qn["gold_pos"].astype(np.int64) >= 100,
          "gold in OLD bridge block": qo["gold_pos"].astype(np.int64) >= 100}
    strata = {}
    for nm, m in st.items():
        e = dict(n=int(m.sum()), share=round(float(m.mean()), 4))
        for k in ("HR_old", "HRCOV_old", "HRCOV_new", "HRCOV_BR_old", "HRCOV_BR_new"):
            e[f"MRR_{k}"] = round(float(S[k][m].mean()), 6)
        e["HRCOV_new_minus_HR"] = boot(S["HR_old"][m], S["HRCOV_new"][m], B=1000)
        e["HRCOV_new_minus_old"] = boot(S["HRCOV_old"][m], S["HRCOV_new"][m], B=1000)
        e["HRCOV_BR_new_minus_old"] = boot(S["HRCOV_BR_old"][m], S["HRCOV_BR_new"][m], B=1000)
        strata[nm] = e
    res["strata"] = strata
    # ---------------- training records + official evaluator
    res["train_new"] = json.load(open(f"{NEW}/train_{b}/train_{b}.json")) if os.path.exists(f"{NEW}/train_{b}/train_{b}.json") else None
    res["train_diag_new"] = json.load(open(f"{NEW}/train_{b}/train_diag_{b}.json")) if os.path.exists(f"{NEW}/train_{b}/train_diag_{b}.json") else None
    if a.systems and os.path.exists(a.systems):
        sj = json.load(open(a.systems)); res["official_systems_raw"] = sj
        off = {}
        for k, v in (sj.get("official") or sj.get("systems") or {}).items():
            harness = None
            arm, sd = k.rsplit("_s", 1)
            key = f"{arm}_s{sd}"
            if key in Rn:
                harness = mets(Rn[key])["MRR"]
            off[k] = dict(official=v, harness_MRR_same_ranks=harness)
        res["official_systems"] = off
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"ts_agg_{b}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: (v["delta"], v["CI95"]) for k, v in cmp_.items()}))


if __name__ == "__main__":
    main()
