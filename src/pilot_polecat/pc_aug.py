#!/usr/bin/env python3
"""pilot_polecat / pc_aug.py -- 6.5 augmentation ORACLE on one base (GPU), pre-registered stop rule.

Mechanism = 6.5 track rescore_lib.Rescorer (read-only import): evidence quads (s, r', c) injected into the most
recent history snapshot of THAT query only (block-diagonal copy), frozen base rescored; LogCL also gets the
global copy-history increment.  Oracle gate = inject ONLY the gold quad (x m, m in {1,3}) whenever the
structural pool (stage-2 bridge top-20 UNION pattern-prior top-20) proposes the gold.  Uniform sample of
--n test queries (rng 7); full-test dMRR estimated as the sample mean of dRR (non-eligible queries dRR = 0),
95% CI by bootstrap over the sample.  Also re-scores the no-injection copy of every eligible query to check the
mechanism reproduces the base (exact filtered rank).
STOP RULE (polecat_decision_rule.json): gated arms run only if the m=3 oracle estimate >= 0.002.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (TRACKS, P, DS, WORK, Splits, split_queries, jdump, log_fn)
sys.path.insert(0, f"{TRACKS}/augmentation/code")
import rescore_lib
rescore_lib.HARNESS = P                       # checkpoint path f"{HARNESS}/dumps/{base}_{DS}_seed42.pt" -> pilot dumps
from rescore_lib import Rescorer, make_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--pool_k", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS)
    Qt = split_queries(S, "test")
    ld = f"{WORK}/lists_{a.base}"
    Lp = np.load(f"{ld}/test_Lpat.npy", mmap_mode="r"); Lb = np.load(f"{ld}/test_Lbr.npy", mmap_mode="r")
    LK = np.load(f"{WORK}/lookup_{a.base}/test.npz")
    rng = np.random.default_rng(7)
    samp = np.sort(rng.choice(len(Qt), min(a.n, len(Qt)), replace=False))
    q = Qt[samp]
    pool = np.concatenate([np.asarray(Lb[samp, :a.pool_k]), np.asarray(Lp[samp, :a.pool_k])], 1)
    elig = (pool == q[:, 2:3]).any(1)
    ei = np.flatnonzero(elig)
    log("sample", len(samp), "eligible", len(ei))
    # copy requests: per eligible query: [no injection, gold x1, gold x3]
    req_q, req_off, req_c = [], [0], []
    for j in ei:
        for m in (0, 1, 3):
            req_q.append(j); req_c.extend([int(q[j, 2])] * m); req_off.append(len(req_c))
    req_q = np.array(req_q, np.int64); req_off = np.array(req_off, np.int64); req_c = np.array(req_c, np.int64)
    RS = Rescorer(a.base, DS, device="cuda")
    filt_fn = make_filter(DS)
    out = RS.run(req_q, q[:, 0], q[:, 1], q[:, 2], q[:, 3], req_off, req_c, filt_fn, log_every=200, tag="oracle")
    rf = out["rank_filt"].reshape(-1, 3)
    base_rank = LK["rank_filt"][samp].astype(np.int64)
    repro = float(np.mean(rf[:, 0] == base_rank[ei]))
    drr = {m: np.zeros(len(samp)) for m in (1, 3)}
    for k, m in ((1, 1), (2, 3)):
        drr[m][ei] = 1.0 / rf[:, k] - 1.0 / rf[:, 0]
    bs = np.random.default_rng(0)
    res = dict(base=a.base, host="serverA (GPU, tkg-gpu:latest)", n_sample=int(len(samp)), pool_k=a.pool_k,
               eligible_frac=float(elig.mean()), reproduction_frac_exact_rank=repro,
               max_abs_rank_diff_noinj=int(np.abs(rf[:, 0] - base_rank[ei]).max()) if len(ei) else 0,
               rescore_s=out["elapsed_s"], copies=int(len(req_q)))
    for m in (1, 3):
        d = drr[m]
        boots = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        res[f"oracle_m{m}"] = dict(full_test_dMRR_est=float(d.mean()), CI95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                   median_new_rank_of_gold=float(np.median(rf[:, 1 if m == 1 else 2])) if len(ei) else None,
                                   frac_eligible_improved=float((d[ei] > 0).mean()) if len(ei) else None)
    # placement oracle on the same sample (gold moved to rank 1 when in pool), for reference
    res["placement_oracle_dMRR_est"] = float(np.mean(np.where(elig, 1.0 - 1.0 / base_rank, 0.0)))
    res["stop_rule"] = dict(threshold=0.002, m3_estimate=res["oracle_m3"]["full_test_dMRR_est"],
                            run_gated_arms=bool(res["oracle_m3"]["full_test_dMRR_est"] >= 0.002))
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"aug_oracle_{a.base}.json"); os.makedirs(f"{WORK}/aug_{a.base}", exist_ok=True)
    np.savez(f"{WORK}/aug_{a.base}/oracle_sample.npz", samp=samp, elig=elig, rf=rf, base_rank=base_rank)
    jdump(res, f"{WORK}/aug_{a.base}/aug_oracle_{a.base}.json")
    os.system(f"chmod -R a+rX {WORK}/aug_{a.base}")
    log("ALL DONE", json.dumps(res["stop_rule"]))


if __name__ == "__main__":
    main()
