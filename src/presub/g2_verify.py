#!/usr/bin/env python3
"""presub / g2_verify.py -- G2 pre-registered verification of the retriever full rows, ALL valid + test rows of one base:
exact filtered gold rank from the full row (same code as g2_features.py) vs the shared-harness dump rank
(RE-GCN / LogCL: gold_rank_filt; TiRGN: BaseDump top-K reconstruction, compared where exact), and the full-row test MRR
vs the residual-track MRR.  No re-ranker / control / claim number is computed.
  G2_DS=ICEWS18 python3 g2_verify.py --base tirgn --out verify_ICEWS18_tirgn.json
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from g2_common import DS, RESIDUAL_MRR, Splits, split_queries, metrics_from_ranks, jdump, log_fn, fullrows
from g2_features import dump_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N = S.n_ent; F = S.filter_index()
    out = dict(ds=DS, base=a.base, host=os.environ.get("HD_HOST", ""))
    for split in ("valid", "test"):
        sro = split_queries(S, split); n = len(sro)
        LP = fullrows(a.base, split); assert LP.shape[0] == n
        rb = np.zeros(n, np.int64); gl = np.zeros(n, np.float32)
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk); b = np.arange(lo, hi)
            s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
            row = np.asarray(LP[lo:hi], np.float32); glp = row[np.arange(len(b)), o]
            fk = S.fkey(s, r, t); nab = (row > glp[:, None]).sum(1)
            for i in range(len(b)):
                j_ = np.searchsorted(F["keys"], fk[i])
                if j_ < len(F["keys"]) and F["keys"][j_] == fk[i]:
                    oth = F["objs"][F["start"][j_]:F["end"][j_]]; oth = oth[oth != o[i]]
                    nab[i] -= int((row[i, oth] > glp[i]).sum())
            rb[b] = nab + 1; gl[b] = glp
            if (lo // a.chunk) % 50 == 0: log(split, hi, "/", n)
        rd, ex, lab = dump_ranks(a.base, split, S, np.arange(n), F)
        dis = np.flatnonzero(ex & (rb != rd))
        out[split] = dict(n=int(n), dump=lab, dump_exact_share=float(ex.mean()), rank_agree_exact_rows=float((rb[ex] == rd[ex]).mean()),
                          n_disagree=int(len(dis)), median_gold_rank_of_disagreements=float(np.median(rb[dis])) if len(dis) else None,
                          max_abs_rank_diff=int(np.abs(rb[ex] - rd[ex]).max()) if ex.any() else None,
                          MRR_fullrow=metrics_from_ranks(rb.astype(np.float64)), MRR_dump=metrics_from_ranks(rd.astype(np.float64)),
                          MRR_delta_fullrow_minus_dump=float(np.mean(1.0 / rb) - np.mean(1.0 / rd)))
        log(split, json.dumps(out[split]))
    out["residual_MRR"] = RESIDUAL_MRR[a.base]
    out["test_MRR_absdiff_vs_residual"] = abs(out["test"]["MRR_fullrow"]["MRR"] - RESIDUAL_MRR[a.base])
    out["pass_rank_agree_0p999"] = bool(out["test"]["rank_agree_exact_rows"] >= 0.999 and out["valid"]["rank_agree_exact_rows"] >= 0.999)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, a.out)
    log("ALL DONE", json.dumps({k: out[k] for k in ("pass_rank_agree_0p999", "test_MRR_absdiff_vs_residual")}))


if __name__ == "__main__":
    main()
