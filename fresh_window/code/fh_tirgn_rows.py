#!/usr/bin/env python3
"""phase2_fresh_window / fh_tirgn_rows.py -- full float32 log-prob rows + harness-schema dump of ONE split for the
GDELTFW TiRGN checkpoint (hd_tirgn.py), by replaying it with ground-truth online history (hd_tirgn_replay.source,
unchanged).  Row order = split_queries (per timestamp: forward block, inverse block), asserted.
  Stage 1:  python3 fh_tirgn_rows.py --ckpt <tirgn_GDELTFW_seed42.pt> --split valid
  Stage 2:  python3 fh_tirgn_rows.py --ckpt <same, hash-checked> --split test
Outputs (work dir): fullrows/tirgn_GDELTFW_<split>_scores_f32.bin (log(hr*p_hist + (1-hr)*p_raw), [n, N] float32) and
base_tirgn/tirgn_scores_GDELTFW_<split>.npz (sro, gold_score, gold_rank_raw, gold_rank_filt (exact, time-aware filter of
all splits, strict '>'), topk_ids/topk_scores K=200, split code, split_bounds) + base_tirgn/rows_<split>_log.json.
GPU memory is capped with torch.cuda.set_per_process_memory_fraction (env FH_GPU_CAP_GB, default 30).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fh_common import DS, WORK, N_ENT, N_REL, DD, FULLROWS, Splits, split_queries, metrics_from_ranks, jdump, log_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--split", required=True, choices=["valid", "test"])
ap.add_argument("--tree", required=True); ap.add_argument("--L", type=int, default=7); ap.add_argument("--hr", type=float, default=0.3)
ap.add_argument("--topk", type=int, default=200); ap.add_argument("--dec_chunk", type=int, default=1024)
a = ap.parse_args()


def main():
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory
        cap = float(os.environ.get("FH_GPU_CAP_GB", "30")) * 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot), 0)
    import hd_tirgn_replay as TRP
    S = Splits(DS); N, R = S.n_ent, S.R
    assert (N, R) == (N_ENT, N_REL)
    F = S.filter_index()
    sro_all = split_queries(S, a.split); n = len(sro_all)
    os.makedirs(f"{WORK}/fullrows", exist_ok=True); os.makedirs(f"{WORK}/base_tirgn", exist_ok=True)
    fr_path = FULLROWS["tirgn"][a.split]
    FR = np.memmap(fr_path, dtype=np.float32, mode="w+", shape=(n, N))
    K = a.topk
    GS = np.zeros(n, np.float32); GR = np.zeros(n, np.int32); GF = np.zeros(n, np.int32)
    TI = np.zeros((n, K), np.int32); TS_ = np.zeros((n, K), np.float32)
    ptr = 0
    cfg = dict(N=N, R=R, dd=DD)
    for sp, ts, sro, fn, trip in TRP.source(cfg, a.ckpt, a.L, (a.split,), dev, a.tree, hr=a.hr, dec_chunk=a.dec_chunk):
        if sp != a.split or sro is None:
            continue
        m = len(sro)
        assert (sro == sro_all[ptr:ptr + m]).all(), f"row order mismatch at ts {ts}"
        with torch.no_grad():
            lp = fn(sro[:, 0], sro[:, 1])
            gold = torch.as_tensor(sro[:, 2], device=lp.device).long()
            glp = lp.gather(1, gold.unsqueeze(1)).squeeze(1)
            fk = S.fkey(sro[:, 0], sro[:, 1], sro[:, 3])
            i0 = np.searchsorted(F["keys"], fk); i0c = np.clip(i0, 0, len(F["keys"]) - 1)
            ri, ci = [], []
            for q_ in np.flatnonzero(F["keys"][i0c] == fk):
                ob = F["objs"][F["start"][i0c[q_]]:F["end"][i0c[q_]]]
                ob = ob[ob != sro[q_, 2]]
                if len(ob): ri.append(np.full(len(ob), q_)); ci.append(ob)
            lf = lp.clone()
            if ri:
                lf[torch.as_tensor(np.concatenate(ri), device=lp.device), torch.as_tensor(np.concatenate(ci), device=lp.device)] = float("-inf")
            rk_f = (lf > glp.unsqueeze(1)).sum(1) + 1; rk_r = (lp > glp.unsqueeze(1)).sum(1) + 1
            tv, ti = torch.topk(lp, K, dim=1)
        FR[ptr:ptr + m] = lp.float().cpu().numpy()
        GS[ptr:ptr + m] = glp.cpu().numpy(); GR[ptr:ptr + m] = rk_r.cpu().numpy(); GF[ptr:ptr + m] = rk_f.cpu().numpy()
        TI[ptr:ptr + m] = ti.cpu().numpy(); TS_[ptr:ptr + m] = tv.cpu().numpy()
        ptr += m
        del lp, lf
    assert ptr == n, (ptr, n)
    FR.flush(); del FR
    code = {"valid": 1, "test": 2}[a.split]
    dump = f"{WORK}/base_tirgn/tirgn_scores_{DS}_{a.split}.npz"
    np.savez(dump, sro=sro_all.astype(np.int32), gold_score=GS, gold_rank_raw=GR, gold_rank_filt=GF, topk_ids=TI, topk_scores=TS_,
             split=np.full(n, code, np.int8), split_bounds=np.array([0, n if code == 1 else 0, n], np.int64))
    # self-check: the written rows are normalised log-probabilities
    chk = np.memmap(fr_path, dtype=np.float32, mode="r", shape=(n, N))
    rs = np.exp(chk[np.linspace(0, n - 1, 200).astype(int)].astype(np.float64)).sum(1)
    out = dict(split=a.split, n=int(n), ckpt=a.ckpt, L=a.L, hr=a.hr, device=dev,
               gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None), fullrows=fr_path, dump=dump,
               rowsum_prob_min=float(rs.min()), rowsum_prob_max=float(rs.max()), seconds=round(time.time() - t0, 1))
    if a.split == "valid":                       # valid metrics may be reported; the test split's metrics are not computed here
        out["valid_filt"] = metrics_from_ranks(GF.astype(np.float64)); out["valid_raw"] = metrics_from_ranks(GR.astype(np.float64))
    jdump(out, f"{WORK}/base_tirgn/rows_{a.split}_log.json"); jdump(out, f"tirgn_rows_{a.split}_log.json")
    os.system(f"chmod -R a+rX {WORK}/fullrows {WORK}/base_tirgn")
    log("ALL DONE", json.dumps({k: out.get(k) for k in ("split", "n", "valid_filt", "rowsum_prob_min", "rowsum_prob_max")}))


if __name__ == "__main__":
    main()
