#!/usr/bin/env python3
"""phase2_fresh_window / fw_sa_official.py -- v78_official.py for GDELTFW (no shipped negatives): score vector = N - position
in the strict RecB_xf history order, promoted list lifted above all (hd_blend.lift); system 'HIST' = the history order.
Rank (primary) = py-tgb Evaluator formula over all entities with the time-aware filter: 1 + #(neg > pos) + 0.5 #(neg == pos),
negatives = all entities except the gold and the other true objects of (s, r, t); harness strict rank (frank) alongside.
Same snapshot loop as v78_official.py; test metrics are written, not printed.
  python3 fw_sa_official.py --ds gdeltfw --label sa --systems SA_W100_s0=promlist.npy,... --out O
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import hd_blend as HB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--label", required=True); ap.add_argument("--systems", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); xi = HB.XiCounts(N, R, dev)
    freq = C.train_freq(cfg["dd"], N); gorder, _ = C.freq_order(freq); gorder_t = torch.as_tensor(gorder, device=dev)
    sro_full = HB.split_queries(Q["test"], R, "test"); n = len(sro_full)
    csr = HB.RecbCSR(f"{C.RECB_XF}/recb_csr_test_{cfg['name']}.npz"); assert (csr.sro == sro_full).all()
    systems = {}
    for kv in [x for x in a.systems.split(",") if x]:
        nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r"); assert len(systems[nm]) == n
    arms = ["HIST"] + list(systems)
    OUT = {k: np.zeros(n, np.float32) for nm in arms for k in (f"rr_{nm}", f"rank_{nm}", f"hrank_{nm}")}
    pos_ = 0
    with torch.no_grad():
        for sp, ts, sro, _, trip, _ in HB.recb_source(cfg, ("test",)):
            if sro is not None:
                p0 = pos_; nq = len(sro); pos_ += nq; idx_all = np.arange(p0, p0 + nq)
                for b0 in range(0, nq, a.bs):
                    lb = np.arange(b0, min(nq, b0 + a.bs)); rows = idx_all[lb]; B = len(lb)
                    r_q, o_q = sro[lb, 1], sro[lb, 2]
                    assert (sro[lb] == sro_full[rows]).all()
                    S = HB.recb_row(csr, rows, r_q, xi.xi(torch.as_tensor(r_q, device=dev)), dev, "tr")
                    pos, _ = C.strict_positions(S, gorder_t)
                    Y0 = (N - pos).float(); gold = torch.as_tensor(o_q, device=dev)
                    fr_, fc_ = Fi.pairs(sro[lb]); fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                    Ys = {"HIST": Y0}
                    for nm, arr in systems.items():
                        Ys[nm] = HB.lift(Y0, torch.as_tensor(np.asarray(arr[rows]).astype(np.int64), device=dev))
                    ar = torch.arange(B, device=dev)
                    M = torch.ones((B, N), dtype=torch.bool, device=dev); M[ar, gold] = False
                    if len(fr_): M[fr, fc] = False
                    for nm, Y in Ys.items():
                        pos_g = Y[ar, gold]; Yn = torch.where(M, Y, torch.full_like(Y, float("-inf")))
                        opt = (Yn > pos_g.unsqueeze(1)).sum(1); pes = (Yn >= pos_g.unsqueeze(1)).sum(1); rk = 0.5 * (opt + pes).double() + 1
                        OUT[f"rr_{nm}"][rows] = (1.0 / rk).float().cpu().numpy(); OUT[f"rank_{nm}"][rows] = rk.float().cpu().numpy()
                        OUT[f"hrank_{nm}"][rows] = HB.frank(Y, fr, fc, gold).float().cpu().numpy()
                        del Yn
                    del Ys, M, S, pos, Y0
            xi.add(trip)
    assert pos_ == n and all((OUT[f"rank_{nm}"] > 0).all() for nm in arms), "unscored rows"
    res = {}
    for nm in arms:
        rk = OUT[f"rank_{nm}"].astype(np.float64); hr = OUT[f"hrank_{nm}"].astype(np.float64)
        res[nm] = dict(official=dict(MRR=float(OUT[f"rr_{nm}"].astype(np.float64).mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean())),
                       harness=dict(MRR=float((1 / hr).mean())), frac_rank_equal=float((np.abs(rk - hr) < 1e-6).mean()))
    meta = dict(ds=a.ds, label=a.label, n=int(n), evaluator="all-entity time-aware filtered rank, ties averaged (py-tgb formula); harness strict alongside",
                metrics=res, elapsed_s=round(time.time() - t0, 1), host=os.environ.get("HD_HOST", ""))
    np.savez(f"{a.out}/official_test_{a.ds}_{a.label}.npz", rows=np.arange(n), ts=sro_full[:, 3].astype(np.int64), **OUT)
    json.dump(meta, open(f"{a.out}/official_test_{a.ds}_{a.label}.json", "w"), indent=1)
    log("ALL DONE", "systems", len(arms))


if __name__ == "__main__":
    main()
