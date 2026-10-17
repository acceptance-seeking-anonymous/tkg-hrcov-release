#!/usr/bin/env python3
"""v78 / v78_official.py -- official py-tgb scoring of history-tail systems (v78_rule_amendment_1 'official_scoring'):
score vector = N - position in the strict RecB_xf history order, promoted list lifted above all (hd_blend.lift); system 'HIST'
= the history order itself.  py-tgb Evaluator with the shipped test negatives (TGB ids == harness ids, asserted), harness
strict filtered rank alongside.  Same loop as hd_blend.py --stage test.
  python3 v78_official.py --ds polecat --label sa --systems name=promlist.npy,... --out O [--smoke N]
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
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--erm", default="")
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); xi = HB.XiCounts(N, R, dev)
    freq = C.train_freq(cfg["dd"], N); gorder, _ = C.freq_order(freq); gorder_t = torch.as_tensor(gorder, device=dev)
    sro_full = HB.split_queries(Q["test"], R, "test"); n = len(sro_full) if not a.smoke else a.smoke
    csr = HB.RecbCSR(f"{C.RECB_XF}/recb_csr_test_{cfg['name']}.npz"); assert (csr.sro == sro_full).all()
    ERM = C.erm_load(a.erm, "test") if a.erm else None
    if ERM is not None: assert (np.asarray(ERM["keys"][:, :3]) == sro_full[:, :3]).all()
    systems = {}
    for kv in [x for x in a.systems.split(",") if x]:
        nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r"); assert len(systems[nm]) == len(sro_full) or a.smoke
    arms = ["HIST"] + list(systems)
    OUT = {k: np.zeros(n, np.float32) for nm in arms for k in (f"rr_{nm}", f"rank_{nm}", f"hrank_{nm}")}
    from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
    from tgb.linkproppred.evaluate import Evaluator
    import importlib.metadata as im, zipfile
    ns = cfg["ns"]
    if not os.path.exists(ns):
        os.makedirs(os.path.dirname(ns), exist_ok=True)
        with zipfile.ZipFile(cfg["zip"]) as z: z.extract(os.path.basename(ns), os.path.dirname(ns))
    sampler = TKGNegativeEdgeSampler(cfg["name"], first_dst_id=0, last_dst_id=N - 1); sampler.load_eval_set(ns, split_mode="test")
    evaluator = Evaluator(name=cfg["name"])
    h2t = np.load(f"{cfg['stage']}/h2t_entity.npy"); t2h = np.load(f"{cfg['stage']}/t2h_entity.npy")
    assert (h2t == np.arange(N)).all() and (t2h == np.arange(N)).all()
    tsraw = {}
    for ln in open(f"{cfg['dd']}/timestamp2id.txt"):
        k, v = ln.split(); tsraw[int(v)] = int(k)
    max_dev = 0.0; pos_ = 0
    with torch.no_grad():
        for sp, ts, sro, _, trip, _ in HB.recb_source(cfg, ("test",)):
            if sro is not None:
                p0 = pos_; nq = len(sro); pos_ += nq; idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(idx_all < n)
                for b0 in range(0, len(loc), a.bs):
                    lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; B = len(lb)
                    s_q, r_q, o_q = sro[lb, 0], sro[lb, 1], sro[lb, 2]
                    assert (sro[lb] == sro_full[rows]).all()
                    S = HB.recb_row(csr, rows, r_q, xi.xi(torch.as_tensor(r_q, device=dev)), dev, "tr")
                    pos, _ = C.order_positions(S, gorder_t, C.erm_dense(ERM, rows, N, dev) if ERM is not None else None)
                    Y0 = (N - pos).float(); gold = torch.as_tensor(o_q, device=dev)
                    fr_, fc_ = Fi.pairs(sro[lb]); fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                    Ys = {"HIST": Y0}
                    for nm, arr in systems.items():
                        Ys[nm] = HB.lift(Y0, torch.as_tensor(np.asarray(arr[rows]).astype(np.int64), device=dev))
                    negs = sampler.query_batch(h2t[s_q], h2t[o_q], np.array([tsraw[int(ts)]] * B), r_q, split_mode="test")
                    ln_ = np.array([len(x) for x in negs])
                    M = torch.zeros((B, N), dtype=torch.bool, device=dev)
                    M[torch.as_tensor(np.repeat(np.arange(B), ln_), device=dev), torch.as_tensor(np.concatenate(negs).astype(np.int64), device=dev)] = True
                    ar = torch.arange(B, device=dev)
                    for nm, Y in Ys.items():
                        pos_g = Y[ar, gold]; Yn = torch.where(M, Y, torch.full_like(Y, float("-inf")))
                        opt = (Yn > pos_g.unsqueeze(1)).sum(1); pes = (Yn >= pos_g.unsqueeze(1)).sum(1); rk = 0.5 * (opt + pes).double() + 1
                        m = evaluator.eval({"y_pred_pos": pos_g.cpu().numpy(), "y_pred_neg": Yn.cpu().numpy(), "eval_metric": ["mrr"]})
                        rr = (1.0 / rk.float()).cpu().numpy(); max_dev = max(max_dev, abs(float(m["mrr"]) - float(rr.mean())))
                        OUT[f"rr_{nm}"][rows] = rr; OUT[f"rank_{nm}"][rows] = rk.cpu().numpy(); OUT[f"hrank_{nm}"][rows] = HB.frank(Y, fr, fc, gold).float().cpu().numpy()
                        del Yn
                    del Ys, M, S, pos, Y0
            xi.add(trip)
    assert all((OUT[f"rank_{nm}"] > 0).all() for nm in arms), "unscored rows"
    res = {}
    for nm in arms:
        rk = OUT[f"rank_{nm}"].astype(np.float64); hr = OUT[f"hrank_{nm}"].astype(np.float64)
        res[nm] = dict(official=dict(MRR=float(OUT[f"rr_{nm}"].astype(np.float64).mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean())),
                       harness=dict(MRR=float((1 / hr).mean())), frac_rank_equal=float((np.abs(rk - hr) < 1e-6).mean()))
    meta = dict(ds=a.ds, label=a.label, n=int(n), evaluator="official py-tgb Evaluator", py_tgb=im.version("py-tgb"), official_vs_formula_max_batch_dev=max_dev,
                metrics=res, elapsed_s=round(time.time() - t0, 1), host=os.environ.get("HD_HOST", ""))
    np.savez(f"{a.out}/official_test_{a.ds}_{a.label}.npz", rows=np.arange(n), ts=sro_full[:n, 3].astype(np.int64), **OUT)
    json.dump(meta, open(f"{a.out}/official_test_{a.ds}_{a.label}.json", "w"), indent=1)
    log("ALL DONE", "systems", len(arms), "max_dev", max_dev)


if __name__ == "__main__":
    main()
