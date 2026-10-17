#!/usr/bin/env python3
"""v78 / v78_ext.py -- EXT_alone and EXT_blend (v78_framing_rule + v78_rule_amendment_1 'EXT').
Method scores from the sibling's top-1000 dumps (harness split order): s = top_scores on top_ids, all other entities at the
floor = minimum listed score of that query (amendment).  EXT_blend: (1 - a) log p_base + a log_softmax(z / tau) over all N,
z = (s - mean_top1000) / sd_top1000 (sd floored at 1e-6; z = 0 if < 2 listed); a in the v7.7 BLEND beta grid (13), tau in
{0.5, 1, 2, 4}.
  --stage valid : exact filtered ranks with average ties on the validation SELECTION part (ts >= 70% cut of the HRcov
                  validation rows) for base, EXT_alone and all 52 blends per method -> sel_ext_<ds>_<base>.json
  --stage test  : official py-tgb Evaluator (shipped negatives) + harness strict rank for base, EXT_alone_<m>, EXT_blend_<m>
  python3 v78_ext.py --ds polecat --base tirgn --stage valid --hist H --methods m=DIR,... --out O [source args]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import hd_blend as HB

AS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
TAUS = [0.5, 1.0, 2.0, 4.0]
GRID = [(a, t) for t in TAUS for a in AS]


def method_rows(M, rows, N, dev):
    ids = torch.as_tensor(np.asarray(M["top_ids"][rows]).astype(np.int64), device=dev)
    sc = torch.as_tensor(np.asarray(M["top_scores"][rows]).astype(np.float64), device=dev)
    fl = sc.min(1).values
    S = fl.unsqueeze(1).repeat(1, N); S.scatter_(1, ids, sc)
    mu = sc.mean(1, keepdim=True); sd = sc.std(1, unbiased=False, keepdim=True).clamp(min=1e-6)
    Z = (S - mu) / sd
    return S, Z


def avg_rank(sc, gold, FM):
    ar = torch.arange(len(gold), device=sc.device); g = sc[ar, gold].unsqueeze(1)
    unf = ~FM; unf[ar, gold] = False
    return 1.0 + ((sc > g) & unf).sum(1).double() + 0.5 * ((sc == g) & unf).sum(1).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--base", required=True); ap.add_argument("--stage", required=True, choices=["valid", "test"])
    ap.add_argument("--hist", required=True); ap.add_argument("--methods", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sel", default=""); ap.add_argument("--ckpt", default=""); ap.add_argument("--tirgn_tree", default=""); ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--fullrows", default=""); ap.add_argument("--store", default=""); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--tag", default="")
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); sp_ = a.stage; sro_full = HB.split_queries(Q[sp_], R, sp_)
    M = {}
    for kv in a.methods.split(","):
        nm, d = kv.split("="); dd = f"{d}/{'val' if sp_ == 'valid' else 'test'}"
        M[nm] = dict(top_ids=np.load(f"{dd}/top_ids.npy", mmap_mode="r"), top_scores=np.load(f"{dd}/top_scores.npy", mmap_mode="r"),
                     keys=np.load(f"{dd}/keys_harness.npy", mmap_mode="r"), rr_off=np.load(f"{dd}/rr_official.npy", mmap_mode="r"))
        assert len(M[nm]["keys"]) == len(sro_full) and (np.asarray(M[nm]["keys"][:, :3]) == sro_full[:, :3]).all(), f"dump rows {nm}"
    if sp_ == "valid":
        Hd = np.load(f"{a.hist}/valid_hist.npz"); rows_all = Hd["rows"].astype(np.int64); tsv = Hd["sro"][:, 3]
        uts = np.unique(tsv); cut = uts[int(0.7 * len(uts))]; rows_req = rows_all[tsv >= cut]
    else:
        rows_req = np.arange(len(sro_full)); cut = None
    if a.smoke: rows_req = rows_req[:a.smoke]
    for nm in M:                                   # adapted dumps: rows inside the learning data are NaN and must not be requested
        dd = f"{dict(kv.split('=') for kv in a.methods.split(','))[nm]}/{'val' if sp_ == 'valid' else 'test'}"
        if os.path.exists(f"{dd}/in_learning_data.npy"):
            assert not np.load(f"{dd}/in_learning_data.npy")[rows_req].any(), f"{nm}: requested rows inside the learning data"
        assert np.isfinite(np.asarray(M[nm]["top_scores"][np.sort(rows_req)[::97]])).all(), f"{nm}: non-finite scores on requested rows"
    n = len(rows_req); req = np.zeros(len(sro_full), bool); req[rows_req] = True; slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(n)
    if a.ckpt:
        import hd_tirgn_replay as TRP
        it = ((s, t, q, f, tr) for s, t, q, f, tr in TRP.source(cfg, a.ckpt, a.L, (sp_,), dev, a.tirgn_tree))
    elif a.store:
        import g8_rows
        it = g8_rows.source(cfg, a.store, (sp_,), dev)
    else:
        vp, tp = a.fullrows.split(","); it = ((s, t, q, f, tr) for s, t, q, f, tr, _ in HB.fullrow_source(cfg, {"valid": vp, "test": tp}, (sp_,), dev))
        MMr = np.memmap({"valid": vp, "test": tp}[sp_], dtype=np.float32, mode="r").reshape(-1, N)
    if sp_ == "valid":
        RK = {"base": np.zeros(n)}
        for nm in M:
            RK[f"EXT_alone_{nm}"] = np.zeros(n)
            for gi in range(len(GRID)): RK[f"EXT_blend_{nm}_{gi}"] = np.zeros(n)
        chk = {nm: 0.0 for nm in M}
    else:
        sel = json.load(open(a.sel)); systems = ["base"] + [f"EXT_alone_{nm}" for nm in M] + [f"EXT_blend_{nm}" for nm in M]
        OUT = {k: np.zeros(n, np.float32) for s_ in systems for k in (f"rr_{s_}", f"rank_{s_}", f"hrank_{s_}")}
        from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
        from tgb.linkproppred.evaluate import Evaluator
        import importlib.metadata as im
        sampler = TKGNegativeEdgeSampler(cfg["name"], first_dst_id=0, last_dst_id=N - 1); sampler.load_eval_set(cfg["ns"], split_mode="test")
        evaluator = Evaluator(name=cfg["name"]); tsraw = {}
        for ln in open(f"{cfg['dd']}/timestamp2id.txt"):
            k, v = ln.split(); tsraw[int(v)] = int(k)
        max_dev = 0.0
    pos = 0; done = 0
    with torch.no_grad():
        for spx, ts, sro, fn, trip in it:
            if sro is None or spx != sp_: continue
            p0 = pos; nq = len(sro); pos += nq; idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[idx_all])
            for b0 in range(0, len(loc), a.bs):
                lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[rows]; B = len(lb)
                assert (sro[lb] == sro_full[rows]).all()
                lp = (torch.as_tensor(np.asarray(MMr[rows]), device=dev) if a.fullrows else fn(sro[lb, 0], sro[lb, 1])).double()
                gold = torch.as_tensor(sro[lb, 2], device=dev); ar = torch.arange(B, device=dev)
                fr_, fc_ = Fi.pairs(sro[lb]); FM = torch.zeros((B, N), dtype=torch.bool, device=dev)
                if len(fr_): FM[torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev)] = True
                if sp_ == "valid":
                    RK["base"][sl] = avg_rank(lp, gold, FM).cpu().numpy()
                    for nm, Md in M.items():
                        S, Z = method_rows(Md, rows, N, dev)
                        ra = avg_rank(S, gold, FM); RK[f"EXT_alone_{nm}"][sl] = ra.cpu().numpy()
                        chk[nm] = max(chk[nm], float(np.abs(1.0 / ra.cpu().numpy() - np.asarray(Md["rr_off"][rows])).max()))
                        for t_ in TAUS:
                            ls = torch.log_softmax(Z / t_, dim=1)
                            for a_ in AS:
                                gi = GRID.index((a_, t_)); RK[f"EXT_blend_{nm}_{gi}"][sl] = avg_rank((1 - a_) * lp + a_ * ls, gold, FM).cpu().numpy()
                            del ls
                        del S, Z
                else:
                    Ys = {"base": lp}
                    for nm, Md in M.items():
                        S, Z = method_rows(Md, rows, N, dev); a_, t_ = sel[nm]["a"], sel[nm]["tau"]
                        Ys[f"EXT_alone_{nm}"] = S; Ys[f"EXT_blend_{nm}"] = (1 - a_) * lp + a_ * torch.log_softmax(Z / t_, dim=1)
                    negs = sampler.query_batch(sro[lb, 0], sro[lb, 2], np.array([tsraw[int(ts)]] * B), sro[lb, 1], split_mode="test")
                    ln_ = np.array([len(x) for x in negs]); Mk = torch.zeros((B, N), dtype=torch.bool, device=dev)
                    Mk[torch.as_tensor(np.repeat(np.arange(B), ln_), device=dev), torch.as_tensor(np.concatenate(negs).astype(np.int64), device=dev)] = True
                    for nm, Y in Ys.items():
                        Y = Y.float(); pg = Y[ar, gold]; Yn = torch.where(Mk, Y, torch.full_like(Y, float("-inf")))
                        opt = (Yn > pg.unsqueeze(1)).sum(1); pes = (Yn >= pg.unsqueeze(1)).sum(1); rk = 0.5 * (opt + pes).double() + 1
                        m_ = evaluator.eval({"y_pred_pos": pg.cpu().numpy(), "y_pred_neg": Yn.cpu().numpy(), "eval_metric": ["mrr"]})
                        rr = (1.0 / rk.float()).cpu().numpy(); max_dev = max(max_dev, abs(float(m_["mrr"]) - float(rr.mean())))
                        OUT[f"rr_{nm}"][sl] = rr; OUT[f"rank_{nm}"][sl] = rk.cpu().numpy(); OUT[f"hrank_{nm}"][sl] = HB.frank(Y, torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev), gold).float().cpu().numpy()
                        del Yn
                    del Ys
                done += B
                del lp, FM
            if (done // 20000) != ((done - len(loc)) // 20000): log(sp_, done, "/", n)
    assert done == n, (done, n)
    if sp_ == "valid":
        res = dict(ds=a.ds, base=a.base, n_select=int(n), cut=int(cut), grid=[list(g) for g in GRID], floor="minimum listed score (amendment)",
                   alone_vs_sibling_official_max_abs_rr=chk, MRR={k: float(np.mean(1.0 / v)) for k, v in RK.items()})
        for nm in M:
            vals = [res["MRR"][f"EXT_blend_{nm}_{gi}"] for gi in range(len(GRID))]; gi = int(np.argmax(vals))
            res[nm] = dict(a=GRID[gi][0], tau=GRID[gi][1], blend_select_MRR=vals[gi], alone_select_MRR=res["MRR"][f"EXT_alone_{nm}"])
        res["base_select_MRR"] = res["MRR"]["base"]
        C.jdump(res, f"{a.out}/sel_ext{a.tag}_{a.ds}_{a.base}.json"); C.jdump(res, f"sel_ext{a.tag}_{a.ds}_{a.base}.json")
        np.savez(f"{a.out}/ranks_ext{a.tag}_valid_{a.ds}_{a.base}.npz", rows=rows_req, **RK)
        log("valid selection written (validation numbers only)")
    else:
        res = {}
        for s_ in systems:
            rk = OUT[f"rank_{s_}"].astype(np.float64); hr = OUT[f"hrank_{s_}"].astype(np.float64)
            res[s_] = dict(official=dict(MRR=float(OUT[f"rr_{s_}"].astype(np.float64).mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean())),
                           harness=dict(MRR=float((1 / hr).mean())))
        meta = dict(ds=a.ds, base=a.base, n=int(n), selected=sel, evaluator="official py-tgb Evaluator", py_tgb=im.version("py-tgb"), max_dev=max_dev, metrics=res)
        np.savez(f"{a.out}/official_test_{a.ds}_v78_ext{a.tag}_{a.base}.npz", rows=rows_req, ts=sro_full[rows_req, 3].astype(np.int64), **OUT)
        json.dump(meta, open(f"{a.out}/official_test_{a.ds}_v78_ext{a.tag}_{a.base}.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
