#!/usr/bin/env python3
"""phase2_fresh_window / fw_ext.py -- v78_ext.py (EXT_alone / EXT_blend of v78_rule_amendment_1) for GDELTFW.
Method scores from the CountTRuCoLa top-1000 dumps (fw_ct_score.py, harness split order): s = top_scores on top_ids, all other
entities at the floor = minimum listed score of that query.  EXT_blend: (1 - a) log p_base + a log_softmax(z / tau) over all N,
z = (s - mean_top1000) / sd_top1000 (sd floored at 1e-6); a in the v7.7 BLEND beta grid (13), tau in {0.5, 1, 2, 4}.
Changes vs v78_ext.py: dump layout <cfgdir>/dump_{val,test}; no shipped negatives -> rank = py-tgb formula over all entities
with the time-aware filter (negatives = all entities except gold and other true objects) and harness strict rank alongside;
--stage test scores ONLY the frozen external comparator (--comparator json: config + variant [+ a, tau]).
  --stage valid : ranks with average ties on the validation SELECTION part (ts >= 70% cut of the HRcov/SA validation rows)
                  for base, EXT_alone and the 52 blends of every config -> sel_ext_<ds>.json + ranks npz
  --stage test  : base + the comparator
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


def ranks_allent(sc, gold, FM):
    """py-tgb formula over all entities with the time-aware filter (FM = other true objects); returns (avg-tie, strict)."""
    ar = torch.arange(len(gold), device=sc.device); g = sc[ar, gold].unsqueeze(1)
    unf = ~FM; unf[ar, gold] = False
    gt = ((sc > g) & unf).sum(1).double(); eq = ((sc == g) & unf).sum(1).double()
    return 1.0 + gt + 0.5 * eq, 1.0 + gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--stage", required=True, choices=["valid", "test"])
    ap.add_argument("--hist", required=True); ap.add_argument("--methods", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fullrows", required=True, help="valid_path,test_path"); ap.add_argument("--comparator", default="")
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args(); t0 = time.time(); log = HB.log_fn(t0)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); sp_ = a.stage; sro_full = HB.split_queries(Q[sp_], R, sp_)
    M = {}
    for kv in a.methods.split(","):
        nm, d = kv.split("="); dd = f"{d}/dump_{'val' if sp_ == 'valid' else 'test'}"
        M[nm] = dict(top_ids=np.load(f"{dd}/top_ids.npy", mmap_mode="r"), top_scores=np.load(f"{dd}/top_scores.npy", mmap_mode="r"),
                     keys=np.load(f"{dd}/keys_harness.npy", mmap_mode="r"), rr_off=np.load(f"{dd}/rr_official.npy", mmap_mode="r"))
        assert len(M[nm]["keys"]) == len(sro_full) and (np.asarray(M[nm]["keys"]) == sro_full).all(), f"dump rows {nm}"
    if sp_ == "valid":
        Hd = np.load(f"{a.hist}/valid_hist.npz"); rows_all = Hd["rows"].astype(np.int64); tsv = Hd["sro"][:, 3]
        uts = np.unique(tsv); cut = uts[int(0.7 * len(uts))]; rows_req = rows_all[tsv >= cut]
    else:
        rows_req = np.arange(len(sro_full)); cut = None
        comp = json.load(open(a.comparator)); assert list(M) == [comp["config"]], "test stage scores only the frozen comparator"
    n = len(rows_req); req = np.zeros(len(sro_full), bool); req[rows_req] = True; slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(n)
    vp, tp = a.fullrows.split(","); it = HB.fullrow_source(cfg, {"valid": vp, "test": tp}, (sp_,), dev)
    MMr = np.memmap({"valid": vp, "test": tp}[sp_], dtype=np.float32, mode="r").reshape(-1, N); assert len(MMr) == len(sro_full)
    if sp_ == "valid":
        RK = {"base": np.zeros(n)}
        for nm in M:
            RK[f"EXT_alone_{nm}"] = np.zeros(n)
            for gi in range(len(GRID)): RK[f"EXT_blend_{nm}_{gi}"] = np.zeros(n)
        chk = {nm: 0.0 for nm in M}
    else:
        cnm = comp["config"]; systems = ["base", "EXT"]
        OUT = {k: np.zeros(n, np.float32) for s_ in systems for k in (f"rr_{s_}", f"rank_{s_}", f"hrank_{s_}")}
    pos = 0; done = 0
    with torch.no_grad():
        for spx, ts, sro, fn, trip, _ in it:
            if sro is None or spx != sp_: continue
            p0 = pos; nq = len(sro); pos += nq; idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[idx_all])
            for b0 in range(0, len(loc), a.bs):
                lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[rows]; B = len(lb)
                assert (sro[lb] == sro_full[rows]).all()
                lp = torch.as_tensor(np.asarray(MMr[rows]), device=dev).double()
                gold = torch.as_tensor(sro[lb, 2], device=dev)
                fr_, fc_ = Fi.pairs(sro[lb]); FM = torch.zeros((B, N), dtype=torch.bool, device=dev)
                if len(fr_): FM[torch.as_tensor(fr_, device=dev), torch.as_tensor(fc_, device=dev)] = True
                if sp_ == "valid":
                    RK["base"][sl] = ranks_allent(lp, gold, FM)[0].cpu().numpy()
                    for nm, Md in M.items():
                        S, Z = method_rows(Md, rows, N, dev)
                        ra = ranks_allent(S, gold, FM)[0]; RK[f"EXT_alone_{nm}"][sl] = ra.cpu().numpy()
                        chk[nm] = max(chk[nm], float(np.abs(1.0 / ra.cpu().numpy() - np.asarray(Md["rr_off"][rows])).max()))
                        for t_ in TAUS:
                            ls = torch.log_softmax(Z / t_, dim=1)
                            for a_ in AS:
                                gi = GRID.index((a_, t_)); RK[f"EXT_blend_{nm}_{gi}"][sl] = ranks_allent((1 - a_) * lp + a_ * ls, gold, FM)[0].cpu().numpy()
                            del ls
                        del S, Z
                else:
                    S, Z = method_rows(M[cnm], rows, N, dev)
                    Y = S if comp["variant"] == "alone" else (1 - comp["a"]) * lp + comp["a"] * torch.log_softmax(Z / comp["tau"], dim=1)
                    for nm_, Y_ in (("base", lp), ("EXT", Y)):
                        rk, hk = ranks_allent(Y_, gold, FM)
                        OUT[f"rr_{nm_}"][sl] = (1.0 / rk).float().cpu().numpy(); OUT[f"rank_{nm_}"][sl] = rk.float().cpu().numpy(); OUT[f"hrank_{nm_}"][sl] = hk.float().cpu().numpy()
                    del S, Z, Y
                done += B
                del lp, FM
            if (done // 20000) != ((done - len(loc)) // 20000): log(sp_, done, "/", n)
    assert done == n, (done, n)
    if sp_ == "valid":
        res = dict(ds=a.ds, n_select=int(n), cut=int(cut), grid=[list(g) for g in GRID], floor="minimum listed score (amendment)",
                   evaluator="all-entity time-aware filtered rank, ties averaged", alone_vs_dump_official_max_abs_rr=chk,
                   MRR={k: float(np.mean(1.0 / v)) for k, v in RK.items()})
        for nm in M:
            vals = [res["MRR"][f"EXT_blend_{nm}_{gi}"] for gi in range(len(GRID))]; gi = int(np.argmax(vals))
            res[nm] = dict(a=GRID[gi][0], tau=GRID[gi][1], blend_select_MRR=vals[gi], alone_select_MRR=res["MRR"][f"EXT_alone_{nm}"])
        res["base_select_MRR"] = res["MRR"]["base"]
        C.jdump(res, f"{a.out}/sel_ext_{a.ds}.json")
        np.savez(f"{a.out}/ranks_ext_valid_{a.ds}.npz", rows=rows_req, **RK)
        log("valid selection written (validation numbers only)", json.dumps({nm: res[nm] for nm in M}))
    else:
        res = {}
        for s_ in systems:
            rk = OUT[f"rank_{s_}"].astype(np.float64); hr = OUT[f"hrank_{s_}"].astype(np.float64)
            res[s_] = dict(official=dict(MRR=float(OUT[f"rr_{s_}"].astype(np.float64).mean()), H1=float((rk <= 1).mean()), H3=float((rk <= 3).mean()), H10=float((rk <= 10).mean())),
                           harness=dict(MRR=float((1 / hr).mean())))
        meta = dict(ds=a.ds, n=int(n), comparator=comp, evaluator="all-entity time-aware filtered rank, ties averaged; strict alongside", metrics=res)
        np.savez(f"{a.out}/official_test_{a.ds}_ext.npz", rows=rows_req, ts=sro_full[rows_req, 3].astype(np.int64), **OUT)
        json.dump(meta, open(f"{a.out}/official_test_{a.ds}_ext.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
