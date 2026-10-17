#!/usr/bin/env python3
"""v78 / v78_hist.py -- strict history order of RecB_xf (v78_rule_amendment_1 'history_order') for the validation rows of the
HRcov feature dirs and every test query.  GPU pass over snapshots (online xi counts, facts with ts < t).
Outputs <out>/{valid,test}_hist.npz: rows (split index), sro, htop [n, 200] (entity ids in history order), hq [n, 200]
(q = S / sum S), r_hist (strict time-aware filtered rank of the gold in the history order), gpos (unfiltered position of
the gold), npos (#entities with S > 0), and a summary json (HIST MRR on valid fit/select parts and test).
  python3 v78_hist.py --ds polecat --out OUT [--bs 512] [--smoke 0]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
import hd_blend as HB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(C.DS)); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--erm", default="")
    a = ap.parse_args(); t0 = time.time(); log = C.log_fn(t0)
    cfg = C.DS[a.ds]; hcfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R)
    freq = C.train_freq(cfg["dd"], N); gorder, grank = C.freq_order(freq); gorder_t = torch.as_tensor(gorder, device=dev)
    sro_full = {sp: HB.split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
    rows_req = {"valid": (np.load(cfg["valid_rows"]).astype(np.int64) if cfg["valid_rows"] else np.arange(len(sro_full["valid"]))),
                "test": np.arange(len(sro_full["test"]))}
    for b, bc in cfg["bases"].items():          # alignment with every HRcov feature dir
        for sp in ("valid", "test"):
            fs = np.load(f"{bc['feat']}/{sp}_q.npz")["sro"].astype(np.int64)
            assert (sro_full[sp][rows_req[sp]] == fs).all(), f"feature rows != split order ({b}, {sp})"
    if a.smoke:      # valid: strided sample across all timestamps (fit and selection parts both present); test: first rows
        st_ = max(1, len(rows_req["valid"]) // a.smoke)
        fidx = {"valid": np.arange(0, len(rows_req["valid"]), st_)[:a.smoke], "test": np.arange(min(a.smoke, len(rows_req["test"])))}
        rows_req = {sp: rows_req[sp][fidx[sp]] for sp in rows_req}
    else:
        fidx = {sp: np.arange(len(v)) for sp, v in rows_req.items()}
    csr = {sp: HB.RecbCSR(f"{C.RECB_XF}/recb_csr_{sp}_{cfg['name']}.npz") for sp in ("valid", "test")}
    for sp in ("valid", "test"): assert (csr[sp].sro == sro_full[sp]).all(), "RecB CSR row order != split order"
    ERM = {sp: C.erm_load(a.erm, sp) for sp in ("valid", "test")} if a.erm else None
    if ERM:
        for sp in ("valid", "test"): assert (np.asarray(ERM[sp]["keys"][:, :3]) == sro_full[sp][:, :3]).all(), "ERM dump rows"
    os.makedirs(a.out, exist_ok=True)
    O = {}
    for sp in ("valid", "test"):
        n = len(rows_req[sp])
        O[sp] = dict(htop=np.full((n, C.KMAX), -1, np.int32), hq=np.zeros((n, C.KMAX), np.float32), r_hist=np.zeros(n, np.int64),
                     gpos=np.zeros(n, np.int64), npos=np.zeros(n, np.int64), ssum=np.zeros(n, np.float64))
    req = {sp: np.zeros(len(sro_full[sp]), bool) for sp in ("valid", "test")}
    slot = {sp: -np.ones(len(sro_full[sp]), np.int64) for sp in ("valid", "test")}
    for sp in ("valid", "test"): req[sp][rows_req[sp]] = True; slot[sp][rows_req[sp]] = np.arange(len(rows_req[sp]))
    xi = HB.XiCounts(N, R, dev); pos_ = {"valid": 0, "test": 0}; done = {"valid": 0, "test": 0}
    with torch.no_grad():
        for sp, ts, sro, _, trip, _ in HB.recb_source(hcfg, ("valid", "test")):
            if sro is not None:
                p0 = pos_[sp]; nq = len(sro); pos_[sp] += nq
                idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[sp][idx_all])
                for b0 in range(0, len(loc), a.bs):
                    lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[sp][rows]; B = len(lb)
                    r_q, o_q = sro[lb, 1], sro[lb, 2]
                    S = HB.recb_row(csr[sp], rows, r_q, xi.xi(torch.as_tensor(r_q, device=dev)), dev, "tr")     # [B, N] f32
                    pos, ent = C.order_positions(S, gorder_t, C.erm_dense(ERM[sp], rows, N, dev) if ERM else None)
                    ar = torch.arange(B, device=dev); gold = torch.as_tensor(o_q, device=dev)
                    pg = pos[ar, gold]
                    fr_, fc_ = Fi.pairs(sro[lb]); nbf = torch.zeros(B, dtype=torch.int64, device=dev)
                    if len(fr_):
                        fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                        above = (pos[fr, fc] < pg[fr]).long(); nbf.index_add_(0, fr, above)
                    tot = S.double().sum(1)
                    top = ent[:, :C.KMAX]; Stop = torch.gather(S, 1, top).double()
                    q = torch.where(tot.unsqueeze(1) > 0, Stop / tot.clamp(min=1e-300).unsqueeze(1), torch.zeros_like(Stop))
                    o = O[sp]
                    o["htop"][sl] = top.int().cpu().numpy(); o["hq"][sl] = q.float().cpu().numpy()
                    o["r_hist"][sl] = (1 + pg - nbf).cpu().numpy(); o["gpos"][sl] = pg.cpu().numpy()
                    o["npos"][sl] = (S > 0).sum(1).cpu().numpy(); o["ssum"][sl] = tot.cpu().numpy()
                    done[sp] += B
                    del S, pos, ent
                if len(loc) and (done[sp] // 50000) != ((done[sp] - len(loc)) // 50000): log(sp, done[sp], "/", len(rows_req[sp]))
            xi.add(trip)
    summ = dict(ds=a.ds, KMAX=C.KMAX, device=dev, host=os.environ.get("HD_HOST", ""), smoke=a.smoke)
    for sp in ("valid", "test"):
        n = len(rows_req[sp]); o = O[sp]; assert done[sp] == n, (sp, done[sp], n)
        sr = sro_full[sp][rows_req[sp]]
        np.savez(f"{a.out}/{sp}_hist.npz", rows=rows_req[sp], fidx=fidx[sp], sro=sr, **o)
        rr = 1.0 / o["r_hist"]
        d = dict(n=int(n), HIST_MRR=float(rr.mean()), gold_in_top50=float((o["gpos"] < 50).mean()),
                 median_npos=float(np.median(o["npos"])), frac_ssum0=float((o["ssum"] <= 0).mean()))
        if sp == "valid":
            t_ = sr[:, 3]; uts = np.unique(t_); cut = uts[int(0.7 * len(uts))]
            d.update(cut=int(cut), HIST_MRR_fit=float(rr[t_ < cut].mean()), HIST_MRR_select=float(rr[t_ >= cut].mean()))
        summ[sp] = d
        if sp == "valid": log(sp, json.dumps(d))          # test summaries are written, not printed (selection discipline)
    summ["elapsed_s"] = round(time.time() - t0, 1)
    C.jdump(summ, f"{a.out}/hist_summary.json"); C.jdump(summ, f"hist_summary_{a.ds}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
