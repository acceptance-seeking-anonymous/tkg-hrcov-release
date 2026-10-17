#!/usr/bin/env python3
"""v7 / v7_g1_features_icews.py = presub g1_features.py with the tkgl-icews entry of pi_blend (B4, Stage B setting).
presub / g1_features.py -- G1 of a_presub_rule.json: base-free learned recurrency.  Candidate pools and re-ranker
features built from RecB only (no base model scores, no base candidates).

Generator: RecB trained mode, clean variant (hd_recb.py CSR of psi + online xi counts, exactly hd_blend.recb_row);
q(o) = RecB(o) / sum_e RecB(e) over ALL entities.  Pool: RecB top-W (W = 150; ties broken by entity id; only
entities with RecB > 0).  19 features per (query, candidate), in the pc_features column layout:
  0-7  generator (HR 'base' slots computed from q): log q, log q - log q1, log RecB rank, in-top-50 flag, q1,
       entropy of the normalised top-50, log q1 - log q2, log tail mass outside the RecB top-200
  8-18 evidence: rs_features Hist / DyadReader (facts with ts < t), identical code and order to pc_features.py
Output (pc_train.py layout, read by g1_train.py): <out>/{valid,test}_{X,logp,cand,filt,babove}.npy + _q.npz with
r_base = time-aware filtered RecB rank of the gold with ties at their average position (the official convention).
  PC_HARNESS=... PC_P=... PC_DD=... python3 g1_features.py --ds polecat --common pc_common --recb_dir D --out O
"""
import os, sys, json, time, argparse, importlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v7_pi_blend as HB

KC = 50
EVID_F = ["ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
          "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj"]
GEN_F = ["log_q", "log_q_minus_top1", "log_recb_rank", "is_top50", "q1", "q_entropy50", "q_margin12", "q_log_tailmass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["polecat", "smallpedia", "icews"])
    ap.add_argument("--common", required=True, help="pc_common | sp_common (registers the dataset in oih_common)")
    ap.add_argument("--recb_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, default=150); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = HB.log_fn(t0); torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PC = importlib.import_module(a.common)
    sys.path.insert(0, f"{PC.TRACKS}/residual/code")
    from rs_features import Hist, DyadReader
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; W = a.W
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R)
    S = PC.Splits(PC.DS); Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad reader", BM.n_transitions, "device", dev)
    sro_full = {sp: HB.split_queries(Q[sp], R, sp) for sp in ("valid", "test")}
    for sp in ("valid", "test"):
        assert (sro_full[sp] == PC.split_queries(S, sp)).all(), "split order mismatch"
    rows_req = {"valid": (np.load(cfg["valid_rows"]) if cfg["valid_rows"] else np.arange(len(sro_full["valid"]))),
                "test": np.arange(len(sro_full["test"]))}
    if a.smoke:
        rows_req = {sp: v[:a.smoke] for sp, v in rows_req.items()}
    csr = {sp: HB.RecbCSR(f"{a.recb_dir}/recb_csr_{sp}_{cfg['name']}.npz") for sp in ("valid", "test")}
    for sp in ("valid", "test"):
        assert (csr[sp].sro == sro_full[sp]).all(), "RecB CSR row order != split order"
    os.makedirs(a.out, exist_ok=True)
    nF = len(GEN_F) + len(EVID_F)
    O = {}
    for sp in ("valid", "test"):
        n = len(rows_req[sp])
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{a.out}/{sp}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        O[sp] = dict(X=mm("X", (n, W, nF), np.float16), LP=mm("logp", (n, W), np.float32), CA=mm("cand", (n, W), np.int32),
                     FI=mm("filt", (n, W), np.bool_), BA=mm("babove", (n, W), np.bool_), rb=np.zeros(n, np.float64),
                     rb_opt=np.zeros(n, np.int64), gp=np.full(n, -1, np.int64), npos=np.zeros(n, np.int64))
    req = {sp: np.zeros(len(sro_full[sp]), bool) for sp in ("valid", "test")}
    slot = {sp: -np.ones(len(sro_full[sp]), np.int64) for sp in ("valid", "test")}
    for sp in ("valid", "test"):
        req[sp][rows_req[sp]] = True; slot[sp][rows_req[sp]] = np.arange(len(rows_req[sp]))
    xi = HB.XiCounts(N, R, dev)
    pos = {"valid": 0, "test": 0}; done = {"valid": 0, "test": 0}
    for sp, ts, sro, _, trip, _ in HB.recb_source(cfg, ("valid", "test")):
        if sro is not None:
            p0 = pos[sp]; nq = len(sro); pos[sp] += nq
            idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[sp][idx_all])
            for b0 in range(0, len(loc), a.bs):
                lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[sp][rows]; B = len(lb)
                s_q, r_q, o_q, t_q = sro[lb, 0], sro[lb, 1], sro[lb, 2], sro[lb, 3]
                Sx = HB.recb_row(csr[sp], rows, r_q, xi.xi(torch.as_tensor(r_q, device=dev)), dev, "tr").double()   # [B, N]
                tot = Sx.sum(1, keepdim=True); q = torch.where(tot > 0, Sx / tot.clamp(min=1e-300), torch.zeros_like(Sx))
                ar = torch.arange(B, device=dev); gold = torch.as_tensor(o_q, device=dev)
                fr_, fc_ = Fi.pairs(sro[lb]); fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                # gold rank, time-aware filtered, ties at their average position
                Sf = Sx.clone()
                if len(fr): Sf[fr, fc] = -1.0
                sg = Sx[ar, gold]; Sf[ar, gold] = -1.0
                gt = (Sf > sg.unsqueeze(1)).sum(1); eq = (Sf == sg.unsqueeze(1)).sum(1)
                rmid = 1.0 + gt.double() + 0.5 * eq.double()
                # pool: RecB top-W, ties by entity id (stable sort of -score over ids 0..N-1)
                order = torch.sort(-Sx, dim=1, stable=True).indices[:, :max(W, 200)]
                sc = torch.gather(Sx, 1, order); qs = torch.gather(q, 1, order)
                cand = torch.where(sc[:, :W] > 0, order[:, :W], torch.full_like(order[:, :W], -1))
                valid = cand >= 0
                filt = torch.zeros((B, W), dtype=torch.bool, device=dev)
                if len(fr):
                    FM = torch.zeros((B, N), dtype=torch.bool, device=dev); FM[fr, fc] = True
                    filt = torch.gather(FM, 1, order[:, :W]) & valid
                isg = (cand == gold.unsqueeze(1)) & valid
                gp = torch.where(isg.any(1), isg.int().argmax(1), torch.full_like(gold, -1))
                ba = (sc[:, :W] > sg.unsqueeze(1)) & valid
                lq = torch.log(qs.clamp(min=1e-300)); q1 = qs[:, 0]
                p50 = qs[:, :KC] / qs[:, :KC].sum(1, keepdim=True).clamp(min=1e-300)
                ent = -(p50 * torch.log(p50.clamp(min=1e-300))).sum(1)
                marg = torch.log(qs[:, 0].clamp(min=1e-300)) - torch.log(qs[:, 1].clamp(min=1e-300))
                tail = torch.log((1.0 - qs[:, :200].sum(1)).clamp(min=1e-12))
                Xf = np.zeros((B, W, nF), np.float32)
                lqn = lq[:, :W].cpu().numpy(); vn = valid.cpu().numpy(); cn = cand.cpu().numpy().astype(np.int64)
                Xf[:, :, 0] = np.maximum(np.where(vn, lqn, -30.0), -30.0)
                Xf[:, :, 1] = Xf[:, :, 0] - np.maximum(lq[:, :1].cpu().numpy(), -30.0)
                Xf[:, :, 2] = np.log(np.arange(1, W + 1))[None, :]
                Xf[:, :KC, 3] = 1.0
                Xf[:, :, 4] = q1.cpu().numpy()[:, None]; Xf[:, :, 5] = ent.cpu().numpy()[:, None]
                Xf[:, :, 6] = np.clip(marg.cpu().numpy(), -30, 30)[:, None]; Xf[:, :, 7] = tail.cpu().numpy()[:, None]
                hf = Hh.feats(s_q, r_q, t_q, np.where(vn, cn, -1))
                A, _ = BM.score(s_q, r_q, t_q, np.where(vn, cn, -1), hf["dyad_hi"])
                Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]
                Xf[:, :, 11] = hf["rec_dyad"]; Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"])
                Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"]); Xf[:, :, 16] = A
                Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
                Xf[~vn] = 0.0
                o_ = O[sp]
                o_["X"][sl] = Xf.astype(np.float16); o_["LP"][sl] = np.where(vn, lqn, -np.inf).astype(np.float32)
                o_["CA"][sl] = cn.astype(np.int32); o_["FI"][sl] = filt.cpu().numpy(); o_["BA"][sl] = ba.cpu().numpy()
                o_["rb"][sl] = rmid.cpu().numpy(); o_["rb_opt"][sl] = (1 + gt).cpu().numpy(); o_["gp"][sl] = gp.cpu().numpy()
                o_["npos"][sl] = (Sx > 0).sum(1).cpu().numpy()
                done[sp] += B
                del Sx, Sf, q, order, sc, qs
            if len(loc) and (done[sp] // 20000) != ((done[sp] - len(loc)) // 20000):
                log(sp, done[sp], "/", len(rows_req[sp]))
        xi.add(trip)
    summ = dict(ds=a.ds, W=W, feat_names=GEN_F + EVID_F, host=os.environ.get("HD_HOST", ""), device=dev)
    for sp in ("valid", "test"):
        o_ = O[sp]; n = len(rows_req[sp])
        assert done[sp] == n, (sp, done[sp], n)
        for k in ("X", "LP", "CA", "FI", "BA"): o_[k].flush()
        np.savez(f"{a.out}/{sp}_q.npz", rows=rows_req[sp], sro=sro_full[sp][rows_req[sp]], r_base=o_["rb"], r_base_opt=o_["rb_opt"],
                 gold_pos=o_["gp"], n_recb_support=o_["npos"])
        rb = o_["rb"]
        summ[sp] = dict(n=int(n), recb_MRR_avgties=float(np.mean(1.0 / rb)), recb_MRR_optimistic=float(np.mean(1.0 / o_["rb_opt"])),
                        gold_in_top50=float(((o_["gp"] >= 0) & (o_["gp"] < KC)).mean()), gold_in_pool=float((o_["gp"] >= 0).mean()),
                        median_recb_support=float(np.median(o_["npos"])))
        log(sp, json.dumps(summ[sp]))
    summ["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(summ, open(f"{a.out}/g1_features_summary.json", "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
