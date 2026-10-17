#!/usr/bin/env python3
"""v7 / v7_scan.py -- one pass over a split with the frozen base's exact score rows (TiRGN replay / float32 full rows), for
  A2 exact full-ranking oracle MRR of several pools (rank_tail = hd_blend.pool_rank r_out convention), and
  B1/B2 inputs: RecB-trained (clean) block RB = first <= 50 RecB > 0 candidates not in the stored base top-50 (descending RecB,
  ties by entity id), with exact base log p, raw base rank, babove, filt; RecB features for the stored PT pool and the RB pool.
No re-ranker is evaluated here.  Pools (unfiltered membership):
  T50 stored (CA[:, :50]), T50r recomputed topk(50) (agreement check), T100 recomputed, U50 = T50r | RecB-tr top-50 (RecB > 0),
  POOLSEL = hd_blend.pool_mask(lp, RecB[var], K) of the valid-selected control, PT = stored HRCOV_PAT pool (bridge masked), RB.
  python3 v7_scan.py --ds polecat --source tirgn --ckpt C --feat F --lists L --layout pc --split test --sel_K 100 --sel_var def --out O
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_blend as HB

KB = 50


def load_pt(feat, lists, layout, split, n_expect):
    pre = f"{feat}/{split}_" + ("A_" if layout == "sp" else "")
    Q = np.load(f"{feat}/{split}_q.npz")
    npb = (Q["n_pat_block"] if layout == "sp" else np.load(f"{lists}/{split}_q.npz")["n_pat_block"][:len(Q["rows"])]).astype(np.int64)
    CA = np.load(pre + "cand.npy", mmap_mode="r")
    rows = Q["rows"].astype(np.int64)
    assert len(rows) == CA.shape[0]
    return rows, CA, npb, Q["sro"].astype(np.int64)


def recb_feats(S, Psi, ids, rank_of):
    """S [B,N] RecB-tr scores, ids [B,W] (-1 pad) -> [B,W,5]: log q, log q - log q1, log RecB rank, has_psi, in RecB top-50."""
    B, W = ids.shape
    ssum = S.sum(1, keepdim=True).clamp(min=1e-30); q1 = (S.max(1, keepdim=True).values / ssum).clamp(min=1e-12)
    idc = ids.clamp(min=0)
    sv = torch.gather(S, 1, idc); qv = (sv / ssum)
    lq = torch.log(qv.clamp(min=1e-12)); lq = torch.where(sv > 0, lq, torch.full_like(lq, -30.0))
    rk = torch.gather(rank_of, 1, idc).float()
    rk = torch.where(sv > 0, rk, torch.full_like(rk, float(S.shape[1])))
    hp = (torch.gather(Psi, 1, idc) > 0).float()
    F = torch.stack([lq, lq - torch.log(q1), torch.log(rk), hp, ((rk <= 50) & (sv > 0)).float()], 2)
    return torch.where((ids >= 0).unsqueeze(2), F, torch.zeros_like(F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--source", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--fullrows", default=""); ap.add_argument("--tirgn_tree", default=""); ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--feat", required=True); ap.add_argument("--lists", default=""); ap.add_argument("--layout", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--recb_dir", required=True)
    ap.add_argument("--sel_K", type=int, required=True); ap.add_argument("--sel_var", required=True)
    ap.add_argument("--b_items", type=int, default=1); ap.add_argument("--valid_rows", default=""); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--out", required=True); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time(); log = HB.log_fn(t0); torch.set_num_threads(a.threads)
    cfg = HB.DSCFG[a.ds]; N, R = cfg["N"], cfg["R"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = {sp: HB.load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    Fi = HB.Filt(Q, N, R); xi = HB.XiCounts(N, R, dev)
    sro_full = HB.split_queries(Q[a.split], R, a.split)
    rows_req, CA, npb, sro_f = load_pt(a.feat, a.lists, a.layout, a.split, None)
    if rows_req.max() >= len(sro_full) or not (sro_full[rows_req] == sro_f).all():      # feature 'rows' are dump rows (polecat)
        if a.split == "test" and len(sro_f) == len(sro_full):
            rows_req = np.arange(len(sro_full))
        else:
            rows_req = np.load(a.valid_rows).astype(np.int64)
    assert (sro_full[rows_req] == sro_f).all(), "feature rows != split order"
    if a.smoke: rows_req = rows_req[:a.smoke]
    n = len(rows_req)
    req = np.zeros(len(sro_full), bool); req[rows_req] = True
    slot = -np.ones(len(req), np.int64); slot[rows_req] = np.arange(n)
    csr = HB.RecbCSR(f"{a.recb_dir}/recb_csr_{a.split}_{cfg['name']}.npz"); assert (csr.sro == sro_full).all()
    os.makedirs(a.out, exist_ok=True)
    P = ["T50", "T50r", "T100", "U50", "POOLSEL", "PT", "RB"]
    O = {f"gin_{p}": np.zeros(n, np.uint8) for p in P}; O.update({f"rtail_{p}": np.zeros(n, np.float32) for p in P})
    O.update(r_strict=np.zeros(n, np.float32), r_ge=np.zeros(n, np.float32), r_raw=np.zeros(n, np.float32), p1=np.zeros(n, np.float32),
             p2=np.zeros(n, np.float32), n_filt_T50=np.zeros(n, np.int16), t50_equal=np.zeros(n, np.uint8), n_rb=np.zeros(n, np.int16),
             gpos_RB=np.full(n, -1, np.int16), gold_lp=np.zeros(n, np.float32))
    if a.b_items:
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{a.out}/{a.split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        RBc = mm("rb_cand", (n, KB), np.int32); RBlp = mm("rb_logp", (n, KB), np.float32); RBrk = mm("rb_brank", (n, KB), np.float32)
        RBba = mm("rb_babove", (n, KB), np.bool_); RBfi = mm("rb_filt", (n, KB), np.bool_)
        FT = mm("recbF_T50", (n, 50, 5), np.float16); FP = mm("recbF_PTblock", (n, 50, 5), np.float16); FR = mm("recbF_RBblock", (n, KB, 5), np.float16)
    if a.source == "tirgn":
        import hd_tirgn_replay as TRP
        it = ((sp, ts, sro, fn, trip) for sp, ts, sro, fn, trip in TRP.source(cfg, a.ckpt, a.L, (a.split,), dev, a.tirgn_tree))
    elif a.source == "fullrows":
        vp, tp = a.fullrows.split(","); it = ((sp, ts, sro, fn, trip) for sp, ts, sro, fn, trip, _ in HB.fullrow_source(cfg, {"valid": vp, "test": tp}, (a.split,), dev))
        MMr = np.memmap({"valid": vp, "test": tp}[a.split], dtype=np.float32, mode="r").reshape(-1, N)
    pos = 0; done = 0; arN = None
    for sp, ts, sro, fn, trip in it:
        if sro is not None and sp == a.split:
            p0 = pos; nq = len(sro); pos += nq
            idx_all = np.arange(p0, p0 + nq); loc = np.flatnonzero(req[idx_all])
            for b0 in range(0, len(loc), a.bs):
                lb = loc[b0:b0 + a.bs]; rows = idx_all[lb]; sl = slot[rows]; B = len(lb)
                s_q, r_q, o_q = sro[lb, 0], sro[lb, 1], sro[lb, 2]
                with torch.no_grad():
                    lp = torch.as_tensor(np.asarray(MMr[rows]), device=dev) if a.source == "fullrows" else fn(s_q, r_q)
                    lp = lp.float()
                    gold = torch.as_tensor(o_q, device=dev); ar = torch.arange(B, device=dev)
                    fr_, fc_ = Fi.pairs(sro[lb]); fr = torch.as_tensor(fr_, device=dev); fc = torch.as_tensor(fc_, device=dev)
                    filt = torch.zeros((B, N), dtype=torch.bool, device=dev)
                    if len(fr): filt[fr, fc] = True
                    filt[ar, gold] = False                                               # filt = OTHER true answers only
                    filt_g = filt.clone(); filt_g[ar, gold] = True                       # other true answers + gold
                    X = xi.xi(torch.as_tensor(r_q, device=dev))
                    S = {v: HB.recb_row(csr, rows, r_q, X, dev, v) for v in set(("tr", a.sel_var))}
                    g_lp = lp[ar, gold]
                    gt = (lp > g_lp.unsqueeze(1)); ge = (lp >= g_lp.unsqueeze(1))
                    O["r_strict"][sl] = (1 + (gt & ~filt_g).sum(1)).float().cpu().numpy()
                    O["r_ge"][sl] = ((ge & ~filt_g).sum(1)).float().cpu().numpy() + 1     # ties counted pessimistically (gold excluded)
                    O["r_raw"][sl] = (1 + (gt & (torch.arange(N, device=dev)[None, :] != gold[:, None])).sum(1)).float().cpu().numpy()
                    t2 = torch.topk(lp, 2, dim=1).values.exp(); O["p1"][sl] = t2[:, 0].cpu().numpy(); O["p2"][sl] = t2[:, 1].cpu().numpy()
                    O["gold_lp"][sl] = g_lp.cpu().numpy()
                    ca = torch.as_tensor(np.asarray(CA[sl, :100]).astype(np.int64), device=dev)
                    jj = torch.arange(100, device=dev)[None, :]
                    nb_ = torch.as_tensor(npb[sl], device=dev)
                    ca_pt = torch.where((jj < 50) | (jj < 50 + nb_[:, None]), ca, torch.full_like(ca, -1))
                    def mask_ids(ids):
                        M = torch.zeros((B, N), dtype=torch.bool, device=dev); m = ids >= 0
                        M[ar[:, None].expand_as(ids)[m], ids[m]] = True; return M
                    t50s = ca[:, :50]; M_T50 = mask_ids(t50s)
                    M_T50r = torch.zeros((B, N), dtype=torch.bool, device=dev); M_T50r.scatter_(1, torch.topk(lp, 50, dim=1).indices, True)
                    M_T100 = torch.zeros((B, N), dtype=torch.bool, device=dev); M_T100.scatter_(1, torch.topk(lp, 100, dim=1).indices, True)
                    tv, ti = torch.topk(S["tr"], 50, dim=1); M_U = M_T50r.clone(); M2 = torch.zeros_like(M_U); M2.scatter_(1, ti, tv > 0); M_U |= M2
                    M_PS = HB.pool_mask(lp, S[a.sel_var], a.sel_K)
                    M_PT = mask_ids(ca_pt)
                    # RB block: RecB-tr > 0, not in stored T50, descending RecB, ties by entity id (stable sort)
                    Str = torch.where(M_T50 | (S["tr"] <= 0), torch.full_like(S["tr"], -1.0), S["tr"])
                    sv, si = torch.sort(Str, dim=1, descending=True, stable=True)
                    rb = torch.where(sv[:, :KB] > 0, si[:, :KB], torch.full_like(si[:, :KB], -1))
                    M_RB = M_T50 | mask_ids(rb)
                    pools = dict(T50=M_T50, T50r=M_T50r, T100=M_T100, U50=M_U, POOLSEL=M_PS, PT=M_PT, RB=M_RB)
                    for p, M in pools.items():
                        gin = M[ar, gold]; unf = M & ~filt_g
                        r_out = 1 + unf.sum(1) + ((~M) & ~filt_g & gt).sum(1)
                        O[f"gin_{p}"][sl] = gin.cpu().numpy().astype(np.uint8); O[f"rtail_{p}"][sl] = r_out.float().cpu().numpy()
                    O["n_filt_T50"][sl] = (M_T50 & filt).sum(1).cpu().numpy().astype(np.int16)
                    O["t50_equal"][sl] = (M_T50 == M_T50r).all(1).cpu().numpy().astype(np.uint8)
                    O["n_rb"][sl] = (rb >= 0).sum(1).cpu().numpy().astype(np.int16)
                    isg = (rb == gold[:, None]); O["gpos_RB"][sl] = torch.where(isg.any(1), isg.float().argmax(1), torch.full_like(gold, -1)).cpu().numpy().astype(np.int16)
                    if a.b_items:
                        rbc = rb.clamp(min=0); lrb = torch.gather(lp, 1, rbc)
                        srt = torch.sort(lp, dim=1).values                                   # ascending
                        brk = (N - torch.searchsorted(srt, lrb.contiguous(), right=True)).float() + 1   # 1 + #entities with lp > cand lp
                        vrb = rb >= 0
                        RBc[sl] = rb.cpu().numpy().astype(np.int32)
                        RBlp[sl] = torch.where(vrb, lrb, torch.full_like(lrb, float("-inf"))).cpu().numpy()
                        RBrk[sl] = torch.where(vrb, brk, torch.full_like(brk, float(N))).cpu().numpy()
                        RBba[sl] = (vrb & (lrb > g_lp.unsqueeze(1))).cpu().numpy(); RBfi[sl] = (vrb & torch.gather(filt, 1, rbc) & (rb != gold[:, None])).cpu().numpy()
                        # RecB rank of every entity under the tr row (1-based, descending, ties by id)
                        sv2, si2 = torch.sort(S["tr"], dim=1, descending=True, stable=True)
                        rank_of = torch.empty_like(si2); rank_of.scatter_(1, si2, torch.arange(1, N + 1, device=dev).unsqueeze(0).expand(B, N))
                        Psi, _ = csr.rows(rows, B, N, dev, "tr", r_q)
                        FT[sl] = recb_feats(S["tr"], Psi, t50s, rank_of).cpu().numpy().astype(np.float16)
                        FP[sl] = recb_feats(S["tr"], Psi, ca_pt[:, 50:], rank_of).cpu().numpy().astype(np.float16)
                        FR[sl] = recb_feats(S["tr"], Psi, rb, rank_of).cpu().numpy().astype(np.float16)
                    del lp, filt, filt_g, gt, ge, pools, S, X
                done += B
            if done and (done // a.bs) % 200 == 0: log(a.split, "done", done, "/", n)
        if sp == a.split or sp in ("train", "valid", "test"):
            xi.add(trip)
    assert done == n, (done, n)
    np.savez(f"{a.out}/{a.split}_scan.npz", rows=rows_req, **O)
    if a.b_items:
        for z in (RBc, RBlp, RBrk, RBba, RBfi, FT, FP, FR): z.flush()
    summ = {p: dict(recall=float(O[f"gin_{p}"].mean()),
                    oracle_MRR=float(np.mean(np.where(O[f"gin_{p}"] > 0, 1.0, 1.0 / O[f"rtail_{p}"].astype(np.float64))))) for p in P}
    summ["base_MRR_strict"] = float(np.mean(1.0 / O["r_strict"])); summ["t50_stored_equal_recomputed"] = float(O["t50_equal"].mean())
    summ["mean_n_rb"] = float(O["n_rb"].mean()); summ["elapsed_s"] = round(time.time() - t0, 1); summ["n"] = int(n)
    json.dump(summ, open(f"{a.out}/{a.split}_scan_summary.json", "w"), indent=1)
    log("DONE", json.dumps({k: (round(v["oracle_MRR"], 5) if isinstance(v, dict) else v) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
