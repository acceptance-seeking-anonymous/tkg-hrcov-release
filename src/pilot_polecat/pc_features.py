#!/usr/bin/env python3
"""pilot_polecat / pc_features.py -- re-ranker feature tensors for one base (CPU).

Candidate set per query: C_cov [W=150] from pc_lists (columns 0-49 = base top-50 = the history re-ranker
pool C_hr).  Per (query, candidate) 30 features:
  0-7   base  : log p (EXACT, full-row lookup for candidates outside the dump top-200), log p - log p1,
                log base rank (exact), in-top-50 flag, p1, top-50 entropy, margin12, log tail mass
  8-18  evid  : residual-track online evidence (rs_features.Hist / DyadReader, facts with ts < t)
  19-29 cov   : pc_lists coverage features (COV_FEATS)
Also LP (fused-score base log-prob), CA, FI (other true answers), BA (exact base log-prob > gold) and per-query
arrays incl. the base-independent strata from pc_prep.  Output: work/feat_<base>/{valid,test}_*.npy, _q.npz
  python3 pc_features.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pc_common import (TRACKS, DS, DUMPS, WORK, KC, W_COV, LP_LEN, LB_LEN, COV_FEATS, Splits, BaseDump, split_queries,
                       metrics_from_ranks, jdump, log_fn, exact_fused_ranks)
sys.path.insert(0, f"{TRACKS}/residual/code")
from rs_features import Hist, DyadReader

BASE_F = ["logp", "logp_minus_top1", "log_base_rank", "is_topKc", "q_p1", "q_entropy50", "q_margin12", "q_log_tailmass"]
EVID_F = ["ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
          "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj"]
FEAT_NAMES = BASE_F + EVID_F + COV_FEATS
GROUPS = dict(base=list(range(0, 8)), evid=list(range(8, 19)), cov=list(range(19, 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N = S.n_ent; F = S.filter_index()
    D = BaseDump(DUMPS[a.base], S); log("dump", D.N, D.K)
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad reader", BM.n_transitions)
    out_dir = f"{WORK}/feat_{a.base}" + ("_smoke" if a.smoke else ""); os.makedirs(out_dir, exist_ok=True)
    sfx = "_smoke" if a.smoke else ""
    ld = f"{WORK}/lists_{a.base}{sfx}"; lk = f"{WORK}/lookup_{a.base}{sfx}"
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    summ = dict(base=a.base, feat_names=FEAT_NAMES, groups=GROUPS, host="serverA (CPU, tkg-gpu:latest)")
    for split in ("valid", "test"):
        t1 = time.time()
        Lq = np.load(f"{ld}/{split}_q.npz"); rows = Lq["rows"]
        LK = np.load(f"{lk}/{split}.npz")
        Ccov = np.load(f"{ld}/{split}_Ccov.npy", mmap_mode="r"); Xcov = np.load(f"{ld}/{split}_Xcov.npy", mmap_mode="r")
        Lp = np.load(f"{ld}/{split}_Lpat.npy", mmap_mode="r"); Lb = np.load(f"{ld}/{split}_Lbr.npy", mmap_mode="r")
        st = dict(np.load(f"{WORK}/strata_{split}.npz"))   # smoke: prep must have been run with the same --smoke
        n = len(rows) if not a.smoke else min(a.smoke, len(rows))
        sro = D.sro[rows[:n]].astype(np.int64)
        assert (sro == st["sro"][:n]).all(), "strata row order mismatch"
        gs = D.gold_score[rows[:n]].astype(np.float64); r_base = D.extra["gold_rank_filt"][rows[:n]].astype(np.int64)
        rank_agree = float((r_base == LK["rank_filt"][:n]).mean())
        assert a.smoke or rank_agree > float(os.environ.get("PC_RANK_AGREE_MIN", "0.99")), f"lookup rank != dump rank ({rank_agree})"   # GPU-nondeterministic near-ties: ~0.2-0.3% differ by <=1-2
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        X_ = mm("X", (n, W_COV, len(FEAT_NAMES)), np.float16); LP_ = mm("logp", (n, W_COV), np.float32)
        CA_ = mm("cand", (n, W_COV), np.int32); FI_ = mm("filt", (n, W_COV), np.bool_); BA_ = mm("babove", (n, W_COV), np.bool_)
        gpos_all = np.full(n, -1, np.int16); p1_all = np.zeros(n, np.float32)
        n_lk_miss = 0
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk); b = np.arange(lo, hi); nb = len(b)
            s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
            tki = D.topk_ids[rows[b]].astype(np.int64); tks = D.topk_scores[rows[b]].astype(np.float64)
            K = tki.shape[1]
            cand = np.asarray(Ccov[b]).astype(np.int64); valid = cand >= 0
            assert (cand[:, :KC] == tki[:, :KC]).all()
            # base log-prob + raw rank: dump top-K where visible, else exact full-row lookup
            lp = np.full((nb, W_COV), -np.inf); brank = np.zeros((nb, W_COV))
            lp[:, :KC] = tks[:, :KC]; brank[:, :KC] = np.arange(1, KC + 1)[None, :]
            req = np.concatenate([np.asarray(Lp[b]), np.asarray(Lb[b])], 1).astype(np.int64)
            lpr = LK["lp_req"][b].astype(np.float64); rkr = LK["rk_req"][b].astype(np.float64)
            glk = LK["gold_lp"][b].astype(np.float64)
            ba = np.zeros((nb, W_COV), bool)
            for i in range(nb):
                vis = {int(e): k for k, e in enumerate(tki[i])}
                rq = {int(e): k for k, e in enumerate(req[i]) if e >= 0}
                for j in range(KC, W_COV):
                    e = int(cand[i, j])
                    if e < 0: break
                    kv = vis.get(e)
                    if kv is not None:
                        lp[i, j] = tks[i, kv]; brank[i, j] = kv + 1; ba[i, j] = tks[i, kv] > gs[b[i]]
                    else:
                        kq = rq.get(e)
                        if kq is None:
                            n_lk_miss += 1; lp[i, j] = -1e9; brank[i, j] = N; continue
                        lp[i, j] = lpr[i, kq]; brank[i, j] = rkr[i, kq]; ba[i, j] = lpr[i, kq] > glk[i]
            fk = S.fkey(s, r, t)
            code = fk[:, None] * N + np.where(valid, cand, 0)
            pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
            istrue = (allTrue[pos] == code) & valid
            isgold = (cand == o[:, None]) & valid
            filt = istrue & ~isgold
            gp = np.where(isgold.any(1), isgold.argmax(1), -1); gpos_all[b] = gp
            hf = Hh.feats(s, r, t, np.where(valid, cand, -1))
            A, mev = BM.score(s, r, t, np.where(valid, cand, -1), hf["dyad_hi"])
            p_top = np.exp(tks[:, :KC]); pn = p_top / p_top.sum(1, keepdims=True); ent50 = -(pn * np.log(pn + 1e-12)).sum(1)
            tail_mass = np.clip(1.0 - np.exp(tks).sum(1), 1e-12, 1.0)
            p1_all[b] = np.exp(tks[:, 0])
            Xf = np.zeros((nb, W_COV, len(FEAT_NAMES)), np.float32)
            lpf = np.where(valid, lp, -30.0)
            Xf[:, :, 0] = np.maximum(lpf, -30.0); Xf[:, :, 1] = Xf[:, :, 0] - tks[:, :1]; Xf[:, :, 2] = np.log(np.maximum(brank, 1))
            Xf[:, :KC, 3] = 1.0
            Xf[:, :, 4] = np.exp(tks[:, :1]); Xf[:, :, 5] = ent50[:, None]; Xf[:, :, 6] = (tks[:, 0] - tks[:, 1])[:, None]
            Xf[:, :, 7] = np.log(tail_mass)[:, None]
            Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]
            Xf[:, :, 11] = hf["rec_dyad"]; Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"])
            Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"]); Xf[:, :, 16] = A
            Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
            Xf[:, :, 19:30] = np.asarray(Xcov[b]).astype(np.float32)
            Xf[~valid] = 0.0
            X_[lo:hi] = Xf.astype(np.float16); LP_[lo:hi] = np.where(valid, lp, -np.inf).astype(np.float32)
            CA_[lo:hi] = cand.astype(np.int32); FI_[lo:hi] = filt; BA_[lo:hi] = ba
            log(split, f"{hi}/{n}")
        for z in (X_, LP_, CA_, FI_, BA_):
            z.flush()
        freq = S.freq[sro[:, 2]]
        stratum = S.stratum(freq); in_hist = S.in_hist(sro[:, 0], sro[:, 2])
        in_top50 = r_base <= KC
        q = dict(rows=rows[:n], sro=sro, r_base=r_base, gold_pos=gpos_all, p1=p1_all, stratum=stratum, in_hist=in_hist,
                 in_top50=in_top50, gs=gs)
        for k in ("nev_s", "nev_o", "dyad", "trip", "pat_any", "pat_hop1", "two_hop", "self_loop", "first_seen", "nb_s"):
            q[k] = st[k][:n]
        for k in ("gpos_pat", "gpos_br", "g_in_union", "g_in_bridge", "br_size", "n_union"):
            q[k] = Lq[k][:n]
        np.savez(f"{out_dir}/{split}_q.npz", **q)
        rk0 = exact_fused_ranks(np.asarray(LP_), np.asarray(CA_), np.asarray(FI_), np.asarray(BA_), gpos_all.astype(np.int64), r_base)
        rb = r_base.astype(np.float64)
        inC = gpos_all >= 0; intop = (gpos_all >= 0) & (gpos_all < KC)
        summ[split] = dict(n=int(n), seconds=round(time.time() - t1, 1), n_lookup_miss=int(n_lk_miss), rank_agree_lookup_vs_dump=rank_agree,
                           identity_f0=float(np.mean(rk0 == r_base)), base=metrics_from_ranks(rb),
                           oracle=dict(top50_at1=metrics_from_ranks(np.where(intop, 1, rb)),
                                       Ccov_at1=metrics_from_ranks(np.where(inC, 1, rb)),
                                       cov_only_at1=metrics_from_ranks(np.where(inC & ~intop, 1, rb))),
                           gold_in_top50=float(intop.mean()), gold_in_Ccov=float(inC.mean()))
        log(split, json.dumps({k: summ[split][k] for k in ("identity_f0", "gold_in_top50", "gold_in_Ccov", "n_lookup_miss")}))
    meta = dict(feat_names=FEAT_NAMES, groups=GROUPS, KC=KC, W=W_COV, base=a.base)
    jdump(meta, f"{out_dir}/meta.json")
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"feat_summary_{a.base}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
