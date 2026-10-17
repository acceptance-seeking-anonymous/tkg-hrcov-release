#!/usr/bin/env python3
"""pilot_icews / pi_features.py -- re-ranker feature tensors for one base (CPU), both relation schemes.

Candidates per query (scheme s): C_cov_s [W=200] from pi_lists (columns 0-49 = base top-50 = the HR pool).
Per (query, candidate) 25 features (pilot-1 pc_features.py minus the 5 bridge features):
  0-7   base : log p (EXACT: dump top-200 where visible, else the pi_replay lookup), log p - log p1, log raw base rank,
               in-top-50 flag, p1, top-50 entropy, margin12, log tail mass
  8-18  evid : residual-track online evidence (rs_features.Hist / DyadReader; FULL history, facts with ts < t);
               scheme rn: Hist / DyadReader / query relation on relation-normalised ids
  19-24 cov  : pi_lists pattern coverage features (COV_FEATS)
Also LP (base log-prob for the fused score), CA, FI (other true answers; ORIGINAL relation ids), BA (exact base log-prob >
gold) and per-query arrays (exact base rank from the replay, strata).  Output: work/feat_<base>_<scheme>/{valid,test}_*.npy
  python3 pi_features.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import (DUMPS, DS, WORK, P, KC, W_MAX, FEAT_NAMES, Splits, SplitsView, relmap2, metrics_from_ranks, jdump, log_fn,
                       exact_fused_ranks_chunked)
from rs_features import Hist, DyadReader

GROUPS = dict(base=list(range(0, 8)), evid=list(range(8, 19)), cov=list(range(19, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--schemes", default="orig,rn")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N, R = S.n_ent, S.R; F = S.filter_index()
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    Dd = np.load(DUMPS[a.base])
    summ = dict(base=a.base, feat_names=FEAT_NAMES, groups=GROUPS, host="serverA (CPU, tkg-gpu:latest)")
    ld = f"{WORK}/lists_{a.base}"; lk = f"{WORK}/lookup_{a.base}"
    for sch in a.schemes.split(","):
        rmap = np.arange(2 * R) if sch == "orig" else relmap2()
        Sv = S if sch == "orig" else SplitsView(S, rmap)
        Hh = Hist(Sv); BM = DyadReader(Sv, Hh); log(sch, "hist + dyad reader", BM.n_transitions)
        out_dir = f"{WORK}/feat_{a.base}_{sch}"; os.makedirs(out_dir, exist_ok=True)
        for split in ("valid", "test"):
            t1 = time.time()
            sro = Dd[f"{split}_sro"].astype(np.int64); n = len(sro)
            LK = np.load(f"{lk}/{split}.npz")
            r_base = LK["rank_filt"].astype(np.int64); gs = LK["gold_lp"].astype(np.float64)
            rank_agree = float((r_base == Dd[f"{split}_rank_filt"]).mean())
            assert rank_agree > 0.99, f"lookup rank != dump rank ({rank_agree})"
            Ccov = np.load(f"{ld}/{split}_{sch}_Ccov.npy", mmap_mode="r"); Xcov = np.load(f"{ld}/{split}_{sch}_Xcov.npy", mmap_mode="r")
            # requested ids in the lookup = [orig C_cov 50..199 | rn C_cov 50..199]
            off = 0 if sch == "orig" else W_MAX - KC
            mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
            X_ = mm("X", (n, W_MAX, len(FEAT_NAMES)), np.float16); LP_ = mm("logp", (n, W_MAX), np.float32)
            CA_ = mm("cand", (n, W_MAX), np.int32); FI_ = mm("filt", (n, W_MAX), np.bool_); BA_ = mm("babove", (n, W_MAX), np.bool_)
            gpos_all = np.full(n, -1, np.int16); p1_all = np.zeros(n, np.float32)
            TKI_ = Dd[f"{split}_topk_ids"]; TKS_ = Dd[f"{split}_topk_scores"]          # loaded once per split (npz members re-read on every access)
            LPR_ = LK["lp_req"]; RKR_ = LK["rk_req"]
            n_miss = 0
            for lo in range(0, n, a.chunk):
                hi = min(n, lo + a.chunk); b = np.arange(lo, hi); nb = len(b)
                s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
                tki = TKI_[b].astype(np.int64); tks = TKS_[b].astype(np.float64)
                K = tki.shape[1]
                cand = np.asarray(Ccov[b]).astype(np.int64); valid = cand >= 0
                assert (cand[:, :KC] == tki[:, :KC]).all()
                lp = np.full((nb, W_MAX), -np.inf); brank = np.full((nb, W_MAX), float(N))
                lp[:, :KC] = tks[:, :KC]; brank[:, :KC] = np.arange(1, KC + 1)[None, :]
                lpr = LPR_[b][:, off:off + W_MAX - KC].astype(np.float64); rkr = RKR_[b][:, off:off + W_MAX - KC].astype(np.float64)
                rq_ids = cand[:, KC:]
                ok = rq_ids >= 0
                # columns >= 50: exact values from the same replay that produced gold_lp / rank_filt (pi_replay lookup)
                lp[:, KC:] = np.where(ok, lpr, -np.inf); brank[:, KC:] = np.where(ok, rkr, N)
                ba = valid & (lp > gs[b][:, None]); ba[:, :KC] = False
                n_miss += int((valid[:, KC:] & (lpr < -1e29)).sum())
                fk = S.fkey(s, r, t)
                code = fk[:, None] * N + np.where(valid, cand, 0)
                pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
                istrue = (allTrue[pos] == code) & valid
                isgold = (cand == o[:, None]) & valid
                filt = istrue & ~isgold
                gp = np.where(isgold.any(1), isgold.argmax(1), -1); gpos_all[b] = gp
                rr = rmap[r]
                hf = Hh.feats(s, rr, t, np.where(valid, cand, -1))
                A, mev = BM.score(s, rr, t, np.where(valid, cand, -1), hf["dyad_hi"])
                p_top = np.exp(tks[:, :KC]); pn = p_top / p_top.sum(1, keepdims=True); ent50 = -(pn * np.log(pn + 1e-12)).sum(1)
                tail_mass = np.clip(1.0 - np.exp(tks).sum(1), 1e-12, 1.0)
                p1_all[b] = np.exp(tks[:, 0])
                Xf = np.zeros((nb, W_MAX, len(FEAT_NAMES)), np.float32)
                lpf = np.where(valid, lp, -30.0)
                Xf[:, :, 0] = np.maximum(lpf, -30.0); Xf[:, :, 1] = Xf[:, :, 0] - tks[:, :1]; Xf[:, :, 2] = np.log(np.maximum(brank, 1))
                Xf[:, :KC, 3] = 1.0
                Xf[:, :, 4] = np.exp(tks[:, :1]); Xf[:, :, 5] = ent50[:, None]; Xf[:, :, 6] = (tks[:, 0] - tks[:, 1])[:, None]
                Xf[:, :, 7] = np.log(tail_mass)[:, None]
                Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]
                Xf[:, :, 11] = hf["rec_dyad"]; Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"])
                Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"]); Xf[:, :, 16] = A
                Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
                Xf[:, :, 19:25] = np.asarray(Xcov[b]).astype(np.float32)
                Xf[~valid] = 0.0
                X_[lo:hi] = Xf.astype(np.float16); LP_[lo:hi] = np.where(valid, lp, -np.inf).astype(np.float32)
                CA_[lo:hi] = cand.astype(np.int32); FI_[lo:hi] = filt; BA_[lo:hi] = ba
                if (lo // a.chunk) % 5 == 0:
                    log(sch, split, f"{hi}/{n}")
            for z in (X_, LP_, CA_, FI_, BA_):
                z.flush()
            st = dict(np.load(f"{WORK}/strata_{split}.npz"))
            assert (st["sro"] == sro).all(), "strata row order"
            q = dict(sro=sro, r_base=r_base, gold_pos=gpos_all, p1=p1_all, gs=gs, in_top50=r_base <= KC, tie_n=LK["tie_n"])
            for k in st:
                if k != "sro":
                    q[k] = st[k]
            Lq = np.load(f"{ld}/{split}_{sch}_q.npz")
            for k in Lq.files:
                q[f"lists_{k}"] = Lq[k]
            np.savez(f"{out_dir}/{split}_q.npz", **q)
            rk0 = exact_fused_ranks_chunked(np.load(f"{out_dir}/{split}_logp.npy", mmap_mode="r"), np.load(f"{out_dir}/{split}_cand.npy", mmap_mode="r"),
                                            np.load(f"{out_dir}/{split}_filt.npy", mmap_mode="r"), np.load(f"{out_dir}/{split}_babove.npy", mmap_mode="r"),
                                            gpos_all.astype(np.int64), r_base)
            inC = gpos_all >= 0; intop = (gpos_all >= 0) & (gpos_all < KC)
            summ[f"{split}_{sch}"] = dict(n=int(n), seconds=round(time.time() - t1, 1), n_lookup_miss=int(n_miss), rank_agree_lookup_vs_dump=rank_agree,
                                         identity_f0=float(np.mean(rk0 == r_base)), gold_in_top50=float(intop.mean()), gold_in_Ccov=float(inC.mean()))
            log(sch, split, json.dumps(summ[f"{split}_{sch}"]))
        jdump(dict(feat_names=FEAT_NAMES, groups=GROUPS, KC=KC, W=W_MAX, base=a.base, scheme=sch), f"{out_dir}/meta.json")
        os.system(f"chmod -R a+rX {out_dir}")
        del Hh, BM
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"feat_summary_{a.base}.json"); jdump(summ, f"{WORK}/feat_summary_{a.base}.json")
    log("ALL DONE")


if __name__ == "__main__":
    main()
