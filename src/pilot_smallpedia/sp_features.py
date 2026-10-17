#!/usr/bin/env python3
"""pilot_smallpedia / sp_features.py -- re-ranker feature tensors for one base and both candidate layouts (CPU).

Per (query, candidate) 42 features = pilot-1's 30 (8 base, 11 online evidence, 11 coverage) + 12 static (ST_FEATS).
Base log-probs, raw ranks, gold filtered rank and 'base log-prob > gold' are EXACT from the full log-prob rows.
Layout A [250] = C_cov | static | degree-matched random;  layout B [150] = top-50 | static.
Output: work/feat_<base>/{split}_{A,B}_{X,logp,cand,filt,babove}.npy + {split}_q.npz
  python3 sp_features.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import (TRACKS, DS, DUMPS, WORK, KC, W_A, W_B, COV_FEATS, ST_FEATS, N_ENT, Splits, BaseDump, split_queries,
                       metrics_from_ranks, jdump, log_fn, exact_fused_ranks, fullrows, MODE_SFX, ST_MODE)
sys.path.insert(0, f"{TRACKS}/residual/code")
from rs_features import Hist, DyadReader

BASE_F = ["logp", "logp_minus_top1", "log_base_rank", "is_topKc", "q_p1", "q_entropy50", "q_margin12", "q_log_tailmass"]
EVID_F = ["ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
          "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj"]
FEAT_NAMES = BASE_F + EVID_F + COV_FEATS + ST_FEATS
GROUPS = dict(base=list(range(0, 8)), evid=list(range(8, 19)), cov=list(range(19, 30)), st=list(range(30, 42)))
STRATA_KEYS = ("nev_s", "nev_o", "dyad", "trip", "pat_any", "pat_hop1", "two_hop", "self_loop", "first_seen", "nb_s",
               "st_S", "st_SS", "st_SSc", "st_MX", "st_enum", "st_enum_size", "st_enum_dm", "st_enum_pop", "union2", "st_deg_s", "st_deg_o",
               "st_enum_S", "st_enum_SS", "st_enum_ST", "st_enum_TS")
LIST_Q = ("gpos_pat", "gpos_br", "gpos_st", "g_in_union", "g_in_bridge", "g_in_static", "br_size", "n_union", "st_size",
          "n_pat_block", "n_br_block", "n_st_block", "n_dm_block", "n_stB_block", "g_st_code", "g_st_ntypes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N = S.n_ent; F = S.filter_index()
    D = BaseDump(DUMPS[a.base], S); log("dump", D.N, D.K)
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad reader", BM.n_transitions)
    sfx = ("_smoke" if a.smoke else "") + MODE_SFX
    out_dir = f"{WORK}/feat_{a.base}{sfx}"; os.makedirs(out_dir, exist_ok=True)
    ld = f"{WORK}/lists_{a.base}{sfx}"
    summ = dict(base=a.base, feat_names=FEAT_NAMES, groups=GROUPS, host="serverA (CPU, tkg-gpu:latest)", static_mode=ST_MODE)
    for split in ("valid", "test"):
        t1 = time.time()
        Lq = np.load(f"{ld}/{split}_q.npz"); rows = Lq["rows"]
        st = dict(np.load(f"{WORK}/strata_{split}.npz"))
        n = len(rows) if not a.smoke else min(a.smoke, len(rows))
        sro_all = split_queries(S, split)
        sro = sro_all[rows[:n]]
        assert (sro == st["sro"][:n]).all(), "strata row order mismatch"
        LP = fullrows(a.base, split)
        drow = D.rows(split)[rows[:n]]
        lay = {"A": (np.load(f"{ld}/{split}_CA.npy", mmap_mode="r"), np.load(f"{ld}/{split}_XAc.npy", mmap_mode="r"),
                     np.load(f"{ld}/{split}_XAs.npy", mmap_mode="r"), W_A),
               "B": (np.load(f"{ld}/{split}_CB.npy", mmap_mode="r"), np.load(f"{ld}/{split}_XBc.npy", mmap_mode="r"),
                     np.load(f"{ld}/{split}_XBs.npy", mmap_mode="r"), W_B)}
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        OUT = {L: dict(X=mm(f"{L}_X", (n, W, len(FEAT_NAMES)), np.float16), LP=mm(f"{L}_logp", (n, W), np.float32),
                       CA=mm(f"{L}_cand", (n, W), np.int32), FI=mm(f"{L}_filt", (n, W), np.bool_), BA=mm(f"{L}_babove", (n, W), np.bool_))
               for L, (_, _, _, W) in lay.items()}
        r_base = np.zeros(n, np.int64); gs = np.zeros(n, np.float64); p1_all = np.zeros(n, np.float32)
        gpos = {L: np.full(n, -1, np.int16) for L in lay}
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk); b = np.arange(lo, hi); nb = len(b)
            s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
            row = np.asarray(LP[rows[b]], np.float32)
            glp = row[np.arange(nb), o]
            fk = S.fkey(s, r, t)
            # exact filtered gold rank from the full row (other true answers at (s, r', t) excluded)
            nab = (row > glp[:, None]).sum(1)
            for i in range(nb):
                j_ = np.searchsorted(F["keys"], fk[i])
                if j_ < len(F["keys"]) and F["keys"][j_] == fk[i]:
                    oth = F["objs"][F["start"][j_]:F["end"][j_]]; oth = oth[oth != o[i]]
                    nab[i] -= int((row[i, oth] > glp[i]).sum())
            r_base[b] = nab + 1; gs[b] = glp
            srow = np.sort(row, axis=1)
            top200 = -np.sort(-np.partition(row, N - 200, axis=1)[:, N - 200:], axis=1)
            p1 = np.exp(top200[:, 0].astype(np.float64)); p1_all[b] = p1
            pt = np.exp(top200[:, :KC].astype(np.float64)); pn = pt / pt.sum(1, keepdims=True); ent50 = -(pn * np.log(pn + 1e-12)).sum(1)
            tail_mass = np.clip(1.0 - np.exp(top200.astype(np.float64)).sum(1), 1e-12, 1.0)
            for L, (Cm, Xc, Xs, W) in lay.items():
                cand = np.asarray(Cm[rows[b]]).astype(np.int64); valid = cand >= 0
                cz = np.maximum(cand, 0)
                lp = row[np.arange(nb)[:, None], cz].astype(np.float64)
                brank = np.empty((nb, W))
                for i in range(nb):
                    brank[i] = N - np.searchsorted(srow[i], lp[i], side="right") + 1
                ba = lp > glp[:, None]
                code = fk[:, None] * N + cz
                pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
                istrue = (allTrue[pos] == code) & valid
                isgold = (cand == o[:, None]) & valid
                filt = istrue & ~isgold
                gp = np.where(isgold.any(1), isgold.argmax(1), -1); gpos[L][b] = gp
                hf = Hh.feats(s, r, t, np.where(valid, cand, -1))
                A_, _ = BM.score(s, r, t, np.where(valid, cand, -1), hf["dyad_hi"])
                Xf = np.zeros((nb, W, len(FEAT_NAMES)), np.float32)
                lpf = np.where(valid, lp, -30.0)
                Xf[:, :, 0] = np.maximum(lpf, -30.0); Xf[:, :, 1] = Xf[:, :, 0] - top200[:, :1]; Xf[:, :, 2] = np.log(np.maximum(brank, 1))
                Xf[:, :KC, 3] = 1.0
                Xf[:, :, 4] = p1[:, None]; Xf[:, :, 5] = ent50[:, None]; Xf[:, :, 6] = (top200[:, 0] - top200[:, 1])[:, None]
                Xf[:, :, 7] = np.log(tail_mass)[:, None]
                Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]
                Xf[:, :, 11] = hf["rec_dyad"]; Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"])
                Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"]); Xf[:, :, 16] = A_
                Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
                Xf[:, :, 19:30] = np.asarray(Xc[rows[b]]).astype(np.float32)
                Xf[:, :, 30:42] = np.asarray(Xs[rows[b]]).astype(np.float32)
                Xf[~valid] = 0.0
                O = OUT[L]
                O["X"][lo:hi] = Xf.astype(np.float16); O["LP"][lo:hi] = np.where(valid, lp, -np.inf).astype(np.float32)
                O["CA"][lo:hi] = cand.astype(np.int32); O["FI"][lo:hi] = filt; O["BA"][lo:hi] = ba
            log(split, f"{hi}/{n}")
        for O in OUT.values():
            for z in O.values(): z.flush()
        freq = S.freq[sro[:, 2]]
        q = dict(rows=rows[:n], sro=sro, r_base=r_base, gold_pos_A=gpos["A"], gold_pos_B=gpos["B"], p1=p1_all, stratum=S.stratum(freq),
                 in_hist=S.in_hist(sro[:, 0], sro[:, 2]), in_top50=r_base <= KC, gs=gs, gold_train_freq=freq,
                 r_dump=D.extra["gold_rank_filt"][drow].astype(np.int64))
        for k in STRATA_KEYS:
            q[k] = st[k][:n]
        for k in LIST_Q:
            q[k] = Lq[k][:n]
        np.savez(f"{out_dir}/{split}_q.npz", **q)
        e = dict(n=int(n), seconds=round(time.time() - t1, 1), rank_agree_fullrow_vs_dump=float((r_base == q["r_dump"]).mean()),
                 max_abs_rank_diff_fullrow_vs_dump=int(np.abs(r_base - q["r_dump"]).max()), base=metrics_from_ranks(r_base.astype(np.float64)),
                 base_dump=metrics_from_ranks(q["r_dump"].astype(np.float64)))
        for L in lay:
            O = OUT[L]
            rk0 = exact_fused_ranks(np.asarray(O["LP"]), np.asarray(O["CA"]), np.asarray(O["FI"]), np.asarray(O["BA"]), gpos[L].astype(np.int64), r_base)
            e[f"identity_f0_{L}"] = float(np.mean(rk0 == r_base)); e[f"gold_in_{L}"] = float((gpos[L] >= 0).mean())
        e["gold_in_top50"] = float((r_base <= KC).mean())
        summ[split] = e
        log(split, json.dumps(e))
    jdump(dict(feat_names=FEAT_NAMES, groups=GROUPS, KC=KC, W_A=W_A, W_B=W_B, base=a.base), f"{out_dir}/meta.json")
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"feat_summary_{a.base}{sfx}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
