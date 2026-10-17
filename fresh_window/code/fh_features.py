#!/usr/bin/env python3
"""phase2_fresh_window / fh_features.py (derived from presub/g2_features.py; changes: dataset via fh_common, --splits so
that Stage 1 touches valid only, verification without a residual-track reference) -- re-ranker feature tensors, W = 100 (CPU).

pilot_smallpedia/sp_features.py restricted to one layout and 25 features = 8 base (exact, from the full rows) + 11 online
evidence (residual-track Hist / DyadReader) + 6 pattern coverage (g2_lists COV6).  HR uses columns 0-18 of the first
50 candidates, HRCOV_PAT all 25 columns of all 100 (g2_train.py).  Output in pc_train layout:
  work/feat_<base>{_smoke}/{valid,test}_{X,logp,cand,filt,babove}.npy + {split}_q.npz (+ feat_summary_<base>.json)
Pre-registered verification of the full rows (summary 'verify'): exact filtered gold rank from the full row vs the
shared-harness dump's gold_rank_filt on the same rows (>= 99.9% equal required) and the full-row test MRR vs the
residual-track MRR of that base.
  python3 g2_features.py --base tirgn [--smoke n]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fh_common import (DS, DUMPS, BaseDump, WORK, KC, W_PAT, COV6, N_ENT, Splits, split_queries, metrics_from_ranks, jdump,
                       log_fn, exact_fused_ranks, fullrows, module_provenance)
from rs_features import Hist, DyadReader

BASE_F = ["logp", "logp_minus_top1", "log_base_rank", "is_topKc", "q_p1", "q_entropy50", "q_margin12", "q_log_tailmass"]
EVID_F = ["ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
          "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj"]
FEAT_NAMES = BASE_F + EVID_F + COV6
GROUPS = dict(base=list(range(0, 8)), evid=list(range(8, 19)), cov=list(range(19, 25)))
STRATA_KEYS = ("nev_s", "nev_o", "dyad", "trip", "pat_any", "pat_hop1", "two_hop", "self_loop", "first_seen", "nb_s")
LIST_Q = ("gpos_pat", "g_in_union", "has_pat", "n_pat_block")


def dump_ranks(base, split, S, rows, F):
    """shared-harness dump filtered gold rank of the split rows `rows` (split order) -> (rank, exact_flag, label).
    RE-GCN / LogCL dumps carry the exact gold_rank_filt; TiRGN dumps do not, so their rank is the harness's own
    top-K reconstruction (BaseDump.filtered_ranks: conservative rank, exact flag where the reconstruction is exact)."""
    key = f"{base}_valid" if (split == "valid" and f"{base}_valid" in DUMPS) else base
    d = np.load(DUMPS[key])
    code = {"valid": 1, "test": 2}[split]
    sp = d["split"] if "split" in d.files else S.split_of_ts(d["sro"][:, 3])
    idx = np.flatnonzero(sp == code)
    ds_ = d["sro"][idx].astype(np.int64); qs = split_queries(S, split)
    assert len(ds_) == len(qs), f"dump {key}: {len(ds_)} {split} rows vs {len(qs)}"
    lab = key
    if (ds_ == qs).all():
        perm = idx
    else:                                      # align identical (s, r', o, ts) tuples (duplicates share the gold rank)
        od = np.lexsort(ds_.T[::-1]); oq = np.lexsort(qs.T[::-1])
        assert (ds_[od] == qs[oq]).all(), f"dump {key} rows are not a permutation of the {split} queries"
        perm = np.empty(len(qs), np.int64); perm[oq] = idx[od]; lab += " (re-aligned)"
    sel = perm[rows]
    if "gold_rank_filt" in d.files:
        return d["gold_rank_filt"][sel].astype(np.int64), np.ones(len(sel), bool), lab + " gold_rank_filt"
    D = BaseDump(DUMPS[key], S)
    rc, ro, ex = D.filtered_ranks(sel, F)
    return rc.astype(np.int64), ex.astype(bool), lab + " top-K reconstruction (BaseDump.filtered_ranks)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--splits", required=True, help="Stage 1: valid; Stage 2: test")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    S = Splits(DS); N = S.n_ent; F = S.filter_index()
    assert N == N_ENT
    keysF = np.repeat(F["keys"], F["end"] - F["start"]); allTrue = np.unique(keysF * N + F["objs"].astype(np.int64))
    Hh = Hist(S); BM = DyadReader(S, Hh); log("hist + dyad reader", BM.n_transitions)
    sfx = "_smoke" if a.smoke else ""
    out_dir = f"{WORK}/feat_{a.base}{sfx}"; os.makedirs(out_dir, exist_ok=True)
    ld = f"{WORK}/lists_{a.base}{sfx}"
    summ = dict(ds=DS, base=a.base, feat_names=FEAT_NAMES, groups=GROUPS, W=W_PAT,
                host=os.environ.get("HD_HOST", "serverA (CPU, tkg-gpu:latest)"), verify={})
    for split in a.splits.split(","):
        assert split in ("valid", "test"), split
        t1 = time.time()
        Lq = np.load(f"{ld}/{split}_q.npz"); rows = Lq["rows"]
        st = dict(np.load(f"{WORK}/strata_{split}.npz"))
        n = len(rows)
        sro_all = split_queries(S, split)
        sro = sro_all[rows]
        assert (sro == st["sro"][:n]).all(), "strata row order mismatch"
        LP = fullrows(a.base, split); assert LP.shape[0] == len(sro_all)
        Cm = np.load(f"{ld}/{split}_CP.npy", mmap_mode="r"); Xc = np.load(f"{ld}/{split}_XPc.npy", mmap_mode="r")
        W = W_PAT
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        O = dict(X=mm("X", (n, W, len(FEAT_NAMES)), np.float16), LP=mm("logp", (n, W), np.float32), CA=mm("cand", (n, W), np.int32),
                 FI=mm("filt", (n, W), np.bool_), BA=mm("babove", (n, W), np.bool_))
        r_base = np.zeros(n, np.int64); gs = np.zeros(n, np.float64); p1_all = np.zeros(n, np.float32); gpos = np.full(n, -1, np.int16)
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk); b = np.arange(lo, hi); nb = len(b)
            s, r, o, t = sro[b, 0], sro[b, 1], sro[b, 2], sro[b, 3]
            row = np.asarray(LP[rows[b]], np.float32)
            glp = row[np.arange(nb), o]
            fk = S.fkey(s, r, t)
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
            cand = np.asarray(Cm[lo:hi]).astype(np.int64); valid = cand >= 0
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
            gpos[b] = np.where(isgold.any(1), isgold.argmax(1), -1)
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
            Xf[:, :, 19:25] = np.asarray(Xc[lo:hi]).astype(np.float32)
            Xf[~valid] = 0.0
            O["X"][lo:hi] = Xf.astype(np.float16); O["LP"][lo:hi] = np.where(valid, lp, -np.inf).astype(np.float32)
            O["CA"][lo:hi] = cand.astype(np.int32); O["FI"][lo:hi] = filt; O["BA"][lo:hi] = ba
            if (lo // a.chunk) % 25 == 0:
                log(split, f"{hi}/{n}")
        for z in O.values(): z.flush()
        freq = S.freq[sro[:, 2]]
        q = dict(rows=rows, sro=sro, r_base=r_base, gold_pos=gpos, p1=p1_all, stratum=S.stratum(freq),
                 in_hist=S.in_hist(sro[:, 0], sro[:, 2]), in_top50=r_base <= KC, gs=gs, gold_train_freq=freq)
        for k in STRATA_KEYS:
            q[k] = st[k][:n]
        for k in LIST_Q:
            q[f"list_{k}"] = Lq[k][:n]
        rd, rex, dkey = dump_ranks(a.base, split, S, rows, F)
        q["r_dump"] = rd; q["r_dump_exact"] = rex
        e = dict(n=int(n), seconds=round(time.time() - t1, 1), base=metrics_from_ranks(r_base.astype(np.float64)), dump=dkey,
                 dump_exact_share=float(rex.mean()),
                 rank_agree_fullrow_vs_dump=float((r_base[rex] == rd[rex]).mean()) if rex.any() else None,
                 rank_agree_all_rows=float((r_base == rd).mean()),
                 max_abs_rank_diff_exact_rows=int(np.abs(r_base[rex] - rd[rex]).max()) if rex.any() else None,
                 base_dump=metrics_from_ranks(rd.astype(np.float64)))
        np.savez(f"{out_dir}/{split}_q.npz", **q)
        rk0 = exact_fused_ranks(np.asarray(O["LP"]), np.asarray(O["CA"]), np.asarray(O["FI"]), np.asarray(O["BA"]), gpos.astype(np.int64), r_base)
        e["identity_f0"] = float(np.mean(rk0 == r_base)); e["gold_in_W100"] = float((gpos >= 0).mean()); e["gold_in_top50"] = float((r_base <= KC).mean())
        summ[split] = e
        log(split, json.dumps(e))
    # forward holdout: no residual-track reference MRR exists for GDELTFW; the pre-registered check is the full-row vs
    # dump filtered-rank agreement (>= 99.9%) on each processed split (dump = fh_tirgn_rows.py, same replay)
    summ["verify"] = {sp: dict(dump=summ[sp]["dump"], dump_exact_share=summ[sp]["dump_exact_share"],
                               rank_agree=summ[sp].get("rank_agree_fullrow_vs_dump"),
                               rank_agree_ok=bool((summ[sp].get("rank_agree_fullrow_vs_dump") or 0) >= 0.999))
                      for sp in a.splits.split(",")}
    summ["verify"]["smoke"] = bool(a.smoke)
    jdump(dict(feat_names=FEAT_NAMES, groups=GROUPS, KC=KC, W=W_PAT, base=a.base, ds=DS), f"{out_dir}/meta.json")
    summ["elapsed_s"] = round(time.time() - t0, 1); summ["modules"] = module_provenance()
    tag = "_".join(a.splits.split(","))
    jdump(summ, f"feat_summary_{a.base}_{tag}{sfx}.json"); jdump(summ, f"{out_dir}/feat_summary_{tag}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE", json.dumps(summ["verify"]))


if __name__ == "__main__":
    main()
