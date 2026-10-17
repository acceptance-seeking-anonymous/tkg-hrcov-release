#!/usr/bin/env python3
"""pilot_smallpedia / sp_prep.py -- BASE-INDEPENDENT preparation (CPU), run while the bases train.

  1. valid sample: VALID_N > #valid rows -> the FULL valid split (rng 12345 kept for parity)  -> work/valid_sample_idx.npy
  2. RecB-lite per-query ranks (datasets-track profile_tkg_v2, patched only to save ranks; lambda on valid)
  3. 6.1 pattern TE tables on the EARLY 70% of train timestamps (pc_prep recipe)              -> work/te_SMALLPEDIA.npz
  4. static path-type precision tables (sp_static.StaticTE) on early-train queries             -> work/ste_SMALLPEDIA.npz
  5. strata (EVAL ONLY) for valid + every test query: pc_prep strata + static reach:
       st_S (static edge s-o), st_SS (static common neighbour, uncapped), st_SSc (capped deg<=cap), st_MX (mixed
       static/temporal 2-hop, uncapped), st_enum (gold in the capped enumerated static candidate set used by the system),
       st_enum_size, density controls of the enumerated set at matched |C| (degree-matched expectation, popularity),
       union2 (any temporal or static <=2-hop path, uncapped), static degree of s / o.         -> work/strata_{valid,test}.npz
  6. uncapped union / temporal-only / static-only 2-hop SET SIZES + controls on a uniform 20k test sample (rng 7)
                                                                                             -> work/reach_sample.npz
  python3 sp_prep.py --workers 24
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import (H, TRACKS, P, DD, DS, WORK, VALID_N, N_ENT, ST_DEG_CAP, Splits, split_queries, profiler_to_split_order,
                       jdump, log_fn, load_static)
from patterns import EventIndex, instantiate, decode
from rt_common import te_keys, M1, M2
import sp_static as SS
import multiprocessing as mp

G = {}


def _strata_chunk(args):
    lo, hi = args
    EI = G["EI"]; Q = G["Q"]; SI = G["SI"]; N = EI.N
    dbin = G["dbin"]; nbin = G["nbin"]; rpop = G["rpop"]; nb = len(nbin)
    n = hi - lo
    spec = (("nev_s", np.int32), ("nev_o", np.int32), ("dyad", bool), ("trip", bool), ("pat_any", bool), ("pat_hop1", bool),
            ("two_hop", bool), ("n_pat", np.int32), ("n_union", np.int32), ("nb_s", np.int32),
            ("st_S", bool), ("st_SS", bool), ("st_SSc", bool), ("st_MX", bool), ("st_enum", bool), ("st_enum_size", np.int32),
            ("st_enum_dm", np.float32), ("st_enum_pop", bool), ("union2", bool), ("st_deg_s", np.int32), ("st_deg_o", np.int32),
            ("st_enum_S", bool), ("st_enum_SS", bool), ("st_enum_ST", bool), ("st_enum_TS", bool))
    out = {k: np.zeros(n, dt) for k, dt in spec}
    for k in range(n):
        s, r, o, t = (int(x) for x in Q[lo + k])
        a, b = EI.indptr[s], EI.indptr[s + 1]
        ps = a + np.searchsorted(EI.ts[a:b], t, "left")
        a2, b2 = EI.indptr[o], EI.indptr[o + 1]
        po = a2 + np.searchsorted(EI.ts[a2:b2], t, "left")
        out["nev_s"][k] = ps - a; out["nev_o"][k] = po - a2
        ds_ = EI.dst[a:ps]; rs_ = EI.rel[a:ps]
        out["dyad"][k] = bool((ds_ == o).any())
        out["trip"][k] = bool(((ds_ == o) & (rs_ == r)).any())
        ns = np.unique(ds_); no = np.unique(EI.dst[a2:po])
        out["nb_s"][k] = len(ns)
        if len(ns) and len(no):
            com = np.intersect1d(ns, no, assume_unique=True)
            com = com[(com != s) & (com != o)]
            out["two_hop"][k] = len(com) > 0
        d = instantiate(EI, s, t, M1, M2)
        if d is not None:
            hit = d["pc_c"] == o
            out["pat_any"][k] = bool(hit.any())
            if hit.any():
                out["pat_hop1"][k] = bool((d["hop"][d["pc_p"][hit]] == 1).any())
            out["n_pat"][k] = len(d["pid"]); out["n_union"][k] = len(np.unique(d["pc_c"]))
        # ---- static reach (uncapped membership tests)
        xs, _ = SI.nbrs(s); xo, _ = SI.nbrs(o)
        out["st_deg_s"][k] = len(xs); out["st_deg_o"][k] = len(xo)
        out["st_S"][k] = bool((xs == o).any())
        us, uo = np.unique(xs), np.unique(xo)
        cs = np.intersect1d(us, uo, assume_unique=True); cs = cs[(cs != s) & (cs != o)]
        out["st_SS"][k] = len(cs) > 0
        out["st_SSc"][k] = bool((SI.deg[cs] <= SI.cap).any()) if len(cs) else False
        mx = False
        if len(us) and len(no):
            c1 = np.intersect1d(us, no, assume_unique=True); mx |= bool(len(c1[(c1 != s) & (c1 != o)]))
        if len(ns) and len(uo):
            c2 = np.intersect1d(ns, uo, assume_unique=True); mx |= bool(len(c2[(c2 != s) & (c2 != o)]))
        out["st_MX"][k] = mx
        out["union2"][k] = out["dyad"][k] or out["two_hop"][k] or out["st_S"][k] or out["st_SS"][k] or mx
        # ---- capped enumerated static candidate set (what the static block can draw from)
        pth = SS.enumerate_paths(SI, EI, s, t)
        C = np.unique(pth["cand"])
        out["st_enum_size"][k] = len(C)
        if len(C):
            hitm = pth["cand"] == o
            out["st_enum"][k] = bool(hitm.any())
            if hitm.any():
                shp = set(pth["shape"][hitm].tolist())
                out["st_enum_S"][k] = SS.SH_S in shp; out["st_enum_SS"][k] = SS.SH_SS in shp
                out["st_enum_ST"][k] = SS.SH_ST in shp; out["st_enum_TS"][k] = SS.SH_TS in shp
            cb = np.bincount(dbin[C], minlength=nb)
            out["st_enum_dm"][k] = cb[dbin[o]] / nbin[dbin[o]]
            out["st_enum_pop"][k] = bool(rpop[o] < len(C))
    return lo, out


def _te_chunk(rows):
    EI = G["EI"]; Q = G["Qtr"]
    keys = {l: [] for l in ("full", "rr", "r2", "p")}; hits = {l: [] for l in keys}
    hop_h = np.zeros(2); hop_n = np.zeros(2)
    for i in rows:
        s, r, o, t = (int(x) for x in Q[i])
        d = instantiate(EI, s, t, M1, M2)
        if d is None:
            continue
        pid = d["pid"]; Pn = len(pid)
        hit = np.bincount(d["pc_p"][d["pc_c"] == o], minlength=Pn) > 0
        hop, rel1, lb1, rel2, lb2 = decode(pid, EI.R)
        for lvl, k in zip(("full", "rr", "r2", "p"), te_keys(r, pid, hop, rel1, rel2, EI.R)):
            keys[lvl].append(k); hits[lvl].append(hit)
        for h in (1, 2):
            m = hop == h; hop_h[h - 1] += hit[m].sum(); hop_n[h - 1] += m.sum()
    out = {}
    for lvl in keys:
        if not keys[lvl]:
            out[lvl] = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0)); continue
        k = np.concatenate(keys[lvl]); h = np.concatenate(hits[lvl]).astype(np.float64)
        uk, inv = np.unique(k, return_inverse=True)
        out[lvl] = (uk, np.bincount(inv, weights=h), np.bincount(inv).astype(np.float64))
    return out, hop_h, hop_n


def _ste_chunk(rows):
    EI = G["EI"]; Q = G["Qtr"]; SI = G["SI"]; R2 = 2 * EI.R
    parts = []
    n_reach = 0
    for i in rows:
        s, r, o, t = (int(x) for x in Q[i])
        pth = SS.enumerate_paths(SI, EI, s, t)
        f = SS.fit_counts(pth, o, r, R2, SI.D1)
        if f is not None:
            parts.append(f); n_reach += int((pth["cand"] == o).any())
    if not parts:
        return None, 0, len(rows)
    return {lvl: SS.merge_counts(parts, lvl) for lvl in ("full", "type", "shape")}, n_reach, len(rows)


def _merge(parts):
    k = np.concatenate([p[0] for p in parts]); h = np.concatenate([p[1] for p in parts]); n = np.concatenate([p[2] for p in parts])
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=h), np.bincount(inv, weights=n)


def _sample_chunk(rows):
    """uncapped 2-hop SET sizes (temporal-entity candidates) of temporal-only, static-only and union graphs."""
    EI = G["EI"]; SI = G["SI"]; Q = G["Q"]; N = EI.N
    dbin = G["dbin"]; nbin = G["nbin"]; rpop = G["rpop"]; nb = len(nbin)
    out = []
    for i in rows:
        s, r, o, t = (int(x) for x in Q[i])
        a, b = EI.indptr[s], EI.indptr[s + 1]; ps = a + np.searchsorted(EI.ts[a:b], t, "left")
        n1t = np.unique(EI.dst[a:ps]); n1s = np.unique(SI.nbrs(s)[0])
        def tem_nb(xs):
            xs = xs[xs < N]
            if not len(xs): return np.zeros(0, np.int64)
            seg = []
            for x in xs.tolist():
                a_, b_ = EI.indptr[x], EI.indptr[x + 1]; p_ = a_ + np.searchsorted(EI.ts[a_:b_], t, "left")
                seg.append(EI.dst[a_:p_])
            return np.concatenate(seg).astype(np.int64) if seg else np.zeros(0, np.int64)
        def st_nb(xs):
            if not len(xs): return np.zeros(0, np.int64)
            return SI.multi_nbrs(xs)[1]
        rec = {}
        T2 = np.unique(np.concatenate([n1t, tem_nb(n1t)]))
        S2 = np.unique(np.concatenate([n1s, st_nb(n1s)]))
        n1u = np.union1d(n1t, n1s)
        U2 = np.unique(np.concatenate([n1u, tem_nb(n1u), st_nb(n1u)]))
        for nm, Cc in (("temporal2", T2), ("static2", S2), ("union2", U2)):
            Cc = Cc[(Cc < N) & (Cc != s)]
            nC = len(Cc)
            cb = np.bincount(dbin[Cc], minlength=nb) if nC else np.zeros(nb, np.int64)
            rec[nm] = (nC, bool((Cc == o).any()) if nC else False, nC / N, cb[dbin[o]] / nbin[dbin[o]], bool(rpop[o] < nC))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--te_n", type=int, default=400000)
    ap.add_argument("--ste_n", type=int, default=100000)
    ap.add_argument("--n_sample", type=int, default=20000)
    ap.add_argument("--skip_recb", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    os.makedirs(WORK, exist_ok=True)
    S = Splits(DS)
    summ = dict(host="serverA (CPU, tkg-gpu:latest container)", workers=a.workers)
    Qv = split_queries(S, "valid"); Qt = split_queries(S, "test"); Qtr = split_queries(S, "train")
    summ["n_rows"] = dict(train=len(Qtr), valid=len(Qv), test=len(Qt))
    rng = np.random.default_rng(12345)
    vidx = np.sort(rng.choice(len(Qv), min(VALID_N, len(Qv)), replace=False))
    if a.smoke:
        vidx = vidx[:a.smoke]; Qt = Qt[:a.smoke]
    np.save(f"{WORK}/valid_sample_idx.npy", vidx)
    summ["valid_sample"] = dict(n=int(len(vidx)), full_valid=bool(len(vidx) == len(Qv)))
    log("rows", summ["n_rows"], "valid sample", len(vidx))
    if not a.skip_recb:
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "profile_tkg_v2.py")).read()   # src/datasets
        anchor = "        G.clear()\n    out[\"profile_seconds_total\"]"
        assert anchor in src
        patched = src.replace(anchor, "        np.save(os.environ['RECB_PREFIX'] + '_test_rank_prof.npy', test_rank)\n"
                                      "        np.save(os.environ['RECB_PREFIX'] + '_val_rank_prof.npy', val_rank)\n" + anchor)
        open(f"{WORK}/profile_tkg_v2_saveranks.py", "w").write(patched)
        env = dict(os.environ, RECB_PREFIX=f"{WORK}/recb")
        t1 = time.time()
        subprocess.run([sys.executable, f"{WORK}/profile_tkg_v2_saveranks.py", "--name", "tkgl-smallpedia", "--dir", DD,
                        "--out", f"{WORK}/recb_profile.json", "--nproc", str(min(16, a.workers))], check=True, env=env)
        prof = json.load(open(f"{WORK}/recb_profile.json"))
        rb = prof["recb_lite"]; j = rb["lambda_grid"].index(rb["lambda_selected"])
        tr_ = np.load(f"{WORK}/recb_test_rank_prof.npy")[:, j]; vr_ = np.load(f"{WORK}/recb_val_rank_prof.npy")[:, j]
        np.save(f"{WORK}/recb_test_rank.npy", tr_[profiler_to_split_order(S, "test")])
        np.save(f"{WORK}/recb_valid_rank.npy", vr_[profiler_to_split_order(S, "valid")])
        summ["recb_lite"] = dict(lambda_selected=rb["lambda_selected"], valid=rb["valid"], test=rb["test"],
                                 valid_MRR_by_lambda=rb["valid_MRR_by_lambda"], seconds=round(time.time() - t1, 1))
        log("recb", json.dumps(summ["recb_lite"]["test"]))
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, S.n_ent, S.R, 1)
    SI = SS.StaticIndex(load_static(), N_ENT, ST_DEG_CAP)
    freq = S.freq
    dbin = np.where(freq == 0, 0, 1 + np.floor(np.log2(np.maximum(freq, 1)))).astype(np.int64)
    nbin = np.bincount(dbin).astype(np.float64)
    rpop = np.empty(N_ENT, np.int64); rpop[np.argsort(-freq, kind="stable")] = np.arange(N_ENT)
    G.update(EI=EI, Qtr=Qtr, SI=SI, dbin=dbin, nbin=nbin, rpop=rpop)
    summ["static_index"] = dict(n_nodes=int(SI.NT), n_prel=int(SI.NP), deg_cap=int(SI.cap), M_T=SS.M_T, MAX_X=SS.MAX_X,
                                n_nodes_deg_gt_cap=int((SI.deg > SI.cap).sum()))
    # ---------------- pattern TE tables (early 70% of train timestamps)
    t1 = time.time()
    uts = np.unique(Qtr[:, 3]); cut = uts[int(0.7 * len(uts))]
    early = np.flatnonzero(Qtr[:, 3] < cut)
    te_rows = np.sort(np.random.default_rng(0).choice(early, min(a.te_n if not a.smoke else 3000, len(early)), replace=False))
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(_te_chunk, np.array_split(te_rows, a.workers * 8), chunksize=1)
    arrs = {}
    for lvl in ("full", "rr", "r2", "p"):
        uk, hh, nn = _merge([p[0][lvl] for p in parts])
        arrs[f"{lvl}_k"] = uk; arrs[f"{lvl}_h"] = hh; arrs[f"{lvl}_n"] = nn
    hop_h = sum(p[1] for p in parts); hop_n = sum(p[2] for p in parts)
    arrs["prior"] = hop_h / np.maximum(hop_n, 1)
    np.savez(f"{WORK}/te_{DS}.npz", **arrs)
    summ["te"] = dict(n_queries=int(len(te_rows)), n_early=int(len(early)), train_ts_cut=int(cut), prior_hit_rate_by_hop=arrs["prior"].tolist(),
                      seconds=round(time.time() - t1, 1))
    log("te", json.dumps(summ["te"]))
    # ---------------- static path-type precision tables (same early window)
    t1 = time.time()
    ste_rows = np.sort(np.random.default_rng(1).choice(early, min(a.ste_n if not a.smoke else 2000, len(early)), replace=False))
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(_ste_chunk, np.array_split(ste_rows, a.workers * 8), chunksize=1)
    sarr = {}
    for lvl in ("full", "type", "shape"):
        ps_ = [p[0][lvl] for p in parts if p[0] is not None]
        k = np.concatenate([x[0] for x in ps_]); h = np.concatenate([x[1] for x in ps_]); z = np.concatenate([x[2] for x in ps_])
        uk, inv = np.unique(k, return_inverse=True)
        sarr[f"{lvl}_k"] = uk; sarr[f"{lvl}_h"] = np.bincount(inv, weights=h); sarr[f"{lvl}_z"] = np.bincount(inv, weights=z)
    np.savez(f"{WORK}/ste_{DS}.npz", **sarr)
    shp_prec = {SS_: float(sarr["shape_h"][i] / max(sarr["shape_z"][i], 1)) for i, SS_ in enumerate(sarr["shape_k"].tolist())}
    summ["static_te"] = dict(n_queries=int(len(ste_rows)), gold_reached=float(sum(p[1] for p in parts) / max(1, sum(p[2] for p in parts))),
                             n_types=int(len(sarr["type_k"])), n_full_keys=int(len(sarr["full_k"])),
                             shape_precision={["S", "SS", "ST", "TS"][int(k)]: v for k, v in shp_prec.items()},
                             shape_hits={["S", "SS", "ST", "TS"][int(k)]: float(sarr["shape_h"][i]) for i, k in enumerate(sarr["shape_k"].tolist())},
                             shape_sizes={["S", "SS", "ST", "TS"][int(k)]: float(sarr["shape_z"][i]) for i, k in enumerate(sarr["shape_k"].tolist())},
                             seconds=round(time.time() - t1, 1))
    log("static te", json.dumps(summ["static_te"]))
    # ---------------- strata
    for split, Q in (("valid", Qv[vidx]), ("test", Qt)):
        t1 = time.time()
        G["Q"] = Q
        n = len(Q); step = 1000
        tasks = [(lo, min(n, lo + step)) for lo in range(0, n, step)]
        res = {}
        with mp.get_context("fork").Pool(a.workers) as pool:
            for lo, o in pool.imap_unordered(_strata_chunk, tasks, chunksize=1):
                res[lo] = o
        keys = list(res[0].keys())
        st = {k: np.concatenate([res[lo][k] for lo, _ in tasks]) for k in keys}
        st["sro"] = Q; st["self_loop"] = Q[:, 0] == Q[:, 2]
        st["first_seen"] = (st["nev_s"] == 0) | (st["nev_o"] == 0)
        np.savez(f"{WORK}/strata_{split}.npz", **st)
        npm = ~st["dyad"]
        summ[f"strata_{split}"] = dict(n=n, seconds=round(time.time() - t1, 1),
                                       shares={k: float(np.mean(st[k])) for k in ("dyad", "trip", "pat_any", "two_hop", "self_loop", "first_seen",
                                                                                  "st_S", "st_SS", "st_SSc", "st_MX", "st_enum", "union2")},
                                       within_no_pair={k: float(np.mean(st[k][npm])) for k in ("two_hop", "st_S", "st_SS", "st_SSc", "st_MX", "st_enum", "union2", "pat_any")},
                                       st_enum_median_size=float(np.median(st["st_enum_size"])),
                                       st_enum_reach_vs_controls=dict(reach=float(st["st_enum"].mean()), degree_matched=float(st["st_enum_dm"].mean()),
                                                                      uniform=float((st["st_enum_size"] / N_ENT).mean()), popularity=float(st["st_enum_pop"].mean())))
        log(split, json.dumps(summ[f"strata_{split}"]))
    # ---------------- uncapped 2-hop set sizes on a uniform test sample
    t1 = time.time()
    G["Q"] = Qt
    samp = np.sort(np.random.default_rng(7).choice(len(Qt), min(a.n_sample if not a.smoke else 500, len(Qt)), replace=False))
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs = [x for part in pool.map(_sample_chunk, np.array_split(samp, a.workers * 8), chunksize=1) for x in part]
    arr = {nm: np.array([r_[nm] for r_ in recs], np.float64) for nm in ("temporal2", "static2", "union2")}
    np.savez(f"{WORK}/reach_sample.npz", idx=samp, **arr)
    summ["reach_sample"] = dict(n=int(len(samp)), seconds=round(time.time() - t1, 1),
                                **{nm: dict(reach=float(v[:, 1].mean()), median_size=float(np.median(v[:, 0])), mean_size_frac=float(v[:, 2].mean()),
                                            degree_matched=float(v[:, 3].mean()), popularity=float(v[:, 4].mean())) for nm, v in arr.items()})
    log("sample", json.dumps(summ["reach_sample"]))
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, "prep_summary.json"); jdump(summ, f"{WORK}/prep_summary.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
