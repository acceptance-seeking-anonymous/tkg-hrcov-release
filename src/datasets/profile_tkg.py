#!/usr/bin/env python3
"""Profile TKG forecasting datasets: recurrence, dyadic history, unseen entities,
history sparsity, relation skew, and a recurrency-baseline MRR under the
time-aware filtered, both-directions, full-ranking protocol.

numpy-only (runs on hosts without pandas).  Usage:
    python3 profile_tkg.py --name ICEWS18 --dir data/ICEWS18 --out prof_ICEWS18.json
    python3 profile_tkg.py --name tkgl-polecat --tgb-zip tkgl-polecat.zip --out ...

Definitions (all 'before t' = strictly earlier timestamp, whole dataset incl.
earlier valid/test facts, i.e. the standard online-history protocol):
  rec            share of test facts (s,r,o,t) whose (s,r,o) occurred before t   [Gastinger Rec]
  drec           share whose (s,r,o) occurred at the previous timestamp           [Gastinger DRec]
  rec_trainval   share whose (s,r,o) occurs in train+valid
  dyad_online    share whose unordered pair {s,o} co-occurred (any relation) before t
  dyad_trainval  same but pair seen in train+valid  (= harness 'hist=yes')
  unseen_*       entity absent from TRAIN (query level: both directions)
  cold_gold      gold entity has < 50 train facts (= harness 'cold')
  hist_sr        #earlier events with the same (query entity, direction-specific relation)
  hist_e         #earlier facts involving the query entity
RecB-lite:  score(c) = sum_k lam^(t-k) over earlier (q,r,c,k)  (time-weighted
  frequency, = Gastinger 'strict' psi_Delta with one GLOBAL lam), seen candidates
  ranked above unseen; ties and unseen candidates broken by the static train
  answer-frequency of c for relation r (a 'relaxed' tie-breaker); remaining
  exact ties count 1/2.  lam chosen on valid MRR from LAMS.  Ranking is over ALL
  entities (1-vs-all), time-aware filtered, forward + inverse queries.
"""
import argparse, json, os, sys, time, zipfile, io, csv
import numpy as np
import multiprocessing as _mp
Pool = _mp.get_context("fork").Pool

LAMS = [0.01, 0.5, 0.9, 0.99, 1.0]

# ----------------------------------------------------------------- loading
def _read_txt(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(float(p[3]))))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)

def load_dir(d):
    return [_read_txt(os.path.join(d, f + ".txt")) for f in ("train", "valid", "test")]

def load_tgb_zip(zpath, log):
    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if n.endswith(".csv") and "edgelist" in n and "static" not in n]
    log["tgb_zip_members"] = z.namelist()[:30]
    if not names:
        raise RuntimeError("no edgelist csv in %s: %s" % (zpath, z.namelist()[:20]))
    name = sorted(names, key=len)[0]
    log["tgb_edgelist"] = name
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
        header = [h.strip().lower() for h in next(rd)]
        log["tgb_header"] = header
        def pick(cands):
            for c in cands:
                if c in header:
                    return header.index(c)
            raise RuntimeError("columns %s not in header %s" % (cands, header))
        it = pick(["ts", "timestamp", "time", "date"]); ih = pick(["head", "src", "source", "subject"])
        itl = pick(["tail", "dst", "destination", "object"]); ir = pick(["relation_type", "relation", "edge_type", "rel", "type"])
        ent, rel = {}, {}
        S, R, O, T = [], [], [], []
        for row in rd:
            h, tl, r = row[ih], row[itl], row[ir]
            S.append(ent.setdefault(h, len(ent))); O.append(ent.setdefault(tl, len(ent)))
            R.append(rel.setdefault(r, len(rel)))
            tv = row[it]
            try:
                T.append(int(float(tv)))
            except ValueError:
                T.append(int(tv.replace("-", "")[:8]))
    a = np.stack([np.array(S), np.array(R), np.array(O), np.array(T)], 1).astype(np.int64)
    # TGB generate_splits: quantiles of timestamps, 70/15/15
    ts = a[:, 3]
    val_time, test_time = np.quantile(ts, [0.70, 0.85])
    tr = a[ts <= val_time]; va = a[(ts > val_time) & (ts <= test_time)]; te = a[ts > test_time]
    log["tgb_split_rule"] = "np.quantile(ts,[0.70,0.85]); train ts<=q70, valid q70<ts<=q85, test ts>q85"
    return [tr, va, te]

# ----------------------------------------------------------------- helpers
def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

def dist(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "share_zero": float((x == 0).mean())}

def count_before(keys_sorted, gkey, tq, T):
    """#events with group key gkey and time < tq, given sorted array of gkey*T+t."""
    lo = np.searchsorted(keys_sorted, gkey * T, "left")
    hi = np.searchsorted(keys_sorted, gkey * T + tq, "left")
    return hi - lo

# ----------------------------------------------------------------- RecB-lite worker
G = {}

def _rr_worker(rr):
    ev = G["ev_by_rr"][rr]          # (q, a, t, split) sorted by t
    E = G["E"]; lams = np.array(LAMS)
    xi_cnt = G["xi_by_rr"].get(rr, {})
    nz = np.sort(np.array(list(xi_cnt.values()), dtype=np.float64))   # nonzero static counts
    n_zero = E - len(nz)
    groups = {}
    out_q, out_rank = [], []
    q, a, t, sp, qid = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3], ev[:, 4]
    bounds = np.flatnonzero(np.diff(t)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(t)]])
    for s0, e0 in zip(starts, ends):
        tt = int(t[s0])
        blk = range(s0, e0)
        if (sp[s0:e0] >= 1).any():   # block holds valid/test events -> evaluate them (history < tt)
            # golds per q at this t over ALL splits (time-aware filter)
            golds = {}
            for i in blk:
                golds.setdefault(int(q[i]), set()).add(int(a[i]))
            for i in blk:
                if sp[i] < 1:
                    continue
                qq, gg = int(q[i]), int(a[i])
                F = golds[qq] - {gg}
                grp = groups.get(qq)
                xg = xi_cnt.get(gg, 0)
                if grp is None:
                    nseen = 0
                    sx = np.zeros(0)
                    gslot = None
                else:
                    cmap, last, acc, xis, n = grp
                    gslot = cmap.get(gg)
                    mask = np.ones(n, bool)
                    for f in F:
                        sl = cmap.get(f)
                        if sl is not None:
                            mask[sl] = False
                if grp is not None and gslot is not None:
                    sc = acc[:n] * lams[None, :] ** (tt - last[:n])[:, None]
                    gs = sc[gslot]
                    xv = xis[:n]
                    gt = (sc > gs[None, :]) | ((sc == gs[None, :]) & (xv > xg)[:, None])
                    eq = (sc == gs[None, :]) & (xv == xg)[:, None]
                    m2 = mask.copy(); m2[gslot] = False
                    rank = 1.0 + gt[m2].sum(0) + 0.5 * eq[m2].sum(0)
                else:
                    if grp is not None:
                        nseen = int(mask.sum())
                        seenF_x = list(xis[:n][mask])
                    else:
                        nseen = 0; seenF_x = []
                    for f in F:   # filtered entities are removed from the unseen pool too
                        if grp is None or grp[0].get(f) is None:
                            seenF_x.append(xi_cnt.get(f, 0))
                        else:
                            seenF_x.append(xis[grp[0][f]])   # already excluded via mask; count once
                    seenF_x = np.array(seenF_x, dtype=np.float64)
                    # all entities with xi > xg, == xg
                    if xg > 0:
                        greater_all = len(nz) - np.searchsorted(nz, xg, "right")
                        equal_all = np.searchsorted(nz, xg, "right") - np.searchsorted(nz, xg, "left")
                    else:
                        greater_all = len(nz); equal_all = n_zero
                    # remove seen + filtered (they are not in the unseen pool)
                    excl = seenF_x
                    # entities in seen∪F counted once (a filtered entity may also be seen: masked out above, so it
                    # appears in xis[mask]==False and only via the F loop) -> no double count
                    g_un = greater_all - int((excl > xg).sum())
                    e_un = equal_all - 1 - int((excl == xg).sum())
                    r = 1.0 + nseen + g_un + 0.5 * max(e_un, 0)
                    rank = np.full(len(LAMS), r)
                if sp[i] == 2 or sp[i] == 1:
                    out_q.append(int(qid[i])); out_rank.append(rank.astype(np.float32))
        # update history with events at tt
        for i in blk:
            qq, aa = int(q[i]), int(a[i])
            grp = groups.get(qq)
            if grp is None:
                grp = [{}, np.zeros(4, np.int64), np.zeros((4, len(LAMS))), np.zeros(4), 0]
                groups[qq] = grp
            cmap = grp[0]
            sl = cmap.get(aa)
            if sl is None:
                n = grp[4]
                if n == len(grp[1]):
                    cap = 2 * n
                    grp[1] = np.resize(grp[1], cap); grp[3] = np.resize(grp[3], cap)
                    acc2 = np.zeros((cap, len(LAMS))); acc2[:n] = grp[2][:n]; grp[2] = acc2
                sl = n; cmap[aa] = sl; grp[4] = n + 1
                grp[1][sl] = tt; grp[2][sl] = 0.0; grp[3][sl] = xi_cnt.get(aa, 0)
            dt = tt - grp[1][sl]
            grp[2][sl] = grp[2][sl] * lams ** dt + 1.0
            grp[1][sl] = tt
    if out_q:
        return np.array(out_q, np.int64), np.stack(out_rank)
    return np.zeros(0, np.int64), np.zeros((0, len(LAMS)), np.float32)

def mrr_block(ranks):
    if len(ranks) == 0:
        return None
    r = np.asarray(ranks, dtype=np.float64)
    return {"MRR": float((1.0 / r).mean()), "H1": float((r <= 1).mean()), "H3": float((r <= 3).mean()),
            "H10": float((r <= 10).mean()), "n": int(len(r))}

# ----------------------------------------------------------------- main profile
def profile(splits, name, nproc=8, do_recb=True, meta=None):
    t0 = time.time()
    meta = meta or {}
    tr, va, te = splits
    lab = np.concatenate([np.zeros(len(tr), np.int64), np.ones(len(va), np.int64), 2 * np.ones(len(te), np.int64)])
    A = np.concatenate([tr, va, te])
    # compact ids
    ents, inv = np.unique(np.concatenate([A[:, 0], A[:, 2]]), return_inverse=True)
    s = inv[:len(A)]; o = inv[len(A):]
    rels, r = np.unique(A[:, 1], return_inverse=True)
    tsv, t = np.unique(A[:, 3], return_inverse=True)
    E, R, T = len(ents), len(rels), len(tsv)
    out = {"name": name, "n_entities_present": int(E), "n_relations_present": int(R), "n_timestamps": int(T),
           "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)), "meta": meta}
    out["split_shares"] = [round(len(x) / len(A), 4) for x in (tr, va, te)]
    ist = lab == 2
    # --- entity sets
    tr_m = lab == 0; tv_m = lab <= 1
    seen_tr = np.zeros(E, bool); seen_tr[s[tr_m]] = True; seen_tr[o[tr_m]] = True
    seen_tv = np.zeros(E, bool); seen_tv[s[tv_m]] = True; seen_tv[o[tv_m]] = True
    freq_tr = np.bincount(np.concatenate([s[tr_m], o[tr_m]]), minlength=E)
    sT, oT, rT, tT = s[ist], o[ist], r[ist], t[ist]
    out["test_entities"] = int(len(np.unique(np.concatenate([sT, oT]))))
    out["unseen_entity_share_of_test_entities"] = float((~seen_tr[np.unique(np.concatenate([sT, oT]))]).mean())
    out["unseen_subject_fact_share"] = float((~seen_tr[sT]).mean())
    out["unseen_object_fact_share"] = float((~seen_tr[oT]).mean())
    out["unseen_any_fact_share"] = float(((~seen_tr[sT]) | (~seen_tr[oT])).mean())
    out["unseen_both_fact_share"] = float(((~seen_tr[sT]) & (~seen_tr[oT])).mean())
    out["unseen_any_fact_share_vs_trainval"] = float(((~seen_tv[sT]) | (~seen_tv[oT])).mean())
    # query level (both directions): query entity / gold unseen in train
    qent = np.concatenate([sT, oT]); gold = np.concatenate([oT, sT])
    out["query_entity_unseen_share"] = float((~seen_tr[qent]).mean())
    out["gold_unseen_share"] = float((~seen_tr[gold]).mean())
    out["cold_gold_share_lt50"] = float((freq_tr[gold] < 50).mean())
    ent_first = np.full(E, 1 << 40, np.int64)
    np.minimum.at(ent_first, s, t); np.minimum.at(ent_first, o, t)
    out["firstseen_any_fact_share"] = float(((ent_first[sT] == tT) | (ent_first[oT] == tT)).mean())
    # --- recurrence
    key = (s * R + r) * E + o
    order = np.lexsort((t, key))
    ks, ts_ = key[order], t[order]
    first_idx = np.concatenate([[0], np.flatnonzero(np.diff(ks)) + 1])
    uk = ks[first_idx]; first_t = ts_[first_idx]
    pos = np.searchsorted(uk, key[ist])
    out["rec"] = float((first_t[pos] < tT).mean())
    kt = np.unique(key * T + t)
    prev = key[ist] * T + (tT - 1)
    out["drec"] = float(np.isin(prev, kt, assume_unique=False).mean() if len(prev) else 0.0)
    ktv = np.unique(key[tv_m])
    out["rec_trainval"] = float(np.isin(key[ist], ktv).mean())
    # --- dyadic
    lo_ = np.minimum(s, o); hi_ = np.maximum(s, o)
    pk = lo_ * E + hi_
    order = np.lexsort((t, pk)); pks, pts = pk[order], t[order]
    fi = np.concatenate([[0], np.flatnonzero(np.diff(pks)) + 1])
    upk = pks[fi]; pfirst = pts[fi]
    pp = np.searchsorted(upk, pk[ist])
    dy_on = pfirst[pp] < tT
    out["dyad_online"] = float(dy_on.mean())
    ptv = np.unique(pk[tv_m])
    dy_tv = np.isin(pk[ist], ptv)
    out["dyad_trainval"] = float(dy_tv.mean())
    out["hist_no_share_trainval"] = 1.0 - out["dyad_trainval"]
    # --- history lengths (query level, both directions)
    qq = np.concatenate([s, o]); rr = np.concatenate([r, r + R]); aa = np.concatenate([o, s]); tq = np.concatenate([t, t])
    gk = qq * (2 * R) + rr
    gks = np.sort(gk * T + tq)
    istq = np.concatenate([ist, ist])
    h_sr = count_before(gks, gk[istq], tq[istq], T)
    eks = np.sort(qq * T + tq)
    h_e = count_before(eks, qq[istq], tq[istq], T)
    out["hist_sr"] = dist(h_sr); out["hist_e"] = dist(h_e)
    # --- relation skew
    rf = np.bincount(r, minlength=R).astype(np.float64)
    p = rf / rf.sum(); srt = np.sort(rf)[::-1]
    k10 = max(1, int(round(0.1 * R)))
    out["relation_gini"] = gini(rf)
    out["relation_top10pct_share"] = float(srt[:k10].sum() / rf.sum())
    out["relation_norm_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(R)) if R > 1 else 0.0
    out["relations_seen_only_in_test"] = int(len(set(np.unique(rT)) - set(np.unique(r[tv_m]))))
    out["test_facts_with_relation_unseen_in_train"] = float((~np.isin(rT, np.unique(r[tr_m]))).mean())
    out["profile_seconds_stats"] = round(time.time() - t0, 1)
    # --- per-query stratum labels for test queries (both directions), in qid order
    nT = int(ist.sum())
    strata = {
        "gold_unseen": ~seen_tr[gold],
        "query_unseen": ~seen_tr[qent],
        "rec_no": ~np.concatenate([first_t[pos] < tT] * 2),
        "dyad_trainval_no": ~np.concatenate([dy_tv, dy_tv]),
        "dyad_online_no": ~np.concatenate([dy_on, dy_on]),
        "cold_gold": freq_tr[gold] < 50,
        "hist_sr_zero": h_sr == 0,
        "fact_any_unseen": np.concatenate([(~seen_tr[sT]) | (~seen_tr[oT])] * 2),
        "fact_any_firstseen": np.concatenate([(ent_first[sT] == tT) | (ent_first[oT] == tT)] * 2),
    }
    out["stratum_shares"] = {k: float(v.mean()) for k, v in strata.items()}
    if do_recb:
        t1 = time.time()
        # events: both directions, with split and query id (test queries get qid in 0..2nT-1; valid get -1-idx)
        isva = np.concatenate([lab == 1, lab == 1])
        qid = np.full(len(qq), -10**12, np.int64)
        qid[istq] = np.arange(2 * nT)
        nV = int(isva.sum()); qid[isva] = -1 - np.arange(nV)
        spl = np.concatenate([lab, lab])
        ev = np.stack([qq, aa, tq, spl, qid], 1)
        # static xi: train answer counts per direction-specific relation
        trq = np.concatenate([tr_m, tr_m])
        xi = {}
        kk = rr[trq] * E + aa[trq]
        u, c = np.unique(kk, return_counts=True)
        for key_, cnt in zip(u.tolist(), c.tolist()):
            xi.setdefault(key_ // E, {})[key_ % E] = cnt
        order = np.argsort(rr, kind="stable")
        rr_s = rr[order]
        cut = np.flatnonzero(np.diff(rr_s)) + 1
        ev_by = {}
        for chunk in np.split(order, cut):
            rid = int(rr[chunk[0]])
            e2 = ev[chunk]
            e2 = e2[np.argsort(e2[:, 2], kind="stable")]
            if (e2[:, 3] >= 1).any():
                ev_by[rid] = e2
        G.update({"ev_by_rr": ev_by, "xi_by_rr": xi, "E": E})
        todo = sorted(ev_by, key=lambda k: -len(ev_by[k]))
        test_rank = np.zeros((2 * nT, len(LAMS)), np.float32)
        val_rank = np.zeros((nV, len(LAMS)), np.float32)
        if nproc > 1:
            with Pool(nproc) as pool:
                res = pool.imap_unordered(_rr_worker, todo, chunksize=1)
                for qi, rk in res:
                    m = qi >= 0
                    test_rank[qi[m]] = rk[m]; val_rank[-1 - qi[~m]] = rk[~m]
        else:
            for k in todo:
                qi, rk = _rr_worker(k)
                m = qi >= 0
                test_rank[qi[m]] = rk[m]; val_rank[-1 - qi[~m]] = rk[~m]
        assert (test_rank > 0).all() and (val_rank > 0).all(), "missing ranks"
        vm = [(1.0 / val_rank[:, j]).mean() for j in range(len(LAMS))]
        j = int(np.argmax(vm))
        out["recb_lite"] = {
            "lambda_grid": LAMS, "valid_MRR_by_lambda": [float(x) for x in vm], "lambda_selected": LAMS[j],
            "valid": mrr_block(val_rank[:, j]), "test": mrr_block(test_rank[:, j]),
            "test_MRR_by_lambda": [float((1.0 / test_rank[:, k]).mean()) for k in range(len(LAMS))],
            "test_by_stratum": {k: {"yes": mrr_block(test_rank[v, j]), "no": mrr_block(test_rank[~v, j])}
                                for k, v in strata.items()},
            "seconds": round(time.time() - t1, 1),
        }
        G.clear()
    out["profile_seconds_total"] = round(time.time() - t0, 1)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir"); ap.add_argument("--tgb-zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--no-recb", action="store_true")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--last-k-ts", default="", help="e.g. 1,1 : valid = 2nd-to-last unique ts block, test = last (festival TKG split)")
    ap.add_argument("--resplit", default="", help="e.g. 0.5,0.2,0.3 : TransFIR-style re-split by unique timestamps")
    a = ap.parse_args()
    log = {}
    meta = json.loads(a.meta)
    if a.tgb_zip:
        sp = load_tgb_zip(a.tgb_zip, log)
    else:
        sp = load_dir(a.dir)
    if a.last_k_ts:
        kv, kt = [int(x) for x in a.last_k_ts.split(",")]
        allf = np.concatenate(sp); times = np.unique(allf[:, 3])
        tt_ = times[-kt]; vt_ = times[-kt - kv]
        sp = [allf[allf[:, 3] < vt_], allf[(allf[:, 3] >= vt_) & (allf[:, 3] < tt_)], allf[allf[:, 3] >= tt_]]
        log["resplit"] = "last-k-ts valid=%d test=%d unique timestamps" % (kv, kt)
    if a.resplit:
        ra = [float(x) for x in a.resplit.split(",")]
        allf = np.concatenate(sp); times = np.unique(allf[:, 3])
        trt = times[:int(len(times) * ra[0])][-1]; vat = times[:int(len(times) * (ra[0] + ra[1]))][-1]
        sp = [allf[allf[:, 3] <= trt], allf[(allf[:, 3] > trt) & (allf[:, 3] <= vat)], allf[allf[:, 3] > vat]]
        log["resplit"] = "TransFIR re_spilt by unique timestamps " + a.resplit
    meta.update(log)
    res = profile(sp, a.name, nproc=a.nproc, do_recb=not a.no_recb, meta=meta)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("name", "n_train", "n_valid", "n_test", "rec", "drec", "dyad_trainval",
                                          "gold_unseen_share", "cold_gold_share_lt50")}),
          json.dumps(res.get("recb_lite", {}).get("test")), res["profile_seconds_total"], flush=True)
