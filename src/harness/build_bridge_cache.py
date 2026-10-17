"""
harness / build_bridge_cache.py -- compute the G12 stage-1 bridge candidate features ONCE
per (dataset, cohort) and store them as memory-mappable .npy files.  Every later track reads these
caches instead of re-running the bridge BFS (the dominant cost).

Cohorts
  g12           G10-G12 coverage-hard cold cohort, reproduced exactly: forward queries, gold train-freq<50,
                gold not in the base top-K row; train capped at 8000 and test at 5000 with
                np.random.default_rng(0).choice (same order of draws as stage2_ranker.cover_hard).
  valid_pool    a uniform sample of valid queries (both directions) restricted to the low-confidence
                trigger pool (base top-1 prob <= valid quantile Q_MAX)  -> fusion-rule tuning.
  test_pool     EVERY test query (both directions) in the low-confidence trigger pool  -> full-test backfill.

Cache layout  <HARNESS>/cache/<dataset>_<base>_<cohort>/
  meta.json         dataset, base, cohort, n_queries, feature names, build stats
  qrow.npy   [nq]   row index into the base dump the query came from (dump named in meta)
  qsplit.npy [nq]   0 train / 1 valid / 2 test
  built.npy  [nq]   bool, False if the bridge reached nothing (stage2 'None')
  reached.npy[nq]   bool, gold in the candidate set
  gold_pos.npy[nq]  int32 index of gold inside the query's slice (-1 if not reached)
  lm_ok.npy  [nq]   bool, event-LM had history for the subject (else lm==0, as in G12)
  qfeat.npy  [nq,6] f32 (q_entropy, q_top1, q_margin12, q_subj_logfreq, q_rel_logfreq, q_cset_logsize)
  offsets.npy[nq+1] int64 slice bounds into the per-candidate arrays
  cands.npy  [M]    int32 candidate entity ids (stage2 insertion order)
  cfeat.npy  [M,8]  f32 (pm_top1, pm_subj, n_interm, idf_score, r2_recur, interm_min_logfreq,
                         interm_mean_logfreq, tight_path_count)
  lm.npy     [M]    f32 raw event-LM utility (multiply by reuse gate for arm4/arm5)
  hard.npy   [M]    bool hub-or-high-path-multiplicity flag
Use bridge_cache.BridgeCache to read it (reconstructs the 16 stage2 features + casc + gated lm).
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import multiprocessing as mp
from oih_common import Splits, BaseDump, DATASETS, HARNESS, jdump
from bridge_features import BridgeIndex, LMUtility, equivalence_check

G = {}


def _work(args):
    lo, hi, shard_path = args
    bi = G["bi"]; lmu = G["lmu"]; sro = G["sro"]; tki = G["tki"]; tks = G["tks"]; rows = G["rows"]
    kref = G["kref"]
    cands_l, cfeat_l, lm_l, hard_l = [], [], [], []
    nq = hi - lo
    built = np.zeros(nq, bool); reached = np.zeros(nq, bool); gpos = np.full(nq, -1, np.int32)
    lmok = np.zeros(nq, bool); qfeat = np.zeros((nq, 6), np.float32); cnt = np.zeros(nq, np.int64)
    for n, i in enumerate(rows[lo:hi]):
        s, r, o, ts = (int(x) for x in sro[i])
        b = bi.build(s, r, o, tki[i, :kref], tks[i, :kref])
        if b is None:
            continue
        built[n] = True
        c = b["cands"]; X = b["X"]
        reached[n] = b["reached"]
        if b["reached"]:
            gpos[n] = int(np.where(c == o)[0][0])
        lm, ok = lmu.scores(s, r, ts, c)
        lmok[n] = ok
        qfeat[n] = X[0, [10, 11, 12, 13, 14, 15]]
        cf = np.stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 7], X[:, 8],
                       b["tight_count"].astype(np.float32)], 1).astype(np.float32)
        cands_l.append(c); cfeat_l.append(cf); lm_l.append(lm); hard_l.append(b["hard"]); cnt[n] = len(c)
    np.savez(shard_path, built=built, reached=reached, gpos=gpos, lmok=lmok, qfeat=qfeat, cnt=cnt,
             cands=np.concatenate(cands_l) if cands_l else np.zeros(0, np.int32),
             cfeat=np.concatenate(cfeat_l) if cfeat_l else np.zeros((0, 8), np.float32),
             lm=np.concatenate(lm_l) if lm_l else np.zeros(0, np.float32),
             hard=np.concatenate(hard_l) if hard_l else np.zeros(0, bool))
    return lo, hi


def select_g12(S, D):
    """Exact replica of stage2_ranker.cover_hard for train (cap 8000) then test (cap 5000)."""
    rng = np.random.default_rng(0)
    sro = D.sro; tki = D.topk_ids; R = S.R
    fwd = sro[:, 1] < R
    gf = S.freq[sro[:, 2]]
    in_row = (tki == sro[:, 2:3]).any(1)
    out = {}
    for spname, cap in (("train", 8000), ("test", 5000)):
        code = {"train": 0, "test": 2}[spname]
        # stage2 iterates dump rows in order and matches (h,r,t,ts) against the split file:
        # forward rows whose quad is in that split  ==  forward rows with that split's ts range.
        cand = np.where(fwd & (D.split == code) & (gf < 50) & ~in_row)[0].tolist()
        if cap and len(cand) > cap:
            cand = [cand[k] for k in rng.choice(len(cand), cap, replace=False)]
        out[spname] = np.array(cand, np.int64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="tirgn")
    ap.add_argument("--cohort", required=True, choices=["g12", "valid_pool", "test_pool", "equiv"])
    ap.add_argument("--q_max", type=float, default=0.3, help="trigger-pool quantile of valid top-1 prob")
    ap.add_argument("--valid_sample", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--n_equiv", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    cfg = DATASETS[a.dataset]
    S = Splits(a.dataset)
    D = BaseDump(cfg["dumps"][a.base], S)
    kref = D.K
    bi = BridgeIndex(S.q["train"], S.n_ent, S.R)
    print(f"[{a.dataset}] dump K={D.K} N={D.N}; BridgeIndex built {bi.build_s}s DEG_CAP={bi.DEG_CAP}", flush=True)
    lmu = LMUtility(S.q["train"], cfg["lm_ckpt"], cfg["lm_vocab"], n_rel_train=len(bi.relfreq),
                    device="cpu", lm_dir=HARNESS)
    stats = dict(dataset=a.dataset, base=a.base, cohort=a.cohort, dump=cfg["dumps"][a.base], kref=int(kref),
                 DEG_CAP=bi.DEG_CAP, bridge_index_s=bi.build_s)

    if a.cohort == "equiv":
        sel = select_g12(S, D)
        rows = np.concatenate([sel["test"][:a.n_equiv], sel["train"][:a.n_equiv // 2]])
        train = [tuple(int(x) for x in q) for q in S.q["train"]]
        res = equivalence_check(bi, D.sro, D.topk_ids, D.topk_scores, train, rows, S.n_ent)
        res.update(stats); res["elapsed_s"] = round(time.time() - t0, 1)
        jdump(res, f"equiv_{a.dataset}.json"); print(json.dumps(res), flush=True)
        return

    sro, tki, tks, dump_used = D.sro, D.topk_ids, D.topk_scores, cfg["dumps"][a.base]
    if a.cohort == "g12":
        sel = select_g12(S, D)
        rows = np.concatenate([sel["train"], sel["test"]])
        qsplit = np.concatenate([np.zeros(len(sel["train"]), np.int8), np.full(len(sel["test"]), 2, np.int8)])
        stats["n_train_q"] = int(len(sel["train"])); stats["n_test_q"] = int(len(sel["test"]))
    else:
        # trigger threshold = Q_MAX quantile of base top-1 prob over ALL valid queries (both directions)
        vd_path = cfg["dumps"].get(f"{a.base}_valid")
        if vd_path:                                    # ICEWS18 TiRGN: valid rows live in a separate dump
            V = BaseDump(vd_path, S)
            v_rows = V.rows("valid")
            v_p1 = np.exp(V.topk_scores[v_rows, 0].astype(np.float64))
        else:
            V = D; v_rows = D.rows("valid"); v_p1 = D.top1_prob(v_rows)
        tau = float(np.quantile(v_p1, a.q_max))
        stats["tau_p1"] = tau; stats["q_max"] = a.q_max; stats["n_valid_all"] = int(len(v_rows))
        if a.cohort == "valid_pool":
            rng = np.random.default_rng(0)
            samp = np.sort(rng.choice(len(v_rows), min(a.valid_sample, len(v_rows)), replace=False))
            srows = v_rows[samp]
            p1 = np.exp(V.topk_scores[srows, 0].astype(np.float64))
            rows = srows[p1 <= tau]
            stats["valid_sample"] = int(len(srows)); stats["valid_sample_rows"] = srows
            sro, tki, tks, dump_used = V.sro, V.topk_ids, V.topk_scores, (vd_path or cfg["dumps"][a.base])
            qsplit = np.full(len(rows), 1, np.int8)
        else:
            t_rows = D.rows("test"); p1 = D.top1_prob(t_rows)
            rows = t_rows[p1 <= tau]
            qsplit = np.full(len(rows), 2, np.int8)
            stats["n_test_all"] = int(len(t_rows))
    stats["n_queries"] = int(len(rows)); stats["dump_rows_from"] = dump_used
    out_dir = f"{HARNESS}/cache/{a.dataset}_{a.base}_{a.cohort}"
    os.makedirs(f"{out_dir}/shards", exist_ok=True)
    G.update(bi=bi, lmu=lmu, sro=sro, tki=tki, tks=tks, rows=rows, kref=kref)
    tasks = [(lo, min(lo + a.chunk, len(rows)), f"{out_dir}/shards/s{lo:08d}.npz")
             for lo in range(0, len(rows), a.chunk)]
    print(f"[{a.dataset}/{a.cohort}] {len(rows)} queries, {len(tasks)} shards, {a.workers} workers", flush=True)
    t1 = time.time(); done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for lo, hi in pool.imap_unordered(_work, tasks):
            done += hi - lo
            if done % (a.chunk * 20) < a.chunk:
                print(f"  {done}/{len(rows)} ({time.time()-t1:.0f}s)", flush=True)
    stats["bfs_s"] = round(time.time() - t1, 1)
    # ---- assemble ----
    parts = [np.load(t[2]) for t in tasks]
    cnt = np.concatenate([p["cnt"] for p in parts])
    offsets = np.zeros(len(rows) + 1, np.int64); offsets[1:] = np.cumsum(cnt)
    np.save(f"{out_dir}/qrow.npy", rows.astype(np.int64)); np.save(f"{out_dir}/qsplit.npy", qsplit)
    for k, nm in (("built", "built"), ("reached", "reached"), ("gpos", "gold_pos"), ("lmok", "lm_ok"), ("qfeat", "qfeat")):
        np.save(f"{out_dir}/{nm}.npy", np.concatenate([p[k] for p in parts]))
    np.save(f"{out_dir}/offsets.npy", offsets)
    for k in ("cands", "cfeat", "lm", "hard"):
        np.save(f"{out_dir}/{k}.npy", np.concatenate([p[k] for p in parts]))
    for t in tasks:
        os.remove(t[2])
    os.rmdir(f"{out_dir}/shards")
    built = np.load(f"{out_dir}/built.npy"); reached = np.load(f"{out_dir}/reached.npy")
    sizes = np.diff(offsets)[built]
    stats.update(n_built=int(built.sum()), reach_over_built=round(float(reached[built].mean()), 4) if built.any() else None,
                 reach_over_all=round(float(reached.mean()), 4), median_cset=int(np.median(sizes)) if len(sizes) else 0,
                 total_candidates=int(offsets[-1]), elapsed_s=round(time.time() - t0, 1),
                 cfeat_names=["pm_top1", "pm_subj", "n_interm", "idf_score", "r2_recur", "interm_min_logfreq",
                              "interm_mean_logfreq", "tight_path_count"],
                 qfeat_names=["q_entropy", "q_top1", "q_margin12", "q_subj_logfreq", "q_rel_logfreq", "q_cset_logsize"])
    if a.cohort == "g12":
        te = qsplit == 2
        stats["gate"] = dict(expanded_reach=round(float(reached[te & built].sum() / max(1, (te & built).sum())), 4),
                             median_cset=int(np.median(np.diff(offsets)[te & built])))
    st = {k: v for k, v in stats.items() if k != "valid_sample_rows"}
    if "valid_sample_rows" in stats:
        np.save(f"{out_dir}/valid_sample_rows.npy", stats["valid_sample_rows"])
    jdump(st, f"{out_dir}/meta.json"); jdump(st, f"cache_meta_{a.dataset}_{a.base}_{a.cohort}.json")
    os.system(f"chmod -R a+rX {out_dir}")
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
