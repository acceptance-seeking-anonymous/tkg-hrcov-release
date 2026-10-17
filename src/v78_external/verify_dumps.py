#!/usr/bin/env python3
"""v78_external / verify_dumps.py -- independent alignment + consistency check of every dump directory, writes MANIFEST.json.

For each <method>/<dataset>/<split> dump:
  1. gold.npy == harness gold, recomputed here from harness_data/<ds>/<split>.txt with the harness split_queries rule
     AND read from the harness's own stored query list (hardening/v76/recb_xf/recb_csr_<split>_<ds>.npz['sro'][:,2]);
     keys_harness.npy == that stored sro (s, r, o, ts_id).
  2. shapes / dtypes; top_ids rows unique; top_scores non-increasing; floor <= last top score.
  3. MRR / H@k recomputed from rr_official / rank_official equal meta.json.
  4. where the gold is in the top-K list, its list score equals gold_score and its 1-based list position >= rank_official
     (the official rank only counts shipped negatives, i.e. other true answers are filtered out), checked as an upper bound.
"""
import os, sys, json, glob
import numpy as np
TR = os.path.expandvars("$WORK_DIR/tracks")
B = f"{TR}/v78_external"

def split_queries(q, R):
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    u_, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q)); out = []
    for u, b, e in zip(u_, st, en):
        m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        out.append(m.copy()); out.append(inv)
    return np.concatenate(out).astype(np.int64)

man = dict(root=f"{B}/dumps", entries=[])
for D in sorted(glob.glob(f"{B}/dumps/*/*/*/")):
    D = D.rstrip("/"); method, ds, split = D.split("/")[-3:]
    hs = {"val": "valid", "test": "test"}[split]
    dd = f"{TR}/datasets/harness_data/{ds}"
    R = int(open(f"{dd}/stat.txt").read().split()[1])
    sro = split_queries(np.loadtxt(f"{dd}/{hs}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4), R)
    csr = np.load(f"{TR}/hardening/v76/recb_xf/recb_csr_{hs}_{ds}.npz")["sro"]
    gold = np.load(f"{D}/gold.npy"); keys = np.load(f"{D}/keys_harness.npy")
    ti = np.load(f"{D}/top_ids.npy", mmap_mode="r"); ts = np.load(f"{D}/top_scores.npy", mmap_mode="r")
    rr = np.load(f"{D}/rr_official.npy"); rk = np.load(f"{D}/rank_official.npy")
    gs = np.load(f"{D}/gold_score.npy"); fl = np.load(f"{D}/floor.npy")
    meta = json.load(open(f"{os.path.dirname(D)}/meta.json"))["splits"][split]
    LD = np.load(f"{D}/in_learning_data.npy") if os.path.exists(f"{D}/in_learning_data.npy") else np.zeros(len(gold), bool)
    Q, K = ti.shape
    chk = dict(rows=int(Q), K=int(K), top_ids_dtype=str(ti.dtype), top_scores_dtype=str(ts.dtype),
               gold_eq_recomputed_harness=bool(np.array_equal(gold, sro[:, 2])), gold_eq_harness_csr_sro=bool(np.array_equal(gold, csr[:, 2])),
               keys_eq_harness_csr_sro=bool(np.array_equal(keys, csr)), n_gold_mismatch=int((gold != csr[:, 2]).sum()))
    bad_sorted = 0; bad_unique = 0; floor_viol = 0; gold_in = 0; pos_lt_rank = 0; score_mismatch = 0
    for c0 in range(0, Q, 20000):
        okr = ~LD[c0:c0 + 20000]
        a = np.asarray(ti[c0:c0 + 20000])[okr]; s = np.asarray(ts[c0:c0 + 20000]).astype(np.float64)[okr]
        gch, gsch, flch, rkch = gold[c0:c0 + 20000][okr], gs[c0:c0 + 20000][okr], fl[c0:c0 + 20000][okr], rk[c0:c0 + 20000][okr]
        bad_sorted += int((np.diff(s, axis=1) > 0).any(1).sum())
        srt = np.sort(a, 1); bad_unique += int((np.diff(srt, axis=1) == 0).any(1).sum())
        floor_viol += int((flch > s[:, -1]).sum())
        hit = a == gch[:, None]; has = hit.any(1); gold_in += int(has.sum())
        pos = hit.argmax(1) + 1
        rows = np.flatnonzero(has)
        score_mismatch += int((np.abs(s[rows, pos[rows] - 1] - gsch[rows].astype(np.float64)) > 1e-6).sum())
        # official rank only counts negatives: rank <= (#list entries > gold + #list entries >= gold - 1)/2 + 1 (rows with gold score above floor)
        g = gsch.astype(np.float64)[:, None]
        sel = rows[(gsch[rows] > flch[rows])]
        ub = 0.5 * ((s[sel] > g[sel]).sum(1) + (s[sel] >= g[sel]).sum(1) - 1) + 1
        pos_lt_rank += int((rkch[sel] > ub + 1e-9).sum())
    rr_, rk_ = rr[~LD], rk[~LD]
    rec = dict(MRR=float(rr_.mean()), H1=float((rk_ <= 1).mean()), H3=float((rk_ <= 3).mean()), H10=float((rk_ <= 10).mean()))
    chk["rows_in_learning_data_excluded"] = int(LD.sum()); chk["learning_rows_all_nan"] = bool(np.isnan(rr[LD]).all()) if LD.any() else None
    chk.update(rows_not_sorted_desc=bad_sorted, rows_with_duplicate_ids=bad_unique, rows_floor_above_last_top=floor_viol,
               gold_in_topk=int(gold_in), rows_official_rank_above_list_bound=pos_lt_rank, rows_gold_score_mismatch=score_mismatch,
               recomputed=rec, meta_match=all(abs(rec[k] - meta[k]) < 1e-9 for k in rec))
    chk["PASS"] = bool(chk["gold_eq_recomputed_harness"] and chk["gold_eq_harness_csr_sro"] and chk["keys_eq_harness_csr_sro"] and bad_sorted == 0
                       and bad_unique == 0 and floor_viol == 0 and pos_lt_rank == 0 and score_mismatch == 0 and chk["meta_match"])
    man["entries"].append(dict(method=method, dataset=ds, split=split, path=D, checks=chk))
    print(method, ds, split, json.dumps(chk), flush=True)
json.dump(man, open(f"{B}/MANIFEST.json", "w"), indent=1)
print("ALL_PASS", all(e["checks"]["PASS"] for e in man["entries"]), len(man["entries"]))
