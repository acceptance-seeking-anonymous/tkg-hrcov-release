"""
harness / bridge_features.py -- the GraphBridge G12 stage-1 candidate generator
(relation-expanded typed 2-hop bridge + degree-budgeted tight set + event-LM utility),
re-implemented with per-entity CSR adjacency so a query costs O(#paths) numpy work instead of
a Python triple loop.  It is a drop-in, FEATURE-IDENTICAL replacement for
stage2_ranker.build_query (G10-G12); `equivalence_check()` verifies that on real queries.

Graph = TRAIN split only (exactly as stage2_ranker.py).  For an inverse-direction query
(r' >= R) the graph relation is r'-R (adjacency is undirected, so this is the natural
extension; G10-G12 themselves only used forward queries).

Per-candidate feature row (16 = stage2 FEATS, same order):
  pm_top1 pm_subj n_interm idf_score r2_recur reached_both cand_logfreq interm_min_logfreq
  interm_mean_logfreq hop_min q_entropy q_top1 q_margin12 q_subj_logfreq q_rel_logfreq q_cset_logsize
plus casc[2] = (tight-set member flag, log1p(tight path count))  [G12 arm5]
plus lm[1]  = event-LM score_query utility (raw; arm4/arm5 multiply by reuse_gate)  [G10-G12]
plus hard[1] = hub-or-high-path-multiplicity flag (arm2 negatives)
"""
import os, sys, time
import numpy as np
from collections import defaultdict, Counter

FEATS = ["pm_top1", "pm_subj", "n_interm", "idf_score", "r2_recur", "reached_both",
         "cand_logfreq", "interm_min_logfreq", "interm_mean_logfreq", "hop_min",
         "q_entropy", "q_top1", "q_margin12", "q_subj_logfreq", "q_rel_logfreq", "q_cset_logsize"]
NF = len(FEATS)
REL_EXP_CAP = 12
MAX_CSET = 30000


class BridgeIndex:
    def __init__(self, train_quads, n_ent, n_rel, rel_exp_cap=REL_EXP_CAP, max_cset=MAX_CSET,
                 keep_python_graph=True):
        t0 = time.time()
        self.N = n_ent; self.R = n_rel; self.max_cset = max_cset
        train = [tuple(int(x) for x in q) for q in train_quads]
        freq = defaultdict(int)
        for (h, r, t, ts) in train:
            freq[h] += 1; freq[t] += 1
        # --- exact replica of stage2_ranker graph construction (order matters for set iteration) ---
        out_edges = defaultdict(list); adj_r = defaultdict(lambda: defaultdict(set))
        for (h, r, t, ts) in train:
            out_edges[h].append((t, r)); out_edges[t].append((h, r))
            adj_r[r][h].add(t); adj_r[r][t].add(h)
        deg = np.zeros(n_ent, np.int64)
        for u in out_edges:
            deg[u] = len(out_edges[u])
        self.DEG_CAP = int(np.percentile(deg[deg > 0], 95))
        self.is_hub = (deg > self.DEG_CAP)
        self.logfreq = np.log1p(np.array([freq.get(e, 0) for e in range(n_ent)], np.float32))
        self.relfreq = Counter(r for (_, r, _, _) in train)
        ent_rels = defaultdict(set)
        for (h, r, t, ts) in train:
            ent_rels[h].add(r); ent_rels[t].add(r)
        rel_cooc = defaultdict(Counter)
        for e, rs in ent_rels.items():
            rl = list(rs)
            for a in rl:
                for b in rl:
                    if a != b:
                        rel_cooc[a][b] += 1
        self.rel_expand = {}
        for r in self.relfreq:
            top = [rr for rr, _ in rel_cooc[r].most_common(rel_exp_cap - 1)]
            self.rel_expand[r] = [r] + top
        self.adj_r = adj_r            # python sets: iteration order reproduces stage2 exactly
        # --- CSR over out_edges: per m, unique (nb, rel) pairs in first-occurrence order + counts ---
        indptr = np.zeros(n_ent + 1, np.int64)
        nb_l = []; rl_l = []; ct_l = []
        uindptr = np.zeros(n_ent + 1, np.int64); unb_l = []      # unique neighbours (any rel), first-occ order
        tot_cnt = np.zeros(n_ent, np.int64)
        for m in range(n_ent):
            lst = out_edges.get(m)
            if lst:
                seen = {}; order = []
                useen = {}; uorder = []
                for (c, r2) in lst:
                    k = (c, r2)
                    if k in seen:
                        seen[k] += 1
                    else:
                        seen[k] = 1; order.append(k)
                    if c not in useen:
                        useen[c] = 1; uorder.append(c)
                nb_l.append(np.array([k[0] for k in order], np.int32))
                rl_l.append(np.array([k[1] for k in order], np.int32))
                ct_l.append(np.array([seen[k] for k in order], np.int32))
                unb_l.append(np.array(uorder, np.int32))
                indptr[m + 1] = indptr[m] + len(order)
                uindptr[m + 1] = uindptr[m] + len(uorder)
                tot_cnt[m] = len(lst)
            else:
                indptr[m + 1] = indptr[m]; uindptr[m + 1] = uindptr[m]
        self.indptr = indptr; self.nb = np.concatenate(nb_l); self.rl = np.concatenate(rl_l)
        self.ct = np.concatenate(ct_l)
        self.uindptr = uindptr; self.unb = np.concatenate(unb_l)
        self.deg = deg
        if not keep_python_graph:
            del out_edges
        self.build_s = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _gather(indptr, ms):
        """flat index array for CSR rows ms (in order) + the row id per element."""
        ms = np.asarray(ms, np.int64)
        if len(ms) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        st = indptr[ms]; ln = indptr[ms + 1] - st
        tot = int(ln.sum())
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        rep = np.repeat(np.arange(len(ms)), ln)
        off = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln)
        return st[rep] + off, ms[rep]

    def m1_set(self, a, rels):
        s = set()
        for fr in rels:
            s |= self.adj_r[fr].get(a, set())
        return s

    # ------------------------------------------------------------------ main
    def build(self, s, r_q, gold, topk_ids, topk_scores):
        """One query.  r_q in [0,2R).  Returns dict or None (no bridge path at all)."""
        R = self.R; N = self.N
        h = int(s); t = int(gold)
        r = int(r_q) if r_q < R else int(r_q) - R          # graph relation
        ids = np.asarray(topk_ids).astype(np.int64); scs = np.asarray(topk_scores).astype(np.float64)
        order = np.argsort(-scs)
        anchors = [int(ids[order][0]), h]
        p = np.exp(scs - scs.max()); p /= p.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum()); ss = np.sort(p)[::-1]
        qtop1 = float(ss[0]); qm12 = float(ss[0] - ss[1])
        first_rels = self.rel_expand.get(r, [r])
        msets = [self.m1_set(a, first_rels) for a in anchors]
        ms_l = [np.fromiter(ms_, np.int64, len(ms_)) for ms_ in msets]
        # ---- path multiplicities per anchor (all relations, with edge multiplicity) ----
        seq_parts = []; pm = []; r2 = np.zeros(N, np.int64)
        for ms in ms_l:
            fi, _ = self._gather(self.indptr, ms)
            nbv = self.nb[fi]; ctv = self.ct[fi]
            pm.append(np.bincount(nbv, weights=ctv, minlength=N).astype(np.int64))
            sel = self.rl[fi] == r
            r2 += np.bincount(nbv[sel], weights=ctv[sel], minlength=N).astype(np.int64)
            ui, _ = self._gather(self.uindptr, ms)
            seq_parts.append(self.unb[ui])
        seq = np.concatenate(seq_parts) if seq_parts else np.zeros(0, np.int64)
        if len(seq) == 0:
            return None
        u, first = np.unique(seq, return_index=True)
        cands = u[np.argsort(first, kind="stable")].astype(np.int64)     # dict-insertion order
        pm_top1 = pm[0][cands]; pm_subj = pm[1][cands]
        if len(cands) > self.max_cset:
            pmt = pm_top1 + pm_subj
            cands = cands[np.argsort(-pmt)[:self.max_cset]]
            pm_top1 = pm[0][cands]; pm_subj = pm[1][cands]
        # ---- distinct-intermediate statistics over the UNION of both anchors' m1 sets ----
        uni = np.fromiter(set(ms_l[0].tolist()) | set(ms_l[1].tolist()), np.int64)
        ui, mrow = self._gather(self.uindptr, uni)
        c_of = self.unb[ui].astype(np.int64); lf_m = self.logfreq[mrow].astype(np.float64)
        ni = np.bincount(c_of, minlength=N)
        idf = np.bincount(c_of, weights=1.0 / (1.0 + lf_m), minlength=N)
        isum = np.bincount(c_of, weights=lf_m, minlength=N)
        imin = np.full(N, 1e9); np.minimum.at(imin, c_of, lf_m)
        nic = ni[cands]
        X = np.zeros((len(cands), 16), np.float32)
        X[:, 0] = pm_top1; X[:, 1] = pm_subj; X[:, 2] = nic; X[:, 3] = idf[cands]
        X[:, 4] = r2[cands]; X[:, 5] = ((pm_top1 > 0) & (pm_subj > 0)).astype(np.float32)
        X[:, 6] = self.logfreq[cands]
        X[:, 7] = np.where(nic > 0, imin[cands], 0.0)
        X[:, 8] = np.where(nic > 0, isum[cands] / np.maximum(nic, 1), 0.0)
        X[:, 9] = 2.0
        X[:, 10] = ent; X[:, 11] = qtop1; X[:, 12] = qm12
        X[:, 13] = self.logfreq[h]; X[:, 14] = np.log1p(self.relfreq.get(r, 0))
        X[:, 15] = np.log1p(len(cands))
        pm_tot = pm_top1 + pm_subj
        hard = self.is_hub[cands] | (pm_tot >= np.percentile(pm_tot, 75) if len(pm_tot) > 4 else False)
        # ---- tight degree-budgeted set: exact relation r, non-hub intermediates (G12 arm5) ----
        tp = np.zeros(N, np.int64)
        for a in anchors:
            mt = [m for m in self.adj_r[r].get(a, set()) if not self.is_hub[m]]
            if mt:
                fi, _ = self._gather(self.indptr, mt)
                tp += np.bincount(self.nb[fi], weights=self.ct[fi], minlength=N).astype(np.int64)
        tpc = tp[cands]
        casc = np.stack([(tpc > 0).astype(np.float32), np.log1p(tpc).astype(np.float32)], 1)
        return dict(cands=cands.astype(np.int32), X=X, casc=casc, hard=np.asarray(hard, bool),
                    reached=bool(np.any(cands == t)), gold=t, tight_count=tpc.astype(np.int32))


class LMUtility:
    """Event-LM per-candidate utility exactly as stage2_ranker.lm_scores (G10-G12)."""

    def __init__(self, train_quads, ckpt, vocab, n_rel_train, device="cpu", lm_dir=None):
        import torch
        if lm_dir and lm_dir not in sys.path:
            sys.path.insert(0, lm_dir)
        from event_lm_score import EventLMScorer
        torch.set_num_threads(1)
        v = np.load(vocab, allow_pickle=True)
        self.ent_role = v["ent_role"].astype(np.int64); self.ent_country = v["ent_country"].astype(np.int64)
        self.role_count = np.bincount(self.ent_role, minlength=int(self.ent_role.max()) + 1)
        self.scorer = EventLMScorer(ckpt, self.ent_role, self.ent_country, num_rel=n_rel_train, device=device)
        train = [tuple(int(x) for x in q) for q in train_quads]
        hist_by_ent = defaultdict(list)
        for (h, r, t, ts) in sorted(train, key=lambda x: -x[3]):
            if len(hist_by_ent[h]) < 40: hist_by_ent[h].append((r, t, ts))
            if len(hist_by_ent[t]) < 40: hist_by_ent[t].append((r, h, ts))
        self.hist_by_ent = hist_by_ent

    def reuse_gate(self, cands):
        rc = self.role_count[self.ent_role[np.asarray(cands, dtype=np.int64)]].astype(np.float64)
        return (1.0 - 1.0 / np.maximum(rc, 1.0)).astype(np.float32)

    def scores(self, s, r_q, ts, cands):
        raw = self.hist_by_ent.get(int(s), [])
        hist = [(rr, nb, max(0, (int(ts) - tt) // 24)) for (rr, nb, tt) in raw if tt < ts][:20]
        if not hist or len(cands) == 0:
            return np.zeros(len(cands), np.float32), False
        try:
            d = self.scorer.score_query(hist, int(r_q), [int(c) for c in cands])
        except Exception:
            return np.zeros(len(cands), np.float32), False
        return np.array([d.get(int(c), 0.0) for c in cands], np.float32), True


# --------------------------------------------------------------------------- reference
def reference_build_query(i, sro, tki, tks, train, N_ENT, cache={}):
    """VERBATIM logic of stage2_ranker.build_query (G12 file md5 e64bd27a.../1adfa274...) minus LM,
    used only by equivalence_check.  Builds its own graph once (cached)."""
    if "g" not in cache:
        freq = defaultdict(int)
        for (h, r, t, ts) in train: freq[h] += 1; freq[t] += 1
        out_edges = defaultdict(list); adj_r = defaultdict(lambda: defaultdict(set))
        for (h, r, t, ts) in train:
            out_edges[h].append((t, r)); out_edges[t].append((h, r))
            adj_r[r][h].add(t); adj_r[r][t].add(h)
        deg = np.zeros(N_ENT, np.int64)
        for u in out_edges: deg[u] = len(out_edges[u])
        DEG_CAP = int(np.percentile(deg[deg > 0], 95)); is_hub = (deg > DEG_CAP)
        logfreq = np.log1p(np.array([freq.get(e, 0) for e in range(N_ENT)], np.float32))
        relfreq = Counter(r for (_, r, _, _) in train)
        ent_rels = defaultdict(set)
        for (h, r, t, ts) in train: ent_rels[h].add(r); ent_rels[t].add(r)
        rel_cooc = defaultdict(Counter)
        for e, rs in ent_rels.items():
            rl = list(rs)
            for a in rl:
                for b in rl:
                    if a != b: rel_cooc[a][b] += 1
        rel_expand = {}
        for r in relfreq:
            top = [rr for rr, _ in rel_cooc[r].most_common(REL_EXP_CAP - 1)]
            rel_expand[r] = [r] + top
        cache["g"] = (out_edges, adj_r, is_hub, logfreq, relfreq, rel_expand)
    out_edges, adj_r, is_hub, logfreq, relfreq, rel_expand = cache["g"]
    NF_ = 16
    h, r, t = int(sro[i, 0]), int(sro[i, 1]), int(sro[i, 2])
    ids = tki[i].astype(np.int64); scs = tks[i].astype(np.float64); order = np.argsort(-scs)
    anchors = [int(ids[order][0]), h]
    p = np.exp(scs - scs.max()); p /= p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum()); ss = np.sort(p)[::-1]
    qtop1 = float(ss[0]); qm12 = float(ss[0] - ss[1])
    first_rels = rel_expand[r]
    agg = {}
    def add_path(c, r2, m, is_top1, hop):
        dd = agg.get(c)
        if dd is None:
            dd = {"pm_top1": 0, "pm_subj": 0, "interm": set(), "idf": 0.0, "r2_recur": 0,
                  "imin": 1e9, "isum": 0.0, "hopmin": hop}; agg[c] = dd
        if is_top1: dd["pm_top1"] += 1
        else: dd["pm_subj"] += 1
        if m not in dd["interm"]:
            lf = logfreq[m]; dd["interm"].add(m); dd["idf"] += 1.0 / (1.0 + lf)
            dd["imin"] = min(dd["imin"], lf); dd["isum"] += lf
        if r2 == r: dd["r2_recur"] += 1
        dd["hopmin"] = min(dd["hopmin"], hop)
    for a_idx, a in enumerate(anchors):
        is_top1 = (a_idx == 0); m1_set = set()
        for fr in first_rels: m1_set |= adj_r[fr].get(a, set())
        for m1 in m1_set:
            for (c, r2) in out_edges.get(m1, ()): add_path(c, r2, m1, is_top1, 2)
    if not agg: return None
    cands = np.array(list(agg.keys()), dtype=np.int64)
    if len(cands) > MAX_CSET:
        pm = np.array([agg[c]["pm_top1"] + agg[c]["pm_subj"] for c in cands])
        cands = cands[np.argsort(-pm)[:MAX_CSET]]
    X = np.zeros((len(cands), NF_), np.float32)
    csl = np.log1p(len(cands)); qsl = logfreq[h]; qrl = np.log1p(relfreq.get(r, 0))
    for j, c in enumerate(cands):
        dd = agg[c]; ni = len(dd["interm"])
        X[j] = [dd["pm_top1"], dd["pm_subj"], ni, dd["idf"], dd["r2_recur"],
                1.0 if (dd["pm_top1"] > 0 and dd["pm_subj"] > 0) else 0.0,
                logfreq[c], dd["imin"] if ni else 0.0, (dd["isum"] / ni) if ni else 0.0,
                float(dd["hopmin"]), ent, qtop1, qm12, qsl, qrl, csl]
    tight_pm = defaultdict(int)
    for a_idx, a in enumerate(anchors):
        for m1 in adj_r[r].get(a, set()):
            if is_hub[m1]: continue
            for (c, r2) in out_edges.get(m1, ()): tight_pm[c] += 1
    casc = np.zeros((len(cands), 2), np.float32)
    for j, c in enumerate(cands):
        tp = tight_pm.get(int(c), 0)
        casc[j] = [1.0 if tp > 0 else 0.0, np.log1p(tp)]
    return cands, X, (t in set(cands.tolist())), t, casc


def equivalence_check(bi, sro, tki, tks, train, rows, N_ENT):
    """Compare BridgeIndex.build against the verbatim reference on `rows`. Returns summary dict."""
    res = dict(n=0, none_agree=0, cand_order_exact=0, cand_set_exact=0, X_maxabs=0.0, X_int_exact=0,
               casc_maxabs=0.0, reached_agree=0, t_ref_s=0.0, t_new_s=0.0)
    for i in rows:
        t0 = time.time(); ref = reference_build_query(i, sro, tki, tks, train, N_ENT); t1 = time.time()
        new = bi.build(sro[i, 0], sro[i, 1], sro[i, 2], tki[i], tks[i]); t2 = time.time()
        res["t_ref_s"] += t1 - t0; res["t_new_s"] += t2 - t1; res["n"] += 1
        if ref is None or new is None:
            res["none_agree"] += int(ref is None and new is None); continue
        res["none_agree"] += 1
        cr, Xr, rr, _, car = ref
        cn = new["cands"].astype(np.int64)
        res["cand_order_exact"] += int(len(cr) == len(cn) and np.array_equal(cr, cn))
        res["cand_set_exact"] += int(set(cr.tolist()) == set(cn.tolist()))
        res["reached_agree"] += int(rr == new["reached"])
        # align by candidate id
        pos = {int(c): j for j, c in enumerate(cn)}
        if set(cr.tolist()) == set(cn.tolist()):
            idx = np.array([pos[int(c)] for c in cr])
            Xn = new["X"][idx]; can = new["casc"][idx]
            res["X_maxabs"] = max(res["X_maxabs"], float(np.max(np.abs(Xn - Xr))))
            res["X_int_exact"] += int(np.array_equal(Xn[:, [0, 1, 2, 4, 5, 9, 15]], Xr[:, [0, 1, 2, 4, 5, 9, 15]]))
            res["casc_maxabs"] = max(res["casc_maxabs"], float(np.max(np.abs(can - car))))
    res["t_ref_s"] = round(res["t_ref_s"], 2); res["t_new_s"] = round(res["t_new_s"], 2)
    return res
