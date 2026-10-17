"""
harness / oih_common.py  -- shared data layer for the three open-idea tracks.

Everything here is CPU/numpy only (runs on the bare host python3 or inside tkg-gpu).

Conventions (read harness_README.md for the full contract):
  * A "query" is one row of a frozen-base score dump: (s, r', o, ts) with r' in [0, 2R).
    r' < R  : forward / tail prediction  (s, r, ?)   gold o
    r' >= R : inverse / head prediction  (o_orig, r'-R, ?) written as (s=o_orig, r', o=s_orig)
    This is exactly how TiRGN / RE-GCN / LogCL score both directions.
  * Splits are identified by TIMESTAMP RANGE (extrapolation splits are time-disjoint),
    which is robust to dumps that were concatenated in any order.
  * Time-aware filtered rank: gold vs all entities, removing other true answers of the
    same (s, r', ts).  From a top-K dump this is EXACT when raw_rank <= K (every entity
    above the gold is visible); for raw_rank > K we subtract only the filter entities we
    can see (conservative, = the project's tirgn_alone_eval convention) and also report an
    optimistic bound that subtracts every filter entity.
"""
import os, json, zipfile, time
import numpy as np
from collections import defaultdict

SCR = os.path.expandvars("$EXT_DIR")
HARNESS = os.path.expandvars("$WORK_DIR")   # harness root (the working tree of all runs)

DATASETS = {
    "ICEWS18": dict(
        dd=f"{SCR}/logcl_redun/data/ICEWS18", n_ent=23033, n_rel=256, step_unit=24,
        dumps={
            "tirgn": f"{SCR}/logcl_redun/tirgn_scores_ICEWS18_traintest.npz",
            # valid rows for TiRGN ICEWS18 live in a separate dump (job)
            "tirgn_valid": f"{SCR}/runs/tirgn_scores_ICEWS18_valid.npz",
            "regcn": f"{HARNESS}/dumps/regcn_scores_ICEWS18_traintest.npz",
            "logcl": f"{HARNESS}/dumps/logcl_scores_ICEWS18_traintest.npz",
        },
        lm_ckpt=f"{SCR}/logcl_redun/event_lm_dropON.pt",
        lm_vocab=f"{SCR}/logcl_redun/event_lm_vocab.npz",
        # reference full-test (both directions, time-aware filtered) MRRs reproduced earlier
        ref=dict(tirgn=dict(MRR=0.337008, H1=0.2316, H3=0.3817, H10=0.5425, n=99090,
                            src="tirgn_run_log.md PHASE 1 (released ckpt, pure-torch shim, both-dir)"),
                 # Task-2 headline numbers (0.2827 / 0.2842) used a head-direction filter bug; the
                 # direction-fixed per-query re-scoring (regcn/logcl_i18_dirfix_perquery.csv) is the
                 # like-for-like reference for the exact filtered ranks written by base_dump.py.
                 regcn=dict(MRR=0.2853, H1=0.1911, H10=0.4714, raw_MRR=0.2721, n=99090, tol=0.005,
                            src="regcn_i18_dirfix_perquery.csv fixed_filt_rank (Task-2 headline 0.2827 was pre-fix)"),
                 logcl=dict(MRR=0.2888, H1=0.1964, H10=0.4704, raw_MRR=0.2751, n=99090, tol=0.005,
                            src="logcl_i18_dirfix_perquery.csv fixed_filt_rank (Task-2 headline 0.2842 was pre-fix)")),
    ),
    "GDELT": dict(
        dd=f"{SCR}/gdelt_armAD/data/GDELT", n_ent=7691, n_rel=240, step_unit=15,
        dumps={
            "tirgn": f"{SCR}/gdelt_tirgn_dump/tirgn_scores_GDELT_traintest.npz",
            "regcn": f"{HARNESS}/dumps/regcn_scores_GDELT_traintest.npz",
            "logcl": f"{HARNESS}/dumps/logcl_scores_GDELT_traintest.npz",
        },
        lm_ckpt=f"{SCR}/gdelt_armAD/event_lm_dropON.pt",
        lm_vocab=f"{SCR}/gdelt_armAD/event_lm_vocab.npz",
        ref=dict(tirgn=dict(MRR=0.216701, H1=0.136289, H3=0.232739, H10=0.376031, n=610482,
                            src="gdelt_tirgn_anchor.json (his7/angle10 ckpt, filter_ent)"),
                 # no direction-fixed GDELT re-scoring exists; Task-2 time-aware numbers carry the head-filter
                 # bug (small on GDELT), so the RAW MRR is the clean like-for-like check.
                 regcn=dict(MRR=0.1970, H1=0.1247, H10=0.3381, raw_MRR=0.19310, n=610482, tol=0.005,
                            src="task2_ecckg_relation_prior_ablation.csv REGCN base; raw from job metrics"),
                 logcl=dict(MRR=0.2105, H1=0.1327, H10=0.3630, n=610482, tol=0.005,
                            src="task2_ecckg_relation_prior_ablation.csv LOGCL base (pre-fix filter)")),
    ),
}

SPLIT_CODE = {"train": 0, "valid": 1, "test": 2}


# ----------------------------------------------------------------------------- io
def load_quads(fp):
    """(h, r, t, ts) int64 array from a whitespace/tab separated split file."""
    out = []
    with open(fp) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 4:
                out.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
    return np.array(out, dtype=np.int64)


def npz_keys(path):
    with zipfile.ZipFile(path) as z:
        return [n[:-4] for n in z.namelist()]


class Splits:
    """Train/valid/test quads + timestamp ranges + gold frequency + co-occurrence."""

    def __init__(self, dataset):
        cfg = DATASETS[dataset]
        self.dataset = dataset
        self.cfg = cfg
        self.n_ent = cfg["n_ent"]; self.R = cfg["n_rel"]
        self.q = {nm: load_quads(f"{cfg['dd']}/{nm}.txt") for nm in ("train", "valid", "test")}
        self.ts_range = {nm: (int(a[:, 3].min()), int(a[:, 3].max())) for nm, a in self.q.items()}
        tr = self.q["train"]
        # entity train frequency (h and t both count) -- identical to stage2_ranker / twoaxis
        self.freq = np.bincount(tr[:, 0], minlength=self.n_ent) + np.bincount(tr[:, 2], minlength=self.n_ent)
        self._cooc = None

    def split_of_ts(self, ts):
        ts = np.asarray(ts)
        out = np.full(ts.shape, -1, np.int8)
        for nm, (a, b) in self.ts_range.items():
            out[(ts >= a) & (ts <= b)] = SPLIT_CODE[nm]
        return out

    @staticmethod
    def stratum(freq):
        f = np.asarray(freq)
        return np.where(f < 50, 0, np.where(f < 500, 1, 2)).astype(np.int8)   # 0 cold 1 mid 2 head

    def cooc_pairs(self):
        """Set of (min,max) entity pairs that co-occur in train+valid (evidence-support axis).
        Encoded as int64 a*N+b for vectorised membership tests."""
        if self._cooc is None:
            tv = np.concatenate([self.q["train"], self.q["valid"]])
            a = np.minimum(tv[:, 0], tv[:, 2]); b = np.maximum(tv[:, 0], tv[:, 2])
            self._cooc = np.unique(a * self.n_ent + b)
        return self._cooc

    def in_hist(self, s, o):
        a = np.minimum(s, o).astype(np.int64); b = np.maximum(s, o).astype(np.int64)
        key = a * self.n_ent + b
        c = self.cooc_pairs()
        pos = np.searchsorted(c, key)
        pos = np.clip(pos, 0, len(c) - 1)
        return (c[pos] == key)

    def filter_index(self, which=("train", "valid", "test")):
        """Time-aware filter: (s, r', ts) -> sorted array of true answers, both directions.
        Returned as a dict keyed by int64 code (see fkey)."""
        quads = np.concatenate([self.q[w] for w in which])
        h, r, t, ts = quads.T
        s_all = np.concatenate([h, t]); r_all = np.concatenate([r, r + self.R])
        o_all = np.concatenate([t, h]); ts_all = np.concatenate([ts, ts])
        key = self.fkey(s_all, r_all, ts_all)
        order = np.lexsort((o_all, key))
        key = key[order]; o_all = o_all[order]
        uk, start = np.unique(key, return_index=True)
        end = np.append(start[1:], len(key))
        return dict(keys=uk, start=start, end=end, objs=o_all)

    def fkey(self, s, r, ts):
        s = np.asarray(s, np.int64); r = np.asarray(r, np.int64); ts = np.asarray(ts, np.int64)
        return (ts * (2 * self.R) + r) * self.n_ent + s


def filter_lookup(F, key):
    """Return the true-answer array for one int64 filter key (empty if none)."""
    i = np.searchsorted(F["keys"], key)
    if i < len(F["keys"]) and F["keys"][i] == key:
        return F["objs"][F["start"][i]:F["end"][i]]
    return F["objs"][:0]


# ----------------------------------------------------------------------------- dumps
class BaseDump:
    """Frozen-base score dump in the shared schema:
         sro[N,4] int32 (s, r', o, ts) | gold_score[N] f32 | gold_rank_raw[N] int32
         topk_ids[N,K] int32 | topk_scores[N,K] f32 (descending)
       optional: split_bounds[3], split[N] int8, gold_rank_filt[N] int32 (exact, from full scores)
    Loads lazily-memory: arrays are materialised on first use (np.load of an .npz decompresses).
    """

    def __init__(self, path, splits=None, keys=("sro", "gold_score", "gold_rank_raw", "topk_ids", "topk_scores")):
        t0 = time.time()
        self.path = path
        d = np.load(path, allow_pickle=False)
        avail = set(d.files)
        for k in keys:
            setattr(self, k, d[k])
        self.extra = {k: d[k] for k in ("gold_rank_filt", "split", "split_bounds") if k in avail}
        self.N = self.sro.shape[0]; self.K = self.topk_ids.shape[1]
        self.splits = splits
        self.split = (self.extra["split"] if "split" in self.extra
                      else (splits.split_of_ts(self.sro[:, 3]) if splits is not None else None))
        self.load_s = round(time.time() - t0, 1)

    def rows(self, split_name, direction="both"):
        code = SPLIT_CODE[split_name]
        m = self.split == code
        if direction == "forward":
            m &= self.sro[:, 1] < self.splits.R
        elif direction == "inverse":
            m &= self.sro[:, 1] >= self.splits.R
        return np.where(m)[0]

    def top1_prob(self, idx):
        """Gold-free base confidence: exp(top-1 log-prob) (scores are log-softmax)."""
        return np.exp(self.topk_scores[idx, 0].astype(np.float64))

    def filtered_ranks(self, idx, F):
        """Time-aware filtered rank for rows idx.  Returns (rank_conservative, rank_optimistic, exact_flag).
        If the dump carries an exact gold_rank_filt (new RE-GCN/LogCL dumps) it is returned for both."""
        if "gold_rank_filt" in self.extra:
            r = self.extra["gold_rank_filt"][idx].astype(np.int64)
            return r, r, np.ones(len(idx), bool)
        R = self.splits.R
        out_c = np.empty(len(idx), np.int64); out_o = np.empty(len(idx), np.int64); ex = np.empty(len(idx), bool)
        sro = self.sro; tki = self.topk_ids; tks = self.topk_scores; gs_all = self.gold_score; raw_all = self.gold_rank_raw
        keys = self.splits.fkey(sro[idx, 0], sro[idx, 1], sro[idx, 3])
        for n, (i, key) in enumerate(zip(idx, keys)):
            raw = int(raw_all[i]); g = int(sro[i, 2])
            fs = filter_lookup(F, key)
            if len(fs) <= 1:            # only the gold itself (or nothing)
                out_c[n] = out_o[n] = raw; ex[n] = True; continue
            others = fs[fs != g]
            gs = gs_all[i]
            ids = tki[i]; sc = tks[i]
            vis = np.isin(ids, others)
            corr = int(np.sum(vis & (sc > gs)))
            n_vis = int(vis.sum())
            out_c[n] = max(1, raw - corr)
            out_o[n] = max(1, raw - corr - (len(others) - n_vis)) if raw > self.K else out_c[n]
            ex[n] = (raw <= self.K) or (n_vis == len(others))
        return out_c, out_o, ex


# ----------------------------------------------------------------------------- metrics
def metrics_from_ranks(rank, ks=(1, 3, 10)):
    r = np.asarray(rank, np.float64)
    out = {"MRR": round(float(np.mean(1.0 / r)), 6) if len(r) else None, "n": int(len(r))}
    for k in ks:
        out[f"H{k}"] = round(float(np.mean(r <= k)), 6) if len(r) else None
    return out


def paired_bootstrap(rr_a, rr_b, B=2000, seed=0, chunk=200):
    """CI of mean(rr_b - rr_a) by resampling queries (paired). Returns dict."""
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64)
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    nz = np.nonzero(d)[0]
    d0 = float(d.mean())
    if len(nz) == 0:
        return {"delta": 0.0, "CI95": [0.0, 0.0], "p_delta_gt0": 0.0, "excludes_zero": False, "n": n, "n_changed": 0}
    # resample: only nonzero deltas affect the numerator; draw how many times each nz query is hit.
    # Equivalent to full multinomial resampling of n queries.
    boots = np.empty(B)
    dn = d[nz]
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        # counts for nonzero queries ~ Multinomial(n, 1/n each) restricted -> Binomial per query is
        # a close approximation for n >> 1 (Poisson(1) bootstrap); use Poisson weights normalised by n.
        w = rng.poisson(1.0, size=(bb, len(nz)))
        boots[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": round(d0, 6), "CI95": [round(float(lo), 6), round(float(hi), 6)],
            "p_delta_gt0": round(float((boots > 0).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(n), "n_changed": int(len(nz)),
            "method": "Poisson(1) paired bootstrap over queries, B=%d" % B}


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
