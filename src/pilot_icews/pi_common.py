"""pilot_icews / pi_common.py -- shared constants + helpers for pilot 3 (TGB 2.0 tkgl-icews), STAGE A.

Stage A (redesign 2026-10-12): bases trained on a contiguous RECENT window of TRAIN (last 20% of train
timestamps); every feature / list / RecB reads the FULL history before t; test = uniform 10% query sample.
Registers dataset "TKGL_ICEWS" in the read-only shared harness's oih_common.DATASETS at import time only.
Also replaces rt_common.te_keys by an icews-safe version (the harness multiplier 2e7 is smaller than the
max 2-hop pattern id (2R*L)^2 + 2R*L = 29.97e6 for R = 391, L = 7 -> key collisions across relations).
"""
import os, sys, json, time
import numpy as np

H = os.environ.get("PI_HARNESS", os.path.expandvars("$WORK_DIR"))
TRACKS = os.environ.get("PI_TRACKS", f"{H}/tracks")
P = os.environ.get("PI_P", f"{TRACKS}/pilot_icews")
DD = os.environ.get("PI_DD", f"{TRACKS}/datasets/harness_data/tkgl-icews")
ZIP = os.environ.get("PI_ZIP", f"{TRACKS}/datasets/raw/tkgl-icews.zip")
RT = f"{TRACKS}/retriever/src"
RS = f"{TRACKS}/residual/code"
for _p in (H, RT, RS, f"{P}/code", os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import oih_common                                    # noqa: E402
from oih_common import Splits, BaseDump, metrics_from_ranks, paired_bootstrap, jdump  # noqa: E402,F401
import rt_common                                     # noqa: E402

DS = "TKGL_ICEWS"
TGB_NAME = "tkgl-icews"
N_ENT, N_REL = int(os.environ.get("PI_NENT", 87856)), int(os.environ.get("PI_NREL", 391))
STAGE = os.environ.get("PI_STAGE", "stageA")     # stageA (window bases, 10% test) | stageB (full train, full test)
WORK = f"{P}/work" if STAGE == "stageA" else f"{P}/work_{STAGE}"
BASES = ("regcn", "logcl", "regcnrn")            # regcnrn = RE-GCN trained on relation-normalised model inputs
DUMPS = {b: f"{P}/dumps/{b}_{STAGE}_scores.npz" for b in BASES}
CKPT = {b: f"{P}/dumps/{b}_{STAGE}_best.pt" for b in BASES}
oih_common.DATASETS[DS] = dict(
    dd=DD, n_ent=N_ENT, n_rel=N_REL, step_unit=1, dumps=dict(DUMPS),
    ref=dict(regcn=dict(MRR=0.182, src="TGB 2.0 leaderboard, RE-GCN (history 3, full train)"),
             cen=dict(MRR=0.187, src="TGB 2.0 leaderboard, CEN"),
             edgeremind=dict(MRR=0.269, src="TGB 2.0 leaderboard, EdgeReMIND (best published, as quoted in dataset_shortlist.md)"),
             recb_tgb=dict(MRR=0.211, src="TGB 2.0 leaderboard, RecurrencyBaseline (as quoted in dataset_shortlist.md)")))

# ---------------- Stage A design constants (fixed BEFORE any test number; mirrored in stageA_decision_rule.json)
WIN_FRAC = 0.20          # base-training window = last 20% of train timestamps
WIN_T0 = int(os.environ.get("PI_WIN_T0", 5749))  # first train day index of the window (train days 0..7186); 1,438 days, 2,266,194 facts
VALID_N = int(os.environ.get("PI_VALID_N", 200000))  # uniform valid sample (rng 12345) for re-ranker training / selection
VALID_SEED = 12345
TEST_FRAC = float(os.environ.get("PI_TEST_FRAC", 0.10))  # uniform test query sample, both directions, same for every arm
TEST_SEED = 31415
EPOCH_GRID = [int(x) for x in os.environ.get("PI_GRID", "1,2,3,4,6,8,10,12").split(",")]
EPOCH_EXT = [int(x) for x in os.environ.get("PI_GRID_EXT", "14,16,18,20").split(",") if x]   # only if the best grid epoch is the last grid point
KC = 50                  # history re-ranker pool = base top-50
B_GRID = [int(x) for x in os.environ.get("PI_BGRID", "50,100,150").split(",")]  # HRCOV history block (pattern-prior candidates outside top-50), valid-selected
W_MAX = KC + max(B_GRID) # 200 candidate columns stored
LP_LEN = int(os.environ.get("PI_LPLEN", 300))  # pattern-prior list length kept per query
KREF = 200               # dump top-K
SEEDS = [0, 1, 2]
COV_FEATS = ["pat_best_util", "pat_best_te", "pat_log_npat", "pat_reach", "pat_log_pos", "is_cov_cand"]
BASE_F = ["logp", "logp_minus_top1", "log_base_rank", "is_topKc", "q_p1", "q_entropy50", "q_margin12", "q_log_tailmass"]
EVID_F = ["ev_log_ntrip", "ev_log_ndyad", "ev_rec_trip", "ev_rec_dyad", "ev_log_nobjrel", "ev_log_nent", "ev_rec_objrel",
          "ev_log_ntrip_recent30", "ev_dyad_lr", "ev_q_log_nquery", "ev_q_log_nsubj"]
FEAT_NAMES = BASE_F + EVID_F + COV_FEATS
SCHEMES = ("orig", "rn")  # relation ids as shipped by TGB / relation-normalised (zero-stripped codes merged)
HOST = "serverA"
DEV = os.environ.get("PI_DEV", "cuda")


def _gpu_mem_cap():
    """optional per-process CUDA memory cap (env PI_GPU_MEM_GB) so a co-scheduled job fails itself instead of starving others."""
    gb = os.environ.get("PI_GPU_MEM_GB")
    if not gb:
        return
    import torch
    if torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(gb) * 2**30 / tot), 0)
        print(f"[gpu-cap] per-process CUDA memory cap {gb} GB of {tot / 2**30:.1f} GB", flush=True)


_gpu_mem_cap()



def fast_global_hist():
    """LogCL GlobalHist with a batched freq_vec (ONE index_put per batch instead of two H2D copies + one indexing kernel
    per query).  Same dict state, same update(); freq_vec returns the identical float32 [B, N] count matrix (indices are
    unique within a row, so assignment order is irrelevant).  Pure speed change; equivalence asserted by pi_ghcheck.py."""
    import torch
    from logcl import GlobalHist

    class FastGlobalHist(GlobalHist):
        def freq_vec(self, s_arr, r_arr, device):
            B = len(s_arr)
            vec = torch.zeros(B, self.num_ent, device=device)
            d = self.d; rows, cols, vals = [], [], []
            for i, key in enumerate(zip(np.asarray(s_arr).tolist(), np.asarray(r_arr).tolist())):
                m = d.get(key)
                if m:
                    k = len(m)
                    cols.append(np.fromiter(m.keys(), np.int64, k)); vals.append(np.fromiter(m.values(), np.float32, k))
                    rows.append(np.full(k, i, np.int64))
            if rows:
                ri = torch.from_numpy(np.concatenate(rows)).to(device); ci = torch.from_numpy(np.concatenate(cols)).to(device)
                vec[ri, ci] = torch.from_numpy(np.concatenate(vals)).to(device)
            return vec
    return FastGlobalHist


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


# ---------------- icews-safe target-encoding keys (replaces rt_common.te_keys at import time)
TE_MULT = np.int64(40_000_000)


def te_keys_icews(r, pid, hop, rel1, rel2, R):
    R2p = 2 * R + 1
    hb = (hop - 1).astype(np.int64)
    k_full = np.int64(r) * TE_MULT + pid
    k_rr = ((np.int64(r) * 2 + hb) * R2p + rel1) * R2p + (rel2 + 1)
    k_r2 = (np.int64(r) * 2 + hb) * R2p + np.where(hop == 2, rel2, rel1)
    return k_full, k_rr, k_r2, pid


rt_common.te_keys = te_keys_icews
assert (2 * N_REL * 7) ** 2 + 2 * N_REL * 7 < int(TE_MULT)


# ---------------- query order (identical to base_dump.py / pc_common.split_queries, vectorised)
def split_queries(S, split):
    """sro [n,4] of one split in EXACT base_dump.py row order: per timestamp (sorted), the forward triples
    (file order, stable) then their inverses; the first TRAIN snapshot is not scored."""
    q = S.q[split]
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    ts = q[:, 3]; R = S.R
    b = np.flatnonzero(np.diff(ts)) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(q)]])
    if split == "train":
        st, en = st[1:], en[1:]
    fwd = np.stack([q[:, 0], q[:, 1], q[:, 2], q[:, 3]], 1)
    inv = np.stack([q[:, 2], q[:, 1] + R, q[:, 0], q[:, 3]], 1)
    parts = []
    for a, e in zip(st, en):
        parts.append(fwd[a:e]); parts.append(inv[a:e])
    return np.concatenate(parts).astype(np.int64)


def n_queries(S, split):
    return 2 * len(S.q[split])


def valid_sample_idx(n_valid_queries):
    rng = np.random.default_rng(VALID_SEED)
    return np.sort(rng.choice(n_valid_queries, min(VALID_N, n_valid_queries), replace=False))


def test_sample_idx(n_test_queries):
    if TEST_FRAC >= 1.0:
        return np.arange(n_test_queries)               # Stage B: full test
    rng = np.random.default_rng(TEST_SEED)
    return np.sort(rng.choice(n_test_queries, int(round(TEST_FRAC * n_test_queries)), replace=False))


# ---------------- relation normalisation
def relmap2(path=None):
    """[2R] map r' -> r'_norm (inverse ids r+R map to norm(r)+R)."""
    path = path or f"{P}/stage/relmap.npy"
    m = np.load(path).astype(np.int64)
    assert len(m) == N_REL
    return np.concatenate([m, m + N_REL])


class SplitsView:
    """A Splits-like object whose quads carry mapped relation ids (for Hist / DyadReader / EventIndex);
    entity ids, timestamps, frequencies unchanged.  Filters / gold / evaluation always use the ORIGINAL Splits."""

    def __init__(self, S, relmap):
        self.dataset = S.dataset; self.cfg = S.cfg; self.n_ent = S.n_ent; self.R = S.R
        self.q = {}
        for k, v in S.q.items():
            w = v.copy(); w[:, 1] = relmap[w[:, 1]]; self.q[k] = w
        self.ts_range = S.ts_range; self.freq = S.freq


# ---------------- filter helpers (vectorised)
def filter_pairs(F, keys, golds):
    """For query keys (int64 fkey) and gold ids: (row, col) of OTHER true answers (time-aware filter)."""
    i = np.searchsorted(F["keys"], keys); i = np.clip(i, 0, len(F["keys"]) - 1)
    hit = F["keys"][i] == keys
    st = np.where(hit, F["start"][i], 0); en = np.where(hit, F["end"][i], 0)
    cnt = en - st
    rows = np.repeat(np.arange(len(keys)), cnt)
    if len(rows) == 0:
        return rows, rows
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cols = F["objs"][np.repeat(st, cnt) + offs]
    keep = cols != golds[rows]
    return rows[keep], cols[keep].astype(np.int64)


def exact_fused_ranks(s, cand, filt, babove, gold_pos, r_base, kc=KC):
    """Filtered rank of the gold in [C_promoted sorted by fused score] ++ base remainder (base order).
    C_promoted = first kc columns (base top-kc) UNION {c beyond kc : s(c) > min_{top-kc} s}.
    babove[n,W] = candidate's EXACT base log-prob > gold's -> exact ranks (pc_common.exact_fused_ranks)."""
    n, W = s.shape
    valid = cand >= 0
    s = np.where(valid, s, -np.inf)
    mint = s[:, :kc].min(1)
    jj = np.arange(W)[None, :]
    prom = valid & ((jj < kc) | (s > mint[:, None]))
    unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0)
    sg = s[ar, g]
    gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & (s > sg[:, None]); above[ar, g] = False
    r_in = 1 + above.sum(1)
    add = (unf & (jj >= kc) & ~babove).sum(1)
    r_out = r_base + add
    return np.where(gin, r_in, r_out).astype(np.int64)


def exact_fused_ranks_chunked(s, cand, filt, babove, gold_pos, r_base, kc=KC, chunk=200000):
    out = np.empty(len(s), np.int64)
    for lo in range(0, len(s), chunk):
        hi = min(len(s), lo + chunk)
        out[lo:hi] = exact_fused_ranks(np.asarray(s[lo:hi]), np.asarray(cand[lo:hi]), np.asarray(filt[lo:hi]),
                                       np.asarray(babove[lo:hi]), np.asarray(gold_pos[lo:hi]), np.asarray(r_base[lo:hi]), kc)
    return out


def rr_mean(ranks_list):
    return np.mean([1.0 / np.asarray(r, np.float64) for r in ranks_list], 0)
