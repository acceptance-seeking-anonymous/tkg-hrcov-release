"""pilot_polecat / pc_common.py -- shared constants + helpers for the tkgl-polecat coverage-stack pilot.

Registers dataset "POLECAT" in the (read-only) shared harness's oih_common.DATASETS at import time, so
every harness module (Splits, BaseDump, filter, metrics, paired_bootstrap) works unchanged.
Nothing here writes into the shared harness or other track dirs.
"""
import os, sys, json, time
import numpy as np

H = os.environ.get("PC_HARNESS", os.path.expandvars("$WORK_DIR"))
TRACKS = os.environ.get("PC_TRACKS", f"{H}/tracks")
P = os.environ.get("PC_P", f"{TRACKS}/pilot_polecat")
DD = os.environ.get("PC_DD", f"{TRACKS}/datasets/harness_data/tkgl-polecat")
RT = f"{TRACKS}/retriever/src"
for _p in (H, RT, f"{P}/code", os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import oih_common                                    # noqa: E402
from oih_common import Splits, BaseDump, metrics_from_ranks, paired_bootstrap, jdump, filter_lookup  # noqa: E402,F401

DS = "POLECAT"
N_ENT, N_REL = 150931, 16
DUMPS = {"regcn": f"{P}/dumps/regcn_scores_POLECAT_traintest.npz",
         "logcl": f"{P}/dumps/logcl_scores_POLECAT_traintest.npz"}
oih_common.DATASETS[DS] = dict(
    dd=DD, n_ent=N_ENT, n_rel=N_REL, step_unit=1, dumps=dict(DUMPS),
    ref=dict(regcn=dict(MRR=0.175, src="TGB 2.0 leaderboard, RE-GCN (DGL, history 3)"),
             cen=dict(MRR=0.184, src="TGB 2.0 leaderboard, CEN")))
CKPT = {"regcn": f"{P}/dumps/regcn_POLECAT_seed42.pt", "logcl": f"{P}/dumps/logcl_POLECAT_seed42.pt"}
WORK = os.environ.get("PC_WORK", f"{P}/work")   # PC_WORK=$P/work_ts for the time-safe bridge rerun

KC = 50            # history re-ranker pool = base top-50
BP = 50            # pattern-prior candidates added to C_cov (outside top-50)
BB = 50            # stage-2 bridge candidates added to C_cov (outside top-50 and the pattern block)
W_COV = KC + BP + BB
LP_LEN = 300       # pattern-prior list kept per query (bump arm B grid <= 300)
LB_LEN = 200       # bridge list kept per query (ConfidenceSwitch B grid <= 200)
D_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
B_PAT = [10, 30, 100, 300]
VALID_N = 200000
SEEDS = [0, 1, 2]
COV_FEATS = ["pat_best_util", "pat_best_te", "pat_log_npat", "pat_reach", "pat_log_pos",
             "br_in", "br_score", "br_log_pos", "br_log_pm_subj", "br_log_ninterm", "is_cov_cand"]


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def split_queries(S, split):
    """sro [n,4] of one split in EXACT base_dump.py row order: per timestamp (sorted), the forward
    triples (file order, stable) then their inverses; the first TRAIN snapshot is not scored."""
    q = S.q[split]
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    ts = q[:, 3]; R = S.R
    out = []
    uts = np.unique(ts)
    for k, u in enumerate(uts):
        if split == "train" and k == 0:
            continue
        m = q[ts == u]
        fwd = np.stack([m[:, 0], m[:, 1], m[:, 2], m[:, 3]], 1)
        inv = np.stack([m[:, 2], m[:, 1] + R, m[:, 0], m[:, 3]], 1)
        out.append(fwd); out.append(inv)
    return np.concatenate(out).astype(np.int64)


def profiler_to_split_order(S, split):
    """profile_tkg_v2 enumerates a split's queries as [all forward facts in file order, all inverse facts];
    return perm such that split_order_array = profiler_array[perm]."""
    q = S.q[split]; n = len(q)
    order = np.argsort(q[:, 3], kind="stable")
    assert (order == np.arange(n)).all(), "split file not time-sorted"
    ts = q[:, 3]
    perm = []
    for u in np.unique(ts):
        idx = np.flatnonzero(ts == u)
        perm.append(idx); perm.append(idx + n)
    return np.concatenate(perm)


def exact_fused_ranks(s, cand, filt, babove, gold_pos, r_base, kc=KC):
    """Filtered rank of the gold in [C_promoted sorted by fused score] ++ base remainder (base order).
    C_promoted = first kc columns (base top-kc) UNION {c beyond kc : s(c) > min_{top-kc} s}.
    babove[n,W] = candidate's EXACT base log-prob > gold's (full-row lookup) -> exact ranks.
    Identical to tracks/residual rs_features.fused_ranks_np, with exact babove."""
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


def bump_ranks_exact(L_ids, L_lp, L_filt, gold, gold_lp, r_base, ck, B_grid=B_PAT, D=D_GRID):
    """Exact filtered rank under s' = s + delta*1[c in top-B of list L] (retriever-track combiner).
    L_ids [n,Lmax] (-1 pad, gold may be inside), L_lp exact base log-probs, L_filt = other true answers,
    gold_lp [n], r_base [n] exact filtered base rank, ck [n,len(D)] = #non-filter entities (excl. gold)
    with log-prob > gold_lp + delta.  Returns ranks [n, len(B), len(D)].  Vectorised over queries.
    gold in top-B : 1 + #(top-B non-filter, non-gold, lp > lp_g) + #(non-B non-filter with lp > lp_g + d)
                    where #(non-B ...) = ck[d] - #(top-B non-filter non-gold with lp > lp_g + d)
    gold not in B : 1 + (r_base-1) - #(B with lp > lp_g) + #(B with lp > lp_g - d)   (B = non-filter, non-gold)"""
    n, Lm = L_ids.shape
    D = np.asarray(D, np.float64)
    valid = (L_ids >= 0) & ~L_filt
    isg = (L_ids == gold[:, None]) & (L_ids >= 0)
    ng = valid & ~isg
    gl = gold_lp.astype(np.float64)[:, None]
    lp = np.where(L_ids >= 0, L_lp.astype(np.float64), -np.inf)
    gpos = np.where(isg.any(1), isg.argmax(1), Lm + 10)
    a0 = np.concatenate([np.zeros((n, 1), np.int64), np.cumsum(ng & (lp > gl), 1)], 1)       # [n, Lm+1]
    out = np.zeros((n, len(B_grid), len(D)), np.int64)
    for bi, B in enumerate(B_grid):
        k = min(B, Lm)
        gin = gpos < k
        nb0 = a0[:, k]
        for di, d in enumerate(D):
            ap = (ng[:, :k] & (lp[:, :k] > gl + d)).sum(1)
            am = (ng[:, :k] & (lp[:, :k] > gl - d)).sum(1)
            r_in = 1 + nb0 + (ck[:, di] - ap)
            r_out = r_base - nb0 + am
            out[:, bi, di] = np.where(gin, r_in, r_out)
    return out


def rr_mean(ranks_list):
    return np.mean([1.0 / np.asarray(r, np.float64) for r in ranks_list], 0)
