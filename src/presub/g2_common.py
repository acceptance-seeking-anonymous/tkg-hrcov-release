"""presub / g2_common.py -- shared constants + helpers for G2 (HRCOV_PAT + recurrency controls on ICEWS18 / GDELT).

Dataset chosen by env G2_DS (ICEWS18 | GDELT); both are registered in the shared harness's oih_common.DATASETS
(RE-GCN split, raw timestamps with step_unit 24 / 15; pattern instantiation uses t_snap = ts // step, as the
retriever track).  Base rows = the retriever track's FULL float32 log-softmax score rows (tracks/retriever/fullscores,
rows in harness split order, verified in g2_features.py against the shared harness dumps).
Track dir P = $PC_P (default tracks/hardening/presub_g2/<DS>); nothing here writes into the shared harness.
"""
import os, sys, json, time
import numpy as np

H = os.environ.get("PC_HARNESS", os.path.expandvars("$WORK_DIR"))
TRACKS = os.environ.get("PC_TRACKS", f"{H}/tracks")
DS = os.environ.get("G2_DS", "ICEWS18")
assert DS in ("ICEWS18", "GDELT"), DS
P = os.environ.get("PC_P", f"{TRACKS}/hardening/presub_g2/{DS}")
RT = f"{TRACKS}/retriever/src"
RS = f"{TRACKS}/residual/code"
for _p in (H, RT, RS, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import oih_common                                    # noqa: E402
from oih_common import Splits, BaseDump, metrics_from_ranks, paired_bootstrap, jdump, filter_lookup  # noqa: E402,F401

CFG = oih_common.DATASETS[DS]
DD = CFG["dd"]
N_ENT, N_REL, STEP = int(CFG["n_ent"]), int(CFG["n_rel"]), int(CFG["step_unit"])
BASES = ("tirgn", "regcn", "logcl")
FS_DIR = os.environ.get("G2_FULLSCORES", f"{TRACKS}/retriever/fullscores")
FULLROWS = {b: {sp: f"{FS_DIR}/{b}_{DS}_{sp}_scores_f32.bin" for sp in ("valid", "test")} for b in BASES}
FULLMETA = {b: {sp: f"{FS_DIR}/{b}_{DS}_{sp}_meta.npz" for sp in ("valid", "test")} for b in BASES}
DUMPS = dict(CFG["dumps"])                           # tirgn (+ tirgn_valid for ICEWS18), regcn, logcl
# residual-track test MRRs (harness exact filtered, from the dumps) that the full rows must reproduce
RESIDUAL_MRR = {"ICEWS18": dict(tirgn=0.337006, logcl=0.287839, regcn=0.286905),
                "GDELT": dict(tirgn=0.216701, logcl=0.211389, regcn=0.197346)}[DS]
RESIDUAL_HR_TIRGN = {"ICEWS18": 0.3664, "GDELT": 0.2656}[DS]   # residual arm B (= HR) on TiRGN
WORK = f"{P}/work"

KC = 50            # history re-ranker pool = base top-50
BP = 50            # pattern-prior block (outside top-50)
W_PAT = KC + BP    # 100: [top50 | <=50 pattern-prior]
LP_LEN = 300       # pattern-prior list kept per query
KREF = 200
VALID_N = 200000   # ICEWS18: > #valid rows -> full valid; GDELT: 200k uniform sample (rng 12345)
SEEDS = [0, 1, 2]
COV6 = ["pat_best_util", "pat_best_te", "pat_log_npat", "pat_reach", "pat_log_pos", "is_cov_cand"]


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def split_queries(S, split):
    """sro [n,4] of one split in EXACT base_dump.py row order (pc_common.split_queries): per timestamp (sorted), the
    forward triples (file order, stable) then their inverses; the first TRAIN snapshot is not scored."""
    q = S.q[split]
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    ts = q[:, 3]; R = S.R
    out = []
    for k, u in enumerate(np.unique(ts)):
        if split == "train" and k == 0:
            continue
        m = q[ts == u]
        out.append(np.stack([m[:, 0], m[:, 1], m[:, 2], m[:, 3]], 1))
        out.append(np.stack([m[:, 2], m[:, 1] + R, m[:, 0], m[:, 3]], 1))
    return np.concatenate(out).astype(np.int64)


def fullrows(base, split):
    """memmap [n_split_rows, N_ENT] float32 log-softmax rows of the base, rows in split_queries order."""
    fn = FULLROWS[base][split]
    n = os.path.getsize(fn) // (4 * N_ENT)
    assert n * 4 * N_ENT == os.path.getsize(fn), fn
    return np.memmap(fn, dtype=np.float32, mode="r", shape=(n, N_ENT))


def exact_fused_ranks(s, cand, filt, babove, gold_pos, r_base, kc=KC):
    """identical to pc_common.exact_fused_ranks: rank of the gold in [C_promoted sorted by fused score] ++ base order."""
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
