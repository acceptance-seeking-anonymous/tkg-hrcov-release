"""phase2_fresh_window / fh_common.py -- shared constants + helpers for the GDELTFW forward holdout of HRCOV_PAT.

Derived from presub_g2/code/g2_common.py (G2 of a_presub_rule.json) with the following changes ONLY:
  * dataset GDELTFW (post-cutoff GDELT 2.0 slice of the status pilot: train 2025-01..06, valid 2025-07, test 2025-08..09,
    daily snapshots, ts ids 0..255, step_unit 1) is registered into oih_common.DATASETS at import time from THIS track's
    copy of the data; the shared harness and the status pilot's files are never modified.
  * one base (tirgn, trained on the GDELTFW train window by hd_tirgn.py); full float32 log-prob rows written by
    fh_tirgn_rows.py into this track (work/fullrows).
  * every module this pipeline imports is a byte-identical copy in this code dir (oih_common, fulltest_eval, patterns,
    rt_common, rs_features); module_provenance() records the file each one was loaded from.
Track dir P = $PC_P (default tracks/phase2_fresh_window/GDELTFW).
"""
import os, sys, json, time
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
H = os.environ.get("PC_HARNESS", "/home/anon/cs_scratch/open_ideas_harness")
FH = os.environ.get("FH_DIR", f"{H}/tracks/phase2_fresh_window")
DS = os.environ.get("G2_DS", "GDELTFW")
assert DS == "GDELTFW", DS
P = os.environ.get("PC_P", f"{FH}/{DS}")
DD = os.environ.get("FH_DD", f"{FH}/data/GDELTFW")
WORK = f"{P}/work"
FS_DIR = os.environ.get("G2_FULLSCORES", f"{WORK}/fullrows")
BASES = ("tirgn",)
# harness-schema dumps written by fh_tirgn_rows.py, one per split (g2_features.dump_ranks uses "<base>_valid" for valid)
TIRGN_DUMPS = {"tirgn_valid": f"{WORK}/base_tirgn/tirgn_scores_GDELTFW_valid.npz", "tirgn": f"{WORK}/base_tirgn/tirgn_scores_GDELTFW_test.npz"}

import oih_common                                    # noqa: E402  (this dir's copy)
import fulltest_eval                                 # noqa: E402,F401  (pre-load this dir's copy before rs_features inserts H)
oih_common.DATASETS["GDELTFW"] = dict(dd=DD, n_ent=1209, n_rel=215, step_unit=1, dumps=dict(TIRGN_DUMPS),
                                       ref={}, lm_ckpt=None, lm_vocab=None)
from oih_common import Splits, BaseDump, metrics_from_ranks, paired_bootstrap, jdump, filter_lookup  # noqa: E402,F401

CFG = oih_common.DATASETS[DS]
N_ENT, N_REL, STEP = int(CFG["n_ent"]), int(CFG["n_rel"]), int(CFG["step_unit"])
FULLROWS = {b: {sp: f"{FS_DIR}/{b}_{DS}_{sp}_scores_f32.bin" for sp in ("valid", "test")} for b in BASES}
DUMPS = dict(CFG["dumps"])

KC = 50            # history re-ranker pool = base top-50
BP = 50            # pattern-prior block (outside top-50)
W_PAT = KC + BP    # 100: [top50 | <=50 pattern-prior]
LP_LEN = 300       # pattern-prior list kept per query
KREF = 200
VALID_N = 200000   # > #valid rows of GDELTFW (189,556) -> full valid (g2_prep logic unchanged)
SEEDS = [0, 1, 2]
COV6 = ["pat_best_util", "pat_best_te", "pat_log_npat", "pat_reach", "pat_log_pos", "is_cov_cand"]


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def module_provenance(names=("oih_common", "fulltest_eval", "patterns", "rt_common", "rs_features")):
    out = {}
    for nm in names:
        m = sys.modules.get(nm)
        if m is not None:
            f = os.path.abspath(m.__file__)
            out[nm] = dict(file=f, in_track_code_dir=f.startswith(CODE))
    return out


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
    """memmap [n_split_rows, N_ENT] float32 log-prob rows of the base, rows in split_queries order."""
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
