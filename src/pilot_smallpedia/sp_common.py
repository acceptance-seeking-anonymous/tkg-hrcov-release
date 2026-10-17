"""pilot_smallpedia / sp_common.py -- shared constants + helpers for the tkgl-smallpedia static-KG fusion pilot.

Derived from pilot_polecat/pc_common.py.  Registers dataset "SMALLPEDIA" in the (read-only) shared harness's
oih_common.DATASETS at import time only; nothing here writes into the shared harness or other track dirs.
Static Wikidata edges (TGB tkgl-smallpedia_static_edgelist.csv) live in $P/stage/static_edges.npz in the shared
id space written by sp_verify.py (temporal entities keep their harness id == TGB id; static-only nodes get the
TGB id >= N_ENT).
"""
import os, sys, json, time
import numpy as np

H = os.environ.get("PC_HARNESS", os.path.expandvars("$WORK_DIR"))
TRACKS = os.environ.get("PC_TRACKS", f"{H}/tracks")
P = os.environ.get("PC_P", f"{TRACKS}/pilot_smallpedia")
PP = f"{TRACKS}/pilot_polecat"                     # pilot-1 code dir (read-only; pc_common helpers reused)
DD = os.environ.get("PC_DD", f"{TRACKS}/datasets/harness_data/tkgl-smallpedia")
ZIP = f"{TRACKS}/datasets/raw/tkgl-smallpedia.zip"
RT = f"{TRACKS}/retriever/src"
for _p in (H, RT, f"{P}/code", os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import oih_common                                    # noqa: E402
from oih_common import Splits, BaseDump, metrics_from_ranks, paired_bootstrap, jdump, filter_lookup  # noqa: E402,F401

DS = "SMALLPEDIA"
TGB_NAME = "tkgl-smallpedia"
N_ENT, N_REL = 47433, 283
BASES = ("regcn", "logcl", "tirgn")
DUMPS = {b: f"{P}/dumps/{b}_scores_{DS}_traintest.npz" for b in BASES}      # symlinks to the valid-selected budget
FULLROWS = {b: {sp: f"{P}/dumps/{b}_{sp}_logp.f32" for sp in ("valid", "test")} for b in BASES}
CKPT = {b: f"{P}/dumps/{b}_{DS}_seed42.pt" for b in BASES}
oih_common.DATASETS[DS] = dict(
    dd=DD, n_ent=N_ENT, n_rel=N_REL, step_unit=1, dumps=dict(DUMPS),
    ref=dict(regcn=dict(MRR=0.594, src="TGB 2.0 leaderboard, RE-GCN"),
             cen=dict(MRR=0.612, src="TGB 2.0 leaderboard, CEN")))
WORK = f"{P}/work"
STATIC = f"{P}/stage/static_edges.npz"

KC = 50            # history re-ranker pool = base top-50
BP = 50            # pattern-prior block
BB = 50            # stage-2 bridge block
BS = 50            # static-KG block (layout A)
BD = 50            # degree-matched random block (density control, layout A)
BS_B = 100         # static block of layout B (top-50 ++ 100 static)
W_A = KC + BP + BB + BS + BD      # 250: [top50 | pat | br | static | dm]
W_B = KC + BS_B                   # 150: [top50 | static]
LP_LEN = 300
LB_LEN = 200
LS_LEN = 200       # static list kept per query
VALID_N = 200000   # > #valid queries (162,066) -> the FULL valid split is used
SEEDS = [0, 1, 2]
# direct-static-edge leakage control (addendum): full = pilot 2 | nodirect = no S-shape paths | nodirect_strict = also no static
# evidence at all for candidates with a direct static edge to s.  Output dirs/files get MODE_SFX.
ST_MODE = os.environ.get("SP_STATIC_MODE", "full")
assert ST_MODE in ("full", "nodirect", "nodirect_strict"), ST_MODE
MODE_SFX = {"full": "", "nodirect": "_nd", "nodirect_strict": "_nds"}[ST_MODE]
COV_FEATS = ["pat_best_util", "pat_best_te", "pat_log_npat", "pat_reach", "pat_log_pos",
             "br_in", "br_score", "br_log_pos", "br_log_pm_subj", "br_log_ninterm", "is_cov_cand"]
ST_FEATS = ["st_1hop", "st_log_n_S", "st_log_n_ST", "st_log_n_TS", "st_log_n_SS", "st_best_te", "st_log_shared_attr",
            "st_log_deg_c", "st_in", "st_log_pos", "is_st_cand", "st_q_log_deg_s"]
PATH_SHAPES = ("S", "ST", "TS", "SS")   # s-p->o | s-p->x-r'->o | s-r'->x-p->o | s-p1->x-p2->o
ST_DEG_CAP = 1000  # intermediate nodes with static degree > cap do not expand (hubs); reach is ALSO reported uncapped


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def split_queries(S, split):
    """sro [n,4] of one split in EXACT base_dump.py row order (identical to pc_common.split_queries)."""
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


def profiler_to_split_order(S, split):
    q = S.q[split]; n = len(q)
    order = np.argsort(q[:, 3], kind="stable")
    assert (order == np.arange(n)).all(), "split file not time-sorted"
    ts = q[:, 3]; perm = []
    for u in np.unique(ts):
        idx = np.flatnonzero(ts == u); perm.append(idx); perm.append(idx + n)
    return np.concatenate(perm)


def fullrows(base, split, mode="r"):
    """memmap [n_split_rows, N_ENT] float32 log-probabilities of the base, rows in split_queries order."""
    fn = FULLROWS[base][split]
    n = os.path.getsize(fn) // (4 * N_ENT)
    return np.memmap(fn, dtype=np.float32, mode=mode, shape=(n, N_ENT))


def load_static():
    z = np.load(STATIC)
    return dict(h=z["h"].astype(np.int64), t=z["t"].astype(np.int64), p=z["p"].astype(np.int64),
                n_nodes=int(z["n_nodes"]), n_prel=int(z["n_prel"]))


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


def rr_mean(ranks_list):
    return np.mean([1.0 / np.asarray(r, np.float64) for r in ranks_list], 0)
