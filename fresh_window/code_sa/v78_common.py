"""v78 / v78_common.py -- shared paths, configuration and the strict history order (v78_rule_amendment_1).
History order: RecB_xf score S desc, then train frequency (head + tail occurrences in train) desc, then entity id asc.
"""
import os, sys, json, time
import numpy as np

O = os.environ.get("V78_WORK", "/home/anon/cs_scratch/open_ideas_harness")
T = f"{O}/tracks"; H = f"{T}/hardening"; V = f"{H}/v78"
RECB_XF = os.environ.get("V78_RECB_XF", f"{H}/v76/recb_xf")   # phase2_fresh_window: GDELTFW RecB_xf dir via env
KMAX = 200            # history top list kept per query (first blocks need <= 75; generator tail mass needs 200)
LP_LEN, PAT_FILL, E1 = 300, -6.0, np.e - 1.0
SA_K = (25, 50, 75)   # first-block sizes of SA_W50 / SA_W100 / SA_W150
HU_K = (25, 50, 75)   # first-block sizes of HR_W50 / (stored-block check) / HR_W150

DS = {
    "polecat": dict(name="tkgl-polecat", dd=f"{T}/datasets/harness_data/tkgl-polecat", N=150931, R=16, layout="pc",
                    code=f"{T}/pc_tirgn/code", common="pc_common", P=f"{T}/pc_tirgn", te=f"{T}/pc_tirgn/work/te_POLECAT.npz",
                    valid_rows=f"{T}/pc_tirgn/work/valid_sample_idx.npy",
                    bases={"tirgn": dict(feat=f"{T}/pc_tirgn/work/feat_tirgn", lists=f"{T}/pc_tirgn/work/lists_tirgn",
                                         dump=f"{T}/pc_tirgn/dumps/tirgn_scores_POLECAT_traintest.npz", mass="top200"),
                           "cogn": dict(feat=f"{H}/g8_cogntke/pc/work/feat_cogn", lists=f"{H}/g8_cogntke/pc/work/lists_cogn",
                                        dump=f"{H}/g8_cogntke/dumps/cogn_scores_POLECAT_traintest.npz", mass="top200")}),
    "smallpedia": dict(name="tkgl-smallpedia", dd=f"{T}/datasets/harness_data/tkgl-smallpedia", N=47433, R=283, layout="sp",
                       code=f"{T}/pilot_smallpedia/code", common="sp_common", P=f"{T}/pilot_smallpedia",
                       te=f"{T}/pilot_smallpedia/work/te_SMALLPEDIA.npz", valid_rows=None,
                       bases={"tirgn": dict(feat=f"{T}/pilot_smallpedia/work/feat_tirgn", lists=None, mass="exact",
                                            rows={"valid": f"{T}/pilot_smallpedia/dumps/tirgn_valid_logp.f32",
                                                  "test": f"{T}/pilot_smallpedia/dumps/tirgn_test_logp.f32"})}),
    # phase2_fresh_window (v79): GDELTFW, base-free SA pools only (no PC/HU pools); common = fw_pc_shim (fh_common + RS_CODE)
    "gdeltfw": dict(name="GDELTFW", dd=f"{T}/phase2_fresh_window/data/GDELTFW", N=1209, R=215, layout="pc",
                    code=f"{T}/phase2_fresh_window/code", common="fw_pc_shim", P=f"{T}/phase2_fresh_window/GDELTFW",
                    te=f"{T}/phase2_fresh_window/GDELTFW/work/te_GDELTFW.npz",
                    valid_rows=f"{T}/phase2_fresh_window/GDELTFW/work/valid_sample_idx.npy", bases={}),
}


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def train_freq(dd, N):
    tr = load_quads(f"{dd}/train.txt")
    return np.bincount(tr[:, 0], minlength=N) + np.bincount(tr[:, 2], minlength=N)


def freq_order(freq):
    """entities ordered by (freq desc, id asc) and the inverse rank."""
    N = len(freq)
    g = np.lexsort((np.arange(N), -freq)).astype(np.int64)
    grank = np.empty(N, np.int64); grank[g] = np.arange(N)
    return g, grank


def feat_pre(base_cfg, layout, split):
    return f"{base_cfg['feat']}/{split}_" + ("A_" if layout == "sp" else "")


def strict_positions(S, gorder_t):
    """S [B, N] float32 (RecB_xf) -> pos [B, N] int64 position of every entity in the strict history order, and the
    ordered entity ids [B, N]. Stable descending sort of S over entities pre-ordered by (freq desc, id asc)."""
    import torch
    Sp = S[:, gorder_t]
    _, idx = torch.sort(Sp, dim=1, descending=True, stable=True)
    ent = gorder_t[idx]
    pos = torch.empty_like(ent)
    ar = torch.arange(S.shape[1], device=S.device).unsqueeze(0).expand_as(ent)
    pos.scatter_(1, ent, ar)
    return pos, ent
