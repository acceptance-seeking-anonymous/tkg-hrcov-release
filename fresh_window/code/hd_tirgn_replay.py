"""hardening / hd_tirgn_replay.py -- TiRGN row source for hd_blend.py (and the TiRGN list lookup): replays a hd_tirgn.py
checkpoint with ground-truth online history (previous L snapshots of all splits) and yields, per snapshot,
(split, ts, sro [n,4] split order, fn(s_q, r_q) -> log-prob rows [b, N], facts).  Scores = log(hr*p_hist + (1-hr)*p_raw)
exactly as hd_tirgn.py / sp_tirgn.py (history vocabulary from the first-occurrence index).
"""
import sys, numpy as np, torch, torch.nn.functional as F


def source(cfg, ckpt, L, splits_needed, dev, tree, hr=0.3, dec_chunk=1024):
    sys.path.insert(0, tree); sys.path.insert(0, f"{tree}/src")
    import dgl_shim  # noqa: F401
    from rgcn.utils import build_sub_graph
    from src.rrgcn import RecurrentRGCN
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hd_tirgn import HistVocab, patched_history_mode, tslice, load_quads
    N, R = cfg["N"], cfg["R"]
    RecurrentRGCN.history_mode = patched_history_mode
    Q = {sp: load_quads(f"{cfg['dd']}/{sp}.txt") for sp in ("train", "valid", "test")}
    HV = HistVocab(np.concatenate([Q["train"], Q["valid"], Q["test"]]), N, R)
    snaps = {}
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
        snaps[sp] = [(int(u), q[b:e]) for u, b, e in zip(uts, st, en)]
    tint = snaps["train"][1][0] - snaps["train"][0][0]; num_times = sum(len(v) for v in snaps.values())
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    Lc = ck["log"]["config"]["L"] if "log" in ck and "config" in ck["log"] else L
    model = RecurrentRGCN("timeconvtranse", "convgcn", N, R, 0, 0, num_times, tint, 200, "sub", hr, sequence_len=Lc,
                          num_bases=100, num_basis=100, num_hidden_layers=2, dropout=0.2, self_loop=True, skip_connect=False,
                          layer_norm=True, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, aggregation="none",
                          weight=0.5, discount=1, angle=10, use_static=False, entity_prediction=True, relation_prediction=False,
                          use_cuda=(str(dev) == "cuda"), gpu=(0 if str(dev) == "cuda" else -1), analysis=False).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    hist = []
    last = max(("train", "valid", "test").index(s) for s in splits_needed)
    with torch.no_grad():
        for spi, sp in enumerate(("train", "valid", "test")):
            if spi > last: break
            for ts, trip in snaps[sp]:
                if sp in splits_needed and len(trip):
                    inv = trip[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
                    sro = np.concatenate([trip, inv]).astype(np.int64)
                    state = {}

                    def fn(s_q, r_q, ts=ts, glist=list(hist[-Lc:]), state=state):
                        if "emb" not in state:
                            evolve, _, r_emb, _, _ = model.forward(glist, None, str(dev) == "cuda")
                            state["emb"] = F.normalize(evolve[-1]); state["r"] = r_emb
                        allt = torch.as_tensor(np.stack([s_q, r_q, np.zeros_like(s_q), np.full(len(s_q), ts)], 1), device=dev).long()
                        V = HV.dense(np.asarray(s_q), np.asarray(r_q), ts, dev)
                        te = model.get_init_time(allt)
                        out = torch.empty((len(s_q), N), device=dev)
                        for c0 in range(0, len(s_q), dec_chunk):
                            sl = slice(c0, c0 + dec_chunk)
                            sr = model.raw_mode(state["emb"], state["r"], tslice(te, sl), allt[sl])
                            sh = model.history_mode(state["emb"], state["r"], tslice(te, sl), allt[sl], V[sl])
                            out[sl] = torch.log(hr * sh + (1 - hr) * sr)
                        return out
                    yield sp, ts, sro, fn, trip[:, :3]
                else:
                    yield sp, ts, None, None, trip[:, :3]
                if len(trip):
                    hist.append(build_sub_graph(N, R, trip[:, :3], str(dev) == "cuda", (0 if str(dev) == "cuda" else -1)))
