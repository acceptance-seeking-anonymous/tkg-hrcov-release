#!/usr/bin/env python3
"""G6 / g6_profile.py -- descriptive latency / memory profiling (g6_protocol.json) of base / HR / HRCOV_PAT on a fixed
10k-query test sample, online over the test snapshots (ground-truth history < t), GPU 1 of serverA only.
Stages: S1 base scoring (TiRGN replay: per-snapshot evolution + per-query decoding -> full log-prob row),
S2 candidate generation (top-50; + pattern-prior block for HRCOV_PAT, pilot-1 recipe), S3 features (8 base + 11 evidence
[+ 6 pattern coverage]), S4 re-ranker MLP (seed-0 model) + fusion alpha*log p + beta*f + promotion order.
Modes: b1 (one query at a time) and batched (all sample queries of a timestamp; GPU decode chunked).
  python3 g6_profile.py --ds polecat --out DIR [--n 10000] [--repeats 3] [--smoke_ts 0]
"""
import os, sys, time, json, argparse, resource, platform, subprocess
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True, choices=["polecat", "smallpedia"]); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=10000); ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--chunk", type=int, default=512); ap.add_argument("--smoke_ts", type=int, default=0)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
O = os.path.expandvars("$WORK_DIR"); T = f"{O}/tracks"; H = f"{T}/hardening"
CFG = {
    "polecat": dict(code=f"{T}/pc_tirgn/code", P=f"{T}/pc_tirgn", common="pc_common", ckpt=f"{H}/tirgn_polecat/tirgn_POLECAT_seed42.pt",
                    te=f"{T}/pc_tirgn/work/te_POLECAT.npz", mass="top200", brank_top50="arange",
                    hr_model=f"{T}/pc_tirgn/work/train_tirgn/models/HR_s0.pt", hr_json=f"{T}/pc_tirgn/work/train_tirgn/train_tirgn.json",
                    hr_ref=f"{T}/pc_tirgn/work/train_tirgn/promlist_HR_s0.npy",
                    pat_model=f"{H}/presub_hcpat/train_polecat_tirgn/models/HRCOV_PAT_s0.pt", pat_json=f"{H}/presub_hcpat/train_polecat_tirgn/train_hcpat.json",
                    pat_ref=f"{H}/presub_hcpat/train_polecat_tirgn/promlist_HRCOV_PAT_s0.npy"),
    "smallpedia": dict(code=f"{T}/pilot_smallpedia/code", P=f"{T}/pilot_smallpedia", common="sp_common", ckpt=f"{T}/pilot_smallpedia/dumps/tirgn_SMALLPEDIA_seed42.pt",
                       te=f"{T}/pilot_smallpedia/work/te_SMALLPEDIA.npz", mass="exact", brank_top50="exact",
                       hr_model=f"{T}/pilot_smallpedia/work/train_tirgn/models/HR_s0.pt", hr_json=f"{T}/pilot_smallpedia/work/train_tirgn/train_tirgn.json",
                       hr_ref=f"{T}/pilot_smallpedia/work/train_tirgn/promlist_HR_s0.npy",
                       pat_model=f"{H}/presub_hcpat2/train_smallpedia_tirgn/models/HRCOV_PAT_s0.pt", pat_json=f"{H}/presub_hcpat2/train_smallpedia_tirgn/train_hcpat.json",
                       pat_ref=f"{H}/presub_hcpat2/train_smallpedia_tirgn/promlist_HRCOV_PAT_s0.npy"),
}[a.ds]
os.environ["PC_P"] = CFG["P"]
sys.path.insert(0, CFG["code"]); sys.path.insert(0, f"{T}/retriever/src"); sys.path.insert(0, f"{T}/residual/code")
import torch, torch.nn.functional as F
torch.set_num_threads(a.threads)
C = __import__(CFG["common"])
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
from rs_features import Hist, DyadReader
KC, BP, LP_LEN, LB_LEN, KREF, PAT_FILL, E1 = 50, 50, 300, 200, 200, -6.0, np.e - 1.0
dev = "cuda"
os.makedirs(a.out, exist_ok=True)


def rss():
    d = {}
    for ln in open("/proc/self/status"):
        if ln.startswith(("VmRSS", "VmHWM")): k, v = ln.split(":"); d[k] = int(v.split()[0]) / 1e6
    return d


def sync(): torch.cuda.synchronize()


T0 = time.time(); setup = {}
def tick(name, t):
    setup[name] = round(time.time() - t, 2); print(f"[{time.time()-T0:8.1f}s] setup {name} {setup[name]}s rss {rss()}", flush=True)

t = time.time(); S = C.Splits(C.DS); N, R = S.n_ent, S.R; allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]); tick("splits", t)
t = time.time(); EI = EventIndex(allq, N, R, 1); te = TETables(CFG["te"]); tick("event_index_te", t)
t = time.time(); Hh = Hist(S); BM = DyadReader(S, Hh); tick("hist_dyad", t)
sro_test = C.split_queries(S, "test").astype(np.int64)
rng = np.random.default_rng(0); samp = np.sort(rng.choice(len(sro_test), min(a.n, len(sro_test)), replace=False))
# ---------------- TiRGN (same construction as hardening/code/hd_tirgn_replay.py)
t = time.time()
tree = f"{T}/pilot_smallpedia/stage/TiRGN"; sys.path.insert(0, tree); sys.path.insert(0, f"{tree}/src"); sys.path.insert(0, f"{H}/code")
import dgl_shim  # noqa: F401
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from hd_tirgn import HistVocab, patched_history_mode, tslice
RecurrentRGCN.history_mode = patched_history_mode
snaps = {}
for sp in ("train", "valid", "test"):
    q = S.q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
    snaps[sp] = [(int(u), q[b:e]) for u, b, e in zip(uts, st, en)]
tint = snaps["train"][1][0] - snaps["train"][0][0]; num_times = sum(len(v) for v in snaps.values())
HV = HistVocab(allq, N, R)
ck = torch.load(CFG["ckpt"], map_location="cpu", weights_only=False)
cfgck = ck.get("log", {}).get("config", {}) if isinstance(ck.get("log"), dict) else {}
Lc = int(cfgck.get("L", 10)); hr = float(cfgck.get("hr", 0.3))
model = RecurrentRGCN("timeconvtranse", "convgcn", N, R, 0, 0, num_times, tint, 200, "sub", hr, sequence_len=Lc,
                      num_bases=100, num_basis=100, num_hidden_layers=2, dropout=0.2, self_loop=True, skip_connect=False,
                      layer_norm=True, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, aggregation="none",
                      weight=0.5, discount=1, angle=10, use_static=False, entity_prediction=True, relation_prediction=False,
                      use_cuda=True, gpu=0, analysis=False).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
prev = (snaps["train"] + snaps["valid"])[-Lc:]
hist0 = [build_sub_graph(N, R, tr[:, :3], True, 0) for _, tr in prev]
tick("tirgn_model_history", t)


def decode_rows(state, s_q, r_q, ts):
    allt = torch.as_tensor(np.stack([s_q, r_q, np.zeros_like(s_q), np.full(len(s_q), ts)], 1), device=dev).long()
    V = HV.dense(np.asarray(s_q), np.asarray(r_q), ts, dev)
    tem = model.get_init_time(allt)
    out = torch.empty((len(s_q), N), device=dev)
    for c0 in range(0, len(s_q), a.chunk):
        sl = slice(c0, c0 + a.chunk)
        sr = model.raw_mode(state["emb"], state["r"], tslice(tem, sl), allt[sl])
        sh = model.history_mode(state["emb"], state["r"], tslice(tem, sl), allt[sl], V[sl])
        out[sl] = torch.log(hr * sh + (1 - hr) * sr)
    return out


# ---------------- re-rankers
class Net(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    def forward(self, x): return self.f(x).squeeze(-1)


def load_rr(mp, jp, arm, W):
    m = torch.load(mp, map_location="cpu", weights_only=False)
    feats = list(m["feats"]) if "feats" in m else list(m["cols"])
    net = Net(len(feats)); net.load_state_dict(m["state_dict"]); net.eval().to(dev)
    J = json.load(open(jp)); sd0 = J["arms"][arm]["seeds"]["0"]
    return dict(net=net, feats=np.asarray(feats), mu=torch.as_tensor(np.asarray(m["mu"], np.float32), device=dev),
                sd=torch.as_tensor(np.asarray(m["sd"], np.float32), device=dev), alpha=float(sd0["cal_alpha"]), beta=float(sd0["cal_beta"]), W=W)


RR = {"HR": load_rr(CFG["hr_model"], CFG["hr_json"], "HR", KC), "HRCOV_PAT": load_rr(CFG["pat_model"], CFG["pat_json"], "HRCOV_PAT", KC + BP)}
REF = {"HR": np.load(CFG["hr_ref"], mmap_mode="r"), "HRCOV_PAT": np.load(CFG["pat_ref"], mmap_mode="r")}
pos_buf = np.full(N, -1, np.int64)


# ---------------- S2: candidates
def pattern_block(s, r, t, tki, tks, lprow):
    """pilot-1 pattern prior (pc_lists / sp_lists query_lists, pattern part only) -> C (top50 ++ <= 50 pattern) and the
    pattern coverage features (COV cols 0-4, 10)."""
    top50 = tki[:KC]
    Lpat = np.full(LP_LEN, -1, np.int64); pat = None
    d = instantiate(EI, s, t, M1, M2)
    if d is not None:
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        if CFG["mass"] == "top200":
            tail = max(1.0 - np.exp(tks).sum(), 1e-12); lp_tail = np.log(tail / max(N - KREF, 1))
            pos_buf[tki] = np.arange(KREF); pk = pos_buf[cc]; lpc = np.where(pk >= 0, tks[np.maximum(pk, 0)], lp_tail); pos_buf[tki] = -1
        else:
            lpc = lprow[cc]
        mass = np.bincount(pp, weights=np.exp(lpc), minlength=P_)
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = te.features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        util = te_h - np.log1p(mass * E1)
        cand, inv = np.unique(cc, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, util[pp])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pp])
        supp = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        if CFG["mass"] == "top200":
            lpcand = np.zeros(len(cand)); lpcand[inv] = lpc
        else:
            lpcand = lprow[cand]
        order = np.lexsort((-lpcand, -supp, -best)); cand_o = cand[order]
        rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        L = cand_o[:LP_LEN]; Lpat[:len(L)] = L
        pat = dict(cand=cand, best=best, bte=bte, npat=npat, rank=rank_o)
    Cl = list(top50.tolist()); seen = set(Cl)
    for e in Lpat:
        if len(Cl) >= KC + BP or e < 0: break
        if int(e) not in seen: Cl.append(int(e)); seen.add(int(e))
    Cn = np.full(KC + BP, -1, np.int64); Cn[:len(Cl)] = Cl; Cv = np.array(Cl, np.int64)
    X = np.zeros((KC + BP, 11), np.float32)
    X[:len(Cl), 0] = PAT_FILL; X[:len(Cl), 4] = np.log1p(LP_LEN + 1); X[:len(Cl), 7] = np.log1p(LB_LEN + 1)
    if pat is not None:
        j = np.searchsorted(pat["cand"], Cv); j = np.minimum(j, len(pat["cand"]) - 1); m = pat["cand"][j] == Cv
        X[:len(Cl)][m, 0] = pat["best"][j[m]]; X[:len(Cl)][m, 1] = pat["bte"][j[m]]; X[:len(Cl)][m, 2] = np.log1p(pat["npat"][j[m]])
        X[:len(Cl)][m, 3] = 1.0; X[:len(Cl)][m, 4] = np.log1p(np.minimum(pat["rank"][j[m]], LP_LEN) + 1)
    X[KC:len(Cl), 10] = 1.0
    return Cn, X


# ---------------- S3: features ;  S4: re-rank
def features(lp, tkv, tki, s, r, t, cand, Xcov):
    """lp [n,N] cuda rows, tkv/tki top-200 (cuda), cand [n,W] numpy -> X [n,W,30] float32 (fp16-rounded, as stored)"""
    n, W = cand.shape; valid = cand >= 0
    ct = torch.as_tensor(np.where(valid, cand, 0), device=dev)
    lpc = lp.gather(1, ct)
    srt, _ = torch.sort(lp, dim=1)
    br = (N - torch.searchsorted(srt, lpc.contiguous(), right=True) + 1).double()
    if CFG["brank_top50"] == "arange":
        br[:, :KC] = torch.arange(1, KC + 1, device=dev, dtype=torch.float64)
    tks = tkv.double()
    p_top = torch.exp(tks[:, :KC]); pn = p_top / p_top.sum(1, keepdim=True); ent50 = -(pn * torch.log(pn + 1e-12)).sum(1)
    tail_mass = torch.clamp(1.0 - torch.exp(tks).sum(1), 1e-12, 1.0)
    Xb = torch.zeros((n, W, 8), device=dev, dtype=torch.float64)
    lpf = torch.where(torch.as_tensor(valid, device=dev), lpc.double(), torch.full_like(lpc.double(), -30.0))
    Xb[:, :, 0] = torch.clamp(lpf, min=-30.0); Xb[:, :, 1] = Xb[:, :, 0] - tks[:, :1]; Xb[:, :, 2] = torch.log(torch.clamp(br, min=1))
    Xb[:, :KC, 3] = 1.0; Xb[:, :, 4] = torch.exp(tks[:, :1]); Xb[:, :, 5] = ent50[:, None]; Xb[:, :, 6] = (tks[:, 0] - tks[:, 1])[:, None]
    Xb[:, :, 7] = torch.log(tail_mass)[:, None]
    Xf = np.zeros((n, W, 30), np.float32)
    Xf[:, :, :8] = Xb.float().cpu().numpy()
    cm = np.where(valid, cand, -1)
    hf = Hh.feats(s, r, t, cm); A, mev = BM.score(s, r, t, cm, hf["dyad_hi"])
    Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]
    Xf[:, :, 11] = hf["rec_dyad"]; Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"])
    Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"]); Xf[:, :, 16] = A
    Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
    if Xcov is not None:
        Xf[:, :, 19:30] = Xcov.astype(np.float16).astype(np.float32)
    Xf[~valid] = 0.0
    return Xf.astype(np.float16).astype(np.float32), torch.where(torch.as_tensor(valid, device=dev), lpc, torch.full_like(lpc, float("-inf")))


def rerank(m, Xf, LP, cand):
    x = (torch.as_tensor(Xf[:, :, m["feats"]], device=dev) - m["mu"]) / m["sd"]
    with torch.no_grad():
        f = m["net"](x)
    valid = torch.as_tensor(cand >= 0, device=dev)
    s = torch.where(valid, m["alpha"] * LP + m["beta"] * f, torch.full_like(f, float("-inf")))
    mint = s[:, :KC].min(1).values
    jj = torch.arange(s.shape[1], device=dev)[None, :]
    prom = valid & ((jj < KC) | (s > mint[:, None]))
    key = torch.where(prom, s, torch.full_like(s, float("-inf")))
    o = torch.argsort(-key, dim=1, stable=True)
    ids = torch.gather(torch.as_tensor(cand, device=dev), 1, o); pk = torch.gather(prom, 1, o)
    return torch.where(pk, ids, torch.full_like(ids, -1)).cpu().numpy()


# ---------------- profiling loop
gpu_name = torch.cuda.get_device_name(0)
by_ts = {}
for k, i in enumerate(samp): by_ts.setdefault(int(sro_test[i, 3]), []).append(int(i))
n_test_ts = {ts: 2 * len(tr) for ts, tr in snaps["test"]}
STAGES = ["S1_decode", "S1_top", "S2_HR", "S3_HR", "S4_HR", "S2_PAT", "S3_PAT", "S4_PAT"]
REC = {mode: {st_: [] for st_ in STAGES} for mode in ("b1", "batched")}
SNAP = {mode: [] for mode in ("b1", "batched")}
MEM = {st_: dict(gpu_peak_GB=0.0) for st_ in STAGES + ["S1_snapshot"]}
AGREE = {"HR": [0, 0, 0, 0], "HRCOV_PAT": [0, 0, 0, 0]}  # [n, full list equal, top-1 equal, gold position equal]
RECB = {mode: {st_: [] for st_ in STAGES} for mode in ("b1", "batched")}   # per-batch wall (s)
BSZ = {mode: [] for mode in ("b1", "batched")}


for st_ in MEM: MEM[st_].update(gpu_resident_before_GB=0.0, host_hwm_GB=None, host_hwm_delta_GB=None)
PROBE = {"on": False}                                   # host-RSS probe: warm-up pass only (never in timed repeats)


def _clear_hwm():
    try:
        with open("/proc/self/clear_refs", "w") as f_: f_.write("5")
        return True
    except Exception:
        return False


def timed(stage, fn, gpu=True):
    torch.cuda.reset_peak_memory_stats()
    if gpu: sync()
    res0 = torch.cuda.memory_allocated() / 1e9
    if PROBE["on"]:
        ok_ = _clear_hwm(); r0 = rss()["VmRSS"]
    t_ = time.perf_counter(); out = fn()
    if gpu: sync()
    dt = time.perf_counter() - t_
    MEM[stage]["gpu_peak_GB"] = max(MEM[stage]["gpu_peak_GB"], torch.cuda.max_memory_allocated() / 1e9)
    MEM[stage]["gpu_resident_before_GB"] = max(MEM[stage]["gpu_resident_before_GB"], res0)
    if PROBE["on"] and ok_:
        h = rss()["VmHWM"]
        MEM[stage]["host_hwm_GB"] = max(MEM[stage]["host_hwm_GB"] or 0.0, h)
        MEM[stage]["host_hwm_delta_GB"] = max(MEM[stage]["host_hwm_delta_GB"] or 0.0, h - r0)
    return out, dt


def process(rows, state, ts, mode, rep, record):
    sro = sro_test[rows]; s, r, o, tt = sro[:, 0], sro[:, 1], sro[:, 2], sro[:, 3]; nb = len(rows)
    with torch.no_grad():
        lp, d1 = timed("S1_decode", lambda: decode_rows(state, s, r, ts))
        (tkv, tki), d2 = timed("S1_top", lambda: torch.topk(lp, KREF, dim=1))
    tki_np = tki.cpu().numpy(); tkv_np = tkv.double().cpu().numpy()
    # HR
    cand_hr, d3 = timed("S2_HR", lambda: tki_np[:, :KC].astype(np.int64), gpu=False)
    (Xh, LPh), d4 = timed("S3_HR", lambda: features(lp, tkv, tki, s, r, tt, cand_hr, None))
    lh, d5 = timed("S4_HR", lambda: rerank(RR["HR"], Xh, LPh, cand_hr))
    # HRCOV_PAT
    def s2p():
        lprow = lp.double().cpu().numpy() if CFG["mass"] == "exact" else None
        outs = [pattern_block(int(s[i]), int(r[i]), int(tt[i]), tki_np[i], tkv_np[i], None if lprow is None else lprow[i]) for i in range(nb)]
        return np.stack([x[0] for x in outs]), np.stack([x[1] for x in outs])
    (cand_p, Xc), d6 = timed("S2_PAT", s2p, gpu=False)
    (Xp, LPp), d7 = timed("S3_PAT", lambda: features(lp, tkv, tki, s, r, tt, cand_p, Xc))
    lpat, d8 = timed("S4_PAT", lambda: rerank(RR["HRCOV_PAT"], Xp, LPp, cand_p))
    if record:
        for st_, d in zip(STAGES, (d1, d2, d3, d4, d5, d6, d7, d8)):
            REC[mode][st_].extend([d / nb] * nb); RECB[mode][st_].append(d)
        BSZ[mode].append(nb)
    if rep == 0 and mode == "batched":
        for arm, lst in (("HR", lh), ("HRCOV_PAT", lpat)):
            ref = np.asarray(REF[arm][rows])
            W = min(lst.shape[1], ref.shape[1])
            AGREE[arm][0] += nb; AGREE[arm][1] += int((lst[:, :W] == ref[:, :W]).all(1).sum()); AGREE[arm][2] += int((lst[:, 0] == ref[:, 0]).sum())
            def gpos(L_):
                hit = (L_[:, :W] == o[:, None]); return np.where(hit.any(1), hit.argmax(1), -1)
            AGREE[arm][3] += int((gpos(lst) == gpos(ref)).sum())
    del lp


test_snaps = snaps["test"][:a.smoke_ts] if a.smoke_ts else snaps["test"]
WALL = {}
for rep in range(a.repeats + 1):
    for mode in ("batched", "b1"):
        PROBE["on"] = (rep == 0)
        hist = list(hist0); tw = time.perf_counter(); rss0 = rss()
        for ts, trip in test_snaps:
            rows = by_ts.get(ts, [])
            if rows:
                with torch.no_grad():
                    def evo():
                        evolve, _, r_emb, _, _ = model.forward(hist[-Lc:], None, True)
                        return dict(emb=F.normalize(evolve[-1]), r=r_emb)
                    state, de = timed("S1_snapshot", evo)
                if mode == "batched":
                    process(rows, state, ts, mode, rep, rep > 0)
                else:
                    for i in rows: process([i], state, ts, mode, rep, rep > 0)
            else:
                de = 0.0
            g, dg = timed("S1_snapshot", lambda: build_sub_graph(N, R, trip[:, :3], True, 0))
            hist.append(g)
            if rep > 0:
                SNAP[mode].append(dict(ts=ts, evolve_s=de, graph_s=dg, n_test_queries=n_test_ts[ts], n_sample=len(rows)))
        WALL[f"{mode}_rep{rep}"] = round(time.perf_counter() - tw, 2)
        print(f"[{time.time()-T0:8.1f}s] rep {rep} {mode} wall {WALL[f'{mode}_rep{rep}']}s rss {rss()}", flush=True)


def stats(v):
    v = np.asarray(v) * 1e3
    return dict(p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95)), mean_ms=float(v.mean()), n=int(len(v)))


res = dict(ds=a.ds, n_sample=int(len(samp)), repeats=a.repeats, chunk=a.chunk, threads=a.threads, setup_s=setup, wall_s=WALL,
           tirgn=dict(L=Lc, hr=hr, ckpt=CFG["ckpt"]), stages={}, end_to_end={}, snapshot={}, memory=MEM, rss_GB=rss(),
           agreement_vs_stored_lists={k: dict(n=v[0], full_list_equal=v[1] / max(v[0], 1), top1_equal=v[2] / max(v[0], 1), gold_position_equal=v[3] / max(v[0], 1)) for k, v in AGREE.items()},
           gpu_max_reserved_GB=torch.cuda.max_memory_reserved() / 1e9, host_rss_probe="VmHWM after /proc/self/clear_refs=5 reset, warm-up pass only",
           batch_size={m: dict(n_batches=len(BSZ[m]), mean=float(np.mean(BSZ[m])) if BSZ[m] else None, p50=float(np.median(BSZ[m])) if BSZ[m] else None,
                               max=int(max(BSZ[m])) if BSZ[m] else None) for m in BSZ},
           batch_wall={})
for mode in ("b1", "batched"):
    res["batch_wall"][mode] = {k: stats(v) for k, v in RECB[mode].items()}
for mode in ("b1", "batched"):
    Rm = {k: np.asarray(v) for k, v in REC[mode].items()}
    res["stages"][mode] = {k: stats(v) for k, v in Rm.items()}
    sn = SNAP[mode]
    evo_tot = sum(x["evolve_s"] for x in sn); g_tot = sum(x["graph_s"] for x in sn)
    snaps_with = [x for x in sn if x["n_sample"] > 0]
    amort = np.mean([(x["evolve_s"] + x["graph_s"]) / max(x["n_test_queries"], 1) for x in sn]) if sn else 0.0
    res["snapshot"][mode] = dict(evolve_ms_mean=1e3 * float(np.mean([x["evolve_s"] for x in snaps_with])) if snaps_with else None,
                                 graph_ms_mean=1e3 * float(np.mean([x["graph_s"] for x in sn])) if sn else None,
                                 amortised_ms_per_query_full_test_traffic=1e3 * float(amort))
    base = Rm["S1_decode"] + Rm["S1_top"]
    hrt = base + Rm["S2_HR"] + Rm["S3_HR"] + Rm["S4_HR"]
    pat = base + Rm["S2_PAT"] + Rm["S3_PAT"] + Rm["S4_PAT"]
    am = amort
    e2e = {}
    for nm, v in (("base", base), ("HR", hrt), ("HRCOV_PAT", pat)):
        e2e[nm] = dict(**stats(v + am), throughput_qps=float(1.0 / np.mean(v + am)), throughput_qps_excl_snapshot=float(1.0 / np.mean(v)))
    for nm in ("HR", "HRCOV_PAT"):
        e2e[nm]["overhead_vs_base_mean"] = float(np.mean(e2e[nm]["mean_ms"]) / e2e["base"]["mean_ms"] - 1.0)
    res["end_to_end"][mode] = e2e
env = dict(host="serverA", gpu=gpu_name, cuda=torch.version.cuda, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
           cpu=subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip(),
           driver=subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0", shell=True, capture_output=True, text=True).stdout.strip(),
           load_at_end=os.getloadavg(), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import scipy; env["scipy"] = scipy.__version__
except Exception:
    pass
res["environment"] = env
json.dump(res, open(f"{a.out}/g6_{a.ds}{'_smoke' if a.smoke_ts else ''}.json", "w"), indent=1)
os.system(f"chmod -R a+rX {a.out}")
print("DONE", json.dumps({m: {k: round(v["mean_ms"], 2) for k, v in res["end_to_end"][m].items()} for m in res["end_to_end"]}), json.dumps(res["agreement_vs_stored_lists"]))
