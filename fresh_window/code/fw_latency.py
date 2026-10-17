#!/usr/bin/env python3
"""phase2_fresh_window / fw_latency.py -- v79 amendment-1 latency protocol (descriptive) for base, BLEND, HRcov, SA and the
blend part of the CountTRuCoLa comparator (CountTRuCoLa rule application is timed by fw_ct_latency.py in its own venv).
Slice: the first --n validation queries (harness order) of the first selection-part timestamp (ts = cut); serverA GPU 0.
Every stage calls the production code of the system (same functions / same formulas, imported or copied verbatim):
  base       TiRGN replay (hd_tirgn_replay.source): encoder (once per timestamp, amortised over the slice) + decoder
  RecB       strict-recurrency psi (fh_recb.walk_pairs + delta_sum, all 14 lambdas as in production) + online xi row
  BLEND      RecB row + selected blend + final ordering
  HRcov      pattern lists (fh_lists.query_lists) + 25 features (fh_features block) + MLP + calibrated fusion + promoted list
  SA         RecB_xf row + strict history order (v78_common.strict_positions) + conf_hist lists (v78_lists.one) + 16 features
             (v78_pools evidence block) + MLP + promoted list
  EXT blend  z-standardised log-softmax blend with the base row (fw_ext.method_rows) + ordering (if the comparator is a blend)
Batch 1: queries one at a time (CPU single process; GPU batch 1).  Batched: production batching (lists: 8 worker processes over
500-query tasks; features: one 4,000-query chunk; GPU stages: one call for the slice).  Median of --reps repeats.
Output: latency.json (seconds per 1,000 queries per stage and end to end, plus the raw repeat times).
"""
import os, sys, json, time, argparse, importlib, statistics
import numpy as np
import torch
FW = "/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window"; CODE = f"{FW}/code"; CS = f"{FW}/code_sa"
sys.path.insert(0, CODE)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--comparator", default=""); ap.add_argument("--ct_dump", default="")
a = ap.parse_args()
dev = "cuda"; torch.set_num_threads(int(os.environ.get("G2_THREADS", "8")))
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:8.1f}s]", *x, flush=True)
def sync(): torch.cuda.synchronize()
def timeit(f, reps=a.reps):
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), ts
RES = dict(n=a.n, reps=a.reps, host=os.environ.get("HD_HOST", ""), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           cpu_threads=torch.get_num_threads(), workers_batched=a.workers, stages={})
def rec(sys_, stage, mode, med, raw, nq):
    RES["stages"].setdefault(sys_, {}).setdefault(stage, {})[mode] = dict(s_per_1000=med / nq * 1000.0, median_s=med, raw_s=raw, n=nq)

# ------------------------------------------------------------------ setup (not timed)
import fh_common as FC
S = FC.Splits(FC.DS); N, R = S.n_ent, S.R; W_ = f"{FC.WORK}"
sro_v = FC.split_queries(S, "valid"); tsv = sro_v[:, 3]; uts = np.unique(tsv); cut = int(uts[int(0.7 * len(uts))])
st0 = int(np.flatnonzero(tsv == cut)[0]); rows = np.arange(st0, st0 + a.n); SRO = sro_v[rows]
RES.update(slice=dict(split="valid", ts=cut, first_row=st0, n=int(a.n), all_same_ts=bool((SRO[:, 3] == cut).all())))
LPv = FC.fullrows("tirgn", "valid"); LPs = np.asarray(LPv[rows], np.float32)
log("slice", RES["slice"])

# ------------------------------------------------------------------ base: TiRGN
import hd_tirgn_replay as TRP
gen = TRP.source(dict(N=N, R=R, dd=FC.DD), f"{W_}/base_tirgn/tirgn_GDELTFW_seed42.pt", 7, ("valid",), dev, f"{FW}/stage/TiRGN", hr=0.3, dec_chunk=4096)
for sp, t_, sro_, fn, trip in gen:
    if sp == "valid" and t_ == cut: break
assert (sro_[:a.n] == SRO).all()
state = fn.__defaults__[2]
def enc1(): state.clear(); fn(SRO[:1, 0], SRO[:1, 1])
def dec_1(): fn(SRO[:1, 0], SRO[:1, 1])
def dec_b1():
    for i in range(a.n): fn(SRO[i:i + 1, 0], SRO[i:i + 1, 1])
def dec_bb(): fn(SRO[:, 0], SRO[:, 1])
m_e, r_e = timeit(enc1); m_d, r_d = timeit(dec_1); enc = max(m_e - m_d, 0.0)
RES["base_encoder_s_per_timestamp"] = enc
m1, r1 = timeit(dec_b1); mb, rb = timeit(dec_bb)
chk = float((fn(SRO[:, 0], SRO[:, 1]).float().cpu().numpy() - LPs).__abs__().max()); RES["base_rows_vs_frozen_max_abs"] = chk
rec("base", "inference", "batch1", m1 + enc, r1, a.n); rec("base", "inference", "batched", mb + enc, rb, a.n)
log("base done", enc, m1, mb, chk)

# ------------------------------------------------------------------ RecB psi + xi (shared by BLEND and SA)
import fh_recb as RB
Q = {sp: RB.load_quads(f"{FC.DD}/{sp}.txt") for sp in ("train", "valid", "test")}
allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
s = np.concatenate([allq[:, 0], allq[:, 2]]); r = np.concatenate([allq[:, 1], allq[:, 1] + R]); o = np.concatenate([allq[:, 2], allq[:, 0]])
uts_ = np.unique(allq[:, 3]); tsmap = np.zeros(int(uts_.max()) + 1, np.int64); tsmap[uts_] = (uts_ - uts_.min())
t = tsmap[np.concatenate([allq[:, 3], allq[:, 3]])]; TT = int(t.max()) + 2
A = {k: {} for k in ("Ft1", "Fo1", "Ks", "Fo2", "Ft2")}; ordr = np.argsort(r, kind="stable"); rb_ = np.searchsorted(r[ordr], np.arange(2 * R + 1))
for rel in range(2 * R):
    ix = ordr[rb_[rel]:rb_[rel + 1]]
    o1 = np.argsort(t[ix], kind="stable"); A["Ft1"][rel] = t[ix][o1]; A["Fo1"][rel] = o[ix][o1]
    k2 = s[ix] * TT + t[ix]; o2 = np.argsort(k2, kind="stable"); A["Ks"][rel] = k2[o2]; A["Fo2"][rel] = o[ix][o2]; A["Ft2"][rel] = t[ix][o2]
tv = np.concatenate([Q["train"], Q["valid"]]); basis = np.zeros(2 * R, bool); basis[np.unique(np.concatenate([tv[:, 1], tv[:, 1] + R]))] = True
RB.G.update(A=A, TT=TT, N=N, R=R, basis=basis, tsmap=tsmap, Dcache={})
sys.path.insert(0, CS)
import hd_blend as HBv       # v78 copy (RecbCSR, recb_row, XiCounts, blend_scores, strict order helpers)
xi = HBv.XiCounts(N, R, dev)
for sp, tt_, sro_x, _, trip_x, _ in HBv.recb_source(HBv.DSCFG["gdeltfw"], ()):
    if sp == "valid" and tt_ >= cut: break
    xi.add(trip_x)
rel_s = SRO[:, 1]; s_s = SRO[:, 0]; t_s = tsmap[SRO[:, 3]]
def psi_b1():
    for i in range(a.n): RB.walk_pairs(int(rel_s[i]), s_s[i:i + 1], t_s[i:i + 1]); RB.G["Dcache"].clear()
def psi_bb():
    for rel in np.unique(rel_s):
        m_ = rel_s == rel; RB.walk_pairs(int(rel), s_s[m_], t_s[m_])
    RB.G["Dcache"].clear()
def xi_b1():
    for i in range(a.n): xi.xi(torch.as_tensor(rel_s[i:i + 1], device=dev))
def xi_bb(): xi.xi(torch.as_tensor(rel_s, device=dev))
mp1, rp1 = timeit(psi_b1); mpb, rpb = timeit(psi_bb); mx1, rx1 = timeit(xi_b1); mxb, rxb = timeit(xi_bb)
rec("RecB", "psi_xi", "batch1", mp1 + mx1, [x + y for x, y in zip(rp1, rx1)], a.n); rec("RecB", "psi_xi", "batched", mpb + mxb, [x + y for x, y in zip(rpb, rxb)], a.n)
log("recb done", mp1, mpb)

# ------------------------------------------------------------------ BLEND scoring (selected config) + ordering
import fh_blend as FB
sel = json.load(open(f"{W_}/controls/sel_gdeltfw.json"))["tirgn"]; bcfg = FB.BLEND[sel["blend_idx"]]
csr_tr = HBv.RecbCSR(f"{W_}/recb/recb_csr_valid_GDELTFW.npz"); csr_xf = HBv.RecbCSR(f"{W_}/recb_xf/recb_csr_valid_GDELTFW.npz")
LPt = torch.as_tensor(LPs, device=dev)
def blend_rows(ix):
    rr = torch.as_tensor(rel_s[ix], device=dev); X = xi.xi(rr); Sr = HBv.recb_row(csr_tr if bcfg[0] == "tr" else csr_tr, rows[ix], rel_s[ix], X, dev, bcfg[0])
    ss = Sr.sum(1, keepdim=True); q = torch.where(ss > 0, Sr / ss.clamp(min=1e-30), torch.zeros_like(Sr))
    sc = FB.blend_scores(LPt[ix], q, bcfg); return torch.argsort(sc, dim=1, descending=True)
def bl_b1():
    for i in range(a.n): blend_rows(np.array([i]))
def bl_bb():
    for lo in range(0, a.n, 512): blend_rows(np.arange(lo, min(a.n, lo + 512)))
m1, r1 = timeit(bl_b1); mb, rb = timeit(bl_bb)
rec("BLEND", "blend_scoring", "batch1", m1, r1, a.n); rec("BLEND", "blend_scoring", "batched", mb, rb, a.n)
RES["BLEND_config"] = list(map(str, bcfg)); log("blend done", m1, mb)

# ------------------------------------------------------------------ HRcov: lists + features + ranker
import fh_lists as FL
from patterns import EventIndex
from rt_common import TETables
FL.G.update(EI=EventIndex(np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]), N, R, FC.STEP), te=TETables(f"{W_}/te_{FC.DS}.npz"), R=R,
            sro=SRO, LP=LPv, tr=rows)
LOUT = {}
def lists_b1():
    for k in range(a.n):
        FL.G["LPchunk"] = LPs[k:k + 1]; FL.G["lo"] = k; LOUT[k] = FL.query_lists(k)
import multiprocessing as mp
def lists_bb():
    tasks = [(lo, min(a.n, lo + 500)) for lo in range(0, a.n, 500)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        out = list(pool.imap(FL._target_chunk, tasks, chunksize=1))
    return out
m1, r1 = timeit(lists_b1); mb, rb = timeit(lists_bb)
CP = np.stack([LOUT[k][0]["CP"] for k in range(a.n)]).astype(np.int64); XPc = np.stack([LOUT[k][0]["XPc"] for k in range(a.n)])
CPf = np.load(f"{W_}/lists_tirgn/valid_CP.npy", mmap_mode="r"); vidx = np.load(f"{W_}/valid_sample_idx.npy")
pos_in_sample = np.searchsorted(vidx, rows); assert (vidx[pos_in_sample] == rows).all()
RES["HRcov_lists_equal_frozen"] = float((np.asarray(CPf[pos_in_sample]).astype(np.int64) == CP).all(1).mean())
rec("HRcov", "lists", "batch1", m1, r1, a.n); rec("HRcov", "lists", "batched", mb, rb, a.n); log("hrcov lists", m1, mb, RES["HRcov_lists_equal_frozen"])
from rs_features import Hist, DyadReader
Fx = S.filter_index(); keysF = np.repeat(Fx["keys"], Fx["end"] - Fx["start"]); allTrue = np.unique(keysF * N + Fx["objs"].astype(np.int64))
Hh = Hist(S); BM = DyadReader(S, Hh)
KC, W = FC.KC, FC.W_PAT
def feat_block(b):          # fh_features.py per-chunk block (verbatim formulas), rows b of the slice
    nb = len(b); s_, r_, o_, t__ = SRO[b, 0], SRO[b, 1], SRO[b, 2], SRO[b, 3]
    row = LPs[b]; glp = row[np.arange(nb), o_]; fk = S.fkey(s_, r_, t__)
    srow = np.sort(row, axis=1); top200 = -np.sort(-np.partition(row, N - 200, axis=1)[:, N - 200:], axis=1)
    p1 = np.exp(top200[:, 0].astype(np.float64)); pt = np.exp(top200[:, :KC].astype(np.float64)); pn = pt / pt.sum(1, keepdims=True)
    ent50 = -(pn * np.log(pn + 1e-12)).sum(1); tail_mass = np.clip(1.0 - np.exp(top200.astype(np.float64)).sum(1), 1e-12, 1.0)
    cand = CP[b]; valid = cand >= 0; cz = np.maximum(cand, 0); lp = row[np.arange(nb)[:, None], cz].astype(np.float64)
    brank = np.empty((nb, W))
    for i in range(nb): brank[i] = N - np.searchsorted(srow[i], lp[i], side="right") + 1
    code = fk[:, None] * N + cz; pos = np.clip(np.searchsorted(allTrue, code), 0, len(allTrue) - 1)
    hf = Hh.feats(s_, r_, t__, np.where(valid, cand, -1)); A_, _ = BM.score(s_, r_, t__, np.where(valid, cand, -1), hf["dyad_hi"])
    Xf = np.zeros((nb, W, 25), np.float32); lpf = np.where(valid, lp, -30.0)
    Xf[:, :, 0] = np.maximum(lpf, -30.0); Xf[:, :, 1] = Xf[:, :, 0] - top200[:, :1]; Xf[:, :, 2] = np.log(np.maximum(brank, 1)); Xf[:, :KC, 3] = 1.0
    Xf[:, :, 4] = p1[:, None]; Xf[:, :, 5] = ent50[:, None]; Xf[:, :, 6] = (top200[:, 0] - top200[:, 1])[:, None]; Xf[:, :, 7] = np.log(tail_mass)[:, None]
    Xf[:, :, 8] = np.log1p(hf["ntrip"]); Xf[:, :, 9] = np.log1p(hf["ndyad"]); Xf[:, :, 10] = hf["rec_trip"]; Xf[:, :, 11] = hf["rec_dyad"]
    Xf[:, :, 12] = np.log1p(hf["nobjrel"]); Xf[:, :, 13] = np.log1p(hf["nent"]); Xf[:, :, 14] = hf["rec_objrel"]; Xf[:, :, 15] = np.log1p(hf["ntrip30"])
    Xf[:, :, 16] = A_; Xf[:, :, 17] = np.log1p(hf["nquery"])[:, None]; Xf[:, :, 18] = np.log1p(hf["nsubj"])[:, None]
    Xf[:, :, 19:25] = XPc[b].astype(np.float32); Xf[~valid] = 0.0
    return Xf, np.where(valid, lp, -np.inf)
XH = {}
def feat_b1():
    for i in range(a.n): XH[i] = feat_block(np.array([i]))
def feat_bb(): XH["all"] = feat_block(np.arange(a.n))
m1, r1 = timeit(feat_b1); mb, rb = timeit(feat_bb)
Xfz = np.load(f"{W_}/feat_tirgn/valid_X.npy", mmap_mode="r")
RES["HRcov_features_vs_frozen_max_abs"] = float(np.abs(XH["all"][0].astype(np.float16).astype(np.float32) - np.asarray(Xfz[pos_in_sample]).astype(np.float32)).max())
rec("HRcov", "features", "batch1", m1, r1, a.n); rec("HRcov", "features", "batched", mb, rb, a.n); log("hrcov feats", m1, mb, RES["HRcov_features_vs_frozen_max_abs"])
import g2_train as G2
Mh = torch.load(f"{W_}/train_tirgn/models/HRCOV_PAT_s0.pt", map_location=dev, weights_only=False)
neth = G2.Net(len(Mh["cols"])).to(dev); neth.load_state_dict(Mh["state_dict"]); neth.eval()
XHa, LPH = XH["all"]; muh = torch.as_tensor(Mh["mu"], device=dev); sdh = torch.as_tensor(Mh["sd"], device=dev)
XHt = torch.as_tensor(XHa[:, :, Mh["cols"]], device=dev); LPHt = torch.as_tensor(LPH, device=dev, dtype=torch.float32); CPt = torch.as_tensor(CP, device=dev)
def hr_score(ix):
    with torch.no_grad():
        f = neth((XHt[ix] - muh) / sdh); sc = Mh["cal_alpha"] * torch.where(torch.isfinite(LPHt[ix]), LPHt[ix], torch.full_like(LPHt[ix], -1e9)) + Mh["cal_beta"] * f
        sc = torch.where(CPt[ix] >= 0, sc, torch.full_like(sc, float("-inf"))); mint = sc[:, :KC].min(1).values
        prom = (torch.arange(W, device=dev)[None, :] < KC) | (sc > mint[:, None]); key = torch.where(prom & (CPt[ix] >= 0), sc, torch.full_like(sc, float("-inf")))
        return torch.gather(CPt[ix], 1, torch.argsort(key, dim=1, descending=True))
def hr_b1():
    for i in range(a.n): hr_score(slice(i, i + 1))
def hr_bb(): hr_score(slice(0, a.n))
m1, r1 = timeit(hr_b1); mb, rb = timeit(hr_bb)
rec("HRcov", "ranker", "batch1", m1, r1, a.n); rec("HRcov", "ranker", "batched", mb, rb, a.n); log("hrcov ranker", m1, mb)

# ------------------------------------------------------------------ SA: RecB_xf row + strict history order + lists + features + ranker
import v78_common as C
freq = C.train_freq(FC.DD, N); gorder, _ = C.freq_order(freq); gorder_t = torch.as_tensor(gorder, device=dev)
HT = {}
def hist_rows(ix):
    rr = rel_s[ix]; Sx = HBv.recb_row(csr_xf, rows[ix], rr, xi.xi(torch.as_tensor(rr, device=dev)), dev, "tr")
    pos, ent = C.strict_positions(Sx, gorder_t); tot = Sx.double().sum(1); top = ent[:, :C.KMAX]
    q = torch.where(tot.unsqueeze(1) > 0, torch.gather(Sx, 1, top).double() / tot.clamp(min=1e-300).unsqueeze(1), torch.zeros_like(top, dtype=torch.float64))
    return top, q, pos
def sh_b1():
    for i in range(a.n): hist_rows(np.array([i]))
def sh_bb():
    out = [hist_rows(np.arange(lo, min(a.n, lo + 512))) for lo in range(0, a.n, 512)]
    HT["top"] = torch.cat([x[0] for x in out]).cpu().numpy(); HT["q"] = torch.cat([x[1] for x in out]).cpu().numpy(); HT["pos"] = torch.cat([x[2] for x in out])
m1, r1 = timeit(sh_b1); mb, rb = timeit(sh_bb)
Hz = np.load(f"{W_}/sa/hist/valid_hist.npz"); jj = np.searchsorted(Hz["rows"], rows)
RES["SA_hist_top75_equal_frozen"] = float((Hz["htop"][jj, :75] == HT["top"][:, :75]).all(1).mean())
rec("SA", "history_order", "batch1", m1, r1, a.n); rec("SA", "history_order", "batched", mb, rb, a.n); log("sa hist", m1, mb, RES["SA_hist_top75_equal_frozen"])
sys.argv = ["v78_lists.py", "--ds", "gdeltfw", "--hist", f"{W_}/sa/hist", "--out", "/tmp/fw_lat_lists", "--splits", "valid"]
VL = importlib.import_module("v78_lists")
VL.POOLS = [p for p in VL.POOLS if p[0] == "SA50"]
allq2 = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]]); h2, rr2, tt2, ts2 = allq2.T.astype(np.int64); TT2 = int(ts2.max()) + 1
s_all = np.concatenate([h2, tt2]); r_all = np.concatenate([rr2, rr2 + R]); o_all = np.concatenate([tt2, h2]); t_all = np.concatenate([ts2, ts2])
csrz = np.load(f"{W_}/recb_xf/recb_csr_valid_GDELTFW.npz")
VL.G.update(EI=FL.G["EI"], te=FL.G["te"], N=N, R=R, TT=TT2, AK=np.sort((r_all * N + o_all) * TT2 + t_all), TK=np.sort(r_all * TT2 + t_all), freq=freq,
            pos_buf=np.full(N, -1, np.int64), base={}, sro=SRO, crow=rows, htop=HT["top"][:, :75].astype(np.int64), ptr=csrz["ptr"], ccand=csrz["cand"],
            cpsi=csrz["psi_tr"], alpha=np.asarray(csrz["alpha_tr"], np.float64))
SL = {}
def sl_b1():
    for i in range(a.n): SL[i] = VL.one(i)
def sl_bb():
    with mp.get_context("fork").Pool(a.workers) as pool:
        return list(pool.imap_unordered(VL.chunk, np.array_split(np.arange(a.n), max(1, a.n // 500)), chunksize=1))
m1, r1 = timeit(sl_b1); mb, rb = timeit(sl_bb)
blk = np.stack([SL[i][1]["SA50"][0] for i in range(a.n)]).astype(np.int64); patX = np.stack([SL[i][1]["SA50"][1] for i in range(a.n)]).astype(np.float32)
bz = np.load(f"{W_}/sa/lists/valid_SA50_blk.npy", mmap_mode="r"); RES["SA_lists_equal_frozen"] = float((np.asarray(bz[jj]).astype(np.int64) == blk).all(1).mean())
rec("SA", "lists", "batch1", m1, r1, a.n); rec("SA", "lists", "batched", mb, rb, a.n); log("sa lists", m1, mb, RES["SA_lists_equal_frozen"])
K = 50; first = HT["top"][:, :K].astype(np.int64); CA = np.concatenate([first, blk], 1)
def sa_feat(b):             # v78_pools.py SA-pool block (evidence + is_extra + pattern columns; SA16 uses 8-19, 21-24)
    s_, r_, t__ = SRO[b, 0], SRO[b, 1], SRO[b, 3]; Uc = CA[b]
    hf = Hh.feats(s_, r_, t__, Uc); A_, _ = BM.score(s_, r_, t__, Uc, hf["dyad_hi"])
    X = np.zeros((len(b), 2 * K, 25), np.float32)
    X[..., 8] = np.log1p(hf["ntrip"]); X[..., 9] = np.log1p(hf["ndyad"]); X[..., 10] = hf["rec_trip"]; X[..., 11] = hf["rec_dyad"]
    X[..., 12] = np.log1p(hf["nobjrel"]); X[..., 13] = np.log1p(hf["nent"]); X[..., 14] = hf["rec_objrel"]; X[..., 15] = np.log1p(hf["ntrip30"])
    X[..., 16] = A_; X[..., 17] = np.log1p(hf["nquery"])[:, None]; X[..., 18] = np.log1p(hf["nsubj"])[:, None]
    X[:, K:, 19] = (Uc[:, K:] >= 0); X[:, :, 20:25] = patX[b]; X[Uc < 0] = 0.0
    return X
XS = {}
def sf_b1():
    for i in range(a.n): sa_feat(np.array([i]))
def sf_bb(): XS["all"] = sa_feat(np.arange(a.n))
m1, r1 = timeit(sf_b1); mb, rb = timeit(sf_bb)
Xsz = np.load(f"{W_}/sa/pools/SA50/valid_X.npy", mmap_mode="r"); cols16 = list(range(8, 19)) + [19, 21, 22, 23, 24]
RES["SA_features_vs_frozen_max_abs"] = float(np.abs(XS["all"][:, :, cols16].astype(np.float16).astype(np.float32) - np.asarray(Xsz[jj])[:, :, cols16].astype(np.float32)).max())
rec("SA", "features", "batch1", m1, r1, a.n); rec("SA", "features", "batched", mb, rb, a.n); log("sa feats", m1, mb, RES["SA_features_vs_frozen_max_abs"])
from v78_train import MLP
Ms = torch.load(f"{W_}/sa/train/models/SA_W100_s0.pt", map_location=dev, weights_only=False)
nets = MLP(len(Ms["cols"])).to(dev); nets.load_state_dict(Ms["state_dict"]); nets.eval()
XSt = torch.as_tensor(XS["all"][:, :, Ms["cols"]], device=dev); mus = torch.as_tensor(Ms["mu"], device=dev); sds = torch.as_tensor(Ms["sd"], device=dev)
CAt = torch.as_tensor(CA, device=dev)
def sa_score(ix):
    with torch.no_grad():
        sc = nets((XSt[ix] - mus) / sds); sc = torch.where(CAt[ix] >= 0, sc, torch.full_like(sc, float("-inf"))); mint = sc[:, :K].min(1).values
        prom = (torch.arange(2 * K, device=dev)[None, :] < K) | (sc > mint[:, None]); key = torch.where(prom & (CAt[ix] >= 0), sc, torch.full_like(sc, float("-inf")))
        return torch.gather(CAt[ix], 1, torch.argsort(key, dim=1, descending=True))
def ss_b1():
    for i in range(a.n): sa_score(slice(i, i + 1))
def ss_bb(): sa_score(slice(0, a.n))
m1, r1 = timeit(ss_b1); mb, rb = timeit(ss_bb)
rec("SA", "ranker", "batch1", m1, r1, a.n); rec("SA", "ranker", "batched", mb, rb, a.n); log("sa ranker", m1, mb)

# ------------------------------------------------------------------ EXT blend part (comparator = blend) + ordering
if a.comparator and a.ct_dump:
    comp = json.load(open(a.comparator)); RES["comparator"] = comp
    sys.path.insert(0, CS); import fw_ext as FE
    Md = dict(top_ids=np.load(f"{a.ct_dump}/top_ids.npy", mmap_mode="r"), top_scores=np.load(f"{a.ct_dump}/top_scores.npy", mmap_mode="r"))
    def ext_rows(ix):
        Sx, Zx = FE.method_rows(Md, rows[ix], N, dev)
        Y = Sx if comp["variant"] == "alone" else (1 - comp["a"]) * LPt[ix].double() + comp["a"] * torch.log_softmax(Zx / comp["tau"], dim=1)
        return torch.argsort(Y, dim=1, descending=True)
    def ex_b1():
        for i in range(a.n): ext_rows(np.array([i]))
    def ex_bb():
        for lo in range(0, a.n, 128): ext_rows(np.arange(lo, min(a.n, lo + 128)))
    m1, r1 = timeit(ex_b1); mb, rb = timeit(ex_bb)
    rec("EXT", "blend_scoring", "batch1", m1, r1, a.n); rec("EXT", "blend_scoring", "batched", mb, rb, a.n); log("ext", m1, mb)
RES["elapsed_s"] = round(time.time() - T0, 1)
json.dump(RES, open(a.out, "w"), indent=1)
log("ALL DONE")
