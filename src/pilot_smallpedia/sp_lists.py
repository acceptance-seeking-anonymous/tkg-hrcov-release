#!/usr/bin/env python3
"""pilot_smallpedia / sp_lists.py -- per-base candidate lists, candidate layouts and coverage + static features (CPU).

Derived from pilot_polecat/pc_lists.py (pattern prior + stage-2 bridge unchanged except: base probabilities of
pattern candidates are EXACT, read from the full log-prob rows, instead of top-200 + uniform tail).  New:
  * static-KG list L_st (sp_static: S / SS / ST / TS typed paths over the union of the static Wikidata graph and the
    temporal history before t; ranked by summed path-type precision, then best precision, then base log-prob)
  * layout A [W=250] = C_cov (pilot-1, 150: top-50 ++ <=50 pattern ++ bridge to 150) | static block (first 50 of L_st
    outside C_cov) | degree-matched random block (one entity per static-block candidate, drawn from the same
    train-degree bin, outside everything before; density control)
  * layout B [W=150] = top-50 | first 100 of L_st outside top-50
  * features per candidate: COV_FEATS (11, pc_lists definition) and ST_FEATS (12) for both layouts.
Base top-K / anchors for valid & test come from the full log-prob rows (sp_fullrows / sp_tirgn); the bridge cohort
(train rows) uses the dump, exactly as pilot 1.
  python3 sp_lists.py --base regcn --workers 16
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sp_common import (DS, DUMPS, WORK, KC, BP, BB, BS, BD, BS_B, W_A, W_B, LP_LEN, LB_LEN, LS_LEN, COV_FEATS, ST_FEATS, SEEDS,
                       N_ENT, ST_DEG_CAP, Splits, BaseDump, split_queries, jdump, log_fn, fullrows, load_static,
                       ST_MODE, MODE_SFX)
from patterns import EventIndex, instantiate, decode
from rt_common import TETables, M1, M2
import sp_static as SS
import multiprocessing as mp

G = {}
E1 = np.e - 1.0
PAT_FILL = -6.0
KREF = 200
ST_FILL = -16.0
W_COV = KC + BP + BB


def make_net(nf):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(nf, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))


def np_mlp(W, X):
    h = np.maximum(X @ W[0] + W[1], 0.0); h = np.maximum(h @ W[2] + W[3], 0.0)
    return (h @ W[4] + W[5])[:, 0]


def bridge_score(X18):
    s = np.zeros(len(X18))
    for (W, mu, sd) in G["nets"]:
        s += np_mlp(W, (X18 - mu) / sd)
    return s / len(G["nets"])


def bridge_one(s, r, o, tki, tks):
    b = G["BI"].build(s, r, o, tki, tks)
    if b is None:
        return None
    return b, np.concatenate([b["X"], b["casc"]], 1).astype(np.float32)


def _cohort_chunk(rows):
    D = G["D"]; out = []
    for i in rows:
        s, r, o, t = (int(x) for x in D.sro[i])
        z = bridge_one(s, r, o, D.topk_ids[i, :KREF], D.topk_scores[i, :KREF])
        if z is None or not z[0]["reached"]:
            out.append((int(i), None, -1)); continue
        b, X18 = z
        out.append((int(i), X18, int(np.flatnonzero(b["cands"] == o)[0])))
    return out


def lookup(sorted_c, C):
    if len(sorted_c) == 0:
        return np.zeros(len(C), np.int64), np.zeros(len(C), bool)
    j = np.minimum(np.searchsorted(sorted_c, C), len(sorted_c) - 1)
    return j, (sorted_c[j] == C) & (C >= 0)


def cov_feats(C, pat, br, ncov_from):
    X = np.zeros((len(C), len(COV_FEATS)), np.float32)
    X[:, 0] = PAT_FILL; X[:, 4] = np.log1p(LP_LEN + 1); X[:, 7] = np.log1p(LB_LEN + 1)
    if pat:
        j, m = lookup(pat["cand"], C)
        X[m, 0] = pat["best"][j[m]]; X[m, 1] = pat["bte"][j[m]]; X[m, 2] = np.log1p(pat["npat"][j[m]]); X[m, 3] = 1.0
        X[m, 4] = np.log1p(np.minimum(pat["rank"][j[m]], LP_LEN) + 1)
    if br:
        j, m = lookup(br["cand"], C)
        X[:, 6] = br["smin"] - 1.0
        X[m, 5] = 1.0; X[m, 6] = br["score"][j[m]]; X[m, 7] = np.log1p(np.minimum(br["rank"][j[m]], LB_LEN) + 1)
        X[m, 8] = np.log1p(br["pm_subj"][j[m]]); X[m, 9] = np.log1p(br["ninterm"][j[m]])
    X[ncov_from:, 10] = 1.0
    X[C < 0] = 0.0
    return X


def st_feats(C, st, s, st_from, direct=None):
    SI = G["SI"]
    X = np.zeros((len(C), len(ST_FEATS)), np.float32)
    X[:, 5] = ST_FILL; X[:, 9] = np.log1p(LS_LEN + 1)
    Cc = np.maximum(C, 0)
    X[:, 7] = np.log1p(SI.deg[Cc]); X[:, 11] = np.log1p(SI.deg[s])
    if direct is not None:          # strict mode: static degree of a direct-edge candidate without its s-c edges
        j, m = lookup(direct[0], C)
        X[m, 7] = np.log1p(SI.deg[Cc[m]] - direct[1][j[m]])
    if st:
        j, m = lookup(st["cand"], C)
        by = st["by"][j[m]]
        X[m, 0] = (by[:, 0] > 0); X[m, 1] = np.log1p(by[:, 0]); X[m, 2] = np.log1p(by[:, 2]); X[m, 3] = np.log1p(by[:, 3])
        X[m, 4] = np.log1p(by[:, 1]); X[m, 5] = np.maximum(st["best"][j[m]], ST_FILL); X[m, 6] = np.log1p(st["shared"][j[m]])
        X[m, 8] = 1.0; X[m, 9] = np.log1p(np.minimum(st["rank"][j[m]], LS_LEN) + 1)
    X[st_from:, 10] = 1.0
    X[C < 0] = 0.0
    return X


def query_lists(k):
    """k = position within the target split -> lists, layouts, features, gold diagnostics."""
    EI = G["EI"]; te = G["te"]; N = G["N"]; R = G["R"]; SI = G["SI"]; STE = G["STE"]
    s, r, o, t = (int(x) for x in G["sro"][k])
    lp = G["LPchunk"][k - G["lo"]].astype(np.float64)
    tki = np.argpartition(-lp, KREF)[:KREF]; tki = tki[np.argsort(-lp[tki], kind="stable")]
    tks = lp[tki]
    top50 = tki[:KC]
    res = dict(Lpat=np.full(LP_LEN, -1, np.int32), Lpat_u=np.zeros(LP_LEN, np.float16), Lbr=np.full(LB_LEN, -1, np.int32),
               Lbr_s=np.zeros(LB_LEN, np.float32), Lst=np.full(LS_LEN, -1, np.int32), Lst_s=np.zeros(LS_LEN, np.float32),
               CA=np.full(W_A, -1, np.int32), CB=np.full(W_B, -1, np.int32),
               XAc=np.zeros((W_A, len(COV_FEATS)), np.float16), XAs=np.zeros((W_A, len(ST_FEATS)), np.float16),
               XBc=np.zeros((W_B, len(COV_FEATS)), np.float16), XBs=np.zeros((W_B, len(ST_FEATS)), np.float16))
    q = dict(gpos_pat=-1, gpos_br=-1, gpos_st=-1, g_in_union=False, g_in_bridge=False, g_in_static=False, br_size=0, n_union=0,
             st_size=0, has_pat=False, has_br=False, has_st=False, g_st_code=-1, g_st_ntypes=0)
    # ---------------- pattern prior (exact base probabilities)
    pat = {}
    d = instantiate(EI, s, t, M1, M2)
    if d is not None:
        q["has_pat"] = True
        pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64); P_ = len(d["pid"])
        mass = np.bincount(pp, weights=np.exp(lp[cc]), minlength=P_)
        hop, rel1, lb1, rel2, lb2 = decode(d["pid"], R)
        te_h = te.features(r, d["pid"], hop, rel1, rel2, R)[:, 8].astype(np.float64)
        util = te_h - np.log1p(mass * E1)
        cand, inv = np.unique(cc, return_inverse=True)
        best = np.full(len(cand), -np.inf); np.maximum.at(best, inv, util[pp])
        bte = np.zeros(len(cand)); np.maximum.at(bte, inv, te_h[pp])
        supp = np.bincount(inv, weights=np.maximum(util[pp], 0.0), minlength=len(cand))
        npat = np.bincount(inv, minlength=len(cand))
        order = np.lexsort((-lp[cand], -supp, -best))
        cand_o = cand[order]; rank_o = np.empty(len(cand), np.int64); rank_o[order] = np.arange(len(cand))
        L = cand_o[:LP_LEN]; res["Lpat"][:len(L)] = L; res["Lpat_u"][:len(L)] = best[order][:LP_LEN]
        q["n_union"] = len(cand)
        gi = np.searchsorted(cand, o)
        if gi < len(cand) and cand[gi] == o:
            q["g_in_union"] = True
            if rank_o[gi] < LP_LEN: q["gpos_pat"] = int(rank_o[gi])
        pat = dict(cand=cand, best=best, bte=bte, npat=npat, rank=rank_o)
    # ---------------- bridge
    br = {}
    z = bridge_one(s, r, o, tki, tks)
    if z is not None:
        b, X18 = z
        q["has_br"] = True
        sc = bridge_score(X18); order = np.argsort(-sc, kind="stable"); bc = b["cands"].astype(np.int64)
        res["Lbr"][:min(LB_LEN, len(bc))] = bc[order][:LB_LEN]; res["Lbr_s"][:min(LB_LEN, len(bc))] = sc[order][:LB_LEN]
        q["br_size"] = len(bc); q["g_in_bridge"] = bool(b["reached"])
        rank_b = np.empty(len(bc), np.int64); rank_b[order] = np.arange(len(bc))
        if b["reached"]:
            gp = int(rank_b[np.flatnonzero(bc == o)[0]])
            if gp < LB_LEN: q["gpos_br"] = gp
        srt = np.argsort(bc)
        br = dict(cand=bc[srt], score=sc[srt], rank=rank_b[srt], pm_subj=b["X"][srt, 1], ninterm=b["X"][srt, 2], smin=float(sc.min()))
    # ---------------- static KG
    st = {}
    pth = SS.enumerate_paths(SI, EI, s, t, nodirect=(ST_MODE != "full"))
    direct = None
    if ST_MODE == "nodirect_strict":   # no static evidence at all for candidates directly linked to s
        x1_, _ = SI.nbrs(s); dn = x1_[(x1_ < N) & (x1_ != s)]
        if len(dn):
            direct = np.unique(dn, return_counts=True)
            kp = ~np.isin(pth["cand"], direct[0])
            pth = {k_: v_[kp] for k_, v_ in pth.items()}
    cs = SS.cand_stats(pth, r, STE, SI, 2 * R)
    if cs is not None:
        q["has_st"] = True
        uc = cs["cand"]
        order = np.lexsort((-lp[uc], -cs["best"], -cs["ssum"]))
        rank_s = np.empty(len(uc), np.int64); rank_s[order] = np.arange(len(uc))
        Ls = uc[order][:LS_LEN]; res["Lst"][:len(Ls)] = Ls; res["Lst_s"][:len(Ls)] = np.log(cs["ssum"][order][:LS_LEN] + 1e-12)
        q["st_size"] = len(uc)
        gi = np.searchsorted(uc, o)
        if gi < len(uc) and uc[gi] == o:
            q["g_in_static"] = True
            if rank_s[gi] < LS_LEN: q["gpos_st"] = int(rank_s[gi])
            gm = pth["cand"] == o
            pg, _ = STE.prec(r, pth["code"][gm], pth["shape"][gm], 2 * R, SI.D1)
            q["g_st_code"] = int(pth["code"][gm][int(np.argmax(pg))]); q["g_st_ntypes"] = int(gm.sum())
        st = dict(cand=uc, by=cs["by"], best=cs["best"], shared=cs["shared"], rank=rank_s)
    # ---------------- layout A: C_cov (pilot 1) | static | degree-matched random
    C = list(top50.tolist()); seen = set(C)
    for e in res["Lpat"]:
        if len(C) >= KC + BP or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    n_pb = len(C)
    for e in res["Lbr"]:
        if len(C) >= W_COV or e < 0: break
        if int(e) not in seen: C.append(int(e)); seen.add(int(e))
    n_cov = len(C)
    CA = np.full(W_A, -1, np.int64); CA[:n_cov] = C
    stb = []
    for e in res["Lst"]:
        if len(stb) >= BS or e < 0: break
        if int(e) not in seen: stb.append(int(e)); seen.add(int(e))
    CA[W_COV:W_COV + len(stb)] = stb
    seen.add(s)
    rng = np.random.default_rng(1000003 * (G["seed_off"] + k) + 17)
    dmb = []
    for e in stb:
        pool = G["bins"][G["dbin"][e]]
        for _ in range(20):
            x = int(pool[rng.integers(len(pool))])
            if x not in seen:
                dmb.append(x); seen.add(x); break
    CA[W_COV + BS:W_COV + BS + len(dmb)] = dmb
    res["CA"][:] = CA
    res["XAc"][:] = cov_feats(CA, pat, br, KC).astype(np.float16)
    res["XAs"][:] = st_feats(CA, st, s, W_COV, direct).astype(np.float16)
    # ---------------- layout B: top50 | first 100 of L_st outside top-50
    CB = np.full(W_B, -1, np.int64); CB[:KC] = top50; seenB = set(top50.tolist()); jb = KC
    for e in res["Lst"]:
        if jb >= W_B or e < 0: break
        if int(e) not in seenB: CB[jb] = int(e); seenB.add(int(e)); jb += 1
    res["CB"][:] = CB
    res["XBc"][:] = cov_feats(CB, pat, br, W_B).astype(np.float16)     # cov block flag off in layout B
    res["XBs"][:] = st_feats(CB, st, s, KC, direct).astype(np.float16)
    q.update(n_pat_block=n_pb - KC, n_br_block=n_cov - n_pb, n_st_block=len(stb), n_dm_block=len(dmb), n_stB_block=jb - KC)
    return res, q


def _target_chunk(args):
    lo, hi = args
    G["LPchunk"] = np.asarray(G["LP"][lo:hi]); G["lo"] = lo
    outs = [query_lists(k) for k in range(lo, hi)]
    res = {k: np.stack([o[0][k] for o in outs]) for k in outs[0][0]}
    q = {k: np.array([o[1][k] for o in outs]) for k in outs[0][1]}
    return lo, res, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--cohort_sample", type=int, default=30000)
    ap.add_argument("--cohort_n", type=int, default=8000)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    import torch
    torch.set_num_threads(8)
    sfx0 = "_smoke" if a.smoke else ""; sfx = sfx0 + MODE_SFX
    ref_dir = f"{WORK}/lists_{a.base}{sfx0}"          # pilot-2 (full-mode) lists: bridge rankers + C_cov reference
    out_dir = f"{WORK}/lists_{a.base}{sfx}"; os.makedirs(out_dir, exist_ok=True)
    S = Splits(DS); N, R = S.n_ent, S.R
    D = BaseDump(DUMPS[a.base], S); log("dump", D.N, D.K, D.load_s)
    summ = dict(base=a.base, host="serverA (CPU, tkg-gpu:latest)", workers=a.workers, static_mode=ST_MODE)
    rows = {sp: D.rows(sp) for sp in ("train", "valid", "test")}
    for sp in ("valid", "test"):
        assert (D.sro[rows[sp]].astype(np.int64) == split_queries(S, sp)).all(), f"dump row order != split order ({sp})"
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    targets = {"valid": vidx, "test": np.arange(len(rows["test"]))}
    if a.smoke:
        targets = {k: v[:a.smoke] for k, v in targets.items()}
    allq = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
    EI = EventIndex(allq, N, R, 1)
    te = TETables(f"{WORK}/te_{DS}.npz")
    STE = SS.StaticTE(f"{WORK}/ste_{DS}.npz")
    SI = SS.StaticIndex(load_static(), N_ENT, ST_DEG_CAP)
    from bridge_features import BridgeIndex
    BI = BridgeIndex(S.q["train"], N, R); log("bridge index", BI.build_s, "s; DEG_CAP", BI.DEG_CAP)
    freq = S.freq
    dbin = np.where(freq == 0, 0, 1 + np.floor(np.log2(np.maximum(freq, 1)))).astype(np.int64)
    bins = [np.flatnonzero(dbin == b_) for b_ in range(dbin.max() + 1)]
    G.update(D=D, EI=EI, te=te, BI=BI, N=N, R=R, SI=SI, STE=STE, dbin=dbin, bins=bins)
    if ST_MODE != "full":
        # leakage-control modes: C_cov must be identical to pilot 2 -> reuse the pilot-2 bridge rankers (no retraining)
        nets = []
        for sd_ in SEEDS:
            ck = torch.load(f"{ref_dir}/bridge_ranker_s{sd_}.pt", map_location="cpu", weights_only=False); w = ck["state_dict"]
            Wt = [w["0.weight"].numpy().T, w["0.bias"].numpy(), w["2.weight"].numpy().T, w["2.bias"].numpy(),
                  w["4.weight"].numpy().T, w["4.bias"].numpy()]
            nets.append((Wt, np.asarray(ck["mu"]).astype(np.float32), np.asarray(ck["sd"]).astype(np.float32)))
        rs_ = json.load(open(f"{ref_dir}/lists_summary.json"))
        summ["bridge_cohort"] = rs_.get("bridge_cohort"); summ["bridge_ranker"] = dict(rs_.get("bridge_ranker", {}), loaded_from=ref_dir)
        log("bridge rankers loaded from", ref_dir)
    else:
        # ---------------- stage-2 bridge ranker on coverage-hard TRAIN queries (pc_lists recipe, dump train rows)
        gf = D.extra["gold_rank_filt"]
        hard = rows["train"][gf[rows["train"]] > KC]
        rng = np.random.default_rng(0)
        samp = rng.choice(hard, min(a.cohort_sample if not a.smoke else 3000, len(hard)), replace=False)
        t1 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_cohort_chunk, np.array_split(samp, a.workers * 8), chunksize=1)
        coh = [x for p in parts for x in p]
        keep = [x for x in coh if x[1] is not None][:(a.cohort_n if not a.smoke else 1000)]
        summ["bridge_cohort"] = dict(n_hard_train=int(len(hard)), n_sampled=len(coh), reach_rate=float(np.mean([x[1] is not None for x in coh])),
                                     n_train=len(keep), median_cset=float(np.median([len(x[1]) for x in keep])), seconds=round(time.time() - t1, 1))
        log("cohort", json.dumps(summ["bridge_cohort"]))
        nets = []; ntr = []
        for sd_ in SEEDS:
            rg = np.random.default_rng(sd_); torch.manual_seed(sd_)
            Xp, Xn = [], []
            for (_, X, gj) in keep:
                n = len(X)
                if n <= 1: continue
                negj = np.delete(np.arange(n), gj)
                take = rg.choice(negj, size=min(64, len(negj)), replace=False)
                Xp.append(np.repeat(X[gj:gj + 1], len(take), 0)); Xn.append(X[take])
            Xp = np.concatenate(Xp); Xn = np.concatenate(Xn)
            allx = np.concatenate([Xp, Xn]); mu = allx.mean(0); sdv = allx.std(0) + 1e-6
            Pp = torch.tensor((Xp - mu) / sdv); Pn = torch.tensor((Xn - mu) / sdv)
            net = make_net(Xp.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            nn_ = len(Pp)
            for ep in range(40 if not a.smoke else 3):
                perm = torch.randperm(nn_)
                for b0 in range(0, nn_, 4096):
                    idx = perm[b0:b0 + 4096]
                    loss = -torch.nn.functional.logsigmoid(net(Pp[idx]).squeeze(1) - net(Pn[idx]).squeeze(1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            Wt = [net[0].weight.detach().numpy().T, net[0].bias.detach().numpy(), net[2].weight.detach().numpy().T,
                  net[2].bias.detach().numpy(), net[4].weight.detach().numpy().T, net[4].bias.detach().numpy()]
            nets.append((Wt, mu.astype(np.float32), sdv.astype(np.float32))); ntr.append(int(nn_))
            torch.save(dict(state_dict=net.state_dict(), mu=mu, sd=sdv, seed=sd_), f"{out_dir}/bridge_ranker_s{sd_}.pt")
    G["nets"] = nets
    if ST_MODE == "full":
        summ["bridge_ranker"] = dict(features="X16 + casc (G12 arm5 without event-LM)", seeds=SEEDS, n_pairs=ntr, epochs=40, n_neg=64)
        log("bridge ranker trained")
    # ---------------- target splits
    for si, (split, tr) in enumerate(targets.items()):
        t1 = time.time(); n = len(tr)
        G["sro"] = split_queries(S, split)[tr]; G["seed_off"] = si * 10_000_000
        LPall = fullrows(a.base, split)
        assert LPall.shape[0] == len(rows[split])
        G["LP"] = LPall[tr] if a.smoke else LPall          # valid sample = full valid -> identity indexing
        if not a.smoke:
            assert (tr == np.arange(len(tr))).all()
        mm = lambda nm, shp, dt: np.lib.format.open_memmap(f"{out_dir}/{split}_{nm}.npy", mode="w+", dtype=dt, shape=shp)
        A = dict(Lpat=mm("Lpat", (n, LP_LEN), np.int32), Lpat_u=mm("Lpat_u", (n, LP_LEN), np.float16),
                 Lbr=mm("Lbr", (n, LB_LEN), np.int32), Lbr_s=mm("Lbr_s", (n, LB_LEN), np.float32),
                 Lst=mm("Lst", (n, LS_LEN), np.int32), Lst_s=mm("Lst_s", (n, LS_LEN), np.float32),
                 CA=mm("CA", (n, W_A), np.int32), CB=mm("CB", (n, W_B), np.int32),
                 XAc=mm("XAc", (n, W_A, len(COV_FEATS)), np.float16), XAs=mm("XAs", (n, W_A, len(ST_FEATS)), np.float16),
                 XBc=mm("XBc", (n, W_B, len(COV_FEATS)), np.float16), XBs=mm("XBs", (n, W_B, len(ST_FEATS)), np.float16))
        step = 500
        tasks = [(lo, min(n, lo + step)) for lo in range(0, n, step)]
        Q = {}
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (lo, res, q) in enumerate(pool.imap(_target_chunk, tasks, chunksize=1)):
                hi = lo + len(next(iter(q.values())))
                for nm, arr in res.items():
                    A[nm][lo:hi] = arr
                for nm, arr in q.items():
                    Q.setdefault(nm, []).append(arr)
                if k % 50 == 0:
                    log(split, hi, "/", n)
        for v in A.values():
            v.flush()
        Q = {k: np.concatenate(v) for k, v in Q.items()}
        np.savez(f"{out_dir}/{split}_q.npz", rows=tr, **Q)
        if ST_MODE != "full":
            rCA = np.load(f"{ref_dir}/{split}_CA.npy", mmap_mode="r"); rXc = np.load(f"{ref_dir}/{split}_XAc.npy", mmap_mode="r")
            neq_c = 0; neq_x = 0
            for lo_ in range(0, n, 20000):
                hi_ = min(n, lo_ + 20000)
                neq_c += int((np.asarray(A["CA"][lo_:hi_, :W_COV]) != np.asarray(rCA[lo_:hi_, :W_COV])).any(1).sum())
                xa_ = np.asarray(A["XAc"][lo_:hi_, :W_COV]); xb_ = np.asarray(rXc[lo_:hi_, :W_COV])
                neq_x += int((~((xa_ == xb_) | (np.isnan(xa_) & np.isnan(xb_)))).any((1, 2)).sum())
            log(split, "C_cov vs pilot-2: rows differing (cand, F_cov)", neq_c, neq_x)
            assert neq_c == 0 and neq_x == 0, f"C_cov differs from pilot 2 ({split}): {neq_c} / {neq_x} rows"
            ccov_identical = True
        else:
            ccov_identical = None
        summ[split] = dict(n=n, seconds=round(time.time() - t1, 1), has_pat=float(Q["has_pat"].mean()), has_br=float(Q["has_br"].mean()),
                           has_st=float(Q["has_st"].mean()), gold_in_union=float(Q["g_in_union"].mean()), gold_in_bridge=float(Q["g_in_bridge"].mean()),
                           gold_in_static=float(Q["g_in_static"].mean()), gold_in_Lst200=float((Q["gpos_st"] >= 0).mean()),
                           gold_in_Lpat300=float((Q["gpos_pat"] >= 0).mean()), gold_in_Lbr200=float((Q["gpos_br"] >= 0).mean()),
                           median_br_size=float(np.median(Q["br_size"])), median_union=float(np.median(Q["n_union"])), median_st_size=float(np.median(Q["st_size"])),
                           mean_blocks={k: float(Q[k].mean()) for k in ("n_pat_block", "n_br_block", "n_st_block", "n_dm_block", "n_stB_block")},
                           ccov_identical_to_pilot2=ccov_identical)
        log(split, json.dumps(summ[split]))
    summ["elapsed_s"] = round(time.time() - t0, 1)
    jdump(summ, f"lists_summary_{a.base}{sfx}.json"); jdump(summ, f"{out_dir}/lists_summary.json")
    os.system(f"chmod -R a+rX {out_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
