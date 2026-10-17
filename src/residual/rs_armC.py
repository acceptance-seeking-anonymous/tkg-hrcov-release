#!/usr/bin/env python3
"""
Residual track -- arm C: G10/G11 utility-as-feature stage-2 ranker (arm4 = X16 + reusability-gated event-LM
utility) trained on the g12 cold coverage-hard cohort (3 seeds, harness recipe), used as a BACKFILL through the
harness ConfidenceSwitch(q, M, B) rule with (q, M, B) selected on the valid sample.  Bridge caches exist only for
TiRGN; for LogCL / RE-GCN the TiRGN-anchored backfill lists are reused (row-matched by (s, r', o, ts)) and the
trigger uses the base's OWN top-1 probability (labelled cross-base reference row).
Fused rank convention = harness fused_rank (identical for conservative dumps; uses the exact base rank for the
untouched part on exact dumps):  gold in head -> head position;  gold in block -> 1 + nh + block position;
otherwise base filtered rank + #(block items not filtered and not visibly above the gold in the base).
"""
import os, sys, json, time, argparse
H = os.path.expandvars("$WORK_DIR"); TRACK = f"{H}/tracks/residual"
sys.path.insert(0, H)
import numpy as np, torch
from oih_common import Splits, BaseDump, DATASETS, metrics_from_ranks, filter_lookup, jdump
from bridge_cache import BridgeCache
from stage2_arms import train_arm, score_with
from fulltest_eval import BaseView, Q_GRID, M_GRID, B_GRID


def score_cache(bc, nets, B=200):
    n = bc.nq; out = [np.full((n, B), -1, np.int32) for _ in nets]
    for j in range(n):
        q = bc.query(j)
        if not q["built"] or len(q["cands"]) == 0:
            continue
        X = bc.arm_features(j, "arm4", q)
        for k, (net, mu, sd) in enumerate(nets):
            s = score_with(net, mu, sd, X); od = np.argsort(-s)[:B]; out[k][j, :len(od)] = q["cands"][od]
    return out


def fr(T, Ts, g, gs, rb, Fs, block_src, M, Bsz):
    head = T[:M]
    for p, e in enumerate(head):
        if e == g:
            return 1 + sum(1 for x in head[:p] if x not in Fs)
    hset = set(head); block = []
    for b in block_src:
        if b < 0: break
        if b in hset: continue
        block.append(b)
        if len(block) >= Bsz: break
    nh = sum(1 for x in head if x not in Fs)
    for j, e in enumerate(block):
        if e == g:
            return 1 + nh + sum(1 for x in block[:j] if x not in Fs)
    vis = dict(zip(T, Ts))
    add = sum(1 for b in block if b not in Fs and not (b in vis and vis[b] > gs))
    return int(rb) + add


def eval_cs(S, F, V, trig_pos, trig_lists, p1_all_valid, qmax, Vval, vtrig_pos, vtrig_lists):
    """valid-select (q, M, B) then apply to test; returns test ranks + record."""
    QG = [q for q in Q_GRID if q <= qmax + 1e-9]
    taus = {q: float(np.quantile(p1_all_valid, q)) for q in QG}
    cfgs = [(M, B) for M in M_GRID for B in B_GRID]
    def grid(view, pos, lists):
        T = view.T[pos].tolist(); Ts = view.Ts[pos].tolist(); out = {}
        oth = [set(int(x) for x in view.others(i)) for i in pos]
        Ls = [l.tolist() for l in lists]
        for (M, B) in cfgs:
            Mv = view.kref if M is None else M
            out[(M, B)] = np.array([fr(T[k], Ts[k], int(view.g[i]), float(view.gs[i]), view.base_rank[i], oth[k], Ls[k], Mv, B)
                                    for k, i in enumerate(pos)], np.int64)
        return out
    vg = grid(Vval, vtrig_pos, vtrig_lists)
    bv = metrics_from_ranks(Vval.base_rank)
    best = None; rows = []
    for q in QG:
        sel = Vval.p1[vtrig_pos] <= taus[q]
        for (M, B) in cfgs:
            rk = Vval.base_rank.copy(); rk[vtrig_pos[sel]] = vg[(M, B)][sel]
            m = metrics_from_ranks(rk); rec = dict(q=q, M=("K" if M is None else M), B=B, valid_MRR=m["MRR"],
                                                  valid_dMRR=round(m["MRR"] - bv["MRR"], 6), n_trig=int(sel.sum()))
            rows.append(rec)
            if best is None or m["MRR"] > best[0]["valid_MRR"]:
                best = (rec, (M, B), q)
    tg = grid(V, trig_pos, trig_lists)
    (M, B), q = best[1], best[2]
    sel = V.p1[trig_pos] <= taus[q]
    rk = V.base_rank.copy(); rk[trig_pos[sel]] = tg[(M, B)][sel]
    return rk, dict(selected=best[0], taus=taus, any_config_improves_valid=bool(best[0]["valid_dMRR"] > 0),
                    n_test_trig=int(sel.sum()), n_test_pool_mapped=int(len(trig_pos)), valid_grid_top5=sorted(rows, key=lambda r: -r["valid_MRR"])[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bases", default="tirgn,logcl,regcn")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
    DS = a.dataset; cfg = DATASETS[DS]; S = Splits(DS); F = S.filter_index()
    out_dir = f"{TRACK}/results/C"; os.makedirs(out_dir, exist_ok=True)
    bg = BridgeCache(DS, "g12"); tr = np.where(bg.qsplit == 0)[0]
    Q = {j: bg.arm_features(j, "arm4") for j in tr if bg.built[j]}
    seeds = [int(x) for x in a.seeds.split(",")]
    nets = []
    for sd_ in seeds:
        net, mu, sd, npairs = train_arm(bg, tr, "arm4", sd_, Q=Q); nets.append((net, mu, sd))
        torch.save(dict(state_dict=net.state_dict(), mu=mu, sd=sd, nf=int(len(mu)), arm="arm4", seed=sd_), f"{out_dir}/arm4_{DS}_seed{sd_}.pt")
    log("arm4 trained", len(nets))
    lists = {}
    for coh in ("valid_pool", "test_pool"):
        bc = BridgeCache(DS, coh); L = score_cache(bc, nets)
        lists[coh] = (np.asarray(bc.qrow).astype(np.int64), L, bc.meta)
        for k, sd_ in enumerate(seeds):
            np.savez(f"{out_dir}/backfill_g12arm4_s{sd_}_{DS}_{coh}.npz", rows=lists[coh][0], ids=L[k])
        log("scored", coh, bc.nq)
    qmax = lists["test_pool"][2]["q_max"]
    vsamp_t = np.load(f"{H}/cache/{DS}_tirgn_valid_pool/valid_sample_rows.npy")
    Dt = BaseDump(cfg["dumps"]["tirgn"], S)
    DtV = BaseDump(cfg["dumps"]["tirgn_valid"], S) if DS == "ICEWS18" else Dt
    key = lambda sro: list(map(tuple, sro.astype(np.int64)))
    res = dict(dataset=DS, host="serverA (CPU, tkg-gpu:latest)", seeds=seeds, q_max=qmax, bases={})
    for base in a.bases.split(","):
        D = Dt if base == "tirgn" else BaseDump(cfg["dumps"][base], S)
        DV = DtV if base == "tirgn" else D
        kref_v = 50 if (DS == "ICEWS18" and base == "tirgn") else D.K
        te_rows = D.rows("test"); va_all = DV.rows("valid")
        if base == "tirgn":
            vrows = vsamp_t; m_v = {int(r): int(r) for r in lists["valid_pool"][0]}; m_t = {int(r): int(r) for r in lists["test_pool"][0]}
        else:
            kv = {k: int(r) for k, r in zip(key(D.sro[va_all]), va_all)}
            kt = {k: int(r) for k, r in zip(key(D.sro[te_rows]), te_rows)}
            vrows = np.array([kv[k] for k in key(DtV.sro[vsamp_t]) if k in kv], np.int64)
            m_v = {int(r): kv.get(k, -1) for r, k in zip(lists["valid_pool"][0], key(DtV.sro[lists["valid_pool"][0]]))}
            m_t = {int(r): kt.get(k, -1) for r, k in zip(lists["test_pool"][0], key(Dt.sro[lists["test_pool"][0]]))}
        Vval = BaseView(S, DV, vrows, F, kref_v); V = BaseView(S, D, te_rows, F, D.K)
        vpos = {int(r): i for i, r in enumerate(vrows)}; tpos = {int(r): i for i, r in enumerate(te_rows)}
        p1v = np.exp(DV.topk_scores[va_all, 0].astype(np.float64))
        brec = dict(n_valid_sample=int(len(vrows)), base_test=metrics_from_ranks(V.base_rank), seeds={})
        rk_all = {}
        for k, sd_ in enumerate(seeds):
            vl = [(vpos[m_v[int(r)]], lists["valid_pool"][1][k][j]) for j, r in enumerate(lists["valid_pool"][0]) if m_v.get(int(r), -1) in vpos]
            tl = [(tpos[m_t[int(r)]], lists["test_pool"][1][k][j]) for j, r in enumerate(lists["test_pool"][0]) if m_t.get(int(r), -1) in tpos]
            vp = np.array([x[0] for x in vl], np.int64); vL = [x[1] for x in vl]
            tp = np.array([x[0] for x in tl], np.int64); tL = [x[1] for x in tl]
            rk, rec = eval_cs(S, F, V, tp, tL, p1v, qmax, Vval, vp, vL)
            rec["test"] = metrics_from_ranks(rk); rec["test_dMRR"] = round(rec["test"]["MRR"] - brec["base_test"]["MRR"], 6)
            brec["seeds"][sd_] = rec; rk_all[f"C_s{sd_}"] = rk.astype(np.int32)
            log(base, "seed", sd_, rec["selected"], rec["test"]["MRR"])
        np.savez(f"{out_dir}/ranksC_{DS}_{base}.npz", rows=te_rows, base=V.base_rank.astype(np.int32), **rk_all)
        res["bases"][base] = brec
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"armC_{DS}.json"); jdump(res, f"{out_dir}/armC_{DS}.json")
    os.system(f"chmod -R a+rX {out_dir}")


if __name__ == "__main__":
    main()
