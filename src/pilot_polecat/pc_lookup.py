#!/usr/bin/env python3
"""pilot_polecat / pc_lookup.py -- full-score-row pass of a frozen base (GPU), two modes.

mode lookup  : replays the base exactly as base_dump.py (ground-truth online history, last 10 snapshots;
               LogCL global copy-history rolled over every earlier snapshot) for the valid sample and every
               test query, and writes, per query, from the FULL 150,931-entity log-prob row:
                 lp_req [n,500]  exact base log-prob of the requested coverage candidates (Lpat ++ Lbr)
                 rk_req [n,500]  raw base rank of each requested candidate (1 + #entities with higher log-prob)
                 ck     [n,6]    #non-filter entities (gold excluded) with log-prob > gold + delta, delta in D_GRID
                 gold_lp, rank_filt (exact, re-derived), tie_n (#non-filter entities tied with the gold)
               and, for TEST, the official py-tgb evaluation of the base: negatives from
               TKGNegativeEdgeSampler.query_batch (time-filtered, TGB ids via the verified id map) scored by
               tgb Evaluator.eval per query (MRR, hits@10), plus the TGB rank of every query.
mode systems : official py-tgb evaluation of re-ranked systems.  --systems name=path.npy,...  each npy is
               [n_test, W] int32: the promoted candidate list in final fused order (-1 pad); the system's
               full score row = base log-prob row with those entities lifted above everything in that order.
  python3 pc_lookup.py --base regcn --mode lookup
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from pc_common import (H, TRACKS, DS, DD, DUMPS, CKPT, WORK, P, D_GRID, LP_LEN, LB_LEN, Splits, split_queries, jdump, log_fn)
from regcn import read_quads, load_stat, group_by_time, make_edges, REGCN
from logcl import LogCL, GlobalHist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", default="lookup")
    ap.add_argument("--systems", default="")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no_official", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = Splits(DS); N, R = S.n_ent, S.R
    ck = torch.load(CKPT[a.base], map_location="cpu", weights_only=False)
    L = 10
    model = (REGCN if a.base == "regcn" else LogCL)(N, R, 200, 2, 0.2, L).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    train = read_quads(os.path.join(DD, "train.txt")); valid = read_quads(os.path.join(DD, "valid.txt")); test = read_quads(os.path.join(DD, "test.txt"))
    pairs = {"train": group_by_time(train), "valid": group_by_time(valid), "test": group_by_time(test)}
    F_ = S.filter_index()
    keysF = np.repeat(F_["keys"], F_["end"] - F_["start"]); allTrue = np.unique(keysF * N + F_["objs"].astype(np.int64))
    D = np.load(DUMPS[a.base])
    dsp = D["split"]; d_gs = D["gold_score"]; d_gf = D["gold_rank_filt"]
    rows_sp = {sp: np.flatnonzero(dsp == c) for sp, c in (("valid", 1), ("test", 2))}
    vidx = np.load(f"{WORK}/valid_sample_idx.npy")
    req = {}
    if a.mode == "lookup":
        req_rows = {"valid": vidx, "test": np.arange(len(rows_sp["test"]))}
        if a.smoke:
            req_rows = {k: v[:a.smoke] for k, v in req_rows.items()}
        for sp in ("valid", "test"):
            ld = f"{WORK}/lists_{a.base}" + ("_smoke" if a.smoke else "")
            Lp = np.load(f"{ld}/{sp}_Lpat.npy", mmap_mode="r"); Lb = np.load(f"{ld}/{sp}_Lbr.npy", mmap_mode="r")
            req[sp] = np.concatenate([np.asarray(Lp), np.asarray(Lb)], 1)
            assert len(req[sp]) == len(req_rows[sp])
    else:
        req_rows = {"test": np.arange(len(rows_sp["test"]))}
        systems = {}
        for kv in a.systems.split(","):
            nm, pth = kv.split("="); systems[nm] = np.load(pth, mmap_mode="r")
    if a.smoke and a.mode != "lookup":
        req_rows = {k: v[:a.smoke] for k, v in req_rows.items()}
    # official evaluator setup (TGB ids)
    official = (not a.no_official) and ("test" in req_rows)
    if official:
        from tgb.linkproppred.tkg_negative_sampler import TKGNegativeEdgeSampler
        from tgb.linkproppred.evaluate import Evaluator
        import importlib.metadata as im
        h2t = np.load(f"{P}/stage/h2t_entity.npy"); t2h = np.load(f"{P}/stage/t2h_entity.npy")
        tsraw = {}
        for ln in open(f"{DD}/timestamp2id.txt"):
            k, v = ln.split(); tsraw[int(v)] = int(k)
        import zipfile
        nsdir = f"{WORK}/tgb_ns"; os.makedirs(nsdir, exist_ok=True)
        with zipfile.ZipFile(f"{TRACKS}/datasets/raw/tkgl-polecat.zip") as z:
            if not os.path.exists(f"{nsdir}/tkgl-polecat_test_ns.pkl"):
                z.extract("tkgl-polecat_test_ns.pkl", nsdir)
        sampler = TKGNegativeEdgeSampler("tkgl-polecat", first_dst_id=0, last_dst_id=N - 1)
        sampler.load_eval_set(f"{nsdir}/tkgl-polecat_test_ns.pkl", split_mode="test")
        evaluator = Evaluator(name="tkgl-polecat")
        tgb_version = im.version("py-tgb")
    Dg = torch.tensor(D_GRID, device=dev, dtype=torch.float32)
    out = {}
    for sp in req_rows:
        n = len(req_rows[sp])
        if a.mode == "lookup":
            out[sp] = dict(lp_req=np.zeros((n, LP_LEN + LB_LEN), np.float32), rk_req=np.zeros((n, LP_LEN + LB_LEN), np.int32),
                           ck=np.zeros((n, len(D_GRID)), np.int32), gold_lp=np.zeros(n, np.float32), rank_filt=np.zeros(n, np.int32),
                           tie_n=np.zeros(n, np.int32))
        if sp == "test" and official:
            out.setdefault(sp, {})
            names = ["base"] if a.mode == "lookup" else list(systems)
            for nm in names:
                out[sp][f"tgb_rr_{nm}"] = np.zeros(n, np.float32); out[sp][f"tgb_h10_{nm}"] = np.zeros(n, np.float32)
                out[sp][f"tgb_rank_{nm}"] = np.zeros(n, np.float32)
    # position (within split order) -> output slot
    slot = {sp: dict(zip(req_rows[sp].tolist(), range(len(req_rows[sp])))) for sp in req_rows}
    hist_edges = []; gh = GlobalHist(N) if a.base == "logcl" else None
    stats = {sp: dict(max_abs_gold_lp_diff=0.0, n_rank_mismatch=0, n=0) for sp in req_rows}
    t_off = 0.0
    with torch.no_grad():
        for sp in ("train", "valid", "test"):
            pos0 = 0
            for k, (ts, triples) in enumerate(pairs[sp]):
                nq = 2 * len(triples)
                if sp in req_rows and len(triples):
                    s_all = np.concatenate([triples[:, 0], triples[:, 2]]); rel_all = np.concatenate([triples[:, 1], triples[:, 1] + R])
                    o_all = np.concatenate([triples[:, 2], triples[:, 0]])
                    loc = [j for j in range(nq) if (pos0 + j) in slot[sp]]
                    if loc:
                        loc = np.array(loc); sl = np.array([slot[sp][pos0 + j] for j in loc])
                        s_q, r_q, o_q = s_all[loc], rel_all[loc], o_all[loc]
                        hist = hist_edges[-L:]
                        for b0 in range(0, len(loc), 1024):
                            bb = slice(b0, b0 + 1024)
                            st = torch.tensor(s_q[bb], device=dev); rt = torch.tensor(r_q[bb], device=dev)
                            if a.base == "regcn":
                                sc, _, _ = model(hist, st, rt)
                            else:
                                sc, _, _, _, _ = model(hist, st, rt, gh.freq_vec(s_q[bb], r_q[bb], dev))
                            lp = F.log_softmax(sc, dim=1)
                            gold = torch.tensor(o_q[bb], device=dev)
                            glp = lp.gather(1, gold.unsqueeze(1)).squeeze(1)
                            # filter: other true answers at (s, r', ts)
                            fk = S.fkey(s_q[bb], r_q[bb], np.full(len(s_q[bb]), ts))
                            lf = lp.clone()
                            ri, ci = [], []
                            for qq, key in enumerate(fk):
                                i0 = np.searchsorted(F_["keys"], key)
                                if i0 < len(F_["keys"]) and F_["keys"][i0] == key:
                                    for oo in F_["objs"][F_["start"][i0]:F_["end"][i0]]:
                                        if int(oo) != int(o_q[bb][qq]):
                                            ri.append(qq); ci.append(int(oo))
                            if ri:
                                lf[torch.tensor(ri, device=dev), torch.tensor(ci, device=dev)] = float("-inf")
                            sls = sl[bb]
                            rk_f = (lf > glp.unsqueeze(1)).sum(1) + 1
                            if a.mode == "lookup":
                                O = out[sp]
                                O["gold_lp"][sls] = glp.cpu().numpy(); O["rank_filt"][sls] = rk_f.cpu().numpy()
                                O["tie_n"][sls] = ((lf == glp.unsqueeze(1)).sum(1) - 1).cpu().numpy()
                                O["ck"][sls] = (lf.unsqueeze(1) > (glp.unsqueeze(1) + Dg.unsqueeze(0)).unsqueeze(2)).sum(2).cpu().numpy()
                                rq = torch.tensor(req[sp][sls], device=dev, dtype=torch.long)
                                lpr = lp.gather(1, rq.clamp(min=0))
                                srt, _ = torch.sort(lp, dim=1)
                                cle = torch.searchsorted(srt, lpr.contiguous(), right=True)
                                O["lp_req"][sls] = torch.where(rq >= 0, lpr, torch.full_like(lpr, -1e30)).cpu().numpy()
                                O["rk_req"][sls] = torch.where(rq >= 0, N - cle + 1, torch.zeros_like(cle)).cpu().numpy()
                                del srt, cle, lpr
                                dr = D_rows = rows_sp[sp][req_rows[sp][sls]]
                                stats[sp]["max_abs_gold_lp_diff"] = max(stats[sp]["max_abs_gold_lp_diff"],
                                                                        float(np.abs(O["gold_lp"][sls] - d_gs[dr]).max()))
                                stats[sp]["n_rank_mismatch"] += int((O["rank_filt"][sls] != d_gf[dr]).sum()); stats[sp]["n"] += len(sls)
                            if sp == "test" and official:
                                t1 = time.time()
                                names = ["base"] if a.mode == "lookup" else list(systems)
                                lpc = lp.cpu().numpy()
                                for nm in names:
                                    Y = lpc
                                    if nm != "base":
                                        Y = lpc.copy()
                                        Cs = np.asarray(systems[nm][sls]).astype(np.int64)
                                        for qq in range(len(sls)):
                                            cc = Cs[qq][Cs[qq] >= 0]
                                            Y[qq, cc] = 1e6 - np.arange(len(cc), dtype=np.float32)
                                    Yt = Y[:, t2h]                                   # columns in TGB id order
                                    src_t = h2t[s_q[bb]]; dst_t = h2t[o_q[bb]]
                                    negs = sampler.query_batch(src_t, dst_t, np.full(len(src_t), tsraw[ts]), r_q[bb], split_mode="test")
                                    for qq in range(len(sls)):
                                        yp = np.array([Yt[qq, dst_t[qq]]]); yn = Yt[qq, negs[qq]][None, :]
                                        m = evaluator.eval({"y_pred_pos": yp, "y_pred_neg": yn, "eval_metric": ["mrr"]})
                                        out[sp][f"tgb_rr_{nm}"][sls[qq]] = m["mrr"]; out[sp][f"tgb_h10_{nm}"][sls[qq]] = m["hits@10"]
                                        out[sp][f"tgb_rank_{nm}"][sls[qq]] = 1 + (yn > yp).sum() + 0.5 * (yn == yp).sum()
                                t_off += time.time() - t1
                            del lp, lf, sc
                if len(triples):
                    hist_edges.append(make_edges(triples, R, dev))
                    if gh is not None:
                        gh.update(triples, R)
                pos0 += nq
                if sp != "train" and k % 50 == 0:
                    log(sp, "snapshot", k, "/", len(pairs[sp]), "official_s", round(t_off))
    res = dict(base=a.base, mode=a.mode, host="serverA (GPU, tkg-gpu:latest)", device=dev,
               gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else None), stats=stats, official_eval_s=round(t_off, 1))
    for sp, O in out.items():
        tag = (f"{WORK}/lookup_{a.base}" if a.mode == "lookup" else f"{WORK}/official_{a.base}") + ("_smoke" if a.smoke else "")
        os.makedirs(tag, exist_ok=True)
        np.savez(f"{tag}/{sp}.npz", rows=req_rows[sp], **O)
        if sp == "test" and official:
            res["official"] = {}
            for nm in (["base"] if a.mode == "lookup" else list(systems)):
                res["official"][nm] = dict(MRR=float(O[f"tgb_rr_{nm}"].mean()), hits10=float(O[f"tgb_h10_{nm}"].mean()),
                                           n=int(len(O[f"tgb_rr_{nm}"])), py_tgb=tgb_version)
            if a.mode == "lookup":
                rf = O["rank_filt"].astype(np.float64); tr_ = O["tgb_rank_base"].astype(np.float64)
                res["official_vs_harness"] = dict(harness_MRR=float(np.mean(1 / rf)), tgb_MRR=float(np.mean(1 / tr_)),
                                                  frac_rank_equal=float(np.mean(np.abs(rf - tr_) < 1e-6)),
                                                  max_abs_rank_diff=float(np.abs(rf - tr_).max()),
                                                  n_with_ties=int((O["tie_n"] > 0).sum()))
        if a.mode == "lookup":
            rf = O["rank_filt"].astype(np.float64)
            res[f"{sp}_harness"] = dict(MRR=float(np.mean(1 / rf)), H1=float(np.mean(rf <= 1)), H3=float(np.mean(rf <= 3)),
                                        H10=float(np.mean(rf <= 10)), n=int(len(rf)))
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, f"{a.mode}_{a.base}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {WORK}")
    log("ALL DONE", json.dumps({k: v for k, v in res.items() if k in ("stats", "official", "official_vs_harness")}))


if __name__ == "__main__":
    main()
