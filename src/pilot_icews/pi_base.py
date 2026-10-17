#!/usr/bin/env python3
"""pilot_icews / pi_base.py -- STAGE A frozen base (RE-GCN | LogCL, harness native-torch models regcn.py / logcl.py)
trained on a contiguous recent WINDOW of train, epoch budget selected on FULL valid, then a sampled dump.

Training = harness base_dump.py recipe (h 200, 2 layers, history 10, dropout 0.2, Adam lr 1e-3 wd 1e-5, grad clip 1.0,
seed 42, LogCL cl_weight 0.1) with ONE change: gradient steps only on train snapshots with day >= WIN_T0 (last 20% of
train days).  History structure is kept intact: RE-GCN's 10-snapshot history of the first window snapshots reaches
back before the window; LogCL's global copy-history is PRE-ROLLED over every train snapshot before the window at the
start of each epoch (so its counts match the online full-history protocol used at evaluation).
Epoch grid: checkpoints + full-valid filtered MRR at EPOCH_GRID; stop after two consecutive grid points below the
running best; if the best is the last grid point, continue over EPOCH_EXT.  Selected = argmax full-valid MRR.
Resumable: every grid checkpoint holds model + optimiser + RNG states; a restart continues from the latest one.
Dump (selected checkpoint; ground-truth online history; valid = train + rolling valid, test = train + valid + rolling
test): for the uniform VALID sample and the uniform 10% TEST sample, per query from the full log-softmax row:
  sro, gold_lp, rank_raw, rank_filt (1 + #non-filter entities with lp > gold), tie_n (#non-filter non-gold entities
  with lp == gold), topk_ids/topk_scores (K = 200).
  python3 pi_base.py --model regcn [--smoke_snaps 100]
"""
import os, sys, json, time, argparse, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from pi_common import (fast_global_hist, H, P, DS, DD, WORK, WIN_T0, relmap2, EPOCH_GRID, EPOCH_EXT, KREF, Splits, split_queries, valid_sample_idx,
                       test_sample_idx, filter_pairs, jdump, log_fn, DUMPS, CKPT, STAGE)
sys.path.insert(0, H)
from regcn import make_edges, get_history, REGCN
from logcl import LogCL, GlobalHist, info_nce


def groups_of(q):
    """[(ts, triples[:, :3])] sorted by ts; identical to regcn.group_by_time (stable order within a timestamp)."""
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    ts = q[:, 3]; b = np.flatnonzero(np.diff(ts)) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(q)]])
    return [(int(ts[a]), q[a:e, :3].copy()) for a, e in zip(st, en)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["regcn", "logcl"])
    ap.add_argument("--relnorm", action="store_true", help="model inputs (train triples, history edges, query relation) use relation-normalised ids; filters / gold stay original")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_snaps", type=int, default=0)
    ap.add_argument("--skip_dump", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    from pi_common import DEV as dev
    smoke = a.smoke_snaps > 0
    mname = a.model + ("rn" if a.relnorm else "")
    tag = f"{mname}_{STAGE}" + ("_smoke" if smoke else "")
    ckdir = f"{P}/dumps/ckpt_{tag}"; os.makedirs(ckdir, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    h_dim, n_layers, L, dropout, lr, clw = 200, 2, 10, 0.2, 1e-3, 0.1
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    S = Splits(DS); N, R = S.n_ent, S.R
    tr_pairs = groups_of(S.q["train"]); va_pairs = groups_of(S.q["valid"]); te_pairs = groups_of(S.q["test"])
    RM2 = relmap2() if a.relnorm else np.arange(2 * R)          # model-input relation map ([2R])
    mrel = lambda g: (np.stack([g[:, 0], RM2[g[:, 1]], g[:, 2]], 1) if a.relnorm else g)
    tr_ts = np.array([p[0] for p in tr_pairs])
    win = [i for i in range(1, len(tr_pairs)) if tr_pairs[i][0] >= WIN_T0]
    if smoke:
        win = win[:a.smoke_snaps]; va_pairs = va_pairs[:max(5, a.smoke_snaps // 4)]; te_pairs = te_pairs[:max(5, a.smoke_snaps // 4)]
    n_win_facts = int(sum(len(tr_pairs[i][1]) for i in win))
    info = dict(model=mname, relnorm_inputs=bool(a.relnorm), stage=STAGE, host="serverA (GPU, tkg-gpu:latest)", seed=a.seed,
                config=dict(h_dim=h_dim, n_layers=n_layers, history_len=L, dropout=dropout, lr=lr, cl_weight=(clw if a.model == "logcl" else None)),
                window=dict(first_day=int(tr_pairs[win[0]][0]), last_day=int(tr_pairs[win[-1]][0]), n_snapshots=len(win), n_facts=n_win_facts,
                            n_train_days=len(tr_pairs), n_train_facts=int(len(S.q["train"]))),
                gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"), torch=torch.__version__, smoke=a.smoke_snaps)
    log("window", info["window"])
    tr_edges = [make_edges(mrel(g), R, dev) for _, g in tr_pairs]
    model = (REGCN if a.model == "regcn" else LogCL)(N, R, h_dim, n_layers, dropout, L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # ---- LogCL global history: pre-rolled over train before the window; full train for evaluation
    gh_pre = gh_trainfull = None
    if a.model == "logcl":
        gh_pre = fast_global_hist()(N)
        for i in range(win[0]):                               # every snapshot before the first trained one (incl. day 0)
            gh_pre.update(mrel(tr_pairs[i][1]), R)
        gh_trainfull = copy.deepcopy(gh_pre)
        for i in range(win[0], len(tr_pairs)):
            gh_trainfull.update(mrel(tr_pairs[i][1]), R)
        log("global history pre-rolled", len(gh_pre.d), "->", len(gh_trainfull.d))
    F_ = S.filter_index()
    log("filter index", len(F_["keys"]))

    def score(hist, s_q, r_q, gh):
        r_q = RM2[r_q]
        st = torch.tensor(s_q, device=dev); rt = torch.tensor(r_q, device=dev)
        if a.model == "regcn":
            sc, _, _ = model(hist, st, rt)
        else:
            sc, _, _, _, _ = model(hist, st, rt, gh.freq_vec(s_q, r_q, dev))
        return sc

    @torch.no_grad()
    def replay(splits, want=None, collect=False):
        """valid eval (want=None: all queries, harness filtered rank only) or dump (want = {split: sorted positions})."""
        model.eval()
        hist_edges = list(tr_edges[-L:]) if not smoke else list(tr_edges[:win[-1] + 1][-L:])
        gh = copy.deepcopy(gh_trainfull) if gh_trainfull is not None else None
        out = {}
        for sp, pairs in (("valid", va_pairs), ("test", te_pairs)):
            if sp not in splits:
                break
            ranks = []; rec = {k: [] for k in ("pos", "sro", "gold_lp", "rank_raw", "rank_filt", "tie_n", "topk_ids", "topk_scores")}
            wset = None if want is None else want[sp]
            pos0 = 0
            for ts, triples in pairs:
                nq = 2 * len(triples)
                s_all = np.concatenate([triples[:, 0], triples[:, 2]]); r_all = np.concatenate([triples[:, 1], triples[:, 1] + R])
                o_all = np.concatenate([triples[:, 2], triples[:, 0]])
                if wset is None:
                    loc = np.arange(nq)
                else:
                    lo_, hi_ = np.searchsorted(wset, [pos0, pos0 + nq])
                    loc = wset[lo_:hi_] - pos0
                hist = hist_edges[-L:]
                for b0 in range(0, len(loc), 4096):
                    lb = loc[b0:b0 + 4096]
                    s_q, r_q, o_q = s_all[lb], r_all[lb], o_all[lb]
                    sc = score(hist, s_q, r_q, gh)
                    lp = F.log_softmax(sc, dim=1)
                    gold = torch.tensor(o_q, device=dev)
                    glp = lp.gather(1, gold.unsqueeze(1))
                    rows, cols = filter_pairs(F_, S.fkey(s_q, r_q, np.full(len(s_q), ts)), o_q)
                    lf = lp.clone()
                    if len(rows):
                        lf[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = float("-inf")
                    rk_f = (lf > glp).sum(1) + 1
                    if not collect:
                        ranks.append(rk_f.cpu().numpy())
                    else:
                        rec["pos"].append(pos0 + lb); rec["sro"].append(np.stack([s_q, r_q, o_q, np.full(len(s_q), ts)], 1))
                        rec["gold_lp"].append(glp.squeeze(1).cpu().numpy()); rec["rank_filt"].append(rk_f.cpu().numpy())
                        rec["rank_raw"].append(((lp > glp).sum(1) + 1).cpu().numpy())
                        rec["tie_n"].append(((lf == glp).sum(1) - 1).cpu().numpy())
                        tv, ti = torch.topk(lp, KREF, dim=1)
                        rec["topk_ids"].append(ti.cpu().numpy().astype(np.int32)); rec["topk_scores"].append(tv.cpu().numpy())
                    del sc, lp, lf
                hist_edges.append(make_edges(mrel(triples), R, dev))
                if gh is not None:
                    gh.update(mrel(triples), R)
                pos0 += nq
            if not collect:
                out[sp] = np.concatenate(ranks)
            else:
                out[sp] = {k: np.concatenate(v) for k, v in rec.items()}
        model.train()
        return out

    # ---------------- resume
    hist_path = f"{ckdir}/grid_log.json"
    glog = json.load(open(hist_path)) if os.path.exists(hist_path) else dict(grid=[], epoch_log=[])
    done = sorted(int(os.path.basename(p)[2:-3]) for p in glob.glob(f"{ckdir}/ep*.pt"))
    start_ep = 0
    if done:
        ck = torch.load(f"{ckdir}/ep{done[-1]}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["rng_cpu"]); (torch.cuda.set_rng_state(ck["rng_cuda"]) if dev == "cuda" else None); np.random.set_state(ck["rng_np"])
        start_ep = done[-1]; log("resumed from epoch", start_ep)
    grid_all = EPOCH_GRID + EPOCH_EXT
    if smoke:
        grid_all = [1, 2]

    def keep_going(ep_done):
        """grid rule: stop after two consecutive grid points below the running best; past the last base-grid
        point (12) continue only while the latest grid point is the best (extension 14..20)."""
        g = glog["grid"]
        if ep_done >= max(grid_all):
            return False
        if not g:
            return True
        best_i = int(np.argmax([x["valid_MRR"] for x in g]))
        if len(g) - 1 - best_i >= 2:
            return False
        if not smoke and ep_done >= EPOCH_GRID[-1]:
            return best_i == len(g) - 1
        return True

    ep = start_ep
    if start_ep and start_ep in grid_all and start_ep not in [x["epoch"] for x in glog["grid"]]:
        rk = replay(("valid",))["valid"].astype(np.float64)       # checkpoint saved but its valid eval was interrupted
        glog["grid"].append(dict(epoch=start_ep, valid_MRR=float(np.mean(1 / rk)), valid_H10=float(np.mean(rk <= 10)),
                                 n_valid=int(len(rk)), eval_s=None, note="evaluated on resume"))
        json.dump(glog, open(hist_path, "w"), indent=1)
    while keep_going(ep):
        model.train(); total = 0.0; te0 = time.time()
        gh = copy.deepcopy(gh_pre) if gh_pre is not None else None
        for i in win:
            triples = mrel(tr_pairs[i][1])
            hist = get_history(tr_edges, i, L)
            s = np.concatenate([triples[:, 0], triples[:, 2]]); rel = np.concatenate([triples[:, 1], triples[:, 1] + R])
            obj = np.concatenate([triples[:, 2], triples[:, 0]])
            s_t = torch.tensor(s, device=dev); rel_t = torch.tensor(rel, device=dev); obj_t = torch.tensor(obj, device=dev)
            if a.model == "regcn":
                scores, _, _ = model(hist, s_t, rel_t)
                loss = F.cross_entropy(scores, obj_t)
            else:
                gf = gh.freq_vec(s, rel, dev)
                scores, zl, zg, _, _ = model(hist, s_t, rel_t, gf)
                loss = F.cross_entropy(scores, obj_t) + clw * info_nce(zl, zg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
            if gh is not None:
                gh.update(triples, R)
        ep += 1
        msg = dict(epoch=ep, loss=round(total / max(1, len(win)), 5), epoch_s=round(time.time() - te0, 1), t_s=round(time.time() - t0))
        glog["epoch_log"].append(msg); log(msg)
        if ep in grid_all:
            torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(), epoch=ep, rng_cpu=torch.get_rng_state(),
                            rng_cuda=(torch.cuda.get_rng_state() if dev == "cuda" else None), rng_np=np.random.get_state()), f"{ckdir}/ep{ep}.pt")
            tv0 = time.time()
            rk = replay(("valid",))["valid"].astype(np.float64)
            g = dict(epoch=ep, valid_MRR=float(np.mean(1 / rk)), valid_H10=float(np.mean(rk <= 10)), n_valid=int(len(rk)),
                     eval_s=round(time.time() - tv0, 1))
            glog["grid"].append(g); log("grid", g)
            json.dump(glog, open(hist_path, "w"), indent=1)
    best = max(glog["grid"], key=lambda x: x["valid_MRR"])
    info.update(grid=glog["grid"], epoch_log=glog["epoch_log"], selected_epoch=best["epoch"], selected_valid_MRR=best["valid_MRR"],
                selection_rule="argmax full-valid filtered MRR over the epoch grid; stop after 2 consecutive grid points below best; "
                               "extend past the last grid point only if it is the best", train_s=round(time.time() - t0))
    ck = torch.load(f"{ckdir}/ep{best['epoch']}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    ckout = CKPT[mname] if not smoke else f"{ckdir}/best.pt"
    torch.save(dict(state_dict=ck["state_dict"], num_ent=N, num_rel=R, epoch=best["epoch"], info=info), ckout)
    info["checkpoint"] = ckout
    if not a.skip_dump:
        nv = 2 * len(S.q["valid"]); nt = 2 * len(S.q["test"])
        want = {"valid": valid_sample_idx(nv), "test": test_sample_idx(nt)}
        if smoke:
            want = {k: v[v < 2 * sum(len(p[1]) for p in (va_pairs if k == "valid" else te_pairs))] for k, v in want.items()}
        td0 = time.time()
        out = replay(("valid", "test"), want=want, collect=True)
        for sp in ("valid", "test"):
            assert (out[sp]["pos"] == want[sp]).all(), sp
        if not smoke:
            for sp in ("valid", "test"):
                ref = split_queries(S, sp)[want[sp]]
                assert (ref == out[sp]["sro"]).all(), f"{sp} row order"
        path = DUMPS[mname] if not smoke else f"{ckdir}/dump_smoke.npz"
        np.savez(path, **{f"{sp}_{k}": v for sp in out for k, v in out[sp].items()})
        info["dump"] = dict(path=path, dump_s=round(time.time() - td0, 1), n_valid=int(len(want["valid"])), n_test=int(len(want["test"])),
                            valid_sample_MRR=float(np.mean(1.0 / out["valid"]["rank_filt"])))
    info["total_s"] = round(time.time() - t0)
    jdump(info, f"base_{tag}.json"); jdump(info, f"{P}/dumps/base_{tag}.json")
    os.system(f"chmod -R a+rX {P}/dumps")
    log("ALL DONE", json.dumps({k: info.get(k) for k in ("selected_epoch", "selected_valid_MRR", "total_s")}))


if __name__ == "__main__":
    main()
