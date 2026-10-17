#!/usr/bin/env python3
"""hardening / hd_base_train.py -- RE-GCN / LogCL (the harness's native-torch regcn.py / logcl.py, h 200, 2 layers,
dropout 0.2, Adam wd 1e-5, grad-norm 1) with a TUNING grid and VALID-BEST epoch selection; dump of the selected
checkpoint in the harness base_dump.py schema + full float32 log-prob rows of every valid / test query.

Differences to harness base_dump.py (fixed 12-epoch budget, history 10, lr 1e-3, no valid selection):
  --history L, --lr, --shuffle (TGB regcn.py iterates training snapshots in random order; RE-GCN only -- LogCL's global
  copy-history is order-dependent), per-epoch FULL-valid time-aware filtered MRR (both directions) with patience,
  the valid-best epoch's weights are kept.  --grid runs several configurations in one process (each from scratch, same
  seed) and dumps only the valid-best configuration.
  python3 hd_base_train.py --model regcn --data_dir D --dataset SMALLPEDIA --out_dir O --grid "3,1e-3,0;10,1e-3,1" --epochs 12
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F

HAR = os.environ.get("HD_HAR", os.path.expandvars("$WORK_DIR"))
sys.path.insert(0, HAR)
from regcn import read_quads, load_stat, group_by_time, make_edges, build_time_index, get_history, REGCN  # noqa: E402
from logcl import LogCL, GlobalHist, info_nce  # noqa: E402

t0 = time.time()
def log(*x): print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["regcn", "logcl"])
    ap.add_argument("--data_dir", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--grid", required=True, help="L,lr,shuffle;...")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--topk", type=int, default=200); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_dump", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cuda"
    num_ent, num_rel = load_stat(a.data_dir)
    train = read_quads(os.path.join(a.data_dir, "train.txt")); valid = read_quads(os.path.join(a.data_dir, "valid.txt"))
    test = read_quads(os.path.join(a.data_dir, "test.txt"))
    tr_pairs = group_by_time(train); va_pairs = group_by_time(valid); te_pairs = group_by_time(test)
    tr_groups = build_time_index(tr_pairs)
    tr_edges = [make_edges(g, num_rel, dev) for g in tr_groups]
    from collections import defaultdict
    filt = defaultdict(list)
    for arr in (train, valid, test):
        for h, r, t, ts in arr:
            filt[(int(h), int(r), int(ts))].append(int(t)); filt[(int(t), int(r) + num_rel, int(ts))].append(int(h))
    tag = f"{a.model}_{a.dataset}"
    os.makedirs(a.out_dir, exist_ok=True)

    def fwd(model, hist, s_all, rel_all, gh, chunk=2048):
        """no-grad scoring in query chunks (per-query logits do not depend on the other queries of the snapshot)."""
        outs = []
        for c0 in range(0, len(s_all), chunk):
            s_ = s_all[c0:c0 + chunk]; r_ = rel_all[c0:c0 + chunk]
            s = torch.tensor(s_, device=dev); rel = torch.tensor(r_, device=dev)
            if a.model == "regcn":
                sc, _, _ = model(hist, s, rel)
            else:
                sc, _, _, _, _ = model(hist, s, rel, gh.freq_vec(s_, r_, dev))
            outs.append(sc)
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    def filt_rank(scores, s_all, rel_all, obj_all, ts):
        gold = torch.tensor(obj_all, device=dev)
        gsc = scores.gather(1, gold.unsqueeze(1)).squeeze(1)
        ri, ci = [], []
        for j in range(len(s_all)):
            for o in filt.get((int(s_all[j]), int(rel_all[j]), int(ts)), ()):
                if o != int(obj_all[j]): ri.append(j); ci.append(o)
        fs = scores.clone()
        if ri: fs[torch.tensor(ri, device=dev), torch.tensor(ci, device=dev)] = float("-inf")
        return gsc, (scores > gsc.unsqueeze(1)).sum(1) + 1, (fs > gsc.unsqueeze(1)).sum(1) + 1

    @torch.no_grad()
    def eval_valid(model, L):
        model.eval(); hist_edges = []; gh = GlobalHist(num_ent) if a.model == "logcl" else None; rr = []
        for (ts, triples) in tr_pairs:
            if len(triples) == 0: continue
            hist_edges.append(make_edges(triples, num_rel, dev))
            if gh is not None: gh.update(triples, num_rel)
        for (ts, triples) in va_pairs:
            if len(triples) == 0: continue
            s_all = np.concatenate([triples[:, 0], triples[:, 2]]); rel_all = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
            obj_all = np.concatenate([triples[:, 2], triples[:, 0]])
            sc = fwd(model, hist_edges[-L:], s_all, rel_all, gh)
            _, _, rk = filt_rank(sc, s_all, rel_all, obj_all, ts); rr.append((1.0 / rk.double()).cpu().numpy())
            hist_edges.append(make_edges(triples, num_rel, dev))
            if gh is not None: gh.update(triples, num_rel)
        return float(np.concatenate(rr).mean())

    grid = [tuple(x.split(",")) for x in a.grid.split(";")]
    G = dict(model=a.model, dataset=a.dataset, host=os.environ.get("HD_HOST", ""), gpu=torch.cuda.get_device_name(0), runs=[])
    best_all = (-1.0, None, None)
    for (Ls, lrs, shs) in grid:
        L, lr, sh = int(Ls), float(lrs), int(shs)
        if sh and a.model == "logcl":
            log("skip shuffled LogCL (order-dependent global history)"); continue
        torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
        model = (REGCN if a.model == "regcn" else LogCL)(num_ent, num_rel, 200, 2, 0.2, L).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        run = dict(L=L, lr=lr, shuffle=sh, epoch_log=[]); best = (-1.0, None, -1); t1 = time.time()
        for ep in range(a.epochs):
            model.train(); total = 0.0
            gh = GlobalHist(num_ent) if a.model == "logcl" else None
            order = list(range(len(tr_groups)))
            if sh: random.shuffle(order)
            for i in order:
                triples = tr_groups[i]
                if len(triples) == 0: continue
                if i >= 1:
                    hist = get_history(tr_edges, i, L)
                    s = np.concatenate([triples[:, 0], triples[:, 2]]); rel = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
                    obj = torch.tensor(np.concatenate([triples[:, 2], triples[:, 0]]), device=dev)
                    if a.model == "regcn":
                        scores, _, _ = model(hist, torch.tensor(s, device=dev), torch.tensor(rel, device=dev))
                        loss = F.cross_entropy(scores, obj)
                    else:
                        scores, zl, zg, _, _ = model(hist, torch.tensor(s, device=dev), torch.tensor(rel, device=dev), gh.freq_vec(s, rel, dev))
                        loss = F.cross_entropy(scores, obj) + 0.1 * info_nce(zl, zg)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    total += loss.item()
                if gh is not None: gh.update(triples, num_rel)
            torch.cuda.empty_cache(); vm = eval_valid(model, L); torch.cuda.empty_cache()
            run["epoch_log"].append(dict(epoch=ep + 1, loss=round(total / max(1, len(tr_groups) - 1), 5), valid_filt_MRR=round(vm, 6),
                                         t_s=round(time.time() - t1)))
            log(f"L{L} lr{lr} sh{sh}", run["epoch_log"][-1])
            if vm > best[0]:
                best = (vm, {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}, ep + 1)
            elif ep + 1 - best[2] >= a.patience:
                break
        run.update(best_epoch=best[2], best_valid_MRR=best[0], train_s=round(time.time() - t1))
        G["runs"].append(run)
        json.dump(G, open(f"{a.out_dir}/{tag}{a.tag}_grid_log.json", "w"), indent=1)
        if best[0] > best_all[0]:
            best_all = (best[0], best[1], dict(L=L, lr=lr, shuffle=sh, epoch=best[2]))
        del model, opt; torch.cuda.empty_cache()
    G["selected"] = dict(best_all[2], valid_MRR=best_all[0])
    json.dump(G, open(f"{a.out_dir}/{tag}{a.tag}_grid_log.json", "w"), indent=1)
    log("selected", G["selected"])
    if a.no_dump: return
    # ------------------------------------------------------------------ dump selected configuration
    L = G["selected"]["L"]
    model = (REGCN if a.model == "regcn" else LogCL)(num_ent, num_rel, 200, 2, 0.2, L).to(dev)
    model.load_state_dict(best_all[1]); model.eval()
    ck = os.path.join(a.out_dir, f"{tag}_seed{a.seed}.pt")
    torch.save({"state_dict": best_all[1], "num_ent": num_ent, "num_rel": num_rel, "selected": G["selected"]}, ck)
    n_rows = 2 * (sum(len(g) for g in tr_groups[1:]) + len(valid) + len(test))
    K = min(a.topk, num_ent)
    SRO = np.zeros((n_rows, 4), np.int32); GS = np.zeros(n_rows, np.float32); GR = np.zeros(n_rows, np.int32)
    GF = np.zeros(n_rows, np.int32); TI = np.zeros((n_rows, K), np.int32); TS_ = np.zeros((n_rows, K), np.float32); SP = np.zeros(n_rows, np.int8)
    MM = {sp: np.memmap(os.path.join(a.out_dir, f"{a.model}_{sp}_logp.f32"), dtype=np.float32, mode="w+",
                        shape=(2 * len({"valid": valid, "test": test}[sp]), num_ent)) for sp in ("valid", "test")}
    mp_ = {"valid": 0, "test": 0}
    ptr = [0]

    @torch.no_grad()
    def score_snapshot(hist, ts, triples, gh, code):
        s_all = np.concatenate([triples[:, 0], triples[:, 2]]); rel_all = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
        obj_all = np.concatenate([triples[:, 2], triples[:, 0]])
        scores = fwd(model, hist, s_all, rel_all, gh)
        gsc, rk_raw, rk_f = filt_rank(scores, s_all, rel_all, obj_all, ts)
        logp = F.log_softmax(scores, dim=1)
        tv, ti = torch.topk(logp, K, dim=1)
        n = len(s_all); p0 = ptr[0]
        SRO[p0:p0 + n] = np.stack([s_all, rel_all, obj_all, np.full(n, ts)], 1)
        GS[p0:p0 + n] = logp.gather(1, torch.tensor(obj_all, device=dev).unsqueeze(1)).squeeze(1).cpu().numpy()
        GR[p0:p0 + n] = rk_raw.cpu().numpy(); GF[p0:p0 + n] = rk_f.cpu().numpy()
        TI[p0:p0 + n] = ti.cpu().numpy(); TS_[p0:p0 + n] = tv.cpu().numpy(); SP[p0:p0 + n] = code
        if code > 0:
            sp = "valid" if code == 1 else "test"
            MM[sp][mp_[sp]:mp_[sp] + n] = logp.cpu().numpy(); mp_[sp] += n
        ptr[0] = p0 + n

    hist_edges = []; gh = GlobalHist(num_ent) if a.model == "logcl" else None
    for code, pairs in ((0, tr_pairs), (1, va_pairs), (2, te_pairs)):
        for i, (ts, triples) in enumerate(pairs):
            if len(triples) == 0: continue
            if not (code == 0 and i == 0):
                score_snapshot(hist_edges[-L:], ts, triples, gh, code)
            hist_edges.append(make_edges(triples, num_rel, dev))
            if gh is not None: gh.update(triples, num_rel)
    assert ptr[0] == n_rows, (ptr[0], n_rows)
    for v in MM.values(): v.flush()

    def mets(rk):
        rk = rk.astype(np.float64)
        return dict(MRR=round(float(np.mean(1 / rk)), 6), H1=round(float(np.mean(rk <= 1)), 6), H3=round(float(np.mean(rk <= 3)), 6),
                    H10=round(float(np.mean(rk <= 10)), 6), n=int(len(rk)))
    n_tr = int((SP == 0).sum()); n_va = int((SP == 1).sum()); n_te = int((SP == 2).sum())
    DL = dict(model=a.model, dataset=a.dataset, selected=G["selected"], valid_filt=mets(GF[SP == 1]), test_filt=mets(GF[SP == 2]),
              train_filt=mets(GF[SP == 0]), rows=dict(train=n_tr, valid=n_va, test=n_te, total=int(n_rows), K=int(K)),
              config=dict(h_dim=200, n_layers=2, history_len=L, dropout=0.2, lr=G["selected"]["lr"], shuffle=G["selected"]["shuffle"],
                          epochs=G["selected"]["epoch"], cl_weight=(0.1 if a.model == "logcl" else None)),
              host=G["host"], total_s=round(time.time() - t0))
    out = os.path.join(a.out_dir, f"{a.model}_scores_{a.dataset}_traintest.npz")
    np.savez(out, sro=SRO, gold_score=GS, gold_rank_raw=GR, gold_rank_filt=GF, topk_ids=TI, topk_scores=TS_, split=SP,
             split_bounds=np.array([n_tr, n_tr + n_va, n_tr + n_va + n_te], np.int64))
    json.dump(DL, open(os.path.join(a.out_dir, f"{tag}_dump_log.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out_dir}")
    log("ALL DONE", json.dumps(dict(selected=G["selected"], valid=DL["valid_filt"])))


if __name__ == "__main__":
    main()
