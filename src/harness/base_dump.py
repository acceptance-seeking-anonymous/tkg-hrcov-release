"""
harness / base_dump.py -- train a frozen base (RE-GCN or LogCL, the project's native-torch
re-implementations regcn.py v6 / logcl.py used in Task 2) and dump train+valid+test scores in the
SAME npz schema as the TiRGN dumps, for BOTH query directions, plus exact time-aware filtered ranks.

Schema written (np.savez, uncompressed):
  sro[N,4] int32 (s, r', o, ts)  r' = r (tail query) or r+R (head query, s = original object)
  gold_score[N] f32  log-softmax score of the gold          gold_rank_raw[N] int32  1+#(score>gold)
  gold_rank_filt[N] int32  exact time-aware filtered rank (other true answers at (s,r',ts) masked)
  topk_ids[N,K] int32 / topk_scores[N,K] f32  (log-softmax, descending)
  split[N] int8 (0 train,1 valid,2 test)   split_bounds[3] int64 (cumulative ends: train, valid, test)
Row order: train (first snapshot unscored, as in training / the TiRGN train dump), valid, test.
History protocol = Task-2 evaluate_and_emit: train rows roll from empty train history; valid rows use
all train + rolling valid; test rows use train + valid + rolling test (ground-truth history).
Training config = Task-2 defaults (h_dim 200, 2 layers, history 10, dropout 0.2, lr 1e-3, seed 42;
epochs 20 ICEWS18 / 12 GDELT; LogCL cl_weight 0.1).  No intermediate validation (it does not change
the trained weights).  Checkpoint saved next to the dump.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from regcn import read_quads, load_stat, group_by_time, make_edges, build_time_index, get_history, REGCN
from logcl import LogCL, GlobalHist, info_nce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["regcn", "logcl"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    epochs = a.epochs or (20 if a.dataset.startswith("ICEWS") else 12)
    h_dim, n_layers, L, dropout, lr, clw = 200, 2, 10, 0.2, 1e-3, 0.1
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.model}_{a.dataset}"
    log = {"model": a.model, "dataset": a.dataset, "device": dev, "epochs": epochs, "seed": a.seed,
           "config": dict(h_dim=h_dim, n_layers=n_layers, history_len=L, dropout=dropout, lr=lr,
                          cl_weight=(clw if a.model == "logcl" else None)), "epoch_log": []}
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    num_ent, num_rel = load_stat(a.data_dir)
    train = read_quads(os.path.join(a.data_dir, "train.txt"))
    valid = read_quads(os.path.join(a.data_dir, "valid.txt"))
    test = read_quads(os.path.join(a.data_dir, "test.txt"))
    tr_pairs = group_by_time(train); va_pairs = group_by_time(valid); te_pairs = group_by_time(test)
    tr_groups = build_time_index(tr_pairs)
    tr_edges = [make_edges(g, num_rel, dev) for g in tr_groups]
    if a.model == "regcn":
        model = REGCN(num_ent, num_rel, h_dim, n_layers, dropout, L).to(dev)
    else:
        model = LogCL(num_ent, num_rel, h_dim, n_layers, dropout, L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # ------------------------------------------------------------------ train (Task-2 run_epoch)
    for ep in range(epochs):
        model.train(); total = 0.0
        gh = GlobalHist(num_ent) if a.model == "logcl" else None
        for i in range(len(tr_groups)):
            triples = tr_groups[i]
            if len(triples) == 0:
                continue
            if i >= 1:
                hist = get_history(tr_edges, i, L)
                s = np.concatenate([triples[:, 0], triples[:, 2]])
                rel = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
                obj = np.concatenate([triples[:, 2], triples[:, 0]])
                s_t = torch.tensor(s, device=dev); rel_t = torch.tensor(rel, device=dev)
                obj_t = torch.tensor(obj, device=dev)
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
                gh.update(triples, num_rel)
        msg = dict(epoch=ep + 1, loss=round(total / max(1, len(tr_groups) - 1), 5), t_s=round(time.time() - t0))
        log["epoch_log"].append(msg); print(f"[{tag}] {msg}", flush=True)
    ck = os.path.join(a.out_dir, f"{tag}_seed{a.seed}.pt")
    torch.save({"state_dict": model.state_dict(), "num_ent": num_ent, "num_rel": num_rel, "log": log}, ck)
    log["checkpoint"] = ck; log["train_s"] = round(time.time() - t0)

    # ------------------------------------------------------------------ filter index (all splits)
    from collections import defaultdict
    filt = defaultdict(list)          # (s, r', ts) -> true answers (both directions)
    for arr in (train, valid, test):
        for h, r, t, ts in arr:
            filt[(int(h), int(r), int(ts))].append(int(t))
            filt[(int(t), int(r) + num_rel, int(ts))].append(int(h))

    n_rows = 2 * (sum(len(g) for g in tr_groups[1:]) + len(valid) + len(test))
    K = min(a.topk, num_ent)
    SRO = np.zeros((n_rows, 4), np.int32); GS = np.zeros(n_rows, np.float32)
    GR = np.zeros(n_rows, np.int32); GF = np.zeros(n_rows, np.int32)
    TI = np.zeros((n_rows, K), np.int32); TS_ = np.zeros((n_rows, K), np.float32)
    SP = np.zeros(n_rows, np.int8)
    ptr = [0]

    @torch.no_grad()
    def score_snapshot(hist, ts, triples, gh, split_code):
        s_all = np.concatenate([triples[:, 0], triples[:, 2]])
        rel_all = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
        obj_all = np.concatenate([triples[:, 2], triples[:, 0]])
        s = torch.tensor(s_all, device=dev); rel = torch.tensor(rel_all, device=dev)
        if a.model == "regcn":
            scores, _, _ = model(hist, s, rel)
        else:
            scores, _, _, _, _ = model(hist, s, rel, gh.freq_vec(s_all, rel_all, dev))
        gold = torch.tensor(obj_all, device=dev)
        gsc = scores.gather(1, gold.unsqueeze(1)).squeeze(1)
        rk_raw = (scores > gsc.unsqueeze(1)).sum(1) + 1
        rows_i, cols_i = [], []
        for j in range(len(s_all)):
            oth = filt.get((int(s_all[j]), int(rel_all[j]), int(ts)), ())
            for o in oth:
                if o != int(obj_all[j]):
                    rows_i.append(j); cols_i.append(o)
        fs = scores.clone()
        if rows_i:
            fs[torch.tensor(rows_i, device=dev), torch.tensor(cols_i, device=dev)] = float("-inf")
        rk_f = (fs > gsc.unsqueeze(1)).sum(1) + 1
        logp = F.log_softmax(scores, dim=1)
        tv, ti = torch.topk(logp, K, dim=1)
        n = len(s_all); p0 = ptr[0]
        SRO[p0:p0 + n] = np.stack([s_all, rel_all, obj_all, np.full(n, ts)], 1)
        GS[p0:p0 + n] = logp.gather(1, gold.unsqueeze(1)).squeeze(1).cpu().numpy()
        GR[p0:p0 + n] = rk_raw.cpu().numpy(); GF[p0:p0 + n] = rk_f.cpu().numpy()
        TI[p0:p0 + n] = ti.cpu().numpy(); TS_[p0:p0 + n] = tv.cpu().numpy(); SP[p0:p0 + n] = split_code
        ptr[0] = p0 + n

    model.eval()
    t1 = time.time()
    hist_edges = []; gh = GlobalHist(num_ent) if a.model == "logcl" else None
    for i, (ts, triples) in enumerate(tr_pairs):           # TRAIN rows
        if len(triples) == 0:
            continue
        if i >= 1:
            score_snapshot(hist_edges[-L:], ts, triples, gh, 0)
        hist_edges.append(make_edges(triples, num_rel, dev))
        if gh is not None: gh.update(triples, num_rel)
    n_tr = ptr[0]
    for (ts, triples) in va_pairs:                           # VALID rows
        if len(triples) == 0: continue
        score_snapshot(hist_edges[-L:], ts, triples, gh, 1)
        hist_edges.append(make_edges(triples, num_rel, dev))
        if gh is not None: gh.update(triples, num_rel)
    n_va = ptr[0] - n_tr
    for (ts, triples) in te_pairs:                           # TEST rows
        if len(triples) == 0: continue
        score_snapshot(hist_edges[-L:], ts, triples, gh, 2)
        hist_edges.append(make_edges(triples, num_rel, dev))
        if gh is not None: gh.update(triples, num_rel)
    n_te = ptr[0] - n_tr - n_va
    assert ptr[0] == n_rows, (ptr[0], n_rows)
    log["dump_s"] = round(time.time() - t1)

    def mets(rk):
        rk = rk.astype(np.float64)
        return dict(MRR=round(float(np.mean(1 / rk)), 6), H1=round(float(np.mean(rk <= 1)), 6),
                    H3=round(float(np.mean(rk <= 3)), 6), H10=round(float(np.mean(rk <= 10)), 6), n=int(len(rk)))
    log["valid_filt"] = mets(GF[SP == 1]); log["test_filt"] = mets(GF[SP == 2]); log["test_raw"] = mets(GR[SP == 2])
    log["test_filt_tail"] = mets(GF[(SP == 2) & (SRO[:, 1] < num_rel)])
    log["train_filt"] = mets(GF[SP == 0])
    log["rows"] = dict(train=int(n_tr), valid=int(n_va), test=int(n_te), total=int(n_rows), K=int(K))
    out = os.path.join(a.out_dir, f"{a.model}_scores_{a.dataset}_traintest.npz")
    np.savez(out, sro=SRO, gold_score=GS, gold_rank_raw=GR, gold_rank_filt=GF, topk_ids=TI,
             topk_scores=TS_, split=SP, split_bounds=np.array([n_tr, n_tr + n_va, n_tr + n_va + n_te], np.int64))
    os.system(f"chmod a+r {out} {ck}")
    log["dump_path"] = out; log["total_s"] = round(time.time() - t0)
    json.dump(log, open(f"{tag}_dump_log.json", "w"), indent=1)        # job cwd -> harvested
    json.dump(log, open(os.path.join(a.out_dir, f"{tag}_dump_log.json"), "w"), indent=1)
    print(json.dumps({k: log[k] for k in ("test_filt", "valid_filt", "rows", "total_s")}), flush=True)


if __name__ == "__main__":
    main()
