#!/usr/bin/env python3
"""hardening / hd_tirgn.py -- TiRGN (authors' model code via the project's pure-torch DGL shim) on a TGB tkgl dataset in
harness format; generalised from pilot_smallpedia/code/sp_tirgn.py (same model, hyper-parameters and training loop).

Differences to sp_tirgn.py: dataset-agnostic (--data_dir/--n_ent/--n_rel), wall-clock cap on training (--max_train_hours),
per-epoch FULL-valid filtered MRR with patience (valid-best epoch kept), per-epoch log written after every epoch,
no full log-prob rows (polecat rows would be 321 GB); dump in the harness base_dump.py schema (train from the 2nd
snapshot, valid, test; top-K log-probs) + checkpoint.
  python3 hd_tirgn.py --data_dir D --n_ent 150931 --n_rel 16 --tree TiRGN --ds POLECAT --epochs 12 --patience 3
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--n_ent", type=int, required=True)
ap.add_argument("--n_rel", type=int, required=True); ap.add_argument("--tree", required=True)
ap.add_argument("--ds", default="POLECAT"); ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--max_train_hours", type=float, default=7.5)
ap.add_argument("--L", type=int, default=10); ap.add_argument("--hr", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--topk", type=int, default=200)
ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--no_dump", action="store_true")
ap.add_argument("--resume_ckpt", default="")
ap.add_argument("--dec_chunk", type=int, default=1024)
if __name__ == "__main__":
    a = ap.parse_args()
    sys.path.insert(0, a.tree); sys.path.insert(0, f"{a.tree}/src")
else:   # imported by hd_tirgn_replay.py (which puts the TiRGN tree on sys.path first): no CLI parsing
    a = None
import dgl_shim  # noqa: E402,F401
from rgcn.utils import build_sub_graph  # noqa: E402
from src.rrgcn import RecurrentRGCN  # noqa: E402

t0 = time.time()
def log(*x): print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


class HistVocab:
    """first-occurrence index over (key = s * 2R + r', o) of ALL facts (both directions)  (sp_tirgn.py)."""
    def __init__(self, quads, N, R):
        s = np.concatenate([quads[:, 0], quads[:, 2]]); r = np.concatenate([quads[:, 1], quads[:, 1] + R])
        o = np.concatenate([quads[:, 2], quads[:, 0]]); t = np.concatenate([quads[:, 3], quads[:, 3]])
        key = s * (2 * R) + r
        order = np.lexsort((t, o, key)); key, o, t = key[order], o[order], t[order]
        first = np.ones(len(key), bool); first[1:] = (key[1:] != key[:-1]) | (o[1:] != o[:-1])
        self.k = key[first]; self.o = o[first]; self.t = t[first]; self.R2 = 2 * R; self.N = N

    def dense(self, s, r, t, dev):
        kq = s.astype(np.int64) * self.R2 + r.astype(np.int64)
        lo = np.searchsorted(self.k, kq, "left"); hi = np.searchsorted(self.k, kq, "right")
        ln = hi - lo
        rows = np.repeat(np.arange(len(kq)), ln)
        idx = np.concatenate([np.arange(x, y) for x, y in zip(lo, hi)]) if ln.sum() else np.zeros(0, np.int64)
        keep = self.t[idx] < t
        V = torch.zeros((len(kq), self.N), device=dev)
        if keep.any():
            V[torch.as_tensor(rows[keep], device=dev), torch.as_tensor(self.o[idx[keep]], device=dev)] = 1.0
        return V


def tslice(te, sl):
    return tuple(x[sl] for x in te) if isinstance(te, (tuple, list)) else te[sl]


def patched_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
    score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=history_vocabulary.float())
    return F.softmax(score_global, dim=1)


def filter_index(qs, N, R):
    quads = np.concatenate(qs)
    h, r, t, ts = quads.T
    s_all = np.concatenate([h, t]); r_all = np.concatenate([r, r + R]); o_all = np.concatenate([t, h]); ts_all = np.concatenate([ts, ts])
    key = (ts_all * (2 * R) + r_all) * N + s_all
    order = np.lexsort((o_all, key)); key = key[order]; o_all = o_all[order]
    uk, start = np.unique(key, return_index=True); end = np.append(start[1:], len(key))
    return dict(keys=uk, start=start, end=end, objs=o_all)


def main():
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; N, R = a.n_ent, a.n_rel
    RecurrentRGCN.history_mode = patched_history_mode
    Q = {sp: load_quads(f"{a.data_dir}/{sp}.txt") for sp in ("train", "valid", "test")}
    allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
    HV = HistVocab(allq, N, R)
    snaps = {}
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
        snaps[sp] = [(int(u), q[b:e]) for u, b, e in zip(uts, st, en)]
    train_times = [u for u, _ in snaps["train"]]
    tint = train_times[1] - train_times[0]
    num_times = sum(len(v) for v in snaps.values())
    log("snapshots", {k: len(v) for k, v in snaps.items()}, "tint", tint, "num_times", num_times)
    model = RecurrentRGCN("timeconvtranse", "convgcn", N, R, 0, 0, num_times, tint, 200, "sub", a.hr, sequence_len=a.L,
                          num_bases=100, num_basis=100, num_hidden_layers=2, dropout=0.2, self_loop=True, skip_connect=False,
                          layer_norm=True, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, aggregation="none",
                          weight=0.5, discount=1, angle=10, use_static=False, entity_prediction=True, relation_prediction=False,
                          use_cuda=True, gpu=0, analysis=False).to(dev)
    torch.cuda.set_device(0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    seq = [("train", j) for j in range(len(snaps["train"]))] + [("valid", j) for j in range(len(snaps["valid"]))] + \
          [("test", j) for j in range(len(snaps["test"]))]
    pos_of = {x: i for i, x in enumerate(seq)}
    gcache = {}

    def G(sp, i):
        k = (sp, i)
        if k not in gcache:
            gcache[k] = build_sub_graph(N, R, snaps[sp][i][1][:, :3], True, 0)
        return gcache[k]

    def hist_graphs(sp, i):
        pos = pos_of[(sp, i)]
        return [G(*x) for x in seq[max(0, pos - a.L):pos]]

    F_ = filter_index([Q["train"], Q["valid"], Q["test"]], N, R)

    def fkey(s, r, ts):
        return (np.asarray(ts, np.int64) * (2 * R) + np.asarray(r, np.int64)) * N + np.asarray(s, np.int64)

    @torch.no_grad()
    def score_snap(sp, i):
        ts, trip = snaps[sp][i]
        glist = hist_graphs(sp, i)
        tt = torch.as_tensor(trip, device=dev).long()
        inv = tt[:, [2, 1, 0, 3]].clone(); inv[:, 1] += R
        allt = torch.cat([tt, inv])
        s_np = allt[:, 0].cpu().numpy(); r_np = allt[:, 1].cpu().numpy()
        V = HV.dense(s_np, r_np, ts, dev)
        evolve, _, r_emb, _, _ = model.forward(glist, None, True)
        emb = F.normalize(evolve[-1])
        te = model.get_init_time(allt)
        lp = torch.empty((len(allt), N), device=dev)
        for c0 in range(0, len(allt), a.dec_chunk):
            sl = slice(c0, c0 + a.dec_chunk)
            sr = model.raw_mode(emb, r_emb, tslice(te, sl), allt[sl]); sh = model.history_mode(emb, r_emb, tslice(te, sl), allt[sl], V[sl])
            lp[sl] = torch.log(a.hr * sh + (1 - a.hr) * sr)
            del sr, sh
        del V
        return allt.cpu().numpy(), lp

    def filt_rank(sro, lp):
        gold = torch.as_tensor(sro[:, 2], device=dev).long()
        glp = lp.gather(1, gold.unsqueeze(1)).squeeze(1)
        fk = fkey(sro[:, 0], sro[:, 1], sro[:, 3])
        i0 = np.searchsorted(F_["keys"], fk)
        ri, ci = [], []
        for q_, (k_, j_) in enumerate(zip(fk, i0)):
            if j_ < len(F_["keys"]) and F_["keys"][j_] == k_:
                for oo in F_["objs"][F_["start"][j_]:F_["end"][j_]]:
                    if int(oo) != int(sro[q_, 2]):
                        ri.append(q_); ci.append(int(oo))
        lf = lp.clone()
        if ri:
            lf[torch.as_tensor(ri, device=dev), torch.as_tensor(ci, device=dev)] = float("-inf")
        rk_f = (lf > glp.unsqueeze(1)).sum(1) + 1; rk_r = (lp > glp.unsqueeze(1)).sum(1) + 1
        return glp, rk_r, rk_f

    def eval_valid(maxsnap=None):
        model.eval(); rr = []
        for i in range(len(snaps["valid"]) if maxsnap is None else maxsnap):
            sro, lp = score_snap("valid", i)
            _, _, rk = filt_rank(sro, lp); rr.append((1.0 / rk.double()).cpu().numpy())
            del lp
        return float(np.concatenate(rr).mean())

    def chunked_loss(glist, triples, V):
        """TiRGN get_loss (entity prediction only, no static / relation head: loss = 0.7 * NLL_entity, mean over
        both directions) with the decoder evaluated in query chunks; the evolve graph is kept (retain_graph) so the
        gradient equals the full-batch gradient."""
        inv = triples[:, [2, 1, 0, 3]].clone(); inv[:, 1] = inv[:, 1] + R
        allt = torch.cat([triples, inv]).to(dev)
        evolve_embs, _, r_emb, _, _ = model.forward(glist, None, True)
        pre_emb = F.normalize(evolve_embs[-1])
        te = model.get_init_time(allt)
        B = len(allt); tot = 0.0
        starts = list(range(0, B, a.dec_chunk))
        for ci, c0 in enumerate(starts):
            sl = slice(c0, c0 + a.dec_chunk)
            sr = model.raw_mode(pre_emb, r_emb, tslice(te, sl), allt[sl]); sh = model.history_mode(pre_emb, r_emb, tslice(te, sl), allt[sl], V[sl])
            sc = torch.log(a.hr * sh + (1 - a.hr) * sr)
            l = 0.7 * F.nll_loss(sc, allt[sl, 2], reduction="sum") / B
            l.backward(retain_graph=(ci < len(starts) - 1)); tot += float(l.detach())
            del sr, sh, sc, l
        return tot

    LOG = dict(model="tirgn", dataset=a.ds, config=vars(a), gpu=torch.cuda.get_device_name(0), epoch_log=[])
    best = (-1.0, None, -1); n_tr = len(snaps["train"])
    if a.resume_ckpt:
        ck = torch.load(a.resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"]); best = (ck["log"]["best_valid_filt_MRR"], ck["state_dict"], ck["log"]["best_epoch"])
        LOG = ck["log"]; a.epochs = 0
        log("resumed from", a.resume_ckpt, "best epoch", best[2])
    stop_reason = "max_epochs"
    for ep in range(a.epochs if not a.smoke else 1):
        model.train(); idx = list(range(n_tr)); random.shuffle(idx); losses = []
        if a.smoke: idx = idx[:a.smoke]
        te0 = time.time()
        for k, i in enumerate(idx):
            if i == 0: continue
            ts, trip = snaps["train"][i]
            glist = [G("train", j) for j in range(max(0, i - a.L), i)]
            tt = torch.as_tensor(trip, device=dev).long()
            s_np = np.concatenate([trip[:, 0], trip[:, 2]]); r_np = np.concatenate([trip[:, 1], trip[:, 1] + R])
            V = HV.dense(s_np, r_np, ts, dev)
            loss = chunked_loss(glist, tt, V)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
            losses.append(loss)
            if k % 200 == 0:
                log("ep", ep, "step", k, "/", len(idx), "loss", round(float(np.mean(losses[-200:])), 4),
                    "s/step", round((time.time() - te0) / (k + 1), 3), "gpu_mem_GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
        t_train_ep = time.time() - te0
        vm = eval_valid(maxsnap=(3 if a.smoke else None))
        LOG["epoch_log"].append(dict(epoch=ep, loss=round(float(np.mean(losses)), 5), valid_filt_MRR=round(vm, 6),
                                     train_epoch_s=round(t_train_ep), t_s=round(time.time() - t0)))
        log(LOG["epoch_log"][-1])
        json.dump(LOG, open(f"{a.out}/tirgn_{a.ds}_epoch_log.json", "w"), indent=1)
        if vm > best[0]:
            best = (vm, {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}, ep)
            torch.save(dict(state_dict=best[1], log=dict(LOG, best_epoch=ep, best_valid_filt_MRR=vm)), f"{a.out}/tirgn_{a.ds}_seed{a.seed}.pt")
        elif ep - best[2] >= a.patience:
            stop_reason = "patience"; break
        if (time.time() - t0) / 3600 > a.max_train_hours:
            stop_reason = "wall_clock_cap"; break
    model.load_state_dict(best[1]); model.eval()
    LOG.update(best_epoch=best[2], best_valid_filt_MRR=best[0], train_s=round(time.time() - t0), stop_reason=stop_reason)
    torch.save(dict(state_dict=best[1], log=LOG), f"{a.out}/tirgn_{a.ds}_seed{a.seed}.pt")
    json.dump(LOG, open(f"{a.out}/tirgn_{a.ds}_epoch_log.json", "w"), indent=1)
    if a.no_dump:
        log("no dump; done"); return
    # ---------------- dump (harness schema)
    K = a.topk
    n_rows = 2 * (sum(len(x[1]) for x in snaps["train"][1:]) + len(Q["valid"]) + len(Q["test"]))
    SRO = np.zeros((n_rows, 4), np.int32); GS = np.zeros(n_rows, np.float32); GR = np.zeros(n_rows, np.int32)
    GF = np.zeros(n_rows, np.int32); TI = np.zeros((n_rows, K), np.int32); TS_ = np.zeros((n_rows, K), np.float32); SP = np.zeros(n_rows, np.int8)
    ptr = 0; t1 = time.time()
    for code, sp in enumerate(("train", "valid", "test")):
        for i in range(len(snaps[sp])):
            if sp == "train" and i == 0: continue
            if a.smoke and i > 2: break
            sro, lp = score_snap(sp, i)
            glp, rk_r, rk_f = filt_rank(sro, lp)
            tv, ti = torch.topk(lp, K, dim=1)
            n = len(sro)
            SRO[ptr:ptr + n] = sro; GS[ptr:ptr + n] = glp.cpu().numpy(); GR[ptr:ptr + n] = rk_r.cpu().numpy()
            GF[ptr:ptr + n] = rk_f.cpu().numpy(); TI[ptr:ptr + n] = ti.cpu().numpy(); TS_[ptr:ptr + n] = tv.cpu().numpy(); SP[ptr:ptr + n] = code
            ptr += n
            del lp
        log("dumped", sp, ptr)
    def mets(rk):
        rk = rk.astype(np.float64)
        return dict(MRR=round(float(np.mean(1 / rk)), 6), H1=round(float(np.mean(rk <= 1)), 6), H3=round(float(np.mean(rk <= 3)), 6),
                    H10=round(float(np.mean(rk <= 10)), 6), n=int(len(rk)))
    SP_ = SP[:ptr]; GF_ = GF[:ptr]
    LOG["valid_filt"] = mets(GF_[SP_ == 1]); LOG["test_filt"] = mets(GF_[SP_ == 2]); LOG["train_filt"] = mets(GF_[SP_ == 0])
    LOG["dump_s"] = round(time.time() - t1)
    n_tr_ = int((SP_ == 0).sum()); n_va = int((SP_ == 1).sum()); n_te = int((SP_ == 2).sum())
    LOG["rows"] = dict(train=n_tr_, valid=n_va, test=n_te, total=int(ptr), K=K)
    out = f"{a.out}/tirgn_scores_{a.ds}_traintest" + ("_smoke" if a.smoke else "") + ".npz"
    np.savez(out, sro=SRO[:ptr], gold_score=GS[:ptr], gold_rank_raw=GR[:ptr], gold_rank_filt=GF[:ptr], topk_ids=TI[:ptr],
             topk_scores=TS_[:ptr], split=SP[:ptr], split_bounds=np.array([n_tr_, n_tr_ + n_va, n_tr_ + n_va + n_te], np.int64))
    LOG["total_s"] = round(time.time() - t0)
    json.dump(LOG, open(f"{a.out}/tirgn_{a.ds}_dump_log.json", "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE", json.dumps({k: LOG.get(k) for k in ("valid_filt", "best_epoch", "rows", "stop_reason")}))


if __name__ == "__main__":
    main()
