#!/usr/bin/env python3
"""v76 / sp_tirgn_valfit.py -- M3a of v76_rule.json: sp_tirgn.py with ONE change: the training snapshots are the train
snapshots plus every validation snapshot with ts < --fit_cut (validation-fit period); epoch selection / early stopping uses
the time-aware filtered MRR of the validation-select snapshots only (ts >= --fit_cut).  History = previous L ground-truth
snapshots across splits (unchanged).  Run with PC_P pointing to a separate directory (dumps/ inside it).
pilot_smallpedia / sp_tirgn.py -- train TiRGN (authors' model code via the project's pure-torch DGL shim) on
tkgl-smallpedia and dump it in the harness base_dump.py schema + full log-prob rows for valid/test (GPU).

Model / hyper-parameters = the ICEWS18 TiRGN run of this project (convgcn encoder, timeconvtranse decoder, 2 layers,
h 200, history 10 train = test, history_rate 0.3, layer-norm, self-loop, n_bases 100, dropouts 0.2, lr 1e-3, grad-norm 1,
entity prediction only; NO static word graph).  Training loop = TiRGN main.py (shuffled train snapshots, history =
previous `L` ground-truth snapshots, loss 0.7 * NLL_entity).  The global-history vocabulary (TiRGN's
tail_history_<t>.npz: objects o seen with (s, r') at any ts < t, all splits) is computed on the fly from a sorted
first-occurrence index instead of the 2.25e9-row sparse files (identical 0/1 vocabulary).  rel_history is not built
(relation prediction is off; it only feeds the relation head).  history_mode's GPU->numpy->GPU float64 copy of the
vocabulary is replaced by vocab.float() (same values).
Selection: epoch with the best FULL-VALID time-aware filtered MRR (both directions), max --epochs, patience 5.
Dump rows: train (from the 2nd snapshot), valid, test; per timestamp the forward triples then their inverses
(= base_dump.py / split_queries order).  Scores = TiRGN's final log( hr * p_hist + (1 - hr) * p_raw ).
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from sp_common import (P, DD, DS, N_ENT, N_REL, FULLROWS, DUMPS, CKPT, Splits, split_queries, jdump, log_fn)

TREE = os.environ.get("TIRGN_TREE", f"{P}/stage/TiRGN")
sys.path.insert(0, TREE); sys.path.insert(0, f"{TREE}/src")
import dgl_shim  # noqa: E402,F401  pure-torch DGL (must precede the repo imports)
from rgcn.utils import build_sub_graph  # noqa: E402
from src.rrgcn import RecurrentRGCN  # noqa: E402


class HistVocab:
    """first-occurrence index over (key = s * 2R + r', o) of ALL facts (both directions)."""
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
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if ln.sum() else np.zeros(0, np.int64)
        keep = self.t[idx] < t
        V = torch.zeros((len(kq), self.N), device=dev)
        if keep.any():
            V[torch.as_tensor(rows[keep], device=dev), torch.as_tensor(self.o[idx[keep]], device=dev)] = 1.0
        return V


def patched_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
    score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=history_vocabulary.float())
    return F.softmax(score_global, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--hr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--fit_cut", type=int, default=105)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    S = Splits(DS); N, R = N_ENT, N_REL
    RecurrentRGCN.history_mode = patched_history_mode
    Q = {sp: S.q[sp] for sp in ("train", "valid", "test")}
    allq = np.concatenate([Q["train"], Q["valid"], Q["test"]])
    HV = HistVocab(allq, N, R)
    snaps = {}
    for sp in ("train", "valid", "test"):
        q = Q[sp]; o = np.argsort(q[:, 3], kind="stable"); q = q[o]
        snaps[sp] = [(int(u), q[q[:, 3] == u]) for u in np.unique(q[:, 3])]
    TRS = [("train", i) for i in range(1, len(snaps["train"]))] + [("valid", j) for j in range(len(snaps["valid"])) if snaps["valid"][j][0] < a.fit_cut]
    SELS = [j for j in range(len(snaps["valid"])) if snaps["valid"][j][0] >= a.fit_cut]
    assert len(SELS) > 0 and len(TRS) > len(snaps["train"]) - 1
    train_times = [u for u, _ in snaps["train"]]
    tint = train_times[1] - train_times[0]
    num_times = sum(len(v) for v in snaps.values())
    model = RecurrentRGCN("timeconvtranse", "convgcn", N, R, 0, 0, num_times, tint, 200, "sub", a.hr, sequence_len=a.L,
                          num_bases=100, num_basis=100, num_hidden_layers=2, dropout=0.2, self_loop=True, skip_connect=False,
                          layer_norm=True, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, aggregation="none",
                          weight=0.5, discount=1, angle=10, use_static=False, entity_prediction=True, relation_prediction=False,
                          use_cuda=True, gpu=0, analysis=False).to(dev)
    torch.cuda.set_device(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    gcache = {}

    def G(sp, i):
        k = (sp, i)
        if k not in gcache:
            gcache[k] = build_sub_graph(N, R, snaps[sp][i][1][:, :3], True, 0)
        return gcache[k]

    def hist_graphs(sp, i):
        """previous L ground-truth snapshots before snapshot i of split sp (all earlier splits included)."""
        seq = [("train", j) for j in range(len(snaps["train"]))] + [("valid", j) for j in range(len(snaps["valid"]))] + \
              [("test", j) for j in range(len(snaps["test"]))]
        pos = seq.index((sp, i))
        return [G(*x) for x in seq[max(0, pos - a.L):pos]]

    F_ = S.filter_index()

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
        sr = model.raw_mode(emb, r_emb, te, allt); sh = model.history_mode(emb, r_emb, te, allt, V)
        lp = torch.log(a.hr * sh + (1 - a.hr) * sr)
        return allt.cpu().numpy(), lp

    def filt_rank(sro, lp):
        gold = torch.as_tensor(sro[:, 2], device=dev).long()
        glp = lp.gather(1, gold.unsqueeze(1)).squeeze(1)
        fk = S.fkey(sro[:, 0], sro[:, 1], sro[:, 3])
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

    def eval_valid():
        model.eval(); rr = []
        for i in SELS:
            sro, lp = score_snap("valid", i)
            _, _, rk = filt_rank(sro, lp); rr.append((1.0 / rk.double()).cpu().numpy())
        return float(np.concatenate(rr).mean())

    LOG = dict(v76="M3a validation-trained base", fit_cut=a.fit_cut, n_train_snapshots=len(snaps["train"]) - 1,
               n_valid_fit_snapshots=sum(1 for x in TRS if x[0] == "valid"), n_valid_select_snapshots=len(SELS),
               valid_fit_facts=int(sum(len(snaps["valid"][j][1]) for sp_, j in TRS if sp_ == "valid")),
               valid_select_facts=int(sum(len(snaps["valid"][j][1]) for j in SELS)), model="tirgn", dataset=DS, host="serverA (GPU, tkg-gpu:latest)", config=vars(a),
               tree=TREE, gpu=torch.cuda.get_device_name(0), epoch_log=[])
    best = (-1.0, None, -1); n_tr = len(snaps["train"])
    for ep in range(a.epochs if not a.smoke else 1):
        model.train(); idx = list(TRS); random.shuffle(idx); losses = []
        if a.smoke: idx = idx[:a.smoke]
        for sp_t, i in idx:
            ts, trip = snaps[sp_t][i]
            glist = hist_graphs(sp_t, i)
            tt = torch.as_tensor(trip, device=dev).long()
            s_np = np.concatenate([trip[:, 0], trip[:, 2]]); r_np = np.concatenate([trip[:, 1], trip[:, 1] + R])
            V = HV.dense(s_np, r_np, ts, dev)
            le, lr_, ls = model.get_loss(glist, tt, None, V, None, True)
            loss = 0.7 * le + 0.3 * lr_ + ls
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
            losses.append(float(loss.item()))
        vm = eval_valid()
        LOG["epoch_log"].append(dict(epoch=ep, loss=round(float(np.mean(losses)), 5), valid_filt_MRR=round(vm, 6), t_s=round(time.time() - t0)))
        log(LOG["epoch_log"][-1])
        if vm > best[0]:
            best = (vm, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        elif ep - best[2] >= a.patience:
            break
    model.load_state_dict(best[1]); model.eval()
    LOG.update(best_epoch=best[2], best_valid_filt_MRR=best[0], train_s=round(time.time() - t0))
    torch.save(dict(state_dict=best[1], log=LOG), CKPT["tirgn"])
    # ---------------- dump (harness schema) + full rows for valid/test
    K = a.topk
    n_rows = 2 * (sum(len(x[1]) for x in snaps["train"][1:]) + len(Q["valid"]) + len(Q["test"]))
    SRO = np.zeros((n_rows, 4), np.int32); GS = np.zeros(n_rows, np.float32); GR = np.zeros(n_rows, np.int32)
    GF = np.zeros(n_rows, np.int32); TI = np.zeros((n_rows, K), np.int32); TS_ = np.zeros((n_rows, K), np.float32); SP = np.zeros(n_rows, np.int8)
    MM = {sp: np.memmap(FULLROWS["tirgn"][sp], dtype=np.float32, mode="w+", shape=(2 * len(Q[sp]), N)) for sp in ("valid", "test")}
    ptr = 0; t1 = time.time()
    for code, sp in enumerate(("train", "valid", "test")):
        mp_ = 0
        for i in range(len(snaps[sp])):
            if sp == "train" and i == 0: continue
            if a.smoke and i > 2: break
            sro, lp = score_snap(sp, i)
            glp, rk_r, rk_f = filt_rank(sro, lp)
            tv, ti = torch.topk(lp, K, dim=1)
            n = len(sro)
            SRO[ptr:ptr + n] = sro; GS[ptr:ptr + n] = glp.cpu().numpy(); GR[ptr:ptr + n] = rk_r.cpu().numpy()
            GF[ptr:ptr + n] = rk_f.cpu().numpy(); TI[ptr:ptr + n] = ti.cpu().numpy(); TS_[ptr:ptr + n] = tv.cpu().numpy(); SP[ptr:ptr + n] = code
            if sp != "train":
                MM[sp][mp_:mp_ + n] = lp.cpu().numpy(); mp_ += n
            ptr += n
            del lp
        log("dumped", sp, ptr)
    for v in MM.values(): v.flush()
    if not a.smoke:
        assert ptr == n_rows, (ptr, n_rows)
        for sp in ("valid", "test"):
            m = SP == {"valid": 1, "test": 2}[sp]
            assert (SRO[m].astype(np.int64) == split_queries(S, sp)).all(), f"row order != split_queries ({sp})"
    def mets(rk):
        rk = rk.astype(np.float64)
        return dict(MRR=round(float(np.mean(1 / rk)), 6), H1=round(float(np.mean(rk <= 1)), 6), H3=round(float(np.mean(rk <= 3)), 6),
                    H10=round(float(np.mean(rk <= 10)), 6), n=int(len(rk)))
    LOG["valid_filt"] = mets(GF[SP == 1]); LOG["test_filt"] = mets(GF[SP == 2]); LOG["test_raw"] = mets(GR[SP == 2])
    LOG["train_filt"] = mets(GF[SP == 0]); LOG["dump_s"] = round(time.time() - t1)
    n_tr_ = int((SP == 0).sum()); n_va = int((SP == 1).sum()); n_te = int((SP == 2).sum())
    LOG["rows"] = dict(train=n_tr_, valid=n_va, test=n_te, total=int(ptr), K=K)
    out = DUMPS["tirgn"] if not a.smoke else DUMPS["tirgn"].replace(".npz", "_smoke.npz")
    np.savez(out, sro=SRO[:ptr], gold_score=GS[:ptr], gold_rank_raw=GR[:ptr], gold_rank_filt=GF[:ptr], topk_ids=TI[:ptr],
             topk_scores=TS_[:ptr], split=SP[:ptr], split_bounds=np.array([n_tr_, n_tr_ + n_va, n_tr_ + n_va + n_te], np.int64))
    LOG["total_s"] = round(time.time() - t0)
    LOG["valid_select_filt"] = mets(GF[(SP == 1) & (SRO[:, 3] >= a.fit_cut)]); LOG["valid_fit_filt_in_sample"] = mets(GF[(SP == 1) & (SRO[:, 3] < a.fit_cut)])
    jdump(LOG, "tirgn_SMALLPEDIA_dump_log.json"); jdump(LOG, f"{P}/dumps/tirgn_{DS}_dump_log.json")
    os.system(f"chmod a+r {P}/dumps/tirgn_* 2>/dev/null")
    log("ALL DONE", json.dumps({k: LOG[k] for k in ("valid_filt", "test_filt", "best_epoch", "rows")}))


if __name__ == "__main__":
    main()
