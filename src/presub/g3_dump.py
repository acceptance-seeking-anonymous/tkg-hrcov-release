#!/usr/bin/env python3
"""presub / g3_dump.py -- G3: replay the TGB-trained RE-GCN checkpoint (g3_tgb_regcn.py) in harness order and write
the smallpedia pipeline's base files, so the unchanged sp_* pipeline runs on base label regcn_tgb:
  <out>/regcn_scores_SMALLPEDIA_traintest.npz   BaseDump schema (sro, gold_score, gold_rank_raw, gold_rank_filt,
                                                topk_ids, topk_scores (log-softmax, top-200), split, split_bounds)
  <out>/regcn_{valid,test}_logp.f32             FULL float32 log-softmax rows over the N temporal entities, rows in
                                                harness split order (per timestamp: forward block, inverse block)
  <out>/g3_dump_log.json                        metrics, row counts, per-query-path verification, timing

Model = TGB's own modules.rrgcn.RecurrentRGCNREGCN built exactly as examples/.../tkgl-smallpedia/regcn.py builds it
(get_args_regcn defaults), state dict loaded strictly.  History protocol = the TGB script's: the last
test_history_len (3) ground-truth snapshots; valid rows roll from the train history, test rows from train + valid.
Graphs are TGB's build_sub_graph over the TGB snapshot layout [forward edges ; inverse edges (r + R)], which is
exactly TGB split_by_time's snapshot (sp_verify: TGB rows == harness rows, identity id maps).  Scores = TGB
ConvTransE logits over all nodes (the script's full-row branch, tanh(emb) product); per-query equality with the
script's candidate branch (samples_of_interest_emb) is checked on the first rows of every evaluated split.
  python3 g3_dump.py --repo <tgb repo copy> --shim_dir <dir> --ckpt <tgb_regcn_ckpt.pt> --dd <harness smallpedia dir>
                     --har <harness root> --out <dir> [--smoke 1]
--smoke 1: train rows of the last train snapshot only, the FIRST valid and the FIRST test snapshot (the rows the
smoke TGB run evaluates), so the official MRR of the smoke test rows must equal the smoke TGB script's test MRR.
"""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--shim_dir", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--dd", required=True); ap.add_argument("--har", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--topk", type=int, default=200); ap.add_argument("--verify_n", type=int, default=64)
ap.add_argument("--rows_block", type=int, default=8192)
a, extra = ap.parse_known_args()
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:8.1f}s]", *x, flush=True)
sys.path.insert(0, a.shim_dir)
import dgl_shim  # noqa: F401
sys.path.append(a.repo)                       # as the TGB script: repo root appended (modules.*)
sys.path.insert(0, a.har)
import torch
import torch.nn.functional as F
from modules.rrgcn import RecurrentRGCNREGCN
from modules.tkg_utils import get_args_regcn
from modules.tkg_utils_dgl import build_sub_graph
from regcn import read_quads, load_stat, group_by_time

os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.cuda.set_device(0)
sys.argv = ["regcn.py"] + extra
args, _ = get_args_regcn(); args.dataset = "tkgl-smallpedia"
N, R = load_stat(a.dd)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
num_nodes = int(sd["dynamic_emb"].shape[0])
num_rels_tgb = 2 * R                           # TGB dataset.num_rels (temporal relations + inverses)
assert num_nodes >= N, (num_nodes, N)
model = RecurrentRGCNREGCN(args.decoder, args.encoder, num_nodes, int(num_rels_tgb / 2), 0, 0, args.n_hidden, args.opn,
                           sequence_len=args.train_history_len, num_bases=args.n_bases, num_basis=args.n_basis,
                           num_hidden_layers=args.n_layers, dropout=args.dropout, self_loop=args.self_loop,
                           skip_connect=args.skip_connect, layer_norm=args.layer_norm, input_dropout=args.input_dropout,
                           hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout, aggregation=args.aggregation,
                           weight=args.weight, discount=args.discount, angle=args.angle, use_static=args.add_static_graph,
                           entity_prediction=args.entity_prediction, relation_prediction=args.relation_prediction,
                           use_cuda=True, gpu=0, analysis=args.run_analysis)
model.cuda()
missing = model.load_state_dict(sd, strict=True)
model.eval()
L = int(args.test_history_len)
log("model", dict(num_nodes=num_nodes, N=N, R=R, L=L, best_epoch=int(ck.get("epoch", -1)), layer_norm=bool(args.layer_norm)))

Q = {sp: read_quads(os.path.join(a.dd, f"{sp}.txt")) for sp in ("train", "valid", "test")}
pairs = {sp: group_by_time(Q[sp]) for sp in ("train", "valid", "test")}
if a.smoke:
    pairs_eval = {"train": len(pairs["train"]) - 1, "valid": 1, "test": 1}
filt = defaultdict(list)                       # harness time-aware filter (identical to hd_base_train.py)
for arr in Q.values():
    for h, r, t, ts in arr:
        filt[(int(h), int(r), int(ts))].append(int(t)); filt[(int(t), int(r) + R, int(ts))].append(int(h))


def tgb_snap(trip):
    inv = np.stack([trip[:, 2], trip[:, 1] + R, trip[:, 0]], 1)
    return np.concatenate([trip[:, :3], inv]).astype(np.int64)


n_rows = {"train": 2 * sum(len(g) for _, g in pairs["train"][1:]), "valid": 2 * len(Q["valid"]), "test": 2 * len(Q["test"])}
if a.smoke:
    n_rows = {"train": 2 * len(pairs["train"][-1][1]), "valid": 2 * len(pairs["valid"][0][1]), "test": 2 * len(pairs["test"][0][1])}
tot = sum(n_rows.values()); K = min(a.topk, N)
SRO = np.zeros((tot, 4), np.int32); GS = np.zeros(tot, np.float32); GR = np.zeros(tot, np.int32); GF = np.zeros(tot, np.int32)
TI = np.zeros((tot, K), np.int32); TV = np.zeros((tot, K), np.float32); SP = np.zeros(tot, np.int8)
MM = {sp: np.memmap(os.path.join(a.out, f"regcn_{sp}_logp.f32"), dtype=np.float32, mode="w+", shape=(n_rows[sp], N))
      for sp in ("valid", "test")}
mp = {"valid": 0, "test": 0}; ptr = [0]
VER = {}


@torch.no_grad()
def score_snapshot(glist, ts, snap, code, verify):
    """embeddings once per snapshot; decoder, ranks, log-softmax, top-K and row writes per block of --rows_block rows
    (a full [n, N] tensor of a large train snapshot does not fit next to another job on a 24 GB card)."""
    evolve_embs, _, r_emb, _, _ = model.forward(glist, None, True)
    emb = F.normalize(evolve_embs[-1]) if model.layer_norm else evolve_embs[-1]
    trip_t = torch.from_numpy(snap).long().cuda()
    n = len(snap)
    s_all, rel_all, obj_all = snap[:, 0], snap[:, 1], snap[:, 2]
    p0 = ptr[0]
    SRO[p0:p0 + n] = np.concatenate([snap, np.full((n, 1), ts)], 1); SP[p0:p0 + n] = code
    sp = {1: "valid", 2: "test"}.get(code)
    for b0 in range(0, n, a.rows_block):
        b1 = min(n, b0 + a.rows_block)
        outs = [model.decoder_ob.forward(emb, r_emb, trip_t[c0:min(b1, c0 + a.chunk)], batch_size_total=n) for c0 in range(b0, b1, a.chunk)]
        sc = (torch.cat(outs) if len(outs) > 1 else outs[0])[:, :N]; del outs
        if verify and b0 == 0:                 # the TGB script's per-query candidate branch, all N nodes as candidates
            allidx = torch.arange(N, device=dev); dmax = 0.0
            for j in range(min(a.verify_n, b1)):
                x = model.decoder_ob.forward(emb, r_emb, trip_t[j:j + 1], samples_of_interest_emb=emb[allidx], batch_size_total=n)
                dmax = max(dmax, float((x[0] - sc[j]).abs().max()))
            VER[code] = dict(n=int(min(a.verify_n, b1)), max_abs_logit_diff=dmax)
        gold = torch.from_numpy(obj_all[b0:b1]).long().cuda()
        gsc = sc.gather(1, gold.unsqueeze(1)).squeeze(1)
        ri, ci = [], []
        for j in range(b0, b1):
            for o in filt.get((int(s_all[j]), int(rel_all[j]), int(ts)), ()):
                if o != int(obj_all[j]): ri.append(j - b0); ci.append(o)
        rk_raw = (sc > gsc.unsqueeze(1)).sum(1) + 1
        if ri:
            fs = sc.clone(); fs[torch.tensor(ri, device=dev), torch.tensor(ci, device=dev)] = float("-inf")
            rk_f = (fs > gsc.unsqueeze(1)).sum(1) + 1; del fs
        else:
            rk_f = rk_raw.clone()
        logp = F.log_softmax(sc, dim=1); del sc
        tv, ti = torch.topk(logp, K, dim=1)
        q0, q1 = p0 + b0, p0 + b1
        GS[q0:q1] = logp.gather(1, gold.unsqueeze(1)).squeeze(1).cpu().numpy()
        GR[q0:q1] = rk_raw.cpu().numpy(); GF[q0:q1] = rk_f.cpu().numpy()
        TI[q0:q1] = ti.cpu().numpy(); TV[q0:q1] = tv.cpu().numpy()
        if sp is not None:
            MM[sp][mp[sp] + b0:mp[sp] + b1] = logp.cpu().numpy()
        del logp, tv, ti
    if sp is not None:
        mp[sp] += n
    ptr[0] = p0 + n


graphs = []                                    # TGB build_sub_graph of every ground-truth snapshot seen so far
for code, sp in ((0, "train"), (1, "valid"), (2, "test")):
    seen = set()
    for i, (ts, trip) in enumerate(pairs[sp]):
        snap = tgb_snap(trip)
        do = not (code == 0 and i == 0)
        if a.smoke:
            do = (code == 0 and i == pairs_eval["train"]) or (code > 0 and i < pairs_eval[sp])
            if code > 0 and i >= pairs_eval[sp]:
                break                          # smoke TGB run cut valid / test to the first snapshot
        if do:
            score_snapshot(graphs[-L:], ts, snap, code, verify=(code not in VER and code > 0) or (code == 0 and code not in VER))
        graphs.append(build_sub_graph(num_nodes, num_rels_tgb, snap, True, 0))
        if len(graphs) > L:
            graphs.pop(0)
    log(sp, "rows", int((SP[:ptr[0]] == code).sum()))
assert ptr[0] == tot, (ptr[0], tot)
assert mp["valid"] == n_rows["valid"] and mp["test"] == n_rows["test"]
for v in MM.values(): v.flush()


def mets(rk):
    rk = rk.astype(np.float64)
    return dict(MRR=round(float(np.mean(1 / rk)), 6), H1=round(float(np.mean(rk <= 1)), 6), H3=round(float(np.mean(rk <= 3)), 6),
                H10=round(float(np.mean(rk <= 10)), 6), n=int(len(rk)))


n_tr = int((SP == 0).sum()); n_va = int((SP == 1).sum()); n_te = int((SP == 2).sum())
out = os.path.join(a.out, "regcn_scores_SMALLPEDIA_traintest.npz")
np.savez(out, sro=SRO, gold_score=GS, gold_rank_raw=GR, gold_rank_filt=GF, topk_ids=TI, topk_scores=TV, split=SP,
         split_bounds=np.array([n_tr, n_tr + n_va, n_tr + n_va + n_te], np.int64))
DL = dict(base="regcn_tgb", ckpt=a.ckpt, best_epoch=int(ck.get("epoch", -1)), smoke=bool(a.smoke), history_len=L,
          num_nodes_model=num_nodes, N=N, R=R, rows=dict(train=n_tr, valid=n_va, test=n_te, K=int(K)),
          valid_filt=mets(GF[SP == 1]), test_filt=mets(GF[SP == 2]), test_raw=mets(GR[SP == 2]),
          per_query_path_check={str(k): v for k, v in VER.items()},
          first_test_snapshot_rows=int(2 * len(pairs["test"][0][1])),
          gpu=torch.cuda.get_device_name(0), host=os.environ.get("HD_HOST", ""), total_s=round(time.time() - t0, 1))
json.dump(DL, open(os.path.join(a.out, "g3_dump_log.json"), "w"), indent=1)
log("ALL DONE", json.dumps({k: DL[k] for k in ("rows", "valid_filt", "test_filt", "per_query_path_check")}))
