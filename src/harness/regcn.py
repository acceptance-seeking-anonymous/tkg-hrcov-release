"""RE-GCN (Li et al., SIGIR 2021) — pure-PyTorch reimplementation, no DGL.

Evolution unit: relation-aware GCN (UnionRGCN-style messages) + time gate over a
window of history graphs; relation embeddings evolve via GRUCell; ConvTransE
decoder. Runs natively on torch 2.8 / sm_120. Emits per-query ranked candidate
lists for the tkg-eval harness.
"""
import os, json, argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- data -----------------------------
def read_quads(path):
    q = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            h, r, t, ts = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            q.append((h, r, t, ts))
    return np.array(q, dtype=np.int64)

def load_stat(data_dir):
    with open(os.path.join(data_dir, "stat.txt")) as f:
        parts = f.readline().split()
    return int(parts[0]), int(parts[1])   # num_ent, num_rel

def group_by_time(quads):
    """Return list of (timestamp, array[[h,r,t],...]) sorted by timestamp."""
    order = np.argsort(quads[:, 3], kind="stable")
    q = quads[order]
    ts = q[:, 3]
    uniq = np.unique(ts)
    groups = []
    for u in uniq:
        m = q[ts == u]
        groups.append((int(u), m[:, :3].copy()))
    return groups

# ----------------------------- model -----------------------------
class UnionRGCNLayer(nn.Module):
    """One RGCN layer with RE-GCN's message: msg = W (h_src + r_emb), mean-agg,
    plus self-loop, skip connection, RReLU."""
    def __init__(self, h_dim, dropout=0.2, self_loop=True, skip=True):
        super().__init__()
        self.h_dim = h_dim
        self.self_loop = self_loop
        self.skip = skip
        self.weight = nn.Parameter(torch.empty(h_dim, h_dim))
        self.loop_weight = nn.Parameter(torch.empty(h_dim, h_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.RReLU()

    def forward(self, h, edges, rel_emb):
        # edges: LongTensor [E,3] (s,r,o) with inverse edges already included
        s, r, o = edges[:, 0], edges[:, 1], edges[:, 2]
        msg = torch.mm(h[s] + rel_emb[r], self.weight)          # [E,H]
        agg = torch.zeros_like(h)
        agg.index_add_(0, o, msg)
        deg = torch.zeros(h.size(0), device=h.device)
        deg.index_add_(0, o, torch.ones_like(o, dtype=torch.float))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg = agg / deg
        if self.self_loop:
            agg = agg + torch.mm(h, self.loop_weight)
        out = self.act(agg)
        out = self.dropout(out)
        if self.skip:
            out = out + h
        return out

class ConvTransE(nn.Module):
    """ConvTransE decoder: score = f(conv([e_s; e_r])) · E^T."""
    def __init__(self, h_dim, num_channels=50, kernel_size=3, dropout=0.2):
        super().__init__()
        self.h_dim = h_dim
        self.inp_drop = nn.Dropout(dropout)
        self.hid_drop = nn.Dropout(dropout)
        self.feat_drop = nn.Dropout(dropout)
        pad = int(math.floor(kernel_size / 2))
        self.conv = nn.Conv1d(2, num_channels, kernel_size, padding=pad)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.fc = nn.Linear(h_dim * num_channels, h_dim)

    def forward(self, e_s, e_r, all_ent):
        b = e_s.size(0)
        x = torch.stack([e_s, e_r], dim=1)          # [B,2,H]
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv(x)                              # [B,C,H]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop(x)
        x = x.view(b, -1)                             # [B, C*H]
        x = self.fc(x)                                # [B,H]
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return torch.mm(x, all_ent.t())               # [B, num_ent]

class REGCN(nn.Module):
    def __init__(self, num_ent, num_rel, h_dim=200, n_layers=2, dropout=0.2,
                 history_len=10, rel_prior=None, ent_prior=None, ent_prior_mode="scalar"):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel          # original relation count (inverse added internally -> 2*num_rel)
        self.h_dim = h_dim
        self.history_len = history_len
        self.ent_emb = nn.Parameter(torch.empty(num_ent, h_dim))
        self.rel_emb = nn.Parameter(torch.empty(2 * num_rel, h_dim))
        nn.init.xavier_normal_(self.ent_emb)
        nn.init.xavier_normal_(self.rel_emb)
        # Option B actor-KG ENTITY prior: [num_ent, h_dim] fixed tensor (0 rows = no
        # profile). Added to the static entity embedding, gated by a learnable scalar
        # init 0 -> starts identical to baseline. self.use_ent_prior=False -> baseline.
        self.use_ent_prior = ent_prior is not None
        self.ent_prior_mode = ent_prior_mode   # "scalar" | "linear"
        if self.use_ent_prior:
            self.register_buffer("ent_prior", ent_prior)
            self.ent_prior_gate = nn.Parameter(torch.zeros(1))
            if ent_prior_mode == "linear":
                # learnable projection of the profile subspace into entity geometry;
                # weights init 0 -> starts identical to baseline, model learns the alignment.
                self.ent_prior_proj = nn.Linear(h_dim, h_dim, bias=False)
                nn.init.zeros_(self.ent_prior_proj.weight)
            elif ent_prior_mode == "twochannel":
                # Option C: a two-channel encoder. The commonsense channel encodes the
                # actor profile into a context vector c = cs_enc(ent_prior). At EVERY
                # recurrence timestep, a state-conditioned gate g = sigmoid(gate_w[h, c])
                # decides per-entity how much commonsense to blend into the evolving
                # entity state, and cs_out projects it in. cs_out init 0 -> the fusion
                # contributes nothing at step 0, so training starts baseline-identical
                # and the model learns both the alignment (cs_enc) and the per-step,
                # per-entity gating (gate_w) jointly.
                self.cs_enc = nn.Linear(h_dim, h_dim)           # profile -> commonsense context
                self.cs_gate = nn.Linear(2 * h_dim, h_dim)      # [state, context] -> per-dim gate
                self.cs_out = nn.Linear(h_dim, h_dim, bias=False)
                nn.init.zeros_(self.cs_out.weight)              # zero fusion at init
        else:
            self.ent_prior = None
        # ECCKG event-type prior (Option A ablation). rel_prior: [num_rel, h_dim] fixed
        # tensor (0 rows = no anchor). Applied additively to BOTH the forward relation
        # and its inverse, gated by a single learnable scalar init 0 -> starts identical
        # to baseline, model chooses how much prior to use. self.use_prior=False -> baseline.
        self.use_prior = rel_prior is not None
        if self.use_prior:
            full = torch.zeros(2 * num_rel, h_dim)
            full[:num_rel] = rel_prior
            full[num_rel:] = rel_prior     # inverse relation shares the event-type prior
            self.register_buffer("rel_prior", full)
            self.prior_gate = nn.Parameter(torch.zeros(1))
        else:
            self.rel_prior = None
        self.layers = nn.ModuleList([UnionRGCNLayer(h_dim, dropout) for _ in range(n_layers)])
        self.time_gate_w = nn.Linear(h_dim, h_dim)
        self.rel_gru = nn.GRUCell(2 * h_dim, h_dim)
        self.decoder = ConvTransE(h_dim, dropout=dropout)
        self.decoder_r = ConvTransE(h_dim, dropout=dropout)  # for relation-side (unused in eval but trained)

    def evolve(self, history_graphs):
        """history_graphs: list of edge-tensors [E,3] (with inverse edges).
        Returns evolved entity + relation embeddings."""
        ent_base = self.ent_emb
        cs_ctx = None
        if self.use_ent_prior:
            if self.ent_prior_mode == "linear":
                ent_base = ent_base + self.ent_prior_proj(self.ent_prior)
            elif self.ent_prior_mode == "twochannel":
                # commonsense channel context, recomputed once per forward (fixed prior)
                cs_ctx = self.cs_enc(self.ent_prior)            # [num_ent, H]
            else:
                ent_base = ent_base + self.ent_prior_gate * self.ent_prior
        h = F.normalize(ent_base)
        # effective static relation embedding = learned + gated ECCKG event-type prior
        rel_base = self.rel_emb
        if self.use_prior:
            rel_base = rel_base + self.prior_gate * self.rel_prior
        r = rel_base
        for edges in history_graphs:
            if edges.numel() == 0:
                continue
            # relation evolution: mean-pool entity emb of endpoints per relation
            r_agg = torch.zeros_like(r)
            cnt = torch.zeros(r.size(0), device=r.device)
            s, rr, o = edges[:, 0], edges[:, 1], edges[:, 2]
            r_agg.index_add_(0, rr, h[s] + h[o])
            cnt.index_add_(0, rr, torch.ones_like(rr, dtype=torch.float))
            cnt = cnt.clamp(min=1.0).unsqueeze(1)
            r_agg = r_agg / cnt
            r_in = torch.cat([rel_base, r_agg], dim=1)   # [2R, 2H]
            r = self.rel_gru(r_in, r)
            r = F.normalize(r)
            # entity evolution via RGCN layers
            cur = h
            for layer in self.layers:
                cur = layer(cur, edges, r)
            cur = F.normalize(cur)
            # time gate
            u = torch.sigmoid(self.time_gate_w(h))
            h = u * cur + (1.0 - u) * h
            # Option C two-channel fusion: per-entity, state-conditioned gate blends
            # the commonsense context into the evolving state (cs_out init 0 -> no-op
            # at start). Applied at every timestep, inside the recurrence.
            if cs_ctx is not None:
                g = torch.sigmoid(self.cs_gate(torch.cat([h, cs_ctx], dim=1)))
                h = F.normalize(h + self.cs_out(g * cs_ctx))
        return h, r

    def forward(self, history_graphs, s, rel):
        h, r = self.evolve(history_graphs)
        scores = self.decoder(h[s], r[rel], h)   # [B, num_ent]
        return scores, h, r

# ----------------------------- edges with inverse -----------------------------
def make_edges(triples, num_rel, device):
    """triples: array [N,3] (h,r,t) -> edge tensor [2N,3] with inverse rel r+num_rel."""
    if len(triples) == 0:
        return torch.empty(0, 3, dtype=torch.long, device=device)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    fwd = np.stack([h, r, t], axis=1)
    inv = np.stack([t, r + num_rel, h], axis=1)
    e = np.concatenate([fwd, inv], axis=0)
    return torch.tensor(e, dtype=torch.long, device=device)

# ----------------------------- train / eval -----------------------------
def build_time_index(groups):
    return [g[1] for g in groups]   # list of [Ni,3] arrays in time order

def build_filter_index(all_quads, num_rel):
    """Time-aware filter sets. Returns dict:
      tail[(h,r,ts)] = set of true tails ; head[(r,t,ts)] = set of true heads."""
    from collections import defaultdict
    tail = defaultdict(set); head = defaultdict(set)
    for h, r, t, ts in all_quads:
        tail[(h, r, ts)].add(t); head[(r, t, ts)].add(h)
    return tail, head

def metrics_from_ranks(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return {"MRR": float(np.mean(1.0/r)), "Hits@1": float(np.mean(r<=1)),
            "Hits@3": float(np.mean(r<=3)), "Hits@10": float(np.mean(r<=10)), "n": int(len(r))}

def get_history(all_snaps_edges, idx, hist_len):
    lo = max(0, idx - hist_len)
    return all_snaps_edges[lo:idx]

def train_and_eval(data_dir, out_prefix, h_dim=200, n_layers=2, epochs=20,
                   lr=1e-3, history_len=10, dropout=0.2, device="cuda",
                   emit_topk=200, eval_split="test", seed=42, sample_keys_file=None,
                   rel_prior_file=None, ent_prior_file=None, ent_prior_mode="scalar"):
    torch.manual_seed(seed); np.random.seed(seed)
    num_ent, num_rel = load_stat(data_dir)
    rel_prior = None
    if rel_prior_file:
        _pz = np.load(rel_prior_file)
        _p = np.zeros((num_rel, h_dim), dtype=np.float32)
        for _rid, _vec in zip(_pz["relation_ids"], _pz["priors"]):
            if 0 <= int(_rid) < num_rel:
                _p[int(_rid)] = _vec
        rel_prior = torch.tensor(_p, device=device)
        print(f"[ECCKG prior] loaded {rel_prior_file}: {(np.abs(_p).sum(1)>0).sum()}/{num_rel} relations have a prior")
    ent_prior = None
    if ent_prior_file:
        _ez = np.load(ent_prior_file)
        _e = np.zeros((num_ent, h_dim), dtype=np.float32)
        for _eid, _vec in zip(_ez["entity_ids"], _ez["priors"]):
            if 0 <= int(_eid) < num_ent:
                _e[int(_eid)] = _vec
        ent_prior = torch.tensor(_e, device=device)
        print(f"[actor-KG prior] loaded {ent_prior_file}: {(np.abs(_e).sum(1)>0).sum()}/{num_ent} entities have a prior")
    train = read_quads(os.path.join(data_dir, "train.txt"))
    valid = read_quads(os.path.join(data_dir, "valid.txt"))
    test  = read_quads(os.path.join(data_dir, "test.txt"))

    tr_groups = build_time_index(group_by_time(train))
    va_groups = group_by_time(valid)
    te_groups = group_by_time(test)

    # precompute edge tensors for every training snapshot
    tr_edges = [make_edges(g, num_rel, device) for g in tr_groups]

    model = REGCN(num_ent, num_rel, h_dim, n_layers, dropout, history_len, rel_prior=rel_prior, ent_prior=ent_prior, ent_prior_mode=ent_prior_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def run_epoch():
        model.train()
        total = 0.0
        # predict snapshot i from previous history_len snapshots
        for i in range(1, len(tr_groups)):
            triples = tr_groups[i]
            if len(triples) == 0:
                continue
            hist = get_history(tr_edges, i, history_len)
            # build training queries: tail prediction (s,r,?) + head via inverse (t, r+R, ?)
            s = np.concatenate([triples[:, 0], triples[:, 2]])
            rel = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
            obj = np.concatenate([triples[:, 2], triples[:, 0]])
            s = torch.tensor(s, device=device)
            rel = torch.tensor(rel, device=device)
            obj = torch.tensor(obj, device=device)
            scores, _, _ = model(hist, s, rel)
            loss = F.cross_entropy(scores, obj)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        return total / max(1, len(tr_groups) - 1)

    # time-aware filter index over ALL quads (train+valid+test), standard protocol
    tail_filt, head_filt = build_filter_index(
        list(map(tuple, train)) + list(map(tuple, valid)) + list(map(tuple, test)), num_rel)

    @torch.no_grad()
    def evaluate_and_emit(groups, emit=False, seed_extra=None, sample_keys=None):
        """Computes RAW and TIME-AWARE-FILTERED ranks on-device against the full
        score matrix (no top-k truncation). Emits top-200 candidate lists only
        for queries in `sample_keys` (the 200-query error-analysis sample)."""
        model.eval()
        hist_edges = list(tr_edges)
        if seed_extra:
            for g in seed_extra:
                hist_edges.append(make_edges(g, num_rel, device))
        raw_ranks = []; filt_ranks = []
        sample_preds = []
        gold_ranks = []   # (gold_entity_id, filtered_rank, target) for every query -> offline stratification
        want = sample_keys or set()
        for (ts, triples) in groups:
            if len(triples) == 0:
                continue
            hist = hist_edges[-history_len:]
            s_all  = np.concatenate([triples[:, 0], triples[:, 2]])
            rel_all= np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
            obj_all= np.concatenate([triples[:, 2], triples[:, 0]])
            tgt    = ["tail"] * len(triples) + ["head"] * len(triples)
            hq = np.concatenate([triples[:, 0], triples[:, 2]])
            rq = np.concatenate([triples[:, 1], triples[:, 1]])
            tq = np.concatenate([triples[:, 2], triples[:, 0]])
            s = torch.tensor(s_all, device=device)
            rel = torch.tensor(rel_all, device=device)
            scores, _, _ = model(hist, s, rel)                    # [B, num_ent]
            gold = torch.tensor(obj_all, device=device)
            gold_score = scores.gather(1, gold.unsqueeze(1)).squeeze(1)
            # RAW rank
            rk_raw = (scores > gold_score.unsqueeze(1)).sum(1) + 1
            raw_ranks.extend(rk_raw.cpu().tolist())
            # TIME-AWARE FILTERED: mask other true answers (same ts) to -inf, keep gold
            filt_scores = scores.clone()
            for j in range(len(s_all)):
                if tgt[j] == "tail":
                    others = tail_filt.get((int(hq[j]), int(rq[j]), int(ts)), ())
                else:
                    others = head_filt.get((int(rq[j]), int(tq[j]), int(ts)), ())
                if others:
                    idx = [o for o in others if o != int(obj_all[j])]
                    if idx:
                        filt_scores[j, torch.tensor(idx, device=device)] = float("-inf")
            gold_fs = filt_scores.gather(1, gold.unsqueeze(1)).squeeze(1)
            rk_filt = (filt_scores > gold_fs.unsqueeze(1)).sum(1) + 1
            filt_ranks.extend(rk_filt.cpu().tolist())
            rkf_cpu = rk_filt.cpu().tolist()
            for j in range(len(s_all)):
                gold_ranks.append((int(obj_all[j]), int(rkf_cpu[j]), tgt[j]))
            # emit top-200 candidates for sampled queries (tail only used downstream)
            if emit and want:
                _, topi = torch.topk(scores, min(emit_topk, scores.size(1)), dim=1)
                topi_c = topi.cpu().numpy()
                for j in range(len(s_all)):
                    k = (int(hq[j]), int(rq[j]), int(tq[j]), int(ts), tgt[j])
                    if k in want:
                        sample_preds.append({
                            "h": int(hq[j]), "r": int(rq[j]), "t": int(tq[j]), "ts": int(ts),
                            "target": tgt[j], "candidates": topi_c[j].tolist(),
                            "raw_rank": int(rk_raw[j].item()), "filtered_rank": int(rk_filt[j].item()),
                        })
            hist_edges.append(make_edges(triples, num_rel, device))
        return (metrics_from_ranks(raw_ranks), metrics_from_ranks(filt_ranks), sample_preds, gold_ranks)

    log = []
    t0 = time.time()
    for ep in range(epochs):
        l = run_epoch()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            vm_raw, vm_filt, _, _ = evaluate_and_emit(va_groups, emit=False)
            log.append({"epoch": ep + 1, "loss": l,
                        "valid_raw_MRR": vm_raw["MRR"], "valid_filt_MRR": vm_filt["MRR"]})
            print(f"[{os.path.basename(data_dir)}] ep {ep+1} loss {l:.4f} "
                  f"valid raw/filt MRR {vm_raw['MRR']:.4f}/{vm_filt['MRR']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # load 200-query sample keys if provided (tail-target error-analysis sample)
    sample_keys = None
    skpath = f"{sample_keys_file}" if sample_keys_file else None
    if skpath and os.path.exists(skpath):
        sk = json.load(open(skpath))
        sample_keys = set((k["h"], k["r"], k["t"], k["ts"], "tail") for k in sk)
        print(f"loaded {len(sample_keys)} sample keys", flush=True)

    # final: emit test predictions — seed history with valid snapshots (post-train, pre-test)
    tm_raw, tm_filt, sample_preds, gold_ranks = evaluate_and_emit(
        te_groups, emit=True, seed_extra=[g[1] for g in va_groups], sample_keys=sample_keys)
    print(f"[{os.path.basename(data_dir)}] TEST raw {tm_raw} | filt {tm_filt} | sample {len(sample_preds)}", flush=True)

    with open(f"{out_prefix}_sample_preds.json", "w") as f:
        json.dump(sample_preds, f)
    # compact per-query gold ranks (tail queries only) for offline cold/mid/head stratification
    import csv as _csv
    with open(f"{out_prefix}_gold_ranks.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["gold_id", "filtered_rank", "target"])
        for gid, rk, tg in gold_ranks:
            if tg == "tail":
                w.writerow([gid, rk, tg])
    prior_gate_val = float(model.prior_gate.detach().cpu().item()) if getattr(model, "use_prior", False) else None
    ent_gate_val = float(model.ent_prior_gate.detach().cpu().item()) if getattr(model, "use_ent_prior", False) else None
    ent_proj_norm = (float(model.ent_prior_proj.weight.detach().norm().cpu().item())
                     if getattr(model, "use_ent_prior", False) and model.ent_prior_mode == "linear" else None)
    cs_out_norm = (float(model.cs_out.weight.detach().norm().cpu().item())
                   if getattr(model, "use_ent_prior", False) and model.ent_prior_mode == "twochannel" else None)
    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump({"full_test_raw": tm_raw, "full_test_time_aware": tm_filt,
                   "train_log": log, "num_ent": num_ent, "num_rel": num_rel,
                   "ecckg_prior": bool(rel_prior_file), "prior_gate": prior_gate_val,
                   "actor_prior": bool(ent_prior_file), "ent_prior_gate": ent_gate_val,
                   "ent_prior_mode": ent_prior_mode, "ent_proj_norm": ent_proj_norm,
                   "cs_out_norm": cs_out_norm,
                   "config": {"h_dim": h_dim, "n_layers": n_layers, "epochs": epochs,
                              "lr": lr, "history_len": history_len, "dropout": dropout}}, f, indent=2)
    return tm_filt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--h_dim", type=int, default=200)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--history_len", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emit_topk", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample_keys_file", default=None)
    ap.add_argument("--rel_prior_file", default=None)
    ap.add_argument("--ent_prior_file", default=None)
    ap.add_argument("--ent_prior_mode", default="scalar", choices=["scalar", "linear", "twochannel"])
    args = ap.parse_args()
    train_and_eval(args.data_dir, args.out_prefix, args.h_dim, args.n_layers,
                   args.epochs, args.lr, args.history_len, args.dropout,
                   args.device, args.emit_topk, sample_keys_file=args.sample_keys_file,
                   rel_prior_file=args.rel_prior_file, ent_prior_file=args.ent_prior_file,
                   ent_prior_mode=args.ent_prior_mode)
