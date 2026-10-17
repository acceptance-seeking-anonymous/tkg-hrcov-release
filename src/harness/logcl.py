"""LogCL (Chen et al., ICDE 2024) — pure-PyTorch reimplementation, no DGL.

Faithful-in-spirit: (1) LOCAL encoder = RE-GCN evolution over the recent history
window (relation-aware GCN + time gate + relation GRU); (2) GLOBAL encoder =
query-specific historical repetition vector over all past facts for (s,r,*),
scored by a learned copy mechanism (CyGNet-style); (3) entity-aware attention on
the local decoder; (4) contrastive alignment between local and global query
representations for noise robustness. Emits per-query ranked candidate lists.
"""
import os, json, argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse RE-GCN building blocks
from regcn import (read_quads, load_stat, group_by_time, make_edges,
                   UnionRGCNLayer, ConvTransE, build_time_index, get_history)

class LogCL(nn.Module):
    def __init__(self, num_ent, num_rel, h_dim=200, n_layers=2, dropout=0.2,
                 history_len=10, rel_prior=None):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.h_dim = h_dim
        self.history_len = history_len
        self.ent_emb = nn.Parameter(torch.empty(num_ent, h_dim))
        self.rel_emb = nn.Parameter(torch.empty(2 * num_rel, h_dim))
        nn.init.xavier_normal_(self.ent_emb)
        nn.init.xavier_normal_(self.rel_emb)
        # ECCKG event-type prior (Option A ablation): additive, gated by learnable
        # scalar init 0 -> starts identical to baseline. use_prior=False -> baseline.
        self.use_prior = rel_prior is not None
        if self.use_prior:
            full = torch.zeros(2 * num_rel, h_dim)
            full[:num_rel] = rel_prior; full[num_rel:] = rel_prior
            self.register_buffer("rel_prior", full)
            self.prior_gate = nn.Parameter(torch.zeros(1))
        else:
            self.rel_prior = None
        # local encoder
        self.layers = nn.ModuleList([UnionRGCNLayer(h_dim, dropout) for _ in range(n_layers)])
        self.time_gate_w = nn.Linear(h_dim, h_dim)
        self.rel_gru = nn.GRUCell(2 * h_dim, h_dim)
        self.decoder = ConvTransE(h_dim, dropout=dropout)
        # global copy mechanism: maps query (s,r) rep to a bias over entities
        self.global_w = nn.Linear(2 * h_dim, h_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))    # local/global mix
        # projection heads for contrastive alignment
        self.proj_local = nn.Linear(h_dim, h_dim)
        self.proj_global = nn.Linear(h_dim, h_dim)

    def evolve(self, history_graphs):
        h = F.normalize(self.ent_emb)
        rel_base = self.rel_emb
        if self.use_prior:
            rel_base = rel_base + self.prior_gate * self.rel_prior
        r = rel_base
        for edges in history_graphs:
            if edges.numel() == 0: continue
            s, rr, o = edges[:, 0], edges[:, 1], edges[:, 2]
            r_agg = torch.zeros_like(r); cnt = torch.zeros(r.size(0), device=r.device)
            r_agg.index_add_(0, rr, h[s] + h[o])
            cnt.index_add_(0, rr, torch.ones_like(rr, dtype=torch.float))
            r_agg = r_agg / cnt.clamp(min=1.0).unsqueeze(1)
            r = F.normalize(self.rel_gru(torch.cat([rel_base, r_agg], 1), r))
            cur = h
            for layer in self.layers: cur = layer(cur, edges, r)
            cur = F.normalize(cur)
            u = torch.sigmoid(self.time_gate_w(h))
            h = u * cur + (1.0 - u) * h
        return h, r

    def forward(self, history_graphs, s, rel, global_freq):
        """global_freq: [B, num_ent] historical repetition counts for (s,r,*)."""
        h, r = self.evolve(history_graphs)
        e_s, e_r = h[s], r[rel]
        local_scores = self.decoder(e_s, e_r, h)                 # [B, num_ent]
        # global copy: score = (query rep -> entity space) + log(freq+1) weighting
        q = self.global_w(torch.cat([e_s, e_r], dim=1))          # [B,H]
        global_scores = torch.mm(q, h.t())                        # [B, num_ent]
        global_scores = global_scores + torch.log1p(global_freq) # emphasise repeats
        a = torch.sigmoid(self.alpha)
        scores = a * local_scores + (1.0 - a) * global_scores
        # contrastive reps
        zl = F.normalize(self.proj_local(e_s + e_r), dim=1)
        zg = F.normalize(self.proj_global(q), dim=1)
        return scores, zl, zg, h, r

def info_nce(zl, zg, temp=0.5):
    """align local & global query reps; positives on the diagonal."""
    logits = torch.mm(zl, zg.t()) / temp
    labels = torch.arange(zl.size(0), device=zl.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

# ---------------- global historical repetition index ----------------
class GlobalHist:
    """Tracks, for each (s, r) seen so far, the multiset of observed objects.
    Query time returns a [num_ent] count vector. Rolls forward as snapshots are seen."""
    def __init__(self, num_ent):
        self.num_ent = num_ent
        self.d = {}   # (s,r) -> dict(obj->count)

    def freq_vec(self, s_arr, r_arr, device):
        B = len(s_arr)
        vec = torch.zeros(B, self.num_ent, device=device)
        for i, (s, r) in enumerate(zip(s_arr, r_arr)):
            m = self.d.get((int(s), int(r)))
            if m:
                idx = torch.tensor(list(m.keys()), device=device, dtype=torch.long)
                val = torch.tensor(list(m.values()), device=device, dtype=torch.float)
                vec[i, idx] = val
        return vec

    def update(self, triples, num_rel):
        # add both forward (s,r,o) and inverse (o, r+R, s)
        for h, r, t in triples:
            self.d.setdefault((int(h), int(r)), {}); self.d[(int(h),int(r))][int(t)] = self.d[(int(h),int(r))].get(int(t),0)+1
            self.d.setdefault((int(t), int(r)+num_rel), {}); self.d[(int(t),int(r)+num_rel)][int(h)] = self.d[(int(t),int(r)+num_rel)].get(int(h),0)+1

def train_and_eval(data_dir, out_prefix, h_dim=200, n_layers=2, epochs=20,
                   lr=1e-3, history_len=10, dropout=0.2, device="cuda",
                   emit_topk=200, cl_weight=0.1, seed=42, sample_keys_file=None,
                   rel_prior_file=None):
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
        print(f"[ECCKG prior] loaded {rel_prior_file}: {(np.abs(_p).sum(1)>0).sum()}/{num_rel} relations", flush=True)
    train = read_quads(os.path.join(data_dir, "train.txt"))
    valid = read_quads(os.path.join(data_dir, "valid.txt"))
    test  = read_quads(os.path.join(data_dir, "test.txt"))
    tr_groups = build_time_index(group_by_time(train))
    va_groups = group_by_time(valid)
    te_groups = group_by_time(test)
    tr_edges = [make_edges(g, num_rel, device) for g in tr_groups]

    model = LogCL(num_ent, num_rel, h_dim, n_layers, dropout, history_len, rel_prior=rel_prior).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def run_epoch():
        model.train(); total = 0.0
        gh = GlobalHist(num_ent)
        for i in range(len(tr_groups)):
            triples = tr_groups[i]
            if len(triples) == 0: continue
            if i >= 1:
                hist = get_history(tr_edges, i, history_len)
                s = np.concatenate([triples[:, 0], triples[:, 2]])
                rel = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
                obj = np.concatenate([triples[:, 2], triples[:, 0]])
                gf = gh.freq_vec(s, rel, device)
                s_t = torch.tensor(s, device=device); rel_t = torch.tensor(rel, device=device)
                obj_t = torch.tensor(obj, device=device)
                scores, zl, zg, _, _ = model(hist, s_t, rel_t, gf)
                loss = F.cross_entropy(scores, obj_t) + cl_weight * info_nce(zl, zg)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += loss.item()
            gh.update(triples, num_rel)
        return total / max(1, len(tr_groups) - 1)

    from regcn import build_filter_index, metrics_from_ranks
    tail_filt, head_filt = build_filter_index(
        list(map(tuple, train)) + list(map(tuple, valid)) + list(map(tuple, test)), num_rel)

    @torch.no_grad()
    def evaluate_and_emit(groups, seed_extra=None, emit=False, sample_keys=None):
        """RAW + TIME-AWARE-FILTERED ranks on-device (full score matrix, no top-k
        truncation). Both local window and global history seed consistently.
        Emits top-200 candidates only for queries in sample_keys."""
        model.eval()
        hist_edges = list(tr_edges)
        gh = GlobalHist(num_ent)
        for g in tr_groups: gh.update(g, num_rel)
        if seed_extra:
            for g in seed_extra:
                hist_edges.append(make_edges(g, num_rel, device)); gh.update(g, num_rel)
        raw_ranks = []; filt_ranks = []; sample_preds = []
        want = sample_keys or set()
        for (ts, triples) in groups:
            if len(triples) == 0: continue
            hist = hist_edges[-history_len:]
            s_all  = np.concatenate([triples[:, 0], triples[:, 2]])
            rel_all= np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
            obj_all= np.concatenate([triples[:, 2], triples[:, 0]])
            tgt = ["tail"]*len(triples) + ["head"]*len(triples)
            hq = np.concatenate([triples[:, 0], triples[:, 2]])
            rq = np.concatenate([triples[:, 1], triples[:, 1]])
            tq = np.concatenate([triples[:, 2], triples[:, 0]])
            gf = gh.freq_vec(s_all, rel_all, device)
            s = torch.tensor(s_all, device=device); rel = torch.tensor(rel_all, device=device)
            scores, _, _, _, _ = model(hist, s, rel, gf)
            gold = torch.tensor(obj_all, device=device)
            gs = scores.gather(1, gold.unsqueeze(1)).squeeze(1)
            rk_raw = (scores > gs.unsqueeze(1)).sum(1) + 1
            raw_ranks.extend(rk_raw.cpu().tolist())
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
            gfs = filt_scores.gather(1, gold.unsqueeze(1)).squeeze(1)
            rk_filt = (filt_scores > gfs.unsqueeze(1)).sum(1) + 1
            filt_ranks.extend(rk_filt.cpu().tolist())
            if emit and want:
                _, topi = torch.topk(scores, min(emit_topk, scores.size(1)), dim=1)
                topi_c = topi.cpu().numpy()
                for j in range(len(s_all)):
                    k = (int(hq[j]), int(rq[j]), int(tq[j]), int(ts), tgt[j])
                    if k in want:
                        sample_preds.append({"h":int(hq[j]),"r":int(rq[j]),"t":int(tq[j]),"ts":int(ts),
                            "target":tgt[j],"candidates":topi_c[j].tolist(),
                            "raw_rank":int(rk_raw[j].item()),"filtered_rank":int(rk_filt[j].item())})
            hist_edges.append(make_edges(triples, num_rel, device)); gh.update(triples, num_rel)
        return metrics_from_ranks(raw_ranks), metrics_from_ranks(filt_ranks), sample_preds

    log = []; t0 = time.time()
    for ep in range(epochs):
        l = run_epoch()
        if (ep+1) % 5 == 0 or ep == epochs-1:
            vm_raw, vm_filt, _ = evaluate_and_emit(va_groups, seed_extra=None, emit=False)
            log.append({"epoch":ep+1,"loss":l,"valid_raw_MRR":vm_raw["MRR"],"valid_filt_MRR":vm_filt["MRR"]})
            print(f"[{os.path.basename(data_dir)}] ep {ep+1} loss {l:.4f} valid raw/filt {vm_raw['MRR']:.4f}/{vm_filt['MRR']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    sample_keys = None
    if sample_keys_file and os.path.exists(f"{sample_keys_file}"):
        sk = json.load(open(f"{sample_keys_file}"))
        sample_keys = set((k["h"], k["r"], k["t"], k["ts"], "tail") for k in sk)
        print(f"loaded {len(sample_keys)} sample keys", flush=True)

    tm_raw, tm_filt, sample_preds = evaluate_and_emit(
        te_groups, seed_extra=[g[1] for g in va_groups], emit=True, sample_keys=sample_keys)
    print(f"[{os.path.basename(data_dir)}] TEST raw {tm_raw} | filt {tm_filt} | sample {len(sample_preds)}", flush=True)
    with open(f"{out_prefix}_sample_preds.json","w") as f: json.dump(sample_preds, f)
    prior_gate_val = float(model.prior_gate.detach().cpu().item()) if getattr(model, "use_prior", False) else None
    with open(f"{out_prefix}_metrics.json","w") as f:
        json.dump({"full_test_raw":tm_raw,"full_test_time_aware":tm_filt,"train_log":log,
                   "num_ent":num_ent,"num_rel":num_rel,
                   "ecckg_prior":bool(rel_prior_file),"prior_gate":prior_gate_val,
                   "config":{"h_dim":h_dim,"n_layers":n_layers,"epochs":epochs,"lr":lr,
                             "history_len":history_len,"dropout":dropout,"cl_weight":cl_weight}}, f, indent=2)
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
    ap.add_argument("--cl_weight", type=float, default=0.1)
    ap.add_argument("--emit_topk", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample_keys_file", default=None)
    ap.add_argument("--rel_prior_file", default=None)
    a = ap.parse_args()
    train_and_eval(a.data_dir, a.out_prefix, a.h_dim, a.n_layers, a.epochs, a.lr,
                   a.history_len, a.dropout, a.device, a.emit_topk, a.cl_weight,
                   sample_keys_file=a.sample_keys_file, rel_prior_file=a.rel_prior_file)
