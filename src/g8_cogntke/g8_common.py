"""G8 / g8_common.py -- CognTKE (commit 4f30966) helpers for tkgl-polecat, option B (g8_deviation_B.json).
* setup(): released HalfTrainer with the README configuration (loader bound to TRED_GNN20).
* cpu_part(): verbatim CPU half of the released Dataloader.get_neighbors_in_period; chain() runs it for all layers
  of one batch in a forked worker (next-layer nodes = lexicographic unique rows, == torch.unique(dim=0, sorted=True)).
* gpu_part(): verbatim GPU half of the released function. Feed: replaces Dataloader.get_neighbors_in_period by
  gpu_part(prefetched array) and checks the node count of every layer against the worker's.
* authors_ranks_gpu(): GPU emulation of the released utils.cal_ranks (fp16 shift, ordinal ties by index, filter).
"""
import os, sys, time, math, json
from collections import deque
import numpy as np
import torch
from scipy.sparse import csr_matrix
import multiprocessing as mp

README_CFG = dict(n_layer=3, dataset='POLECAT', batch_size=128, model_name='TRED_GNN', window_size=15, hidden_dim=64,
                  max_global_window_size=5000, epochs=20, gpu=0, tag='g8', train_mode='half', lr=0.005, attention_dim=5,
                  act='idd', lamb=0.00012, dropout=0.25, time_dim=16, single_timestamp_layer_numbers=2, gnn_mode='unique',
                  time_mode='embedding')


def setup(cfg, repo):
    sys.path.insert(0, repo)
    import utils, base_model
    from utils import EnhancedDict
    from models.TRED_GNN import TRED_GNN20
    base_model.load_model = lambda name: TRED_GNN20      # released load_model('TRED_GNN') finds no such class
    opts = EnhancedDict(dict(cfg)); opts.path = './data/POLECAT/'; opts.disable_bar = True
    opts.tag = f"L{opts.n_layer}" + opts.tag
    utils.gpu_setting(0)
    tr = base_model.HalfTrainer(opts)
    return tr, utils, base_model


def cpu_part(D, nodes, target_timestamp, step):
    step = min(step, target_timestamp)
    start_time_stamp = max(target_timestamp - step, 0)
    node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(D.num_entity, nodes.shape[0]))
    edge_list = []
    for i in range(step):
        edge_1hot = D.fact_sub_matrix[start_time_stamp + i].dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        timestamp = np.ones((edges[1].size, 1)) * (step - i)
        edge_list.append(np.concatenate([np.expand_dims(edges[1], 1), D.graph_extended[start_time_stamp + i][edges[0]], timestamp], axis=1))
    idd = np.vstack([nodes[:, 0], nodes[:, 1], np.ones(shape=(nodes.shape[0])) * (D.num_relation_extended - 1),
                     nodes[:, 1], np.zeros(nodes[:, 1].shape)]).T
    edge_list.append(idd)
    return np.concatenate(edge_list, axis=0)


_D = None


def chain(args):
    ts, s_arr, steps = args
    D = _D
    nodes = np.stack([np.arange(len(s_arr)), np.asarray(s_arr)], 1).astype(np.int64)
    arrs, counts = [], [len(nodes)]
    for li, st in enumerate(steps):
        a = cpu_part(D, nodes, ts, st).astype(np.int64)      # == torch.LongTensor(float64 array) for integral values
        assert a.max() < 2 ** 31 and a.min() >= 0
        arrs.append(a.astype(np.int32))
        if li + 1 < len(steps):
            nodes = np.unique(a[:, [0, 3]], axis=0); counts.append(len(nodes))
    return arrs, counts


def gpu_part(D, arr):
    sampled_edges = torch.from_numpy(arr).cuda().long()
    head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
    tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
    mask = sampled_edges[:, 2] == (D.num_relation * 2)
    _, old_idx = head_index[mask].sort()
    old_nodes_new_idx = tail_index[mask][old_idx]
    sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
    sampled_edges[:, [4, 5, 6]] = sampled_edges[:, [5, 6, 4]]
    sampled_edges = sampled_edges.contiguous()
    return tail_nodes, sampled_edges, old_nodes_new_idx


class Feed:
    def __init__(self, D):
        self.D = D; self.cur = None; self.counts = None; self.layer = 0
    def set(self, item):
        self.cur = deque(item[0]); self.counts = item[1]; self.layer = 0
    def __call__(self, nodes, target_timestamp, step):
        assert nodes.shape[0] == self.counts[self.layer], (nodes.shape, self.counts, self.layer)
        self.layer += 1
        return gpu_part(self.D, self.cur.popleft())
    def done(self):
        return len(self.cur) == 0


class Prefetcher:
    def __init__(self, D, steps, workers=4, depth=8):
        global _D
        _D = D
        self.steps = steps; self.depth = depth
        self.pool = mp.get_context('fork').Pool(workers)
    def run(self, jobs):
        jobs = list(jobs); q = deque(); nxt = 0
        while nxt < len(jobs) and len(q) < self.depth:
            ts, s = jobs[nxt]; q.append(self.pool.apply_async(chain, ((ts, s, self.steps),))); nxt += 1
        while q:
            r = q.popleft().get()
            if nxt < len(jobs):
                ts, s = jobs[nxt]; q.append(self.pool.apply_async(chain, ((ts, s, self.steps),))); nxt += 1
            yield r
    def close(self):
        self.pool.terminate()


_EPS16 = None


def authors_ranks_gpu(scores16, labels, fr, fc):
    """released utils.cal_ranks on the GPU: s = scores - min + 1e-6 in fp16 (numpy half semantics: 1e-6 cast to fp16),
    rank = 1 + #(non-filtered e: s_e > s_g) + #(non-filtered e: s_e == s_g and e < g); NaN rows -> NaN."""
    global _EPS16
    if _EPS16 is None or _EPS16.device != scores16.device:
        _EPS16 = torch.tensor(1e-6, dtype=torch.float16, device=scores16.device)
    assert scores16.dtype == torch.float16
    B, N = scores16.shape
    sh = (scores16 - scores16.min(dim=1, keepdim=True)[0]) + _EPS16
    F = torch.zeros((B, N), dtype=torch.bool, device=sh.device); F[fr, fc] = True
    ar = torch.arange(B, device=sh.device)
    g = sh[ar, labels].unsqueeze(1)
    idx = torch.arange(N, device=sh.device).unsqueeze(0)
    above = ((sh > g) | ((sh == g) & (idx < labels.unsqueeze(1)))) & ~F
    rank = 1.0 + above.sum(1).double()
    nanrow = torch.isnan(sh).any(1) | (torch.isinf(sh) & ~F).any(1)
    rank[nanrow] = float('nan')
    return rank


def tieavg_ranks_gpu(scores16, labels, fr, fc):
    """harness / TGB convention on the model's own distribution: log_softmax(fp32 row); rank = 1 + #greater + 0.5 #ties
    over non-filtered entities (the gold is in the filter)."""
    lp = torch.log_softmax(scores16.float(), dim=1)
    B, N = lp.shape
    F = torch.zeros((B, N), dtype=torch.bool, device=lp.device); F[fr, fc] = True
    ar = torch.arange(B, device=lp.device); g = lp[ar, labels].unsqueeze(1)
    return 1.0 + ((lp > g) & ~F).sum(1).double() + 0.5 * ((lp == g) & ~F).sum(1).double()


def filter_pairs(D, ts, db_list):
    fi = [(j, o) for j, tri in enumerate(db_list) for o in D.time_filter[ts][(tri[0], tri[1])]]
    fi = np.array(fi, dtype=np.int64).T
    return torch.as_tensor(fi[0]).cuda(), torch.as_tensor(fi[1]).cuda()


def metrics(rk):
    rk = np.asarray(rk, np.float64)
    return dict(MRR=float(np.mean(1.0 / rk)), H1=float(np.mean(rk <= 1)), H3=float(np.mean(rk <= 3)),
                H10=float(np.mean(rk <= 10)), H100=float(np.mean(rk <= 100)), n=int(len(rk)), n_nan=int(np.isnan(rk).sum()))
