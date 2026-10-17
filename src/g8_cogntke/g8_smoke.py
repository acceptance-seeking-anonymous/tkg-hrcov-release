# G8 smoke + cost estimate for CognTKE on tkgl-polecat. Run with cwd = CognTKE repo root (commit 4f30966).
# Measures: data load, per-batch train time vs timestamp, eval time (authors' path, batch 128) and
# dump path (no_grad+autocast, GPU log-softmax) at several batch sizes, peak GPU memory, reached-entity
# statistics (authors' 'cover': gold score != 0). Weights are random / a few smoke steps only.
import os, sys, time, json, argparse, resource, math, traceback
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--n_layer', type=int, default=4)
ap.add_argument('--shim', default='')
ap.add_argument('--nb_train', type=int, default=8)
ap.add_argument('--nb_eval', type=int, default=6)
ap.add_argument('--time_dim', type=int, default=32)
ap.add_argument('--skip_authors_eval', type=int, default=0)
ap.add_argument('--bss', default='128,512,1024')
a = ap.parse_args()
res = {'n_layer': a.n_layer, 'time_dim': a.time_dim}
try:
    import torch_scatter
    res['torch_scatter'] = getattr(torch_scatter, '__version__', '?') + ' @ ' + os.path.dirname(torch_scatter.__file__)
except Exception as e:
    res['torch_scatter_import_error'] = repr(e)[:300]
    sys.path = [p for p in sys.path if 'pylib_ts' not in p]
    for k in list(sys.modules):
        if k.startswith('torch_scatter'): del sys.modules[k]
    sys.path.insert(0, a.shim)
    import torch_scatter
    res['torch_scatter'] = torch_scatter.__version__ + ' @ ' + os.path.dirname(torch_scatter.__file__)
import torch
sys.path.insert(0, os.getcwd())
import utils, base_model
from utils import EnhancedDict
from models.TRED_GNN import TRED_GNN20
base_model.load_model = lambda name: TRED_GNN20   # released load_model looks for class 'TRED_GNN' which the module does not define
torch.manual_seed(0); np.random.seed(0)
opts = EnhancedDict(dict(n_layer=a.n_layer, dataset='POLECAT', batch_size=128, model_name='TRED_GNN', window_size=15,
    hidden_dim=64, max_global_window_size=5000, epochs=20, gpu=0, tag='g8smoke', train_mode='half', lr=0.001,
    attention_dim=5, act='idd', lamb=0.00012, dropout=0.25, time_dim=a.time_dim, single_timestamp_layer_numbers=2,
    gnn_mode='unique', time_mode='embedding'))
opts.path = './data/POLECAT/'; opts.disable_bar = True; opts.tag = f"L{opts.n_layer}" + opts.tag
utils.gpu_setting(0)
res['gpu'] = torch.cuda.get_device_name(0); res['torch'] = torch.__version__
t0 = time.time(); tr = base_model.HalfTrainer(opts); res['t_load_s'] = time.time() - t0
D = tr.data
res['time_length'] = [D.time_length_train, D.time_length_valid, D.time_length_test, D.time_length]
res['n_queries'] = [int(sum(D.data_splited[t].shape[0] for t in range(0, D.time_length_train))),
                    int(sum(D.data_splited[t].shape[0] for t in range(D.time_length_train, D.time_length_train + D.time_length_valid))),
                    int(sum(D.data_splited[t].shape[0] for t in range(D.time_length_train + D.time_length_valid, D.time_length)))]
res['time2id_identity'] = bool(all(D.time2id[k] == int(k) for k in D.time2id))
res['num_entity'] = D.num_entity; res['num_relation'] = D.num_relation
nq = np.array([D.data_splited[t].shape[0] for t in range(D.time_length)])
res['batches_per_split_b128'] = [int(np.ceil(nq[a.n_layer:D.time_length_train] / 128).sum()),
                                 int(np.ceil(nq[D.time_length_train:D.time_length_train + D.time_length_valid] / 128).sum()),
                                 int(np.ceil(nq[D.time_length_train + D.time_length_valid:] / 128).sum())]
# retrieval timer
_orig = D.get_neighbors_in_period
acc = {'ret': 0.0}
def timed(nodes, ts, step):
    s = time.time(); o = _orig(nodes, ts, step); acc['ret'] += time.time() - s; return o
D.get_neighbors_in_period = timed
M = tr.tred_gnn
params = list(M.parameters())
def sync(): torch.cuda.synchronize()
# ---------------- train timing ----------------
Ttr = D.time_length_train
train_ts = sorted(set([a.n_layer, 150, 300, 600, 900, Ttr - 1]))
res['train'] = []
for ts in train_ts:
    n = nq[ts]; nb = min(a.nb_train, int(math.ceil(n / 128))); tt = []; rr = []; cov = []; ents = []
    torch.cuda.reset_peak_memory_stats()
    try:
        M.train()
        for i in range(nb):
            idx = range(i * 128, min((i + 1) * 128, n))
            db = torch.tensor(D.get_batch(ts, idx)).cuda()
            sync(); acc['ret'] = 0.0; s = time.time()
            tr.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                scores = M(ts, db[:, 0], db[:, 1])
                loss, vn, tn = tr.cal_loss(db, scores)
            tr.scalar.scale(loss).backward(); tr.scalar.step(optimizer=tr.optimizer); tr.scalar.update()
            for para in params:
                pd_ = para.data.clone(); fl = pd_ != pd_; pd_[fl] = np.random.random(); para.data.copy_(pd_)
            sync(); tt.append(time.time() - s); rr.append(acc['ret']); cov.append(vn / tn); ents.append(float((scores != 0).sum()) / tn)
        res['train'].append({'ts': ts, 'nq': int(n), 'nb': nb, 'sec_per_batch': float(np.mean(tt[1:] if nb > 2 else tt)),
                             'ret_sec_per_batch': float(np.mean(rr[1:] if nb > 2 else rr)), 'cover': float(np.mean(cov)),
                             'reached_per_query': float(np.mean(ents)), 'peak_gpu_GB': torch.cuda.max_memory_allocated() / 1e9})
    except torch.cuda.OutOfMemoryError as e:
        res['train'].append({'ts': ts, 'OOM': repr(e)[:200], 'peak_gpu_GB': torch.cuda.max_memory_allocated() / 1e9}); torch.cuda.empty_cache()
    print('train', res['train'][-1], flush=True)
# ---------------- eval timing: authors' path (batch 128, CPU rankdata) ----------------
Tv0 = D.time_length_train; Tt0 = D.time_length_train + D.time_length_valid
eval_ts = [Tv0, Tv0 + D.time_length_valid // 2, Tt0 - 1, Tt0, Tt0 + D.time_length_test // 2, D.time_length - 1]
M.eval(); res['eval_authors'] = []
for ts in (eval_ts if not a.skip_authors_eval else []):
    n = nq[ts]; nb = min(a.nb_eval, int(math.ceil(n / 128))); tt = []; rr = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(nb):
        idx = range(i * 128, min((i + 1) * 128, n))
        db = torch.tensor(D.get_batch(ts, idx)).cuda()
        sync(); acc['ret'] = 0.0; s = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            scores = M(ts, db[:, 0], db[:, 1])
        loss, vn, tn = tr.cal_loss(db, scores)
        sc = scores.data.cpu().numpy()
        fi = np.array([(j, o) for j, tri in enumerate(db.tolist()) for o in D.time_filter[ts][(tri[0], tri[1])]]).T
        filt = np.zeros_like(sc); filt[fi[0], fi[1]] = 1
        rk = utils.cal_ranks(sc, db[:, 2].data.cpu().numpy(), filt)
        sync(); tt.append(time.time() - s); rr.append(acc['ret'])
    res['eval_authors'].append({'ts': ts, 'nq': int(n), 'nb': nb, 'sec_per_batch': float(np.mean(tt[1:])), 'ret_sec_per_batch': float(np.mean(rr[1:])),
                                'peak_gpu_GB': torch.cuda.max_memory_allocated() / 1e9})
    print('eval_authors', res['eval_authors'][-1], flush=True)
# ---------------- dump path: larger batches, GPU log-softmax, reached stats ----------------
res['dump'] = []
for bs in [int(x) for x in a.bss.split(',')]:
    for ts in [Tv0, Tt0, Tt0 + D.time_length_test // 2, D.time_length - 1]:
        n = nq[ts]; nb = min(3 if bs < 1024 else 2, int(math.ceil(n / bs))); tt = []; rr = []
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(nb):
                idx = range(i * bs, min((i + 1) * bs, n))
                db = torch.tensor(D.get_batch(ts, idx)).cuda()
                sync(); acc['ret'] = 0.0; s = time.time()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    scores = M(ts, db[:, 0], db[:, 1])
                lp = torch.log_softmax(scores.float(), dim=1)
                nz = (scores != 0); nnz = nz.sum(1)
                top = torch.topk(lp, 200, dim=1)
                gold_reached = nz[torch.arange(len(db)), db[:, 2]].float().mean().item()
                sync(); tt.append(time.time() - s); rr.append(acc['ret'])
            rec = {'bs': bs, 'ts': ts, 'nb': nb, 'sec_per_query': float(np.sum(tt) / sum(min(bs, n - i * bs) for i in range(nb))),
                   'ret_share': float(np.sum(rr) / np.sum(tt)), 'peak_gpu_GB': torch.cuda.max_memory_allocated() / 1e9,
                   'nnz_mean': float(nnz.float().mean()), 'nnz_max': int(nnz.max()), 'gold_reached_last_batch': gold_reached}
            if bs == 128 and i == 0 and nb == 1: pass
        except torch.cuda.OutOfMemoryError as e:
            rec = {'bs': bs, 'ts': ts, 'OOM': repr(e)[:200]}; torch.cuda.empty_cache()
        res['dump'].append(rec); print('dump', rec, flush=True)
# batch-size invariance: first 128 queries of a test ts at bs 128 vs inside a 2048 batch
ts = Tt0; n = nq[ts]
db = torch.tensor(D.get_batch(ts, range(0, min(512, n)))).cuda()
with torch.no_grad(), torch.cuda.amp.autocast():
    s_small = M(ts, db[:128, 0], db[:128, 1]).float(); s_big = M(ts, db[:, 0], db[:, 1])[:128].float()
res['bs_invariance_max_abs_logit_diff'] = float((s_small - s_big).abs().max())
res['bs_invariance_same_support'] = bool(((s_small != 0) == (s_big != 0)).all())
# gold reached share over full test timestamps (structural coverage; weights irrelevant for support)
gr = []; nnzs = []
for ts in [Tt0, Tt0 + 100, Tt0 + 200, D.time_length - 1]:
    n = nq[ts]
    for i in range(int(math.ceil(n / 512))):
        db = torch.tensor(D.get_batch(ts, range(i * 512, min((i + 1) * 512, n)))).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            sc = M(ts, db[:, 0], db[:, 1])
        nz = sc != 0
        gr.append(nz[torch.arange(len(db)), db[:, 2]].cpu().numpy()); nnzs.append(nz.sum(1).cpu().numpy())
gr = np.concatenate(gr); nnzs = np.concatenate(nnzs)
res['test_slice_gold_reached_share'] = float(gr.mean()); res['test_slice_n'] = int(len(gr))
res['test_slice_nnz'] = {'mean': float(nnzs.mean()), 'median': float(np.median(nnzs)), 'p95': float(np.percentile(nnzs, 95)), 'max': int(nnzs.max())}
res['host_maxrss_GB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
res['nq_per_ts'] = nq.tolist()
json.dump(res, open(a.out, 'w'), indent=1)
print('DONE', json.dumps({k: v for k, v in res.items() if k != 'nq_per_ts'})[:3000])
