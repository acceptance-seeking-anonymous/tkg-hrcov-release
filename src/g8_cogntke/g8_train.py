"""G8 / g8_train.py -- CognTKE (commit 4f30966) on tkgl-polecat, option B of g8_deviation_B.json.
Released HalfTrainer training step verbatim (autocast, GradScaler, NaN reset, timestamps in order, batch 128), README
configuration, 20 epochs; subgraph retrieval prefetched (--mode B) or released (--mode A); after every epoch the FULL
valid set is evaluated with the GPU emulation of the released cal_ranks (selection metric, ordinal ties) plus the
tie-averaged rank on log_softmax (information only); no test evaluation; valid-best state_dict kept (released best_weight
logic: replace iff best < v_mrr); resumable from ckpt_last.pt (model, optimizer, scaler, RNG, log).
  python3 g8_train.py --run_dir R [--mode B] [--epochs 20] [--smoke 0]
"""
import os, sys, time, json, argparse, math, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g8_common as C
ap = argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True); ap.add_argument('--mode', default='B', choices=['A', 'B'])
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--depth', type=int, default=8); ap.add_argument('--eval_bs', type=int, default=512)
ap.add_argument('--smoke', type=int, default=0, help='>0: only this many train / valid batches per epoch')
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:9.1f}s]", *x, flush=True)
os.makedirs(a.run_dir, exist_ok=True)
cfg = dict(C.README_CFG); cfg['epochs'] = a.epochs
torch.manual_seed(0); np.random.seed(0); random.seed(0)          # the released main.py sets no seed
tr, utils, BM = C.setup(cfg, os.getcwd())
D = tr.data; M = tr.tred_gnn; opts = tr.opts
steps = [opts.max_global_window_size] + [opts.window_size] * (opts.n_layer - 1)
Ttr, Tv = D.time_length_train, D.time_length_valid
nq = np.array([D.data_splited[t].shape[0] for t in range(D.time_length)])
if a.mode == 'B':
    pf = C.Prefetcher(D, steps, workers=a.workers, depth=a.depth); feed = C.Feed(D); D.get_neighbors_in_period = feed
def blist(t0, t1, bs):
    return [(ts, i * bs, min((i + 1) * bs, nq[ts])) for ts in range(t0, t1) for i in range(int(math.ceil(nq[ts] / bs)))]
train_b = blist(opts.n_layer, Ttr, opts.batch_size); valid_b = blist(Ttr, Ttr + Tv, a.eval_bs)
if a.smoke:
    train_b = train_b[600 * 17:600 * 17 + a.smoke]; valid_b = valid_b[:a.smoke]
def items(bl):
    if a.mode == 'B':
        return pf.run([(ts, D.data_splited[ts][b0:b1, 0]) for ts, b0, b1 in bl])
    return (None for _ in bl)
LOG = dict(cfg=dict(opts), mode=a.mode, steps=steps, n_train_batches=len(train_b), n_valid_batches=len(valid_b),
           host=os.environ.get('G8_HOST', ''), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
           deviation='g8_deviation_B.json', epochs=[], best_epoch=None)
start, best_mrr = 0, 0.0
ck = f"{a.run_dir}/ckpt_last.pt"
if os.path.exists(ck):
    S = torch.load(ck, map_location='cpu', weights_only=False)
    M.load_state_dict(S['model']); tr.optimizer.load_state_dict(S['optimizer']); tr.scalar.load_state_dict(S['scaler'])
    torch.set_rng_state(S['rng_torch']); torch.cuda.set_rng_state(S['rng_cuda']); np.random.set_state(S['rng_np']); random.setstate(S['rng_py'])
    LOG = S['log']; start = S['epoch']; best_mrr = S['best_mrr']; LOG.setdefault('resumed_at', []).append(start)
    log('resumed after epoch', start, 'best', best_mrr)
params = list(M.parameters())
for ep in range(start, a.epochs):
    tr.now_epoch = ep + 1
    M.train(); t0 = time.time(); counter = np.zeros(4)
    for k, ((ts, b0, b1), item) in enumerate(zip(train_b, items(train_b))):
        if item is not None: feed.set(item)
        data_batched = torch.tensor(D.get_batch(ts, range(b0, b1))).cuda()
        tr.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            scores = M(ts, data_batched[:, 0], data_batched[:, 1])
            loss, valid_num, total_num = tr.cal_loss(data_batched, scores)
        tr.scalar.scale(loss).backward()
        tr.scalar.step(optimizer=tr.optimizer)
        tr.scalar.update()
        counter[0] += total_num; counter[1] += valid_num; counter[2] += loss.item(); counter[3] += float((scores != 0).sum())
        for para in params:                                   # avoid NaN (released)
            para_data = para.data.clone(); flag = para_data != para_data
            para_data[flag] = np.random.random(); para.data.copy_(para_data)
        if item is not None: assert feed.done()
        if k % 2000 == 0:
            el = time.time() - t0; log(f"ep {ep+1} batch {k}/{len(train_b)} ts {ts} {el:.0f}s eta_epoch {el / max(k, 1) * len(train_b):.0f}s loss {loss.item() / max(total_num, 1):.4f}")
    c1, c2, c3 = counter[1] / counter[0], counter[2] / counter[0], counter[3] / counter[0]
    t1 = time.time()
    M.eval(); rk_a, rk_t = [], []; vc = np.zeros(3)
    for (ts, b0, b1), item in zip(valid_b, items(valid_b)):
        if item is not None: feed.set(item)
        db = torch.tensor(D.get_batch(ts, range(b0, b1))).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            scores = M(ts, db[:, 0], db[:, 1])
        with torch.no_grad():
            loss, vn, tn = tr.cal_loss(db, scores)
            vc += (tn, vn, loss.item())
            fr, fc = C.filter_pairs(D, ts, db.tolist())
            rk_a.append(C.authors_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy())
            rk_t.append(C.tieavg_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy())
        if item is not None: assert feed.done()
    va = C.metrics(np.concatenate(rk_a)); vt = C.metrics(np.concatenate(rk_t))
    t2 = time.time()
    result = dict(epoch=ep + 1, cover_rate_train=float(c1), loss_train=float(c2), average_entity_num=float(c3),
                  v_mrr=va['MRR'], v_h1=va['H1'], v_h3=va['H3'], v_h10=va['H10'], v_h100=va['H100'], v_cover=float(vc[1] / vc[0]),
                  v_loss=float(vc[2] / vc[0]), v_n=va['n'], v_n_nan=va['n_nan'], v_tieavg=vt, t_mrr=None,
                  time_train=t1 - t0, time_valid=t2 - t1)
    LOG['epochs'].append(result)
    if best_mrr < result['v_mrr']:                           # released: if self.best_mrr < v_mrr -> save best
        best_mrr = result['v_mrr']; LOG['best_epoch'] = ep + 1
        torch.save(M.state_dict(), f"{a.run_dir}/best_state.pt.tmp"); os.replace(f"{a.run_dir}/best_state.pt.tmp", f"{a.run_dir}/best_state.pt")
    S = dict(model=M.state_dict(), optimizer=tr.optimizer.state_dict(), scaler=tr.scalar.state_dict(), rng_torch=torch.get_rng_state(),
             rng_cuda=torch.cuda.get_rng_state(), rng_np=np.random.get_state(), rng_py=random.getstate(), log=LOG, epoch=ep + 1, best_mrr=best_mrr)
    torch.save(S, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    torch.save(M.state_dict(), f"{a.run_dir}/state_ep{ep+1:02d}.pt")
    json.dump(LOG, open(f"{a.run_dir}/epoch_log.json", 'w'), indent=1, default=str)
    os.system(f"chmod -R a+rX {a.run_dir}")
    log('EPOCH', json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in result.items() if k != 'v_tieavg'}), 'tieavg', round(vt['MRR'], 5))
LOG['total_s'] = time.time() - T0
json.dump(LOG, open(f"{a.run_dir}/epoch_log.json", 'w'), indent=1, default=str)
os.system(f"chmod -R a+rX {a.run_dir}")
if a.mode == 'B': pf.close()
log('TRAIN DONE best_epoch', LOG['best_epoch'], 'best v_mrr', best_mrr)
