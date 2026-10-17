"""G8 / g8_verify.py -- pre-training identity checks for option B (g8_deviation_B.json), README configuration.
(1) prefetched subgraphs (worker CPU half + released GPU half) == released Dataloader.get_neighbors_in_period
    (torch.equal of tail_nodes, sampled_edges, old_nodes_new_idx for every layer) on sample batches of train/valid/test;
(2) GPU ranker == released utils.cal_ranks (CPU) per query on a valid slice (3 valid timestamps, all queries, batch 128);
(3) timing of the training step with and without prefetch.  Weights: README init + 40 smoke steps (no CognTKE MRR).
"""
import os, sys, time, json, argparse, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g8_common as C
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--skip_sub', type=int, default=0)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
tr, utils, BM = C.setup(C.README_CFG, os.getcwd())
D = tr.data; M = tr.tred_gnn; opts = tr.opts
steps = [opts.max_global_window_size] + [opts.window_size] * (opts.n_layer - 1)
orig = D.get_neighbors_in_period
nq = np.array([D.data_splited[t].shape[0] for t in range(D.time_length)])
Ttr, Tv = D.time_length_train, D.time_length_valid; Tt0 = Ttr + Tv
res = {'cfg': dict(opts), 'steps': steps}
pf = C.Prefetcher(D, steps, workers=a.workers, depth=8)
feed = C.Feed(D)
params = list(M.parameters())


def train_step(ts, idx):
    db = torch.tensor(D.get_batch(ts, idx)).cuda()
    tr.optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        scores = M(ts, db[:, 0], db[:, 1]); loss, vn, tn = tr.cal_loss(db, scores)
    tr.scalar.scale(loss).backward(); tr.scalar.step(optimizer=tr.optimizer); tr.scalar.update()
    for para in params:
        pd_ = para.data.clone(); fl = pd_ != pd_; pd_[fl] = np.random.random(); para.data.copy_(pd_)


# ---- 40 smoke steps (released path) so scores are not at init
import copy
init_state = copy.deepcopy(M.state_dict())
M.train()
for i in range(40):
    ts = 600 + i // 10; train_step(ts, range((i % 10) * 128, min((i % 10 + 1) * 128, nq[ts])))
torch.cuda.synchronize()
# ---- (1) subgraph identity
rec = []
def recording(nodes, ts, step):
    o = orig(nodes, ts, step); rec.append(o); return o
samples = [(t, bs, b) for t in (3, 150, 600, 1192) for bs, b in ((128, 0), (128, 1))] + \
          [(t, bs, 0) for t in (Ttr, Ttr + Tv // 2, Tt0 - 1) for bs in (128, 512)] + [(t, 512, 0) for t in (Tt0, Tt0 + 150, D.time_length - 1)]
M.eval(); sub = []; all_ok = True
for ts, bs, b in (samples if not a.skip_sub else []):
    idx = range(b * bs, min((b + 1) * bs, nq[ts])); db = torch.tensor(D.get_batch(ts, idx)).cuda()
    rec.clear(); D.get_neighbors_in_period = recording
    with torch.no_grad(), torch.cuda.amp.autocast():
        s1 = M(ts, db[:, 0], db[:, 1]).float()
    R1 = list(rec)
    item = next(pf.run([(ts, D.data_splited[ts][idx][:, 0])]))
    rec.clear()
    def rec_feed(nodes, t_, st_):
        o = feed(nodes, t_, st_); rec.append(o); return o
    feed.set(item); D.get_neighbors_in_period = rec_feed
    with torch.no_grad(), torch.cuda.amp.autocast():
        s2 = M(ts, db[:, 0], db[:, 1]).float()
    R2 = list(rec); D.get_neighbors_in_period = orig
    eq = [all(torch.equal(x, y) for x, y in zip(r1, r2)) for r1, r2 in zip(R1, R2)]
    ok = len(R1) == len(R2) == opts.n_layer and all(eq) and feed.done()
    all_ok &= ok
    sub.append(dict(ts=int(ts), bs=bs, batch=b, n_q=len(idx), layers_equal=eq, n_edges=[int(r[1].shape[0]) for r in R1],
                    n_nodes=[int(r[0].shape[0]) for r in R1], scores_support_equal=bool(((s1 != 0) == (s2 != 0)).all()),
                    scores_max_abs_diff=float((s1 - s2).abs().max()), scores_frac_equal=float((s1 == s2).float().mean())))
    print('sub', sub[-1], flush=True)
res['subgraph_identity'] = dict(all_equal=bool(all_ok), samples=sub)
# ---- (2) ranker identity on a valid slice (released evaluate_base inner loop, batch 128). cal_ranks returns, for a row whose
# shifted / filtered scores contain NaN, all N entries of that row (NaN); the GPU ranker returns one NaN -> expand before comparing.
def ranker_check(tag):
    out = dict(n_rows=0, n_mismatch=0, n_nan_rows_gpu=0, n_batches=0, n_batches_len_mismatch=0, max_abs_score=0.0, n_ties_broken=0)
    tc = tg = 0.0
    for ts in (Ttr, Ttr + Tv // 2, Tt0 - 1):
        n = nq[ts]
        for i in range(int(math.ceil(n / 128))):
            idx = range(i * 128, min((i + 1) * 128, n)); db = torch.tensor(D.get_batch(ts, idx)).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                scores = M(ts, db[:, 0], db[:, 1])
            out['max_abs_score'] = max(out['max_abs_score'], float(scores.float().abs().max()))
            t0 = time.time()
            sc = scores.data.cpu().numpy()
            fi = [(j, o) for j, tri in enumerate(db.tolist()) for o in D.time_filter[ts][(tri[0], tri[1])]]
            fi = np.array(fi).T; filt = np.zeros_like(sc); filt[fi[0], fi[1]] = 1
            rc = np.asarray(utils.cal_ranks(sc, db[:, 2].data.cpu().numpy(), filt), np.float64); tc += time.time() - t0
            t0 = time.time()
            fr, fc = C.filter_pairs(D, ts, db.tolist())
            rg = C.authors_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy()
            rt = C.tieavg_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy(); tg += time.time() - t0
            exp = np.concatenate([np.full(sc.shape[1], np.nan) if np.isnan(v) else np.array([v]) for v in rg])
            out['n_batches'] += 1; out['n_rows'] += len(rg); out['n_nan_rows_gpu'] += int(np.isnan(rg).sum())
            ok_ = ~np.isnan(rg); out['n_ties_broken'] += int((rg[ok_] != rt[ok_]).sum())
            if len(exp) != len(rc):
                out['n_batches_len_mismatch'] += 1; out['n_mismatch'] += len(rg); continue
            same = (exp == rc) | (np.isnan(exp) & np.isnan(rc))
            out['n_mismatch'] += int((~same).sum())
    out['identical'] = bool(out['n_mismatch'] == 0 and out['n_batches_len_mismatch'] == 0)
    out['n_rows_compared_finite'] = out['n_rows'] - out['n_nan_rows_gpu']
    out['sec_cpu_ranker'] = tc; out['sec_gpu_ranker'] = tg
    print('ranker', tag, out, flush=True)
    return out
M.eval()
r_smoke = ranker_check('after_40_steps')
cur = copy.deepcopy(M.state_dict()); M.load_state_dict(init_state); r_init = ranker_check('init'); M.load_state_dict(cur)
res['ranker_identity'] = dict(after_40_steps=r_smoke, init=r_init,
                              identical=bool(r_smoke['identical'] and r_init['identical']),
                              n_finite_rows_compared=r_smoke['n_rows_compared_finite'] + r_init['n_rows_compared_finite'])
# ---- (3) timing: released vs prefetched training step (40 batches each at ts 600..603 and 1188..1191)
M.train(); tim = {}
for mode in ('released', 'prefetch'):
    jobs = [(ts, b) for ts in (600, 601, 1190, 1191) for b in range(10) if b * 128 < nq[ts]]
    torch.cuda.synchronize(); t0 = time.time()
    if mode == 'released':
        D.get_neighbors_in_period = orig
        for ts, b in jobs: train_step(ts, range(b * 128, min((b + 1) * 128, nq[ts])))
    else:
        D.get_neighbors_in_period = feed
        gen = pf.run([(ts, D.data_splited[ts][b * 128:min((b + 1) * 128, nq[ts]), 0]) for ts, b in jobs])
        for (ts, b), item in zip(jobs, gen):
            feed.set(item); train_step(ts, range(b * 128, min((b + 1) * 128, nq[ts]))); assert feed.done()
        D.get_neighbors_in_period = orig
    torch.cuda.synchronize(); tim[mode] = (time.time() - t0) / len(jobs)
res['train_sec_per_batch'] = tim
res['peak_gpu_GB'] = torch.cuda.max_memory_allocated() / 1e9
res['PASS_ranker'] = bool(res['ranker_identity']['identical'] and res['ranker_identity']['n_finite_rows_compared'] >= 2000)
pf.close()
json.dump(res, open(a.out, 'w'), indent=1, default=str)
print('DONE PASS_ranker', res['PASS_ranker'], json.dumps(tim))
