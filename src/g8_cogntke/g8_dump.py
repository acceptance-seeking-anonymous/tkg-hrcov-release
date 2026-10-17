"""G8 / g8_dump.py -- score the valid-selected CognTKE checkpoint on tkgl-polecat in HARNESS query order and write
 (1) harness BaseDump npz (pc_common schema): train (from the 2nd snapshot) / valid / test rows; sro, gold log p, raw rank
     (1 + #higher), strict filtered rank (as hd_tirgn.py), top-K ids / log p (stable sort: ties in entity-id order);
 (2) exact sparse rows for valid and test: per query the reached entities (score != 0) with their fp16 logits and the fp32
     log-sum-exp of the FULL row (unreached entities at logit 0, released convention) -> log p = logit - lse exactly as here.
Row convention: x = scores.float() (fp16 autocast forward, released), lse = logsumexp(x), lp = x - lse.
  python3 g8_dump.py --state best_state.pt --out D [--splits train,valid,test] [--bs 512]
"""
import os, sys, time, json, argparse, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g8_common as C
ap = argparse.ArgumentParser()
ap.add_argument('--state', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--splits', default='train,valid,test'); ap.add_argument('--bs', type=int, default=512)
ap.add_argument('--topk', type=int, default=200); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--smoke', type=int, default=0)
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:9.1f}s]", *x, flush=True)
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
tr, utils, BM = C.setup(C.README_CFG, os.getcwd())
D = tr.data; M = tr.tred_gnn; opts = tr.opts
M.load_state_dict(torch.load(a.state, map_location='cuda')); M.eval()
steps = [opts.max_global_window_size] + [opts.window_size] * (opts.n_layer - 1)
pf = C.Prefetcher(D, steps, workers=a.workers, depth=8); feed = C.Feed(D); D.get_neighbors_in_period = feed
N, R = D.num_entity, D.num_relation
def load_q(sp):
    q = np.loadtxt(f"{opts.path}/{sp}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)
    o = np.argsort(q[:, 3], kind='stable'); q = q[o]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q))
    out = []
    for u, b, e in zip(uts, st, en):
        trip = q[b:e]; inv = trip[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R
        out.append((int(u), np.concatenate([trip, inv]).astype(np.int64)))
    return out
K = a.topk; splits = a.splits.split(','); code = {'train': 0, 'valid': 1, 'test': 2}
BD = {k: [] for k in ('sro', 'gold_score', 'gold_rank_raw', 'gold_rank_filt', 'topk_ids', 'topk_scores', 'split')}
summ = dict(state=a.state, cfg=dict(opts), K=K, bs=a.bs, splits={})
for sp in ('train', 'valid', 'test'):
    if sp not in splits: continue
    snaps = load_q(sp)
    if sp == 'train': snaps = snaps[1:]
    if a.smoke: snaps = snaps[:a.smoke]
    jobs = [(ts, sro[b0:b0 + a.bs]) for ts, sro in snaps for b0 in range(0, len(sro), a.bs)]
    sparse = sp in ('valid', 'test')
    ids_l, val_l, nnz_l, lse_l = [], [], [], []
    ga, gt, gs, reach = [], [], [], []
    t1 = time.time()
    for k, ((ts, sro), item) in enumerate(zip(jobs, pf.run([(ts, s[:, 0]) for ts, s in jobs]))):
        feed.set(item)
        s_t = torch.as_tensor(sro[:, 0]).cuda(); r_t = torch.as_tensor(sro[:, 1]).cuda(); gold = torch.as_tensor(sro[:, 2]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            sc = M(ts, s_t, r_t)
        assert feed.done()
        with torch.no_grad():
            x = sc.float(); lse = torch.logsumexp(x, 1); lp = x - lse.unsqueeze(1)
            B = len(sro); ar = torch.arange(B, device='cuda')
            glp = lp[ar, gold]
            fr, fc = C.filter_pairs(D, ts, sro[:, :2].tolist())
            Fm = torch.zeros((B, N), dtype=torch.bool, device='cuda'); Fm[fr, fc] = True; Fm[ar, gold] = False
            rk_raw = (lp > glp.unsqueeze(1)).sum(1) + 1
            rk_f = ((lp > glp.unsqueeze(1)) & ~Fm).sum(1) + 1
            sv, si = torch.sort(lp, dim=1, descending=True, stable=True)
            BD['sro'].append(sro.astype(np.int32)); BD['gold_score'].append(glp.cpu().numpy()); BD['gold_rank_raw'].append(rk_raw.cpu().numpy().astype(np.int32))
            BD['gold_rank_filt'].append(rk_f.cpu().numpy().astype(np.int32)); BD['topk_ids'].append(si[:, :K].cpu().numpy().astype(np.int32))
            BD['topk_scores'].append(sv[:, :K].cpu().numpy()); BD['split'].append(np.full(B, code[sp], np.int8))
            nzm = sc != 0
            reach.append(nzm[ar, gold].cpu().numpy())
            if sparse:
                rr, cc = torch.nonzero(nzm, as_tuple=True)
                ids_l.append(cc.int().cpu().numpy()); val_l.append(sc[rr, cc].cpu().numpy()); nnz_l.append(nzm.sum(1).cpu().numpy()); lse_l.append(lse.cpu().numpy())
                ga.append(C.authors_ranks_gpu(sc, gold, fr, fc).cpu().numpy()); gt.append(C.tieavg_ranks_gpu(sc, gold, fr, fc).cpu().numpy())
            gs.append(rk_f.cpu().numpy())
            if a.smoke and k == 0 and sparse:
                np.save(f"{a.out}/smoke_lp_{sp}.npy", lp[:64].cpu().numpy())
        if k % 500 == 0: log(sp, k, '/', len(jobs), 'ts', ts, f"{time.time()-t1:.0f}s")
    reach = np.concatenate(reach); rkf = np.concatenate(gs).astype(np.float64)
    ss = dict(n=int(len(reach)), seconds=round(time.time() - t1, 1), gold_reached_share=float(reach.mean()),
              strict_filt=C.metrics(rkf))
    if sparse:
        nnz = np.concatenate(nnz_l).astype(np.int64); ptr = np.zeros(len(nnz) + 1, np.int64); ptr[1:] = np.cumsum(nnz)
        np.save(f"{a.out}/rows_{sp}_ptr.npy", ptr); np.save(f"{a.out}/rows_{sp}_ids.npy", np.concatenate(ids_l))
        np.save(f"{a.out}/rows_{sp}_val16.npy", np.concatenate(val_l)); np.save(f"{a.out}/rows_{sp}_lse.npy", np.concatenate(lse_l))
        np.save(f"{a.out}/rows_{sp}_sro.npy", np.concatenate([s for _, s in jobs]).astype(np.int32))
        ss.update(authors_metric=C.metrics(np.concatenate(ga)), tieavg_filt=C.metrics(np.concatenate(gt)),
                  nnz=dict(mean=float(nnz.mean()), median=float(np.median(nnz)), max=int(nnz.max())))
        np.save(f"{a.out}/reach_{sp}.npy", reach)
    summ['splits'][sp] = ss; log('split done', sp, json.dumps({k: v for k, v in ss.items()}))
    json.dump(summ, open(f"{a.out}/dump_summary.json", 'w'), indent=1)
cat = {k: np.concatenate(v) for k, v in BD.items()}
spl = cat['split']; bounds = np.array([int((spl == 0).sum()), int((spl <= 1).sum()), int(len(spl))], np.int64)
np.savez(f"{a.out}/cogn_scores_POLECAT_traintest.npz", split_bounds=bounds, **cat)
summ['total_s'] = round(time.time() - T0); summ['rows'] = dict(train=int((spl == 0).sum()), valid=int((spl == 1).sum()), test=int((spl == 2).sum()))
json.dump(summ, open(f"{a.out}/dump_summary.json", 'w'), indent=1)
pf.close(); os.system(f"chmod -R a+rX {a.out}")
log('DUMP DONE', json.dumps(summ['rows']))
