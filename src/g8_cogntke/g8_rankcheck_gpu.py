"""G8 ranker identity, part 1 (GPU): README model after 40 smoke steps (as g8_verify.py), valid slice (2 valid timestamps,
all queries, batch 128): saves the fp16 score rows, labels, filters and the GPU-emulated ranks for the CPU comparison."""
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g8_common as C
out = sys.argv[1]
torch.manual_seed(0); np.random.seed(0)
tr, utils, BM = C.setup(C.README_CFG, os.getcwd())
D = tr.data; M = tr.tred_gnn
nq = np.array([D.data_splited[t].shape[0] for t in range(D.time_length)])
params = list(M.parameters()); M.train()
for i in range(40):
    ts = 600 + i // 10; idx = range((i % 10) * 128, min((i % 10 + 1) * 128, nq[ts]))
    db = torch.tensor(D.get_batch(ts, idx)).cuda(); tr.optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        scores = M(ts, db[:, 0], db[:, 1]); loss, vn, tn = tr.cal_loss(db, scores)
    tr.scalar.scale(loss).backward(); tr.scalar.step(optimizer=tr.optimizer); tr.scalar.update()
    for para in params:
        pd_ = para.data.clone(); fl = pd_ != pd_; pd_[fl] = np.random.random(); para.data.copy_(pd_)
M.eval(); Ttr, Tv = D.time_length_train, D.time_length_valid
SC, LAB, FR, FC, RG, RT, BID = [], [], [], [], [], [], []; row0 = 0
for ts in (Ttr, Ttr + Tv // 2):
    n = nq[ts]
    for i in range(int(math.ceil(n / 128))):
        idx = range(i * 128, min((i + 1) * 128, n)); db = torch.tensor(D.get_batch(ts, idx)).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            scores = M(ts, db[:, 0], db[:, 1])
        fr, fc = C.filter_pairs(D, ts, db.tolist())
        RG.append(C.authors_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy()); RT.append(C.tieavg_ranks_gpu(scores, db[:, 2], fr, fc).cpu().numpy())
        SC.append(scores.cpu().numpy()); LAB.append(db[:, 2].cpu().numpy()); FR.append(fr.cpu().numpy() + row0); FC.append(fc.cpu().numpy())
        BID.append(np.full(len(idx), len(BID))); row0 += len(idx)
np.save(f"{out}/rk_scores.npy", np.concatenate(SC)); np.savez(f"{out}/rk_meta.npz", lab=np.concatenate(LAB), fr=np.concatenate(FR), fc=np.concatenate(FC),
        rank_gpu=np.concatenate(RG), rank_tieavg=np.concatenate(RT), bid=np.concatenate([np.full(len(x), j) for j, x in enumerate(LAB)]))
print('saved', row0, 'rows', np.concatenate(SC).dtype)
