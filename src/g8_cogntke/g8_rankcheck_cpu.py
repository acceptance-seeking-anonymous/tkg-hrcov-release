"""G8 ranker identity, part 2 (CPU): the released utils.cal_ranks (verbatim body) per 128-row batch on the saved fp16 score
rows, compared with the GPU emulation.  Run once with the container's numpy/scipy and once with an authors'-era stack."""
import sys, json, numpy as np, scipy
from scipy.stats import rankdata
def cal_ranks(scores, labels, filters):                    # verbatim from CognTKE utils.py (commit 4f30966)
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-6
    full_rank = rankdata(-scores, method='ordinal', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='ordinal', axis=1)
    mask = np.zeros_like(scores)
    mask[np.arange(mask.shape[0]), labels] = 1
    ranks = (full_rank - filter_rank + 1) * mask
    ranks = ranks[np.nonzero(ranks)]
    return ranks.tolist()
d = sys.argv[1]; tag = sys.argv[2]
S = np.load(f"{d}/rk_scores.npy", mmap_mode='r'); Mt = np.load(f"{d}/rk_meta.npz")
lab, fr, fc, rg, bid = Mt['lab'], Mt['fr'], Mt['fc'], Mt['rank_gpu'], Mt['bid']
out = dict(tag=tag, numpy=np.__version__, scipy=scipy.__version__, n_rows=int(len(lab)), n_batches=int(bid.max() + 1),
           n_len_mismatch_batches=0, n_mismatch=0, rank_dtype=None)
allc = []
for b in range(int(bid.max()) + 1):
    rows = np.flatnonzero(bid == b); r0 = rows[0]
    sc = np.asarray(S[rows]); assert sc.dtype == np.float16
    m = (fr >= r0) & (fr <= rows[-1]); filt = np.zeros_like(sc); filt[fr[m] - r0, fc[m]] = 1
    if out['rank_dtype'] is None:
        out['rank_dtype'] = str(rankdata(-(sc - np.min(sc, axis=1, keepdims=True) + 1e-6), method='ordinal', axis=1).dtype)
    rc = np.asarray(cal_ranks(sc, lab[rows], filt), np.float64)
    if len(rc) != len(rows):
        out['n_len_mismatch_batches'] += 1; out['n_mismatch'] += len(rows); continue
    allc.append(rc); out['n_mismatch'] += int((rc != rg[rows]).sum())
out['identical'] = bool(out['n_mismatch'] == 0 and out['n_len_mismatch_batches'] == 0)
if allc:
    rc = np.concatenate(allc); out['cpu_mrr'] = float(np.mean(1 / rc))
out['gpu_mrr'] = float(np.mean(1 / rg)); out['n_rows_tie_broken_vs_tieavg'] = int((rg != Mt['rank_tieavg']).sum())
print(json.dumps(out)); json.dump(out, open(f"{d}/rankcheck_{tag}.json", 'w'), indent=1)
