#!/usr/bin/env python3
"""v76 / v76_rbfeat.py (= hardening hd_rbfeat.py + --recb_dir / --W)
hardening / hd_rbfeat.py -- RecB score features for every (query, candidate) of an existing re-ranker feature tensor.

For the pilot feature tensors (polecat feat_<b>/{split}_cand.npy, smallpedia feat_<b>/{split}_A_cand.npy) adds three
columns per candidate c of query (s, r', t), from the CLEAN TRAINED RecB (hd_recb.py CSR: psi at the valid-selected
lambda_r; alpha_r; xi from the online fact counts of relation r' before t):
   rb_log_q   = log(1e-7 + q(c)),  q = RecB / sum_e RecB(e),  RecB = 1000 alpha_r psi + 1000 (1 - alpha_r) xi
   rb_log_psi = log(1e-7 + psi(c))      rb_log_xi = log(1e-7 + xi(c))
psi/xi depend on (s, r', t) only, so CSR rows are matched on that key.  Output [n, W, 3] float16 memmap.
  python3 hd_rbfeat.py --ds polecat --cand <feat>/test_cand.npy --q <feat>/test_q.npz --split test --out <dir>/test_RB.npy
"""
import os, sys, json, time, argparse
import numpy as np

HAR = os.path.expandvars("$WORK_DIR"); TR = f"{HAR}/tracks"; HD = f"{TR}/hardening"
CFG = {"polecat": ("tkgl-polecat", f"{TR}/datasets/harness_data/tkgl-polecat"),
       "smallpedia": ("tkgl-smallpedia", f"{TR}/datasets/harness_data/tkgl-smallpedia")}
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--cand", required=True); ap.add_argument("--q", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--recb_dir", default=f"{HD}/work/recb"); ap.add_argument("--W", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
name, dd = CFG[a.ds]
st = open(f"{dd}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
z = np.load(f"{a.recb_dir}/recb_csr_{a.split}_{name}.npz")
ptr, cand_c, psi_c, csro = z["ptr"], z["cand"], z["psi_tr"].astype(np.float64), z["sro"]
alpha = np.asarray(z["alpha_tr"], np.float64)
ck = (csro[:, 3] * (2 * R) + csro[:, 1]) * N + csro[:, 0]
o = np.argsort(ck, kind="stable"); cks = ck[o]
Q = np.load(a.q); sro = Q["sro"].astype(np.int64)
qk = (sro[:, 3] * (2 * R) + sro[:, 1]) * N + sro[:, 0]
pos = np.searchsorted(cks, qk); assert (cks[np.clip(pos, 0, len(cks) - 1)] == qk).all(), "query key missing in CSR"
crow = o[pos]                                   # CSR row of each feature row
CA = np.load(a.cand, mmap_mode="r"); n, W = CA.shape
if a.W: CA = CA[:, :a.W]; W = a.W
OUT = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(n, W, 3))
# global CSR key (row * N + cand) is increasing (cand sorted within row)
gkey = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr)) * N + cand_c.astype(np.int64)
psum = np.add.reduceat(psi_c, ptr[:-1]) if len(psi_c) else np.zeros(len(ptr) - 1)
psum = np.where(np.diff(ptr) > 0, psum, 0.0)
# facts for xi (both directions), time order
fq = np.concatenate([np.loadtxt(f"{dd}/{sp}.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4) for sp in ("train", "valid", "test")])
fr = np.concatenate([fq[:, 1], fq[:, 1] + R]); fo = np.concatenate([fq[:, 2], fq[:, 0]]); ft = np.concatenate([fq[:, 3], fq[:, 3]])
of = np.argsort(ft, kind="stable"); fr, fo, ft = fr[of], fo[of], ft[of]
C = np.zeros((2 * R, N), np.int32); tot = np.zeros(2 * R, np.int64); hp = 0
order = np.argsort(sro[:, 3], kind="stable")
uts, s0 = np.unique(sro[order, 3], return_index=True); e0 = np.append(s0[1:], len(order))
for u, b, e in zip(uts, s0, e0):
    h = np.searchsorted(ft, u, "left")
    if h > hp:
        np.add.at(C, (fr[hp:h], fo[hp:h]), 1); tot += np.bincount(fr[hp:h], minlength=2 * R); hp = h
    rows = order[b:e]
    for c0 in range(0, len(rows), 20000):
        rr = rows[c0:c0 + 20000]
        ca = np.asarray(CA[rr]).astype(np.int64); valid = ca >= 0; cc = np.where(valid, ca, 0)
        rel = sro[rr, 1]; cr = crow[rr]
        xi = C[rel[:, None], cc] / np.maximum(tot[rel], 1)[:, None]
        key = cr[:, None] * N + cc
        p = np.searchsorted(gkey, key); pc = np.clip(p, 0, len(gkey) - 1)
        hit = (gkey[pc] == key) & valid
        psi = np.where(hit, psi_c[pc], 0.0)
        al = alpha[rel][:, None]
        rb = 1000 * al * psi + 1000 * (1 - al) * xi
        ssum = 1000 * alpha[rel] * psum[cr] + 1000 * (1 - alpha[rel]) * (tot[rel] > 0)
        q = np.where(ssum[:, None] > 0, rb / np.maximum(ssum, 1e-30)[:, None], 0.0)
        X = np.stack([np.log(1e-7 + q), np.log(1e-7 + psi), np.log(1e-7 + xi)], -1)
        X[~valid] = 0.0
        OUT[rr] = X.astype(np.float16)
OUT.flush()
json.dump(dict(recb_dir=a.recb_dir, ds=a.ds, split=a.split, n=int(n), W=int(W), feats=["rb_log_q", "rb_log_psi", "rb_log_xi"],
               frac_cand_with_psi=float(np.mean(np.asarray(OUT[:, :, 1]) > np.log(2e-7))), elapsed_s=round(time.time() - t0, 1)),
          open(a.out.replace(".npy", ".json"), "w"), indent=1)
print("done", n, W, round(time.time() - t0, 1), flush=True)
