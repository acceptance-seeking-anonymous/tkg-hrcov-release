
import numpy as np, os, json, sys
out=[]
for p in open("neginf_files.txt").read().split():
    d, b = os.path.split(p)
    cands = [b.replace("logp", "cand")]
    if b.endswith("_lk_logp.npy"): cands = [b.replace("lk_logp", "lk_cand"), b.replace("_lk_logp.npy", "_req.npy")]
    cp = next((os.path.join(d, c) for c in cands if os.path.exists(os.path.join(d, c))), None)
    if cp is None and "/lk/" in p:
        up = os.path.dirname(d); sp = b.split("_")[0]; q = os.path.join(up, f"{sp}_req.npy")
        cp = q if os.path.exists(q) else None
    L = np.load(p, mmap_mode="r"); rec = dict(path=p, cand=cp, shape=list(L.shape))
    if cp is None: rec["status"] = "no paired candidate array"; out.append(rec); continue
    C = np.load(cp, mmap_mode="r")
    if C.shape[0] != L.shape[0]: rec["status"] = f"shape mismatch {C.shape}"; out.append(rec); continue
    w = min(L.shape[1], C.shape[1]) if L.ndim > 1 and C.ndim > 1 else None
    ninf_valid = ninf_pad = 0; ninf_beyond = 0
    for lo in range(0, L.shape[0], 200000):
        l = np.asarray(L[lo:lo+200000]); c = np.asarray(C[lo:lo+200000])
        if l.ndim > 1 and l.shape[1] > w: ninf_beyond += int(np.isneginf(l[:, w:]).sum()); l = l[:, :w]
        c = c[:, :w] if c.ndim > 1 else c
        m = np.isneginf(l); ninf_valid += int((m & (c >= 0)).sum()); ninf_pad += int((m & (c < 0)).sum())
    rec.update(neginf_at_valid_candidate=ninf_valid, neginf_at_padding=ninf_pad, neginf_beyond_cand_width=ninf_beyond, status="ok")
    out.append(rec); print(os.path.basename(p), ninf_valid, ninf_pad, ninf_beyond, flush=True)
json.dump(out, open("neginf_masks.json", "w"), indent=1)
