#!/usr/bin/env python3
"""rescore_check.py -- recompute every re-ranker score from the saved models (CPU) and count non-finite values.
For each arm/seed: the ranker output f on all candidate slots of the arm's window W and on the valid slots (cand >= 0),
the fused score alpha * logp + beta * f on valid slots, the base logp on valid slots, and the model tensors (weights, mu, sd).
No row is dropped or changed; read-only.
  python3 rescore_check.py --feat FD --arm HR=train.json:models_dir --arm HRCOV=... --out rescore_check.json [--splits valid,test]
"""
import os, json, time, argparse
import numpy as np
import torch


class Net(torch.nn.Module):          # identical to pc_train.py / hcpat_train.py / v7_btrain.py (mlp64)
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def nf(x):
    x = np.asarray(x); return dict(n=int(x.size), nan=int(np.isnan(x).sum()), posinf=int((x == np.inf).sum()), neginf=int((x == -np.inf).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", required=True); ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--splits", default="valid,test"); ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--label", default=""); ap.add_argument("--pre", default="", help="file prefix after split_, e.g. A_ for the sp layout")
    a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(int(os.environ.get("RS_THREADS", "12")))
    arms = []
    for spec in a.arm:
        nm, rest = spec.split("=", 1); tj, md = rest.split(":", 1); J = json.load(open(tj)); A = J["arms"][nm]
        arms.append((nm, A, md))
    out = dict(name="rescore_check", label=a.label, feat=a.feat, arms={}, splits=a.splits.split(","))
    for split in a.splits.split(","):
        X = np.load(f"{a.feat}/{split}_{a.pre}X.npy", mmap_mode="r"); LP = np.load(f"{a.feat}/{split}_{a.pre}logp.npy", mmap_mode="r")
        CA = np.load(f"{a.feat}/{split}_{a.pre}cand.npy", mmap_mode="r"); n = X.shape[0]
        for nm, A, md in arms:
            W = int(A.get("W", 100))
            for sd_, srec in sorted(A["seeds"].items()):
                ck = torch.load(f"{md}/{nm}_s{sd_}.pt", map_location="cpu", weights_only=False)
                cols = list(ck["cols"]); mu = np.asarray(ck["mu"], np.float32); sdv = np.asarray(ck["sd"], np.float32)
                net = Net(len(cols)); net.load_state_dict(ck["state_dict"]); net.eval()
                par = {k: nf(v.numpy()) for k, v in ck["state_dict"].items()}
                al, be = srec.get("cal_alpha"), srec.get("cal_beta")
                acc = dict(f_all=dict(n=0, nan=0, posinf=0, neginf=0), f_valid=dict(n=0, nan=0, posinf=0, neginf=0),
                           fused_valid=dict(n=0, nan=0, posinf=0, neginf=0), logp_valid=dict(n=0, nan=0, posinf=0, neginf=0),
                           X_cols_valid=dict(n=0, nan=0, posinf=0, neginf=0))
                def add(k, d):
                    for kk in ("n", "nan", "posinf", "neginf"): acc[k][kk] += d[kk]
                with torch.no_grad():
                    for lo in range(0, n, a.chunk):
                        xr = np.asarray(X[lo:lo + a.chunk, :W][:, :, cols], np.float32)
                        ca = np.asarray(CA[lo:lo + a.chunk, :W]); lp = np.asarray(LP[lo:lo + a.chunk, :W], np.float64); v = ca >= 0
                        f = net(torch.from_numpy((xr - mu) / sdv)).numpy()
                        add("f_all", nf(f)); add("f_valid", nf(f[v])); add("logp_valid", nf(lp[v])); add("X_cols_valid", nf(xr[v]))
                        if al is not None and be is not None: add("fused_valid", nf(al * lp[v] + be * f[v].astype(np.float64)))
                bad = sum(acc[k]["nan"] + acc[k]["posinf"] + acc[k]["neginf"] for k in acc) + sum(p["nan"] + p["posinf"] + p["neginf"] for p in par.values())
                bad += int(not np.isfinite(mu).all()) + int(not np.isfinite(sdv).all())
                rec = out["arms"].setdefault(f"{nm}_s{sd_}", dict(W=W, n_features=len(cols), cal_alpha=al, cal_beta=be,
                                                                params_finite=all(p["nan"] + p["posinf"] + p["neginf"] == 0 for p in par.values()),
                                                                mu_sd_finite=bool(np.isfinite(mu).all() and np.isfinite(sdv).all()), splits={}))
                rec["splits"][split] = dict(n_queries=int(n), **acc, n_nonfinite_total=int(bad))
                print(f"[{time.time() - t0:7.1f}s] {split} {nm}_s{sd_} W={W} nonfinite={bad} f_valid={acc['f_valid']['n']}", flush=True)
    out["all_finite"] = bool(all(s["n_nonfinite_total"] == 0 for r in out["arms"].values() for s in r["splits"].values()))
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1); print("ALL_FINITE", out["all_finite"])


if __name__ == "__main__":
    main()
