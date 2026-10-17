#!/usr/bin/env python3
"""pilot_polecat / pc_train_diag.py -- POST-HOC attribution diagnostics (not pre-registered; valid-selected like
the primary arms; reported as diagnostics only).  Same B-arm recipe as HRCOV (pc_train), with the coverage
block restricted:
  HRCOV_PAT    top-50 + pattern-prior block only               (columns 0-99)
  HRCOV_BR     top-50 + stage-2 bridge block only              (columns 0-49, 100-149)
  HRCOV_DYAD   top-50 + coverage candidates WITH an (s, c) event before t   (history-aware pool widening)
  HRCOV_NODYAD top-50 + coverage candidates WITHOUT any (s, c) event before t (genuinely new partners)
Masked columns are removed from the candidate set (cand = -1, gold outside -> conservative base rank path).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pc_train as T
from pc_common import WORK, KC, W_COV, SEEDS, jdump, log_fn

NDYAD_COL = 9   # FEAT_NAMES index of ev_log_ndyad

def colmask(Dd, name):
    n, W = Dd["CA"].shape
    jj = np.arange(W)[None, :]
    if name == "HRCOV_PAT": return np.broadcast_to(jj < 100, (n, W))
    if name == "HRCOV_BR": return np.broadcast_to((jj < KC) | (jj >= 100), (n, W))
    nd = np.asarray(Dd["X"][:, :, NDYAD_COL], np.float32) > 0
    if name == "HRCOV_DYAD": return (jj < KC) | nd
    if name == "HRCOV_NODYAD": return (jj < KC) | ~nd
    raise ValueError(name)

def apply_mask(Dd, name):
    m = colmask(Dd, name)
    Dd = dict(Dd); Dd["CA"] = np.where(m, Dd["CA"], -1); Dd["LP"] = np.where(m, Dd["LP"], -np.inf)
    gp = Dd["Q"]["gold_pos"].astype(np.int64); ok = gp >= 0
    keep = np.zeros(len(gp), bool); keep[ok] = m[np.flatnonzero(ok), gp[ok]]
    Q = dict(Dd["Q"]); Q["gold_pos"] = np.where(keep, gp, -1).astype(np.int16); Dd["Q"] = Q
    return Dd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", default="HRCOV_PAT,HRCOV_BR,HRCOV_DYAD,HRCOV_NODYAD")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fd = f"{WORK}/feat_{a.base}"; rd = f"{WORK}/train_{a.base}"
    Vall0 = T.load_split(fd, "valid"); Dte0 = T.load_split(fd, "test")
    ts = Vall0["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    out = dict(base=a.base, host=f"serverA ({dev})", note="post-hoc attribution diagnostics, not pre-registered", arms={})
    ranks = {}
    cols = T.COLS["all"]; W = W_COV
    for arm in a.arms.split(","):
        Vall = apply_mask(Vall0, arm); Dte = apply_mask(Dte0, arm)
        Dsel = T.subset(Vall, sel_idx); Vsel = T.view(Dsel, W); Vte = T.view(Dte, W)
        gpv = Vall["Q"]["gold_pos"][fit_idx].astype(np.int64)
        fit_tr = fit_idx[gpv >= 0]
        if len(fit_tr) > 120000:
            fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, 120000, replace=False))
        Dfit = T.subset(Vall, fit_tr); Vf = T.view(Dfit, W); m = Vf["CA"] >= 0
        Z = Dfit["X"][:, :, cols].astype(np.float32)[m]
        sub = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
        mu = sub.mean(0).astype(np.float32); sd = (sub.std(0) + 1e-4).astype(np.float32)
        Xfit = torch.tensor((Dfit["X"][:, :, cols].astype(np.float32) - mu) / sd, device=dev)
        mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev)
        rec = dict(n_fit_queries=int(len(fit_tr)), seeds={}, gold_in_set_test=float((Vte["gp"] >= 0).mean()))
        for sd_ in SEEDS:
            torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
            net = T.Net(len(cols)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            best = (-1.0, None, -1); nfit = len(fit_tr)
            for ep in range(30):
                net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev)
                for b0 in range(0, nfit, 512):
                    idx = perm[b0:b0 + 512]
                    z = net(Xfit[idx]).masked_fill(~mfit[idx], -1e30)
                    loss = torch.nn.functional.cross_entropy(z, yfit[idx]); opt.zero_grad(); loss.backward(); opt.step()
                net.eval(); fsel = T.predict(net, Dsel["X"], mu, sd, cols, W, dev)
                msel = T.metr(T.ranks_of(Vsel, fsel))
                if msel["MRR"] > best[0]: best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                elif ep - best[2] >= 5: break
            net.load_state_dict(best[1]); net.eval()
            fsel = T.predict(net, Dsel["X"], mu, sd, cols, W, dev); fte = T.predict(net, Dte["X"], mu, sd, cols, W, dev)
            al, be = T.calibrate_ab(Vsel, fsel)
            rs = T.ranks_of(Vsel, al * Vsel["LP"] + be * fsel); rt = T.ranks_of(Vte, al * Vte["LP"] + be * fte)
            rec["seeds"][sd_] = dict(best_epoch=best[2], cal_alpha=al, cal_beta=be, select=T.metr(rs), test=T.metr(rt))
            ranks[f"{arm}_s{sd_}"] = rt.astype(np.int32)
            log(arm, sd_, rec["seeds"][sd_]["select"]["MRR"], rec["seeds"][sd_]["test"]["MRR"])
        out["arms"][arm] = rec
        del Xfit
    np.savez(f"{rd}/ranks_test_diag.npz", **ranks); np.savez("ranks_test_diag_%s.npz" % a.base, **ranks)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_diag_{a.base}.json"); jdump(out, f"{rd}/train_diag_{a.base}.json")
    log("ALL DONE")

if __name__ == "__main__":
    main()
