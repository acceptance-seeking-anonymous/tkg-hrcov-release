#!/usr/bin/env python3
"""pilot_smallpedia / sp_train.py -- history re-ranker arms with / without static-KG candidates and features (GPU).

Recipe = pilot_polecat/pc_train.py "B standalone, calibrated": MLP d->64->64->1 (zero-init last layer), listwise CE over
the arm's candidate set, early stopping on valid-select MRR of f alone (patience 5, <= 30 epochs), then
s = alpha * log p_base + beta * f with (alpha, beta) = argmin valid-select listwise NLL (LBFGS).  Valid timestamps
< 70% quantile -> fit, rest -> select.  Test is never used for a choice.  Exact fused ranks (promotion rule).
Arms (layout, columns, features):
  HR            A  0-49                    hist (19)                  history re-ranker
  HRCOV         A  0-149                   hist + cov (30)            pilot-1 coverage stack = REFERENCE
  HRCOV_ST      A  0-199                   hist + cov + static (42)   PRIMARY: + 50 static-KG candidates + static features
  HRCOV_DM      A  0-149 + 200-249         hist + cov + static (42)   density control: 50 degree-matched random instead
  HRCOV_STFEAT  A  0-149                   hist + cov + static (42)   static features without static candidates
  HRCOV_PAT     A  top-50 + pattern block  hist + cov (30)            attribution (pilot-1 recommendation)
  HR_ST         B  0-99                    hist + static (31)         secondary: HR + 50 static candidates
  STONLY        B  0-149                   hist + static (31)         static-only candidates at HRCOV's W = 150
  python3 sp_train.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sp_common import (WORK, KC, BP, BB, BS, BD, W_A, W_B, SEEDS, metrics_from_ranks, jdump, log_fn, exact_fused_ranks, MODE_SFX, ST_MODE)

H_, C_, S_ = list(range(0, 19)), list(range(19, 30)), list(range(30, 42))
ARMS = {"HR": ("A", list(range(0, 50)), H_, None),
        "HRCOV": ("A", list(range(0, 150)), H_ + C_, None),
        "HRCOV_ST": ("A", list(range(0, 200)), H_ + C_ + S_, None),
        "HRCOV_DM": ("A", list(range(0, 150)) + list(range(200, 250)), H_ + C_ + S_, None),
        "HRCOV_STFEAT": ("A", list(range(0, 150)), H_ + C_ + S_, None),
        "HRCOV_PAT": ("A", list(range(0, 150)), H_ + C_, "pat_only"),
        "HR_ST": ("B", list(range(0, 100)), H_ + S_, None),
        "STONLY": ("B", list(range(0, 150)), H_ + S_, None)}
SAVE_LISTS = ("HR", "HRCOV", "HRCOV_ST", "HR_ST")


class Net(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        torch.nn.init.zeros_(self.f[-1].weight); torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load_split(fd, split):
    Q = dict(np.load(f"{fd}/{split}_q.npz"))
    out = dict(Q=Q)
    for L in ("A", "B"):
        out[L] = dict(X=np.load(f"{fd}/{split}_{L}_X.npy", mmap_mode="r"), LP=np.load(f"{fd}/{split}_{L}_logp.npy"),
                      CA=np.load(f"{fd}/{split}_{L}_cand.npy"), FI=np.load(f"{fd}/{split}_{L}_filt.npy"), BA=np.load(f"{fd}/{split}_{L}_babove.npy"))
    return out


def view(Dd, arm, idx=None):
    """arm view: candidate columns (ordered), per-query masks, gold position within the view."""
    L, cols, feats, mask = ARMS[arm]
    Z = Dd[L]; sel = slice(None) if idx is None else idx
    cols = np.asarray(cols)
    CA = Z["CA"][sel][:, cols].astype(np.int64)
    if mask == "pat_only":
        npb = Dd["Q"]["n_pat_block"][sel]
        j = np.arange(len(cols))[None, :]
        CA = np.where((j >= KC) & (j >= KC + npb[:, None]), -1, CA)
    o = Dd["Q"]["sro"][sel][:, 2]
    isg = (CA == o[:, None]) & (CA >= 0)
    gp = np.where(isg.any(1), isg.argmax(1), -1)
    return dict(LP=np.where(CA >= 0, Z["LP"][sel][:, cols], -np.inf), CA=CA, FI=Z["FI"][sel][:, cols], BA=Z["BA"][sel][:, cols],
                gp=gp, rb=Dd["Q"]["r_base"][sel].astype(np.int64), cols=cols, feats=feats, L=L)


def X_of(Dd, V, idx=None, chunk=None):
    Z = Dd[V["L"]]["X"]
    sel = slice(None) if idx is None else idx
    x = np.asarray(Z[sel] if idx is not None else Z)
    return x


def ranks_of(V, s):
    return exact_fused_ranks(s, V["CA"], V["FI"], V["BA"], V["gp"], V["rb"])


def metr(rk):
    return metrics_from_ranks(rk, ks=(1, 3, 10, 50))


def predict(net, Z, V, mu, sd, dev, rows=None, chunk=20000):
    n = len(V["CA"]); out = np.empty((n, len(V["cols"])), np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            rr = np.arange(lo, min(n, lo + chunk)) if rows is None else rows[lo:lo + chunk]
            x = np.asarray(Z[rr], np.float32)[:, V["cols"]][:, :, V["feats"]]
            x = (x - mu) / sd
            out[lo:lo + chunk] = net(torch.from_numpy(x).to(dev)).cpu().numpy()
    return out


def calibrate_ab(V, f, iters=60):
    m = (V["CA"] >= 0) & ~V["FI"]; ok = V["gp"] >= 0
    lp = torch.tensor(np.where(m, V["LP"], 0.0)[ok], dtype=torch.float64)
    ff = torch.tensor(f[ok], dtype=torch.float64); mk = torch.tensor(m[ok]); y = torch.tensor(V["gp"][ok])
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = (ab[0] * lp + ab[1] * ff).masked_fill(~mk, -1e30)
        loss = torch.nn.functional.cross_entropy(z, y); loss.backward(); return loss
    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())


def prom_lists(V, s):
    n, W = s.shape
    valid = V["CA"] >= 0
    s = np.where(valid, s, -np.inf)
    mint = s[:, :KC].min(1)
    prom = valid & ((np.arange(W)[None, :] < KC) | (s > mint[:, None]))
    key = np.where(prom, s, -np.inf)
    o = np.argsort(-key, axis=1, kind="stable")
    ids = np.take_along_axis(V["CA"], o, 1); pk = np.take_along_axis(prom, o, 1)
    return np.where(pk, ids, -1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_max", type=int, default=120000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    sfx = ("_smoke" if a.smoke else "") + MODE_SFX
    fd = f"{WORK}/feat_{a.base}{sfx}"; rd = f"{WORK}/train_{a.base}{sfx}"; os.makedirs(f"{rd}/models", exist_ok=True)
    Vall = load_split(fd, "valid"); Dte = load_split(fd, "test")
    ts = Vall["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    if a.smoke:
        fit_idx = np.arange(0, len(ts), 2); sel_idx = np.arange(1, len(ts), 2)
    log("loaded valid", len(ts), "fit", len(fit_idx), "select", len(sel_idx), "test", len(Dte["Q"]["r_base"]), dev)
    out = dict(base=a.base, host=f"serverA ({dev}, tkg-gpu:latest, torch {torch.__version__})", valid_cut_ts=int(cut),
               n_fit_rows=int(len(fit_idx)), n_select=int(len(sel_idx)), n_test=int(len(Dte["Q"]["r_base"])), cfg=vars(a), arms={},
               static_mode=ST_MODE)
    rb_te = Dte["Q"]["r_base"].astype(np.int64); rb_sel = Vall["Q"]["r_base"][sel_idx].astype(np.int64)
    out["base_select"] = metr(rb_sel); out["base_test"] = metr(rb_te)
    ranks_out = dict(base=rb_te.astype(np.int32)); sel_out = dict(base=rb_sel.astype(np.int32))
    seeds = [int(x) for x in a.seeds.split(",")]
    for arm in a.arms.split(","):
        L, cols, feats, mask = ARMS[arm]
        Vf_all = view(Vall, arm, fit_idx)
        okf = Vf_all["gp"] >= 0
        fit_tr = fit_idx[okf]
        if len(fit_tr) > a.fit_max:
            fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
        Vf = view(Vall, arm, fit_tr); Vs = view(Vall, arm, sel_idx); Vt = view(Dte, arm)
        out.setdefault("identity_test", {})[arm] = float(np.mean(ranks_of(Vt, Vt["LP"]) == Vt["rb"]))
        Xraw = np.asarray(Vall[L]["X"][fit_tr], np.float32)[:, cols][:, :, feats]
        m = Vf["CA"] >= 0
        Z = Xraw[m]
        sub = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
        mu = sub.mean(0).astype(np.float32); sd = (sub.std(0) + 1e-4).astype(np.float32)
        Xfit = torch.tensor((Xraw - mu) / sd, device=dev); del Xraw, Z, sub
        mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev)
        nfit = len(fit_tr)
        rec = dict(layout=L, n_cols=len(cols), features=len(feats), mask=mask, n_fit_queries=int(nfit), seeds={})
        for sd_ in seeds:
            torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
            net = Net(len(feats)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
            hist = []; best = (-1.0, None, -1)
            for ep in range(a.epochs if not a.smoke else 2):
                net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
                for b0 in range(0, nfit, a.bs):
                    idx = perm[b0:b0 + a.bs]
                    z = net(Xfit[idx]).masked_fill(~mfit[idx], -1e30)
                    loss = torch.nn.functional.cross_entropy(z, yfit[idx])
                    opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                net.eval()
                fsel = predict(net, Vall[L]["X"], Vs, mu, sd, dev, rows=sel_idx)
                msel = metr(ranks_of(Vs, fsel))
                hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                if msel["MRR"] > best[0]:
                    best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                elif ep - best[2] >= a.patience:
                    break
            net.load_state_dict(best[1]); net.eval()
            torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, feats=feats, arm=arm, seed=sd_), f"{rd}/models/{arm}_s{sd_}.pt")
            fsel = predict(net, Vall[L]["X"], Vs, mu, sd, dev, rows=sel_idx); fte = predict(net, Dte[L]["X"], Vt, mu, sd, dev)
            srec = dict(best_epoch=best[2], history=hist, alone_select=metr(ranks_of(Vs, fsel)), alone_test=metr(ranks_of(Vt, fte)))
            al, be = calibrate_ab(Vs, fsel)
            srec.update(cal_alpha=al, cal_beta=be)
            ssel = al * Vs["LP"] + be * fsel; ste = al * Vt["LP"] + be * fte
            rs = ranks_of(Vs, ssel); rt = ranks_of(Vt, ste)
            srec.update(select=metr(rs), test=metr(rt))
            s_ = np.where(Vt["CA"] >= 0, ste, -np.inf); mint = s_[:, :KC].min(1)
            promc = (s_[:, KC:] > mint[:, None]) & (Vt["CA"][:, KC:] >= 0)
            srec["test_promoted_extra_per_query"] = float(promc.sum(1).mean()); srec["test_frac_queries_with_promotion"] = float(promc.any(1).mean())
            if L == "A" and len(cols) > 150:
                blk = promc[:, 100:150]            # columns 150-199 of the view = static block (HRCOV_ST) / dm block (HRCOV_DM)
                srec["test_promoted_block4_per_query"] = float(blk.sum(1).mean())
            ranks_out[f"{arm}_s{sd_}"] = rt.astype(np.int32); sel_out[f"{arm}_s{sd_}"] = rs.astype(np.int32)
            if arm in SAVE_LISTS and (sd_ == seeds[0] or os.environ.get("SP_SAVE_LISTS_ALL_SEEDS") == "1"):   # G3 runs saved every seed
                np.save(f"{rd}/promlist_{arm}_s{sd_}.npy", prom_lists(Vt, ste))
            rec["seeds"][sd_] = srec
            log(arm, "seed", sd_, "best_ep", best[2], "select", srec["select"]["MRR"])
        rec["n_params"] = int(sum(p.numel() for p in net.parameters()))
        out["arms"][arm] = rec
        del Xfit, mfit, yfit
        torch.cuda.empty_cache()
        np.savez(f"{rd}/ranks_test.npz", **ranks_out); np.savez(f"{rd}/ranks_select.npz", **sel_out)
        jdump(out, f"{rd}/train_{a.base}.json")
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_{a.base}{sfx}.json"); jdump(out, f"{rd}/train_{a.base}.json")
    os.system(f"chmod -R a+rX {rd}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
