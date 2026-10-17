#!/usr/bin/env python3
"""pilot_icews / pi_train.py -- Stage A history re-ranker arms (GPU), valid-only selection, RELNORM variant.

Recipe = pilot-1/2 "B standalone, calibrated" (pc_train.py / sp_train.py): MLP d->64->64->1 (zero-init last layer),
listwise CE over the arm's candidate set, Adam 1e-3, batch 512, <= 30 epochs, early stopping on valid-select MRR of f
alone (patience 5); then s = alpha * log p_base + beta * f with (alpha, beta) = argmin valid-select listwise NLL (LBFGS).
Valid days < 70% quantile of the valid sample's days -> fit (<= 120,000 queries with the gold in the candidate set),
rest -> select.  Test is never used for a choice.  Exact fused ranks (pilot-1 promotion rule).
Arms (columns of the scheme's 200-column layout, feature indices):
  HR         cols 0-49   feats 0-18 (8 base + 11 evidence)
  HRCOV_B    cols 0-(49+B), B in {50,100,150}; feats 0-24 (+ 6 pattern coverage)
Selection (pre-registered): B* = argmax 3-seed-mean select MRR over HRCOV_B; A* = argmax over {HR, HRCOV_B*};
then A*_RN = A* trained/evaluated on the 'rn' feature tensors (same columns / features / recipe).
  python3 pi_train.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from pi_common import (WORK, STAGE, KC, B_GRID, SEEDS, metrics_from_ranks, jdump, log_fn, exact_fused_ranks_chunked)

H_, C_ = list(range(0, 19)), list(range(19, 25))
ARMS = {"HR": (list(range(0, KC)), H_)}
for B in B_GRID:
    ARMS[f"HRCOV{B}"] = (list(range(0, KC + B)), H_ + C_)


class Net(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        torch.nn.init.zeros_(self.f[-1].weight); torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load_split(fd, split):
    return dict(Q=dict(np.load(f"{fd}/{split}_q.npz")), X=np.load(f"{fd}/{split}_X.npy", mmap_mode="r"),
                LP=np.load(f"{fd}/{split}_logp.npy"), CA=np.load(f"{fd}/{split}_cand.npy"), FI=np.load(f"{fd}/{split}_filt.npy"),
                BA=np.load(f"{fd}/{split}_babove.npy"))


def view(Dd, cols, idx=None):
    sel = slice(None) if idx is None else idx
    cols = np.asarray(cols)
    CA = Dd["CA"][sel][:, cols].astype(np.int64)
    o = Dd["Q"]["sro"][sel][:, 2]
    isg = (CA == o[:, None]) & (CA >= 0)
    gp = np.where(isg.any(1), isg.argmax(1), -1)
    return dict(LP=np.where(CA >= 0, Dd["LP"][sel][:, cols], -np.inf), CA=CA, FI=Dd["FI"][sel][:, cols], BA=Dd["BA"][sel][:, cols],
                gp=gp, rb=Dd["Q"]["r_base"][sel].astype(np.int64), cols=cols)


def ranks_of(V, s):
    return exact_fused_ranks_chunked(s, V["CA"], V["FI"], V["BA"], V["gp"], V["rb"])


def metr(rk):
    return metrics_from_ranks(rk, ks=(1, 3, 10, 50))


def predict(net, Z, V, feats, mu, sd, dev, rows=None, chunk=20000):
    n = len(V["CA"]); out = np.empty((n, len(V["cols"])), np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            rr = np.arange(lo, min(n, lo + chunk)) if rows is None else rows[lo:lo + chunk]
            x = np.asarray(Z[rr], np.float32)[:, V["cols"]][:, :, feats]
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


def run_arm(arm, cols, feats, Vall, Dte, fit_idx, sel_idx, seeds, a, dev, rd, log, tagname, save_all_proms=False):
    Vf_all = view(Vall, cols, fit_idx)
    fit_tr = fit_idx[Vf_all["gp"] >= 0]
    if len(fit_tr) > a.fit_max:
        fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
    Vf = view(Vall, cols, fit_tr); Vs = view(Vall, cols, sel_idx); Vt = view(Dte, cols)
    ident = float(np.mean(ranks_of(Vt, Vt["LP"]) == Vt["rb"]))
    Xraw = np.asarray(Vall["X"][fit_tr], np.float32)[:, cols][:, :, feats]
    m = Vf["CA"] >= 0
    Z = Xraw[m]
    sub = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
    mu = sub.mean(0).astype(np.float32); sd = (sub.std(0) + 1e-4).astype(np.float32)
    Xfit = torch.tensor((Xraw - mu) / sd, device=dev); del Xraw, Z, sub
    mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev)
    nfit = len(fit_tr)
    rec = dict(n_cols=len(cols), features=len(feats), n_fit_queries=int(nfit), identity_test=ident, seeds={})
    ranks_te, ranks_sel, prom0 = {}, {}, None
    for sd_ in seeds:
        torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
        net = Net(len(feats)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
        hist = []; best = (-1.0, None, -1)
        for ep in range(a.epochs):
            net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
            for b0 in range(0, nfit, a.bs):
                idx = perm[b0:b0 + a.bs]
                z = net(Xfit[idx]).masked_fill(~mfit[idx], -1e30)
                loss = torch.nn.functional.cross_entropy(z, yfit[idx])
                opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
            net.eval()
            fsel = predict(net, Vall["X"], Vs, feats, mu, sd, dev, rows=sel_idx)
            msel = metr(ranks_of(Vs, fsel))
            hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
            if msel["MRR"] > best[0]:
                best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
            elif ep - best[2] >= a.patience:
                break
        net.load_state_dict(best[1]); net.eval()
        torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, feats=feats, arm=tagname, seed=sd_), f"{rd}/models/{tagname}_s{sd_}.pt")
        fsel = predict(net, Vall["X"], Vs, feats, mu, sd, dev, rows=sel_idx); fte = predict(net, Dte["X"], Vt, feats, mu, sd, dev)
        srec = dict(best_epoch=best[2], history=hist, alone_select=metr(ranks_of(Vs, fsel)))
        al, be = calibrate_ab(Vs, fsel)
        srec.update(cal_alpha=al, cal_beta=be)
        ssel = al * Vs["LP"] + be * fsel; ste = al * Vt["LP"] + be * fte
        rs = ranks_of(Vs, ssel); rt = ranks_of(Vt, ste)
        srec["select"] = metr(rs)                      # test metrics are computed only at aggregation
        s_ = np.where(Vt["CA"] >= 0, ste, -np.inf); mint = s_[:, :KC].min(1)
        promc = (s_[:, KC:] > mint[:, None]) & (Vt["CA"][:, KC:] >= 0)
        srec["test_promoted_extra_per_query"] = float(promc.sum(1).mean()) if promc.shape[1] else 0.0
        ranks_te[f"{tagname}_s{sd_}"] = rt.astype(np.int32); ranks_sel[f"{tagname}_s{sd_}"] = rs.astype(np.int32)
        if save_all_proms:                                  # Stage B: every seed's promoted list to disk (official 3-seed scoring)
            np.save(f"{rd}/promlist_{tagname}_s{sd_}.npy", prom_lists(Vt, ste))
        elif sd_ == seeds[0]:
            prom0 = prom_lists(Vt, ste)
        rec["seeds"][sd_] = srec
        log(tagname, "seed", sd_, "best_ep", best[2], "select", srec["select"]["MRR"])
    rec["select_MRR_mean"] = float(np.mean([rec["seeds"][s]["select"]["MRR"] for s in seeds]))
    del Xfit, mfit, yfit
    torch.cuda.empty_cache()
    return rec, ranks_te, ranks_sel, prom0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_max", type=int, default=120000)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    from pi_common import DEV as dev
    torch.set_num_threads(8)
    rd = f"{WORK}/train_{a.base}"; os.makedirs(f"{rd}/models", exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    D = {sch: (load_split(f"{WORK}/feat_{a.base}_{sch}", "valid"), load_split(f"{WORK}/feat_{a.base}_{sch}", "test")) for sch in ("orig", "rn")}
    Vall, Dte = D["orig"]
    assert (D["rn"][0]["Q"]["sro"] == Vall["Q"]["sro"]).all() and (D["rn"][1]["Q"]["sro"] == Dte["Q"]["sro"]).all()
    ts = Vall["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    log("valid", len(ts), "fit", len(fit_idx), "select", len(sel_idx), "test", len(Dte["Q"]["r_base"]))
    out = dict(base=a.base, host=f"serverA ({dev}, tkg-gpu:latest, torch {torch.__version__})",
               valid_cut_day=int(cut), n_fit_rows=int(len(fit_idx)), n_select=int(len(sel_idx)), n_test=int(len(Dte["Q"]["r_base"])), cfg=vars(a), arms={})
    rb_sel = Vall["Q"]["r_base"][sel_idx].astype(np.int64)
    out["base_select"] = metr(rb_sel)
    ranks_out = dict(base=Dte["Q"]["r_base"].astype(np.int32)); sel_out = dict(base=rb_sel.astype(np.int32))
    proms = {}
    for arm, (cols, feats) in ARMS.items():
        if STAGE != "stageA" and os.path.exists(f"{rd}/arm_{arm}.json"):          # resume (Stage B)
            rec = json.load(open(f"{rd}/arm_{arm}.json")); rec["seeds"] = {int(k): v for k, v in rec["seeds"].items()}
            z = np.load(f"{rd}/ranks_arm_{arm}.npz"); rt = {k[3:]: z[k] for k in z.files if k.startswith("te_")}; rs = {k[3:]: z[k] for k in z.files if k.startswith("se_")}
            p0 = None; log(arm, "resumed from disk")
        else:
            rec, rt, rs, p0 = run_arm(arm, cols, feats, Vall, Dte, fit_idx, sel_idx, seeds, a, dev, rd, log, arm,
                                      save_all_proms=(STAGE != "stageA" and arm.startswith("HRCOV")))
            if STAGE != "stageA":
                np.savez(f"{rd}/ranks_arm_{arm}.npz", **{f"te_{k}": v for k, v in rt.items()}, **{f"se_{k}": v for k, v in rs.items()})
                jdump(rec, f"{rd}/arm_{arm}.json")
        out["arms"][arm] = rec; ranks_out.update(rt); sel_out.update(rs); proms[arm] = p0
        np.savez(f"{rd}/ranks_test.npz", **ranks_out); np.savez(f"{rd}/ranks_select.npz", **sel_out); jdump(out, f"{rd}/train_{a.base}.json")
    covs = {B: out["arms"][f"HRCOV{B}"]["select_MRR_mean"] for B in B_GRID}
    Bstar = max(B_GRID, key=lambda B: covs[B])
    cand = {"HR": out["arms"]["HR"]["select_MRR_mean"], f"HRCOV{Bstar}": covs[Bstar]}
    Astar = max(cand, key=lambda k: cand[k])
    rule_txt = "B* = argmax 3-seed-mean valid-select MRR over HRCOV_B; A* = argmax over {HR, HRCOV_B*}"
    if STAGE != "stageA":                                  # Stage B rule: the claim system and its RN variant are HRCOV_B*
        out["selection_stageA_style_A_star"] = Astar; Astar = f"HRCOV{Bstar}"
        rule_txt = "stageB_rule.json: B* = argmax 3-seed-mean valid-select MRR over HRCOV_B (ties -> smaller B); RN arm = HRCOV_B*"
    out["selection"] = dict(HRCOV_select_MRR_by_B=covs, B_star=Bstar, A_star=Astar, A_candidates_select_MRR=cand, rule=rule_txt)
    log("selection", json.dumps(out["selection"]))
    cols, feats = ARMS[Astar]
    Vrn, Trn = D["rn"]
    rec, rt, rs, p0 = run_arm(f"{Astar}_RN", cols, feats, Vrn, Trn, fit_idx, sel_idx, seeds, a, dev, rd, log, f"{Astar}_RN")
    out["arms"][f"{Astar}_RN"] = rec; ranks_out.update(rt); sel_out.update(rs); proms[f"{Astar}_RN"] = p0
    if STAGE == "stageA":
        for nm in ("HR", f"HRCOV{Bstar}", f"{Astar}_RN"):
            np.save(f"{rd}/promlist_{nm}_s{seeds[0]}.npy", proms[nm])
        out["promlists"] = [f"promlist_{nm}_s{seeds[0]}.npy" for nm in ("HR", f"HRCOV{Bstar}", f"{Astar}_RN")]
    else:                                                  # all promoted lists are KEPT 
        out["promlists"] = [f"promlist_HRCOV{Bstar}_s{s_}.npy" for s_ in seeds]
        assert all(os.path.exists(f"{rd}/{f}") for f in out["promlists"]), out["promlists"]
    np.savez(f"{rd}/ranks_test.npz", **ranks_out); np.savez(f"{rd}/ranks_select.npz", **sel_out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_{a.base}.json"); jdump(out, f"{rd}/train_{a.base}.json")
    os.system(f"chmod -R a+rX {rd}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
