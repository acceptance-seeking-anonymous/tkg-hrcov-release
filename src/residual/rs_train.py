#!/usr/bin/env python3
"""
Residual-objective track (idea 6.2) -- steps 2-3: train arms on VALID-fit, select on VALID-select, score FULL test.

Module f(q,c): MLP d->64->64->1 (the G10-G12 stage-2 architecture), last layer zero-initialised, identical for
every trained arm with the same feature set (matched capacity).  Candidate set C(q) from rs_features.py.

  R*    residual-in-loss  : s(c) = log p_base(c) + g * f(q,c),   loss = listwise CE of softmax_C(s)
                            (= maximise log p(gold|base+evidence) - log p(gold|base) restricted to C)
        gates g: none | reuse g(c)=1-1/rolecount(c) | conf g(q)=1-p1(q)
  A     base-agnostic     : f trained with listwise CE of softmax_C(f) (base logits NOT in the loss), then
                            s = log p_base + lam * g_bm * f with (lam, gate) chosen on valid-select under the
                            BridgeMem rule (max MRR s.t. MRR, H@1, H@3, H@10 do not decrease vs the base)
  B     post-hoc calib.   : same standalone f, s = alpha*log p_base + beta*f, (alpha,beta) = argmin valid NLL
  D     BridgeMem control : s = log p_base + lam * g * A_dyad(q,c) (count reader, base-agnostic, train facts only),
                            g in {1, 4u(1-u)}, BridgeMem selection rule
  COPY  diagnostic        : s = log p_base + lam * log1p(n(s,r',c)) , lam by valid-select MRR
  *_ev  evidence-only feature set (no base-score / u_pred columns)
Valid split: valid timestamps < 70% quantile -> fit, rest -> select (test never used for any choice).
"""
import os, sys, json, time, argparse
H = os.path.expandvars("$WORK_DIR"); TRACK = f"{H}/tracks/residual"
sys.path.insert(0, H); sys.path.insert(0, f"{TRACK}/code")
import numpy as np
import torch
from oih_common import metrics_from_ranks, jdump
from rs_features import fused_ranks_np, KC, FEAT_NAMES, GROUPS

LAM_A = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
LAM_D = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
LAM_C = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


class Net(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(),
                                     torch.nn.Linear(h, 1))
        torch.nn.init.zeros_(self.f[-1].weight); torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x).squeeze(-1)


SUFFIX = ""


def load_split(DS, base, split):
    d = f"{TRACK}/feat/{DS}_{base}{SUFFIX}"
    out = dict(X=np.load(f"{d}/{split}_X.npy", mmap_mode="r"), LP=np.load(f"{d}/{split}_logp.npy"),
               CA=np.load(f"{d}/{split}_cand.npy"), FI=np.load(f"{d}/{split}_filt.npy"),
               BA=np.load(f"{d}/{split}_babove.npy"), GR=np.load(f"{d}/{split}_greuse.npy").astype(np.float32))
    out["Q"] = dict(np.load(f"{d}/{split}_q.npz"))
    return out


def subset(Dd, idx):
    o = {k: (v[idx] if k != "Q" else None) for k, v in Dd.items()}
    o["X"] = np.asarray(Dd["X"][idx]); o["Q"] = {k: v[idx] for k, v in Dd["Q"].items()}
    return o


def metr(rk):
    return metrics_from_ranks(rk, ks=(1, 3, 10, 50))


def ranks_of(Dd, s):
    return fused_ranks_np(s, Dd["CA"], Dd["FI"], Dd["BA"], Dd["Q"]["gold_pos"].astype(np.int64), Dd["Q"]["r_base"])


def gate_arr(Dd, gate):
    n, W = Dd["LP"].shape
    if gate == "none":
        return np.ones((n, W), np.float32)
    if gate == "reuse":
        return Dd["GR"]
    if gate == "conf":
        return np.broadcast_to((1.0 - Dd["Q"]["p1"]).astype(np.float32)[:, None], (n, W)).copy()
    if gate == "bm":
        u = Dd["Q"]["u_bm"].astype(np.float32)
        return np.broadcast_to((4 * u * (1 - u))[:, None], (n, W)).copy()
    raise ValueError(gate)


def predict(net, X, mu, sd, cols, chunk=20000):
    n = X.shape[0]; out = np.empty(X.shape[:2], np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            x = (np.asarray(X[lo:lo + chunk][:, :, cols], np.float32) - mu) / sd
            out[lo:lo + chunk] = net(torch.from_numpy(x)).numpy()
    return out


def bm_select(Dsel, base_m, fsel, gates, lams, gsel_cache):
    """BridgeMem selection: maximise valid-select MRR s.t. none of MRR/H1/H3/H10 decreases vs the base."""
    grid = []
    for gname in gates:
        g = gsel_cache[gname]
        for lam in lams:
            m = metr(ranks_of(Dsel, Dsel["LP"] + lam * g * fsel))
            feas = all(m[k] >= base_m[k] for k in ("MRR", "H1", "H3", "H10"))
            grid.append(dict(gate=gname, lam=lam, feasible=bool(feas), **m))
    feas = [x for x in grid if x["feasible"]]
    best = max(feas, key=lambda x: x["MRR"]) if feas else dict(gate="none", lam=0.0, feasible=True, **base_m)
    best_unc = max(grid, key=lambda x: x["MRR"])
    return best, best_unc, grid


def calibrate_ab(Dsel, fsel, iters=60):
    m = (Dsel["CA"] >= 0) & ~Dsel["FI"]
    ok = Dsel["Q"]["gold_pos"] >= 0
    lp = torch.tensor(np.where(m, Dsel["LP"], 0.0)[ok], dtype=torch.float64)
    f = torch.tensor(fsel[ok], dtype=torch.float64); mk = torch.tensor(m[ok]); y = torch.tensor(Dsel["Q"]["gold_pos"][ok].astype(np.int64))
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        z = (ab[0] * lp + ab[1] * f).masked_fill(~mk, -1e30)
        loss = torch.nn.functional.cross_entropy(z, y); loss.backward(); return loss
    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="tirgn")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="R0,Rreuse,Rconf,A,R0ev,Aev")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--fit_max", type=int, default=120000)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = a.suffix
    torch.set_num_threads(a.threads)
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
    DS, base = a.dataset, a.base
    res_dir = f"{TRACK}/results/{DS}_{base}{a.suffix}"; os.makedirs(f"{res_dir}/models", exist_ok=True)
    Vall = load_split(DS, base, "valid")
    ts = Vall["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    rng0 = np.random.default_rng(0)
    fit_tr = fit_idx[Vall["Q"]["gold_pos"][fit_idx] >= 0]
    if len(fit_tr) > a.fit_max:
        fit_tr = np.sort(rng0.choice(fit_tr, a.fit_max, replace=False))
    Dfit = subset(Vall, fit_tr); Dsel = subset(Vall, sel_idx)
    del Vall
    Dte = load_split(DS, base, "test")
    log("loaded", "fit", len(fit_tr), "of", len(fit_idx), "select", len(sel_idx), "test", Dte["LP"].shape)
    is_tirgn = base == "tirgn"
    cols_full = GROUPS["base"] + GROUPS["upred"] + GROUPS["evid"] + (GROUPS["bridge"] if is_tirgn else [])
    cols_ev = GROUPS["evid"] + (GROUPS["bridge"] if is_tirgn else [])
    FS = {"full": cols_full, "evid": cols_ev}
    ARM = {"R0": ("resid", "none", "full"), "Rreuse": ("resid", "reuse", "full"), "Rconf": ("resid", "conf", "full"),
           "A": ("stand", "none", "full"), "R0ev": ("resid", "none", "evid"), "Aev": ("stand", "none", "evid")}
    # standardisation over fit candidates
    mfit = (Dfit["CA"] >= 0)
    stats = {}
    for fs, cols in FS.items():
        Z = Dfit["X"][:, :, cols].astype(np.float32)[mfit]
        sub = Z[rng0.choice(len(Z), min(len(Z), 3000000), replace=False)]
        stats[fs] = (sub.mean(0).astype(np.float32), (sub.std(0) + 1e-4).astype(np.float32))
    mfitT = torch.tensor(mfit & ~Dfit["FI"]); yfit = torch.tensor(Dfit["Q"]["gold_pos"].astype(np.int64))
    lpfit = torch.tensor(np.where(mfit, Dfit["LP"], 0.0).astype(np.float32))
    gsel_cache = {g: gate_arr(Dsel, g) for g in ("none", "reuse", "conf", "bm")}
    gte_cache = {g: gate_arr(Dte, g) for g in ("none", "reuse", "conf", "bm")}
    gfit_cache = {g: torch.tensor(gate_arr(Dfit, g)) for g in ("none", "reuse", "conf")}
    base_sel = metr(Dsel["Q"]["r_base"]); base_te = metr(Dte["Q"]["r_base"])
    id_sel = float(np.mean(ranks_of(Dsel, Dsel["LP"]) == Dsel["Q"]["r_base"]))
    id_te = float(np.mean(ranks_of(Dte, Dte["LP"]) == Dte["Q"]["r_base"]))
    out = dict(dataset=DS, base=base, host="serverA (CPU, tkg-gpu:latest, torch %s, %d threads)" % (torch.__version__, a.threads),
               n_fit_rows=int(len(fit_idx)), n_fit_train=int(len(fit_tr)), n_select=int(len(sel_idx)), n_test=int(len(Dte["LP"])),
               valid_cut_ts=int(cut), base_select=base_sel, base_test=base_te, identity_select=id_sel, identity_test=id_te,
               feature_sets={k: [FEAT_NAMES[c] for c in v] for k, v in FS.items()}, arms={}, cfg=vars(a))
    ranks_out = dict(base=Dte["Q"]["r_base"].astype(np.int32), rows=Dte["Q"]["rows"])
    sel_ranks_out = dict(base=Dsel["Q"]["r_base"].astype(np.int32))
    greuse_all_zero = bool((Dfit["GR"][mfit] == 0).all())
    out["greuse_all_zero"] = greuse_all_zero
    seeds = [int(x) for x in a.seeds.split(",")]
    for arm in a.arms.split(","):
        kind, gate, fs = ARM[arm]; cols = FS[fs]; mu, sd = stats[fs]
        rec = dict(kind=kind, gate=gate, features=fs, n_features=len(cols), seeds={})
        if gate == "reuse" and greuse_all_zero:
            rec["degenerate"] = "g(c)=1-1/rolecount(c) is 0 for every candidate (unique roles): fused score == base"
            for sd_ in seeds:
                ranks_out[f"{arm}_s{sd_}"] = Dte["Q"]["r_base"].astype(np.int32)
                sel_ranks_out[f"{arm}_s{sd_}"] = Dsel["Q"]["r_base"].astype(np.int32)
                rec["seeds"][sd_] = dict(select=base_sel, test=base_te, best_epoch=-1)
            out["arms"][arm] = rec; continue
        Xfit = torch.tensor((Dfit["X"][:, :, cols].astype(np.float32) - mu) / sd)
        for sd_ in seeds:
            torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
            net = Net(len(cols)); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
            rec["n_params"] = int(sum(p.numel() for p in net.parameters()))
            nfit = Xfit.shape[0]; hist = []; best = (-1.0, None, -1)
            gfit = gfit_cache[gate] if kind == "resid" else None
            for ep in range(a.epochs):
                net.train(); perm = torch.from_numpy(rng.permutation(nfit)); tl = 0.0; nb = 0
                for b0 in range(0, nfit, a.bs):
                    idx = perm[b0:b0 + a.bs]
                    f = net(Xfit[idx])
                    z = lpfit[idx] + gfit[idx] * f if kind == "resid" else f
                    z = z.masked_fill(~mfitT[idx], -1e30)
                    loss = torch.nn.functional.cross_entropy(z, yfit[idx])
                    opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                net.eval()
                fsel = predict(net, Dsel["X"], mu, sd, cols)
                ssel = Dsel["LP"] + gsel_cache[gate] * fsel if kind == "resid" else fsel
                msel = metr(ranks_of(Dsel, ssel))
                hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                if msel["MRR"] > best[0]:
                    best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                elif ep - best[2] >= a.patience:
                    break
            net.load_state_dict(best[1]); net.eval()
            torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_), f"{res_dir}/models/{arm}_s{sd_}.pt")
            fsel = predict(net, Dsel["X"], mu, sd, cols); fte = predict(net, Dte["X"], mu, sd, cols)
            srec = dict(best_epoch=best[2], history=hist)
            if kind == "resid":
                rs = ranks_of(Dsel, Dsel["LP"] + gsel_cache[gate] * fsel); rt = ranks_of(Dte, Dte["LP"] + gte_cache[gate] * fte)
                srec.update(select=metr(rs), test=metr(rt))
                ranks_out[f"{arm}_s{sd_}"] = rt.astype(np.int32); sel_ranks_out[f"{arm}_s{sd_}"] = rs.astype(np.int32)
            else:
                # standalone ranker alone (diagnostic)
                rs0 = ranks_of(Dsel, fsel); rt0 = ranks_of(Dte, fte)
                srec.update(alone_select=metr(rs0), alone_test=metr(rt0))
                ranks_out[f"{arm}alone_s{sd_}"] = rt0.astype(np.int32)
                # A-style mixing (BridgeMem rule)
                best_m, best_unc, grid = bm_select(Dsel, base_sel, fsel, ("none", "bm"), LAM_A, gsel_cache)
                rt = ranks_of(Dte, Dte["LP"] + best_m["lam"] * gte_cache[best_m["gate"]] * fte)
                rs = ranks_of(Dsel, Dsel["LP"] + best_m["lam"] * gsel_cache[best_m["gate"]] * fsel)
                srec.update(mix_selected=dict(gate=best_m["gate"], lam=best_m["lam"]), mix_select=metr(rs), mix_test=metr(rt),
                            mix_best_unconstrained=dict(gate=best_unc["gate"], lam=best_unc["lam"], MRR=best_unc["MRR"]), mix_grid=grid)
                ranks_out[f"{arm}mix_s{sd_}"] = rt.astype(np.int32); sel_ranks_out[f"{arm}mix_s{sd_}"] = rs.astype(np.int32)
                # B-style post-hoc calibration (valid NLL)
                al, be = calibrate_ab(Dsel, fsel)
                rt = ranks_of(Dte, al * Dte["LP"] + be * fte); rs = ranks_of(Dsel, al * Dsel["LP"] + be * fsel)
                srec.update(cal_alpha=al, cal_beta=be, cal_select=metr(rs), cal_test=metr(rt))
                ranks_out[f"{arm}cal_s{sd_}"] = rt.astype(np.int32); sel_ranks_out[f"{arm}cal_s{sd_}"] = rs.astype(np.int32)
            rec["seeds"][sd_] = srec
            log(arm, "seed", sd_, "best_ep", best[2], json.dumps({k: srec[k]["MRR"] for k in srec if isinstance(srec[k], dict) and "MRR" in srec[k]}))
        out["arms"][arm] = rec
        del Xfit
    # ---- D: BridgeMem-mechanism control (dyadic transition LR residual, feature column ev_dyad_lr) ----
    jd = FEAT_NAMES.index("ev_dyad_lr")
    Asel = np.asarray(Dsel["X"][:, :, jd], np.float32); Ate = np.asarray(Dte["X"][:, :, jd], np.float32)
    best_m, best_unc, grid = bm_select(Dsel, base_sel, Asel, ("none", "bm"), LAM_D, gsel_cache)
    rt = ranks_of(Dte, Dte["LP"] + best_m["lam"] * gte_cache[best_m["gate"]] * Ate)
    rs = ranks_of(Dsel, Dsel["LP"] + best_m["lam"] * gsel_cache[best_m["gate"]] * Asel)
    out["arms"]["D"] = dict(kind="bridgemem_control", selected=dict(gate=best_m["gate"], lam=best_m["lam"]), select=metr(rs),
                            test=metr(rt), best_unconstrained=dict(gate=best_unc["gate"], lam=best_unc["lam"], MRR=best_unc["MRR"]),
                            grid=grid, frac_cands_with_dyad_window=float((Ate[Dte["CA"] >= 0] != 0).mean()))
    ranks_out["D"] = rt.astype(np.int32); sel_ranks_out["D"] = rs.astype(np.int32)
    log("D", json.dumps(out["arms"]["D"]["selected"]), out["arms"]["D"]["test"]["MRR"])
    # ---- COPY diagnostic ----
    jc = FEAT_NAMES.index("ev_log_ntrip")
    Csel = np.asarray(Dsel["X"][:, :, jc], np.float32); Cte = np.asarray(Dte["X"][:, :, jc], np.float32)
    grid = [dict(lam=l, **metr(ranks_of(Dsel, Dsel["LP"] + l * Csel))) for l in LAM_C]
    bc = max(grid, key=lambda x: x["MRR"])
    rt = ranks_of(Dte, Dte["LP"] + bc["lam"] * Cte)
    out["arms"]["COPY"] = dict(kind="copy_prior_diagnostic", selected_lam=bc["lam"], select=dict((k, bc[k]) for k in ("MRR", "H1", "H3", "H10", "H50")),
                               test=metr(rt), grid=grid)
    ranks_out["COPY"] = rt.astype(np.int32)
    np.savez(f"{res_dir}/ranks_test.npz", **ranks_out); np.savez(f"{res_dir}/ranks_select.npz", **sel_ranks_out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_{DS}_{base}{a.suffix}.json"); jdump(out, f"{res_dir}/train_{DS}_{base}.json")
    os.system(f"chmod -R a+rX {res_dir}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
