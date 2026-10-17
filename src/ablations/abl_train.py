#!/usr/bin/env python3
"""ablations_A / abl_train.py -- POST-HOC ABLATION (not a decision; ablation_rule.json): hcpat_train.py with feature-group
column subsets on the HRCOV_PAT candidate set (W 100): FULL (25), A10_noBase (= A11 minus base, 17), A11_noEvid (14), A11_noCov (19).
Original docstring: presub / hcpat_train.py -- DEFINITIONAL RE-RUN (not part of the hardening decision): HRCOV_PAT on an
existing hardening HRCOV feature directory.  HRCOV_PAT = the HRCOV recipe without the bridge block: candidates = base top-50
++ the first <= 50 pattern-prior candidates (columns >= 50 + n_pat_block of the HRCOV layout, i.e. bridge candidates, are
masked), W = 100; features = 8 base + 11 evidence + the 6 pattern coverage features (pat_best_util, pat_best_te,
pat_log_npat, pat_reach, pat_log_pos, is_cov_cand); the 5 bridge features (br_*) are dropped.  Same definition as the
pre-registered G2 HRCOV_PAT.  Re-ranker code identical to g2_train.py / g1_train.py (= pc_train.py core); seeds 0-2.
  python3 hcpat_train.py --layout pc --feat <pc feat dir> --lists <pc lists dir> --out <dir> --label polecat_tirgn
  python3 hcpat_train.py --layout sp --feat <sp feat dir> --out <dir> --label smallpedia_regcn_tuned
"""
import os, sys, json, time, argparse
import numpy as np
import torch

KC = 50
SEEDS = [0, 1, 2]
ARMS = {"FULL": ("stand", 100, "pat"), "A10_noBase": ("stand", 100, "noBase"), "A11_noEvid": ("stand", 100, "noEvid"),
        "A11_noCov": ("stand", 100, "noCov")}
COLS = {"pat": list(range(0, 24)) + [29],                       # HRCOV layout: 0-7 base, 8-18 evidence, 19-23 pattern cov, 24-28 bridge, 29 is_cov_cand
        "noBase": list(range(8, 24)) + [29], "noEvid": list(range(0, 8)) + list(range(19, 24)) + [29], "noCov": list(range(0, 19))}
SAVE_LISTS = tuple(ARMS)
LAYOUT = {}


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def metrics_from_ranks(rk, ks=(1, 3, 10)):
    rk = np.asarray(rk, np.float64)
    out = dict(MRR=float(np.mean(1.0 / rk)))
    for k in ks: out[f"H{k}"] = float(np.mean(rk <= k))
    out["n"] = int(len(rk)); return out


def fused_ranks_float(s, cand, filt, babove, gold_pos, r_base, kc=KC):
    """pc_common.exact_fused_ranks with a float (average-tie) r_base for golds outside the promoted list."""
    n, W = s.shape
    valid = cand >= 0
    s = np.where(valid, s, -np.inf)
    mint = s[:, :kc].min(1)
    jj = np.arange(W)[None, :]
    prom = valid & ((jj < kc) | (s > mint[:, None]))
    unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0)
    sg = s[ar, g]
    gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & (s > sg[:, None]); above[ar, g] = False
    r_in = 1 + above.sum(1)
    add = (unf & (jj >= kc) & ~babove).sum(1)
    r_out = r_base + add
    return np.where(gin, r_in, r_out).astype(np.float64)


class Net(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(),
                                     torch.nn.Linear(h, 1))
        torch.nn.init.zeros_(self.f[-1].weight); torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load_split(fd, split):
    """HRCOV feature dir -> pc_train-layout dict, bridge columns masked (CA = -1, LP = -inf), gold_pos recomputed."""
    L = LAYOUT
    pre = f"{fd}/{split}_" + ("A_" if L["layout"] == "sp" else "")
    Q = dict(np.load(f"{fd}/{split}_q.npz"))
    if L["layout"] == "sp":
        npb = Q["n_pat_block"].astype(np.int64); gp = Q["gold_pos_A"].astype(np.int64)
    else:
        Lq = np.load(f"{L['lists']}/{split}_q.npz")
        assert (Lq["rows"][:len(Q["rows"])] == Q["rows"]).all(), "lists / features row order"
        npb = Lq["n_pat_block"][:len(Q["rows"])].astype(np.int64); gp = Q["gold_pos"].astype(np.int64)
    CA = np.load(pre + "cand.npy")[:, :100].astype(np.int64)
    j = np.arange(CA.shape[1])[None, :]
    bridge = (j >= 50) & (j >= 50 + npb[:, None])
    CA = np.where(bridge, -1, CA)
    LP = np.load(pre + "logp.npy")[:, :100]; LP = np.where(CA >= 0, LP, -np.inf).astype(np.float32)
    gp = np.where((gp >= 0) & (gp < 100) & ((gp < 50) | (gp < 50 + npb)), gp, -1)
    o = Q["sro"][:, 2]; isg = (CA == o[:, None]) & (CA >= 0)
    assert (np.where(isg.any(1), isg.argmax(1), -1) == gp).all(), "gold position after masking"
    Q["gold_pos"] = gp; Q["n_pat_block"] = npb
    X = np.load(pre + "X.npy", mmap_mode="r")
    return dict(X=X, LP=LP, CA=CA.astype(np.int32), FI=np.load(pre + "filt.npy")[:, :100], BA=np.load(pre + "babove.npy")[:, :100], Q=Q)


def subset(Dd, idx):
    o = {k: (v[idx] if k not in ("Q", "X") else None) for k, v in Dd.items()}
    o["X"] = np.asarray(Dd["X"][idx]); o["Q"] = {k: v[idx] for k, v in Dd["Q"].items()}
    return o


def view(Dd, W):
    """restrict to the first W candidate columns; gold_pos >= W -> -1 (gold outside this arm's set)."""
    gp = Dd["Q"]["gold_pos"].astype(np.int64)
    return dict(LP=Dd["LP"][:, :W], CA=Dd["CA"][:, :W], FI=Dd["FI"][:, :W], BA=Dd["BA"][:, :W],
                gp=np.where(gp < W, gp, -1), rb=Dd["Q"]["r_base"].astype(np.float64))


def ranks_of(V, s):
    return fused_ranks_float(s, V["CA"], V["FI"], V["BA"], V["gp"], V["rb"])


def metr(rk):
    return metrics_from_ranks(rk, ks=(1, 3, 10, 50))


def predict(net, X, mu, sd, cols, W, dev, chunk=20000):
    n = X.shape[0]; out = np.empty((n, W), np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            x = (np.asarray(X[lo:lo + chunk, :W][:, :, cols], np.float32) - mu) / sd
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
    """promoted candidate ids in final fused order (-1 pad) -> official-evaluator systems input."""
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
    ap.add_argument("--feat", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layout", required=True, choices=["pc", "sp"]); ap.add_argument("--lists", default=""); ap.add_argument("--label", required=True)
    ap.add_argument("--arms", default="FULL,A10_noBase,A11_noEvid,A11_noCov")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_max", type=int, default=120000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    LAYOUT.update(layout=a.layout, lists=a.lists)
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("G2_THREADS", "8")))
    sfx = "_smoke" if a.smoke else ""
    fd = a.feat; rd = a.out; os.makedirs(f"{rd}/models", exist_ok=True)
    Vall = load_split(fd, "valid")
    ts = Vall["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    if a.smoke:                                   # smoke rows span ~1 timestamp: alternate rows instead
        fit_idx = np.arange(0, len(ts), 2); sel_idx = np.arange(1, len(ts), 2)
    Dsel = subset(Vall, sel_idx)
    Dte = load_split(fd, "test")
    log("loaded valid", len(ts), "fit rows", len(fit_idx), "select", len(sel_idx), "test", Dte["LP"].shape, "device", dev)
    out = dict(label=a.label, layout=a.layout, definition="POST-HOC ABLATION on the HRCOV_PAT candidate set (W 100, bridge columns masked); arms = feature-group column subsets", cols={k: COLS[ARMS[k][2]] for k in ARMS}, host=f"{os.environ.get('HD_HOST', '')} ({dev}, torch {torch.__version__})", valid_cut_ts=int(cut),
               n_fit_rows=int(len(fit_idx)), n_select=int(len(sel_idx)), n_test=int(len(Dte["LP"])), cfg=vars(a), arms={})
    WS = sorted({ARMS[x][1] for x in a.arms.split(",")} | {KC})
    Vte = {W: view(Dte, W) for W in WS}; Vsel = {W: view(Dsel, W) for W in WS}
    out["base_select"] = metr(Vsel[KC]["rb"]); out["base_test"] = metr(Vte[KC]["rb"])
    out["identity_test"] = {W: float(np.mean(np.abs(ranks_of(Vte[W], Vte[W]["LP"]) - Vte[W]["rb"]) < 1e-9)) for W in WS}
    log("identity", out["identity_test"])
    ranks_out = dict(base=Vte[KC]["rb"].astype(np.float32)); sel_out = dict(base=Vsel[KC]["rb"].astype(np.float32))
    seeds = [int(x) for x in a.seeds.split(",")]
    fit_cache = {}
    for arm in a.arms.split(","):
        kind, W, cs = ARMS[arm]; cols = COLS[cs]
        gpv = Vall["Q"]["gold_pos"][fit_idx].astype(np.int64)
        fit_tr = fit_idx[(gpv >= 0) & (gpv < W)]
        if len(fit_tr) > a.fit_max:
            fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
        if len(fit_tr) < 10:
            out["arms"][arm] = dict(kind=kind, W=W, features=cs, skipped=f"only {len(fit_tr)} fit queries"); continue
        if (W, cs) not in fit_cache:
            fit_cache.clear()
            Dfit = subset(Vall, fit_tr); Vf = view(Dfit, W)
            m = Vf["CA"] >= 0
            Z = Dfit["X"][:, :W][:, :, cols].astype(np.float32)[m]
            sub = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
            mu = sub.mean(0).astype(np.float32); sd = (sub.std(0) + 1e-4).astype(np.float32)
            Xfit = torch.tensor((Dfit["X"][:, :W][:, :, cols].astype(np.float32) - mu) / sd, device=dev)
            mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev)
            lpfit = torch.tensor(np.where(m, Vf["LP"], 0.0).astype(np.float32), device=dev)
            fit_cache[(W, cs)] = (Xfit, mfit, yfit, lpfit, mu, sd, len(fit_tr))
            del Z, sub, Dfit
        Xfit, mfit, yfit, lpfit, mu, sd, nfit = fit_cache[(W, cs)]
        rec = dict(kind=kind, W=W, features=cs, n_features=len(cols), n_fit_queries=int(nfit), seeds={})
        for sd_ in seeds:
            torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
            net = Net(len(cols)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
            hist = []; best = (-1.0, None, -1)
            for ep in range(a.epochs if not a.smoke else 2):
                net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
                for b0 in range(0, nfit, a.bs):
                    idx = perm[b0:b0 + a.bs]
                    f = net(Xfit[idx])
                    z = lpfit[idx] + f if kind == "resid" else f
                    z = z.masked_fill(~mfit[idx], -1e30)
                    loss = torch.nn.functional.cross_entropy(z, yfit[idx])
                    opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                net.eval()
                fsel = predict(net, Dsel["X"], mu, sd, cols, W, dev)
                ssel = Vsel[W]["LP"] + fsel if kind == "resid" else fsel
                msel = metr(ranks_of(Vsel[W], ssel))
                hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                if msel["MRR"] > best[0]:
                    best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                elif ep - best[2] >= a.patience:
                    break
            net.load_state_dict(best[1]); net.eval()
            torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_), f"{rd}/models/{arm}_s{sd_}.pt")
            fsel = predict(net, Dsel["X"], mu, sd, cols, W, dev); fte = predict(net, Dte["X"], mu, sd, cols, W, dev)
            srec = dict(best_epoch=best[2], history=hist)
            if kind == "resid":
                ssel = Vsel[W]["LP"] + fsel; ste = Vte[W]["LP"] + fte
            else:
                srec.update(alone_select=metr(ranks_of(Vsel[W], fsel)), alone_test=metr(ranks_of(Vte[W], fte)))
                al, be = calibrate_ab(Vsel[W], fsel)
                srec.update(cal_alpha=al, cal_beta=be)
                ssel = al * Vsel[W]["LP"] + be * fsel; ste = al * Vte[W]["LP"] + be * fte
            rs = ranks_of(Vsel[W], ssel); rt = ranks_of(Vte[W], ste)
            srec.update(select=metr(rs), test=metr(rt))
            # promoted coverage candidates (how often the coverage block enters the final list)
            if W > KC:
                vte = Vte[W]; s_ = np.where(vte["CA"] >= 0, ste, -np.inf); mint = s_[:, :KC].min(1)
                promc = (s_[:, KC:] > mint[:, None]) & (vte["CA"][:, KC:] >= 0)
                srec["test_promoted_cov_per_query"] = float(promc.sum(1).mean()); srec["test_frac_queries_with_promotion"] = float(promc.any(1).mean())
            ranks_out[f"{arm}_s{sd_}"] = rt.astype(np.float32); sel_out[f"{arm}_s{sd_}"] = rs.astype(np.float32)
            if arm in SAVE_LISTS:
                np.save(f"{rd}/promlist_{arm}_s{sd_}.npy", prom_lists(Vte[W], ste))
            rec["seeds"][sd_] = srec
            log(arm, "seed", sd_, "best_ep", best[2], "select", srec["select"]["MRR"], "test", srec["test"]["MRR"])
        rec["n_params"] = int(sum(p.numel() for p in net.parameters()))
        out["arms"][arm] = rec
    np.savez(f"{rd}/ranks_test.npz", **ranks_out); np.savez(f"{rd}/ranks_select.npz", **sel_out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_abl_{a.label}{sfx}.json"); jdump(out, f"{rd}/train_abl.json")
    os.system(f"chmod -R a+rX {rd}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
