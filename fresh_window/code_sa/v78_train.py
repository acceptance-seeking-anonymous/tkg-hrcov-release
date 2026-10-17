#!/usr/bin/env python3
"""v78 / v78_train.py -- v7.8 re-ranker arms (v78_rule_amendment_1).  hcpat_train.py recipe generalised: 2x64 MLP (or the
REL_LIN / global linear scorer), listwise masked softmax CE, Adam 1e-3 wd 1e-5, bs 512, <= 30 epochs, patience 5 on the
selection-part MRR of the exact rank under the arm's own tail order, fit rows = fit-part queries with gold in pool (cap 120,000,
seed 0), seeds 0-2; optional calibrated fusion alpha log p_base + beta f (calibrate_ab on selection rows).
Arm = (pool, column set, model, fuse, tail).  Pools: v78_pools.py dirs (25-column RB layout) or 'PT:<base>' = the stored
HRcov pool (hcpat_train.load_split, 25 HRcov features).
Outputs <out>/: ranks_select.npz / ranks_test.npz (harness exact ranks, key <arm>_s<seed>), promlist_<arm>_s<seed>.npy (final
order, official scoring), train_v78.json (selection metrics; test metrics written, not printed).
  python3 v78_train.py --ds polecat --pools P --arms SA_W100,SA_W50,... --out O [--bases tirgn,cogn] [--smoke N]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C

SA16 = list(range(8, 19)) + [19, 21, 22, 23, 24]
SAGEN24 = list(range(0, 8)) + SA16
FULL25 = list(range(0, 25))
PAT25 = list(range(0, 24)) + [29]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def arms_for(bases):
    A = {"SA_W100": ("SA50", SA16, "mlp64", False, "hist"), "SA_W50": ("SA25", SA16, "mlp64", False, "hist"),
         "SA_W150": ("SA75", SA16, "mlp64", False, "hist"), "SA_GEN": ("SA50", SAGEN24, "mlp64", False, "hist")}
    for b in bases:
        A[f"LAD_C_{b}"] = (f"PC_{b}", SA16, "mlp64", False, "hist")
        A[f"LAD_S_{b}"] = (f"PC_{b}", FULL25, "mlp64", True, "hist")
        A[f"LAD_T_{b}"] = (f"PC_{b}", FULL25, "mlp64", True, "base")
        A[f"HR_W50_{b}"] = (f"HU25_{b}", FULL25, "mlp64", True, "base")
        A[f"HR_W150_{b}"] = (f"HU75_{b}", FULL25, "mlp64", True, "base")
        A[f"REL_LIN_{b}"] = (f"PT:{b}", PAT25, "rellin", True, "base")
        A[f"REL_LIN_global_{b}"] = (f"PT:{b}", PAT25, "lin", True, "base")
    return A


def metr(rk, ks=(1, 3, 10, 50)):
    rk = np.asarray(rk, np.float64); out = dict(MRR=float(np.mean(1.0 / rk)))
    for k in ks: out[f"H{k}"] = float(np.mean(rk <= k))
    out["n"] = int(len(rk)); return out


def fused_ranks_float(s, cand, filt, babove, gold_pos, r_tail, kc):
    n, W = s.shape; valid = cand >= 0; s = np.where(valid, s, -np.inf); mint = s[:, :kc].min(1)
    jj = np.arange(W)[None, :]; prom = valid & ((jj < kc) | (s > mint[:, None])); unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0); sg = s[ar, g]; gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & (s > sg[:, None]); above[ar, g] = False
    r_in = 1 + above.sum(1); add = (unf & (jj >= kc) & ~babove).sum(1)
    return np.where(gin, r_in, r_tail + add).astype(np.float64)


def prom_lists(CA, s, kc):
    n, W = s.shape; valid = CA >= 0; s = np.where(valid, s, -np.inf); mint = s[:, :kc].min(1)
    prom = valid & ((np.arange(W)[None, :] < kc) | (s > mint[:, None])); key = np.where(prom, s, -np.inf)
    o = np.argsort(-key, axis=1, kind="stable"); ids = np.take_along_axis(CA, o, 1); pk = np.take_along_axis(prom, o, 1)
    return np.where(pk, ids, -1).astype(np.int32)


class RelLin(torch.nn.Module):
    def __init__(self, d, nrel, offsets=True):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(d)); self.b = torch.nn.Parameter(torch.zeros(1)); self.offsets = offsets
        if offsets:
            self.D = torch.nn.Parameter(torch.zeros(nrel, d)); self.c = torch.nn.Parameter(torch.zeros(nrel))

    def forward(self, x, rel):
        w = self.w.unsqueeze(0) + (self.D[rel] if self.offsets else 0.0)
        z = (x * w.unsqueeze(1)).sum(-1) + self.b
        return z + (self.c[rel].unsqueeze(1) if self.offsets else 0.0)

    def penalty(self):
        return (self.D ** 2).sum() + (self.c ** 2).sum() if self.offsets else torch.zeros((), device=self.w.device)


class MLP(torch.nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        torch.nn.init.zeros_(self.f[-1].weight); torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, x, rel=None):
        return self.f(x).squeeze(-1)

    def penalty(self):
        return torch.zeros((), device=self.f[0].weight.device)


def load_v78(pdir, split, tail, n_lim):
    Q = dict(np.load(f"{pdir}/{split}_q.npz")); K = int(Q["K"]); n = len(Q["rows"]) if not n_lim else min(n_lim, len(Q["rows"]))
    CA = np.load(f"{pdir}/{split}_cand.npy")[:n].astype(np.int64)
    BA = np.load(f"{pdir}/{split}_bah.npy")[:n] if tail == "hist" else np.load(f"{pdir}/{split}_bab.npy")[:n]
    rt = Q["r_hist"][:n] if tail == "hist" else Q["r_base"][:n]
    return dict(X=np.load(f"{pdir}/{split}_X.npy", mmap_mode="r"), LP=np.load(f"{pdir}/{split}_logp.npy")[:n], CA=CA,
                FI=np.load(f"{pdir}/{split}_filt.npy")[:n] & (CA >= 0), BA=BA & (CA >= 0), gp=Q["gold_pos"][:n].astype(np.int64),
                rt=rt.astype(np.float64), sro=Q["sro"][:n].astype(np.int64), K=K)


def load_pt(bcfg, layout, split, n_lim):
    """stored HRcov pool (hcpat_train.load_split): bridge columns masked, W = 100, base tail."""
    pre = C.feat_pre(bcfg, layout, split); Q = dict(np.load(f"{bcfg['feat']}/{split}_q.npz"))
    n = len(Q["rows"]) if not n_lim else min(n_lim, len(Q["rows"]))
    if layout == "sp": npb = Q["n_pat_block"].astype(np.int64)[:n]; gp = Q["gold_pos_A"].astype(np.int64)[:n]
    else:
        Lq = np.load(f"{bcfg['lists']}/{split}_q.npz"); assert (Lq["rows"][:len(Q["rows"])] == Q["rows"]).all()
        npb = Lq["n_pat_block"][:n].astype(np.int64); gp = Q["gold_pos"].astype(np.int64)[:n]
    CA = np.load(pre + "cand.npy", mmap_mode="r")[:n, :100].astype(np.int64); j = np.arange(100)[None, :]
    CA = np.where((j >= 50) & (j >= 50 + npb[:, None]), -1, CA)
    LP = np.asarray(np.load(pre + "logp.npy", mmap_mode="r")[:n, :100]); LP = np.where(CA >= 0, LP, -np.inf).astype(np.float32)
    gp = np.where((gp >= 0) & (gp < 100) & ((gp < 50) | (gp < 50 + npb)), gp, -1)
    sro = Q["sro"][:n].astype(np.int64); isg = (CA == sro[:, 2:3]) & (CA >= 0)
    assert (np.where(isg.any(1), isg.argmax(1), -1) == gp).all(), "PT gold position"
    return dict(X=np.load(pre + "X.npy", mmap_mode="r"), LP=LP, CA=CA, FI=np.asarray(np.load(pre + "filt.npy", mmap_mode="r")[:n, :100]) & (CA >= 0),
                BA=np.asarray(np.load(pre + "babove.npy", mmap_mode="r")[:n, :100]) & (CA >= 0), gp=gp, rt=Q["r_base"][:n].astype(np.float64), sro=sro, K=50)


def feats(D, idx, cols, W, chunk=40000):
    return np.concatenate([np.asarray(D["X"][idx[lo:lo + chunk]][:, :W][:, :, cols], np.float32) for lo in range(0, len(idx), chunk)])


def predict(net, D, idx, cols, W, mu, sd, rel, dev, chunk=20000):
    out = np.empty((len(idx), W), np.float32)
    with torch.no_grad():
        for lo in range(0, len(idx), chunk):
            ii = idx[lo:lo + chunk]
            x = (np.asarray(D["X"][ii][:, :W][:, :, cols], np.float32) - mu) / sd
            out[lo:lo + chunk] = net(torch.from_numpy(x).to(dev), torch.as_tensor(rel[ii], device=dev)).cpu().numpy()
    return out


def calibrate_ab(LP, CA, FI, gp, f, iters=60):
    m = (CA >= 0) & ~FI; ok = gp >= 0
    lp = torch.tensor(np.where(m, LP, 0.0)[ok], dtype=torch.float64); ff = torch.tensor(f[ok], dtype=torch.float64)
    mk = torch.tensor(m[ok]); y = torch.tensor(gp[ok])
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); z = (ab[0] * lp + ab[1] * ff).masked_fill(~mk, -1e30)
        loss = Fn.cross_entropy(z, y); loss.backward(); return loss
    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(C.DS)); ap.add_argument("--pools", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_max", type=int, default=120000); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); log = C.log_fn(t0)
    cfg = C.DS[a.ds]; dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("V7_THREADS", "8")))
    ARMS = arms_for(list(cfg["bases"])); arms = a.arms.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(f"{a.out}/models", exist_ok=True)
    out = dict(ds=a.ds, rule="v78_framing_rule + v78_rule_amendment_1", host=f"{os.environ.get('HD_HOST', '')} ({dev}, torch {torch.__version__})",
               cfg=vars(a), arms={})
    rk_path = f"{a.out}/ranks_test.npz"; sel_path = f"{a.out}/ranks_select.npz"
    ranks_out = dict(np.load(rk_path)) if os.path.exists(rk_path) else {}; sel_out = dict(np.load(sel_path)) if os.path.exists(sel_path) else {}
    if os.path.exists(f"{a.out}/train_v78.json"): out["arms"] = json.load(open(f"{a.out}/train_v78.json"))["arms"]
    cache = {}
    for arm in arms:
        pool, cols, model, fuse, tail = ARMS[arm]
        key = (pool, tail)
        if key not in cache:
            cache.clear()
            if pool.startswith("PT:"):
                b = pool.split(":")[1]; Dv = load_pt(cfg["bases"][b], cfg["layout"], "valid", 0); Dt = load_pt(cfg["bases"][b], cfg["layout"], "test", a.smoke)
            else:
                Dv = load_v78(f"{a.pools}/{pool}", "valid", tail, 0); Dt = load_v78(f"{a.pools}/{pool}", "test", tail, a.smoke)
            cache[key] = (Dv, Dt)
        Dv, Dt = cache[key]; K = Dv["K"]; W = 2 * K if not pool.startswith("PT:") else 100
        ts = Dv["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
        fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]; te_idx = np.arange(len(Dt["gp"]))
        if a.smoke: fit_idx = fit_idx[:a.smoke]; sel_idx = sel_idx[:a.smoke // 2]
        gpv = Dv["gp"][fit_idx]; fit_tr = fit_idx[gpv >= 0]; n_ret = len(fit_tr)
        if len(fit_tr) > a.fit_max: fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
        V = lambda Dd, idx: dict(LP=Dd["LP"][idx], CA=Dd["CA"][idx], FI=Dd["FI"][idx], BA=Dd["BA"][idx], gp=Dd["gp"][idx], rt=Dd["rt"][idx])
        Vf, Vs, Vt = V(Dv, fit_tr), V(Dv, sel_idx), V(Dt, te_idx)
        relv = Dv["sro"][:, 1]; relt = Dt["sro"][:, 1]; nrel = 2 * cfg["R"]
        x = feats(Dv, fit_tr, cols, W)
        m = Vf["CA"] >= 0; Z = x[m]; subz = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
        mu = subz.mean(0).astype(np.float32); sd = (subz.std(0) + 1e-4).astype(np.float32); del Z, subz
        Xfit = torch.tensor((x - mu) / sd, device=dev); del x
        mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev); rfit = torch.as_tensor(relv[fit_tr], device=dev)
        if tail == "base":
            ident = float(np.mean(np.abs(fused_ranks_float(np.where(Vt["CA"] >= 0, Vt["LP"], -np.inf), Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K) - Vt["rt"]) < 1e-9))
        elif pool.startswith("SA"):
            ident = float(np.mean(np.abs(fused_ranks_float(-np.tile(np.arange(W, dtype=np.float64), (len(te_idx), 1)), Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K) - Vt["rt"]) < 1e-9))
        else:
            ident = None
        rec = dict(pool=pool, cols=cols, model=model, fuse=fuse, tail=tail, K=K, W=W, n_features=len(cols), valid_cut_ts=int(cut),
                   n_fit_retained=int(n_ret), n_fit_queries=int(len(fit_tr)), n_select=int(len(sel_idx)), n_test=int(len(te_idx)),
                   identity_tail_order_test=ident, recall_select=float((Vs["gp"] >= 0).mean()), seeds={})
        log(arm, pool, "K", K, "W", W, "fit", len(fit_tr), "sel", len(sel_idx), "test", len(te_idx), "identity", ident)
        lam_grid = LAMBDAS if model == "rellin" else [None]
        cand_runs = {}
        for lam in lam_grid:
            for sd_ in seeds:
                torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
                net = (MLP(len(cols)) if model == "mlp64" else RelLin(len(cols), nrel, offsets=(model == "rellin"))).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5); hist = []; best = (-1.0, None, -1); nfit = len(fit_tr)
                for ep in range(a.epochs if not a.smoke else 2):
                    net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
                    for b0 in range(0, nfit, a.bs):
                        idx = perm[b0:b0 + a.bs]; z = net(Xfit[idx], rfit[idx]).masked_fill(~mfit[idx], -1e30)
                        loss = Fn.cross_entropy(z, yfit[idx])
                        if lam is not None: loss = loss + lam * net.penalty()
                        opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                    net.eval(); fsel = predict(net, Dv, sel_idx, cols, W, mu, sd, relv, dev)
                    msel = metr(fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K))
                    hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                    if msel["MRR"] > best[0]: best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                    elif ep - best[2] >= a.patience: break
                net.load_state_dict(best[1]); net.eval()
                fsel = predict(net, Dv, sel_idx, cols, W, mu, sd, relv, dev); fte = predict(net, Dt, te_idx, cols, W, mu, sd, relt, dev)
                srec = dict(best_epoch=best[2], history=hist, lam=lam)
                rs_a = fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K)
                rt_a = fused_ranks_float(fte, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K)
                srec.update(alone_select=metr(rs_a), alone_test=metr(rt_a))
                if fuse:
                    al, be = calibrate_ab(Vs["LP"], Vs["CA"], Vs["FI"], Vs["gp"], fsel); srec.update(cal_alpha=al, cal_beta=be)
                    ssel = al * np.where(np.isfinite(Vs["LP"]), Vs["LP"], -1e9) + be * fsel; ste = al * np.where(np.isfinite(Vt["LP"]), Vt["LP"], -1e9) + be * fte
                else:
                    ssel, ste = fsel, fte
                rs = fused_ranks_float(ssel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K)
                rt = fused_ranks_float(ste, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K)
                srec.update(select=metr(rs), test=metr(rt))
                vte = np.where(Vt["CA"] >= 0, ste, -np.inf); mint = vte[:, :K].min(1)
                promc = (vte[:, K:] > mint[:, None]) & (Vt["CA"][:, K:] >= 0)
                srec.update(test_promoted_block_per_query=float(promc.sum(1).mean()), test_frac_with_promotion=float(promc.any(1).mean()))
                cand_runs[(lam, sd_)] = (srec, rs, rt, prom_lists(Vt["CA"], ste, K), best[1], net)
                log(arm, "lam", lam, "seed", sd_, "best_ep", best[2], "select MRR", round(srec["select"]["MRR"], 5))
        if model == "rellin":
            msel = {lam: np.mean([cand_runs[(lam, s)][0]["select"]["MRR"] for s in seeds]) for lam in lam_grid}
            lam_star = max(lam_grid, key=lambda l: msel[l]); rec["lambda_select_MRR"] = {str(k): v for k, v in msel.items()}; rec["lambda_star"] = lam_star
            rec["lambda_grid_seed_records"] = {str(l): {s: dict(select=cand_runs[(l, s)][0]["select"]) for s in seeds} for l in lam_grid}
        else:
            lam_star = None
        for sd_ in seeds:
            srec, rs, rt, pl, st, net = cand_runs[(lam_star, sd_)]
            torch.save(dict(state_dict=st, mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_, model=model, lam=lam_star), f"{a.out}/models/{arm}_s{sd_}.pt")
            np.save(f"{a.out}/promlist_{arm}_s{sd_}.npy", pl)
            ranks_out[f"{arm}_s{sd_}"] = rt.astype(np.float32); sel_out[f"{arm}_s{sd_}"] = rs.astype(np.float32)
            rec["seeds"][sd_] = srec
        rec["n_params"] = int(sum(p.numel() for p in cand_runs[(lam_star, seeds[0])][5].parameters()))
        out["arms"][arm] = rec
        del Xfit, mfit, yfit, rfit, cand_runs
        np.savez(rk_path, **ranks_out); np.savez(sel_path, **sel_out); C.jdump(out, f"{a.out}/train_v78.json")
        log(arm, "done; select MRR per seed", [round(rec["seeds"][s]["select"]["MRR"], 5) for s in seeds])
    out["elapsed_s"] = round(time.time() - t0, 1)
    C.jdump(out, f"{a.out}/train_v78.json"); C.jdump(out, f"train_v78_{a.ds}_{arms[0]}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
