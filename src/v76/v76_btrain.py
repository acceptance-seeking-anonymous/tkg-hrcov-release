#!/usr/bin/env python3
"""v76 / v76_btrain.py = v7_btrain.py + M4 arms (v76_rule.json): pool PT50 (PT pool columns 50-99 masked = base top-50),
column sets with 3 extra RecB features (+RBtr: trained RecB, +RBxf: cross-fitted RecB; v76_rbfeat.py {split}_RB.npy over the
stored PT candidate tensor), arms HR19 / HR19_RBtr / HR19_RBxf / PAT_RBtr / PAT_RBxf.  Everything else identical to v7.
v7 / v7_btrain.py -- B1 / B2 / B3 / C1 / A5 re-ranker arms (tois_v7_rule.json), hcpat_train.py recipe generalised.
Pools: PT = stored HRCOV_PAT pool (base top-50 ++ <= 50 pattern-prior, bridge columns masked; pc/sp feature layout, 30 cols)
       RB = base top-50 ++ <= 50 RecB-trained candidates (v7_rbfeat.py, 25 cols: 0-18 base+evid, 19 is_extra, 20-24 RecB feats)
Every arm: same fit rows (valid ts < 70% cut, gold in the arm's pool, cap 120,000 seed 0), selection rows (valid ts >= cut),
listwise loss, Adam lr 1e-3 wd 1e-5, bs 512, <= 30 epochs, patience 5 on selection MRR of the ranker score, seeds 0-2,
calibrated alpha/beta fusion with the base log-prob (LBFGS on selection rows).  Reported variants: 'alone' (ranker score)
and 'fused' (alpha log p + beta f); QBLEND has its own score.  Exact fused rank (= the evaluator's statistic) for select/test;
promoted lists written for official scoring.
  python3 v7_btrain.py --layout pc --feat F --lists L --rb RBDIR --out O --label polecat_tirgn --arms MLP64_PAT,LIN_PAT,...
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as Fn

KC = 50
ARMS = {"MLP64_PAT": ("PT", "pat25", "mlp64", "masked"), "LIN_PAT": ("PT", "pat25", "lin", "masked"),
        "MLP16_PAT": ("PT", "pat25", "mlp16", "masked"), "QBLEND": ("PT", "qblend", "qblend", "masked"),
        "PT_common": ("PT", "pt_common", "mlp64", "masked"), "RB_common": ("RB", "rb_common", "mlp64", "masked"),
        "RB_gen": ("RB", "rb_gen", "mlp64", "masked"), "NB_PAT": ("PT", "nb17", "mlp64", "masked"),
        "EV_PT": ("PT", "ev_pt", "mlp64", "masked"), "EV_RB": ("RB", "ev_rb", "mlp64", "masked"),
        "noDyadLR": ("PT", "nodyad", "mlp64", "masked"), "UNMASKED": ("PT", "pat25", "mlp64", "unmasked"),
        "MULTIPOS": ("PT", "pat25", "mlp64", "multipos"),
        "HR19": ("PT50", "hr19", "mlp64", "masked"), "HR19_RBtr": ("PT50", "hr19+RBtr", "mlp64", "masked"),
        "HR19_RBxf": ("PT50", "hr19+RBxf", "mlp64", "masked"), "PAT_RBtr": ("PT", "pat25+RBtr", "mlp64", "masked"),
        "PAT_RBxf": ("PT", "pat25+RBxf", "mlp64", "masked"),
        "C2_UTIL12": ("C2:UTIL12", "rb_gen", "mlp64", "masked"), "C2_CONF12": ("C2:CONF12", "rb_gen", "mlp64", "masked"),
        "C2_UTIL1": ("C2:UTIL1", "rb_gen", "mlp64", "masked"), "C2_UTIL2": ("C2:UTIL2", "rb_gen", "mlp64", "masked")}
PAT25 = list(range(0, 24)) + [29]
COLS = {"pat25": PAT25, "pt_common": list(range(0, 19)) + [29], "nb17": list(range(8, 24)) + [29], "ev_pt": list(range(8, 19)) + [29],
        "nodyad": [c for c in PAT25 if c != 16], "rb_common": list(range(0, 20)), "rb_gen": list(range(0, 25)), "ev_rb": list(range(8, 20)),
        "qblend": [4, 5, 6, 7, 17, 18], "hr19": list(range(0, 19))}
ALONE_LISTS = ("MLP64_PAT", "NB_PAT", "EV_PT", "EV_RB")


def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def metr(rk, ks=(1, 3, 10, 50)):
    rk = np.asarray(rk, np.float64); out = dict(MRR=float(np.mean(1.0 / rk)))
    for k in ks: out[f"H{k}"] = float(np.mean(rk <= k))
    out["n"] = int(len(rk)); return out


def fused_ranks_float(s, cand, filt, babove, gold_pos, r_base, kc=KC):
    n, W = s.shape; valid = cand >= 0; s = np.where(valid, s, -np.inf); mint = s[:, :kc].min(1)
    jj = np.arange(W)[None, :]; prom = valid & ((jj < kc) | (s > mint[:, None])); unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0); sg = s[ar, g]; gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & (s > sg[:, None]); above[ar, g] = False
    r_in = 1 + above.sum(1); add = (unf & (jj >= kc) & ~babove).sum(1)
    return np.where(gin, r_in, r_base + add).astype(np.float64)


def prom_lists(CA, s):
    n, W = s.shape; valid = CA >= 0; s = np.where(valid, s, -np.inf); mint = s[:, :KC].min(1)
    prom = valid & ((np.arange(W)[None, :] < KC) | (s > mint[:, None])); key = np.where(prom, s, -np.inf)
    o = np.argsort(-key, axis=1, kind="stable"); ids = np.take_along_axis(CA, o, 1); pk = np.take_along_axis(prom, o, 1)
    return np.where(pk, ids, -1).astype(np.int32)


def make_net(kind, d):
    if kind == "lin":
        net = torch.nn.Sequential(torch.nn.Linear(d, 1))
    elif kind == "mlp16":
        net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    elif kind == "qblend":
        net = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    else:
        net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    return net


def load_pt(fd, lists, layout, rbdir, split, extra=None):
    pre = f"{fd}/{split}_" + ("A_" if layout == "sp" else "")
    Q = dict(np.load(f"{fd}/{split}_q.npz"))
    if layout == "sp":
        npb = Q["n_pat_block"].astype(np.int64); gp = Q["gold_pos_A"].astype(np.int64)
    else:
        Lq = np.load(f"{lists}/{split}_q.npz"); npb = Lq["n_pat_block"][:len(Q["rows"])].astype(np.int64); gp = Q["gold_pos"].astype(np.int64)
    CA = np.load(pre + "cand.npy")[:, :100].astype(np.int64); j = np.arange(100)[None, :]
    CA = np.where((j >= 50) & (j >= 50 + npb[:, None]), -1, CA)
    LP = np.load(pre + "logp.npy")[:, :100]; LP = np.where(CA >= 0, LP, -np.inf).astype(np.float32)
    gp = np.where((gp >= 0) & (gp < 100) & ((gp < 50) | (gp < 50 + npb)), gp, -1)
    o = Q["sro"][:, 2]; isg = (CA == o[:, None]) & (CA >= 0)
    assert (np.where(isg.any(1), isg.argmax(1), -1) == gp).all(), "PT gold position"
    out = dict(X=np.load(pre + "X.npy", mmap_mode="r"), LP=LP, CA=CA.astype(np.int32), FI=np.load(pre + "filt.npy")[:, :100] & (CA >= 0),
               BA=np.load(pre + "babove.npy")[:, :100] & (CA >= 0), gp=gp, rb=Q["r_base"].astype(np.float64), ts=Q["sro"][:, 3].astype(np.int64))
    if rbdir and os.path.exists(f"{rbdir}/{split}_PT_recbF.npy"):
        out["RF"] = np.load(f"{rbdir}/{split}_PT_recbF.npy", mmap_mode="r")
    for nm, d_ in (extra or {}).items():
        if d_:
            E = np.load(f"{d_}/{split}_RB.npy", mmap_mode="r"); assert E.shape[0] == out["X"].shape[0] and E.shape[1] >= 100, (nm, E.shape)
            out[nm] = E
    return out


def restrict50(Dp):
    """PT50: the PT pool with columns 50-99 masked (base top-50 only, the HR pool)."""
    D = dict(Dp); CA = Dp["CA"].copy(); CA[:, 50:] = -1
    D["CA"] = CA; D["LP"] = np.where(CA >= 0, Dp["LP"], -np.inf).astype(np.float32)
    D["FI"] = Dp["FI"] & (CA >= 0); D["BA"] = Dp["BA"] & (CA >= 0)
    D["gp"] = np.where((Dp["gp"] >= 0) & (Dp["gp"] < 50), Dp["gp"], -1)
    return D


def load_rb(rbdir, split):
    Q = dict(np.load(f"{rbdir}/{split}_q.npz"))
    CA = np.load(f"{rbdir}/{split}_cand.npy")
    return dict(X=np.load(f"{rbdir}/{split}_X.npy", mmap_mode="r"), LP=np.load(f"{rbdir}/{split}_logp.npy"), CA=CA,
                FI=np.load(f"{rbdir}/{split}_filt.npy") & (CA >= 0), BA=np.load(f"{rbdir}/{split}_babove.npy") & (CA >= 0),
                gp=Q["gold_pos"].astype(np.int64), rb=Q["r_base"].astype(np.float64), ts=Q["sro"][:, 3].astype(np.int64))


def sub(D, idx):
    return {k: (v[idx] if k not in ("X", "RF", "RBtr", "RBxf") else v) for k, v in D.items()}


def feats(D, idx, cols, kind, chunk=50000):
    """float32 [len(idx), 100, d] (qblend: [len(idx), d] query features + g [len(idx), 100]); chunked memmap reads."""
    if kind == "qblend":
        q = np.concatenate([np.asarray(D["X"][idx[lo:lo + chunk], 0][:, cols], np.float32) for lo in range(0, len(idx), chunk)])
        lq = np.concatenate([np.asarray(D["RF"][idx[lo:lo + chunk], :100, 0], np.float32) for lo in range(0, len(idx), chunk)])
        g = np.where(lq > -29.9, np.log(np.exp(lq) + 1e-5), np.log(1e-5)).astype(np.float32)
        return q, g
    if isinstance(cols, tuple):    # (base cols, extra feature array name)
        bc, ex = cols
        return np.concatenate([np.concatenate([np.asarray(D["X"][idx[lo:lo + chunk], :100][:, :, bc], np.float32),
                                               np.asarray(D[ex][idx[lo:lo + chunk], :100], np.float32)], -1) for lo in range(0, len(idx), chunk)]), None
    return np.concatenate([np.asarray(D["X"][idx[lo:lo + chunk], :100][:, :, cols], np.float32) for lo in range(0, len(idx), chunk)]), None


def predict(net, x, g, LP, kind, mu, sd, dev, chunk=20000):
    n = x.shape[0]; out = np.empty((n, 100), np.float32)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            xx = torch.from_numpy((x[lo:lo + chunk] - mu) / sd).to(dev)
            if kind == "qblend":
                w = Fn.softplus(net(xx)).squeeze(-1)
                lp = torch.from_numpy(np.where(np.isfinite(LP[lo:lo + chunk]), LP[lo:lo + chunk], -1e9).astype(np.float32)).to(dev)
                out[lo:lo + chunk] = (lp + w.unsqueeze(1) * torch.from_numpy(g[lo:lo + chunk]).to(dev)).cpu().numpy()
            else:
                out[lo:lo + chunk] = net(xx).squeeze(-1).cpu().numpy()
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
    ap.add_argument("--feat", required=True); ap.add_argument("--lists", default=""); ap.add_argument("--layout", required=True)
    ap.add_argument("--rb", default=""); ap.add_argument("--c2", default=""); ap.add_argument("--out", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--arms", required=True); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--fit_max", type=int, default=120000)
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--rbtr", default=""); ap.add_argument("--rbxf", default="")
    a = ap.parse_args(); t0 = time.time(); log = log_fn(t0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("V7_THREADS", "8")))
    os.makedirs(f"{a.out}/models", exist_ok=True)
    arms = a.arms.split(","); pools = sorted({ARMS[x][0] for x in arms})
    D = {}
    for p in pools:
        for sp in ("valid", "test"):
            if p == "PT50": continue
            D[(p, sp)] = load_pt(a.feat, a.lists, a.layout, a.rb, sp, dict(RBtr=a.rbtr, RBxf=a.rbxf)) if p == "PT" else (load_rb(f"{a.c2}/{p.split(':')[1]}", sp) if p.startswith("C2:") else load_rb(a.rb, sp))
    if "PT50" in pools:
        for sp in ("valid", "test"):
            if ("PT", sp) not in D: D[("PT", sp)] = load_pt(a.feat, a.lists, a.layout, a.rb, sp, dict(RBtr=a.rbtr, RBxf=a.rbxf))
            D[("PT50", sp)] = restrict50(D[("PT", sp)])
    ts = D[(pools[0], "valid")]["ts"]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    nte = len(D[(pools[0], "test")]["gp"]); te_idx = np.arange(nte)
    if a.smoke:
        fit_idx = fit_idx[:a.smoke]; sel_idx = sel_idx[:a.smoke // 2]; te_idx = te_idx[:a.smoke]
    for p in pools:   # identical row order across pools
        assert (D[(p, "valid")]["ts"] == ts).all() and len(D[(p, "test")]["gp"]) == nte
    out = dict(label=a.label, rule="v76_rule.json; recipe tois_v7_rule.json", host=f"{os.environ.get('HD_HOST', '')} ({dev}, torch {torch.__version__})",
               valid_cut_ts=int(cut), n_fit_population=int(len(fit_idx)), n_select=int(len(sel_idx)), n_test=int(len(te_idx)), cfg=vars(a), arms={})
    def view(p, sp, idx):
        Dd = D[(p, sp)]; return dict(LP=Dd["LP"][idx], CA=Dd["CA"][idx], FI=Dd["FI"][idx], BA=Dd["BA"][idx], gp=Dd["gp"][idx], rb=Dd["rb"][idx])
    ranks_out, sel_out = {}, {}
    for p in pools:
        Vt = view(p, "test", te_idx); ranks_out[f"base_{p.replace(':', '_')}"] = Vt["rb"].astype(np.float32)
        p_ = p.replace(":", "_"); out[f"identity_{p_}"] = float(np.mean(np.abs(fused_ranks_float(Vt["LP"], Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rb"]) - Vt["rb"]) < 1e-9))
        out[f"recall_test_{p_}"] = float((Vt["gp"] >= 0).mean())
    log("loaded", {k: v for k, v in out.items() if k.startswith(("identity", "recall"))}, "fit pop", len(fit_idx), "sel", len(sel_idx), "test", len(te_idx), dev)
    seeds = [int(s) for s in a.seeds.split(",")]
    for arm in arms:
        p, ck, kind, loss_kind = ARMS[arm]
        cols = (COLS[ck.split("+")[0]], ck.split("+")[1]) if "+" in ck else COLS[ck]; Dv = D[(p, "valid")]
        gpv = Dv["gp"][fit_idx]; fit_tr = fit_idx[gpv >= 0]
        n_ret = len(fit_tr)
        if len(fit_tr) > a.fit_max: fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
        Vf = view(p, "valid", fit_tr); Vs = view(p, "valid", sel_idx); Vt = view(p, "test", te_idx)
        x, g = feats(Dv, fit_tr, cols, kind)
        xs_, gs_ = feats(Dv, sel_idx, cols, kind); xt_, gt_ = feats(D[(p, "test")], te_idx, cols, kind)
        if kind == "qblend":
            mu = x.mean(0).astype(np.float32); sd = (x.std(0) + 1e-4).astype(np.float32)
        else:
            m = Vf["CA"] >= 0; Z = x[m]; subz = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
            mu = subz.mean(0).astype(np.float32); sd = (subz.std(0) + 1e-4).astype(np.float32); del Z, subz
        Xfit = torch.tensor((x - mu) / sd, device=dev); del x
        valid_c = torch.tensor(Vf["CA"] >= 0, device=dev); fi = torch.tensor(Vf["FI"], device=dev)
        yfit = torch.tensor(Vf["gp"], device=dev)
        lpfit = torch.tensor(np.where(Vf["CA"] >= 0, Vf["LP"], -1e9).astype(np.float32), device=dev)
        gfit = torch.tensor(g, device=dev) if g is not None else None
        ncol = (len(cols[0]) + 3) if isinstance(cols, tuple) else len(cols)
        rec = dict(pool=p, cols=(list(cols[0]) + [cols[1] + "x3"]) if isinstance(cols, tuple) else cols, model=kind, loss=loss_kind, n_features=ncol, n_fit_retained=int(n_ret), n_fit_queries=int(len(fit_tr)), seeds={})
        nfit = len(fit_tr)
        for sd_ in seeds:
            torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
            net = make_net(kind, ncol).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
            hist = []; best = (-1.0, None, -1)
            for ep in range(a.epochs if not a.smoke else 2):
                net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
                for b0 in range(0, nfit, a.bs):
                    idx = perm[b0:b0 + a.bs]
                    if kind == "qblend":
                        w = Fn.softplus(net(Xfit[idx])).squeeze(-1); z = lpfit[idx] + w.unsqueeze(1) * gfit[idx]
                    else:
                        z = net(Xfit[idx]).squeeze(-1)
                    vc = valid_c[idx]; y = yfit[idx]
                    if loss_kind == "masked":
                        loss = Fn.cross_entropy(z.masked_fill(~(vc & ~fi[idx]), -1e30), y)
                    elif loss_kind == "unmasked":
                        loss = Fn.cross_entropy(z.masked_fill(~vc, -1e30), y)
                    else:   # multipos: -log sum_{true answers in pool} softmax over the unmasked pool
                        zz = z.masked_fill(~vc, -1e30); pos = (fi[idx] | Fn.one_hot(y, zz.shape[1]).bool()) & vc
                        loss = (torch.logsumexp(zz, 1) - torch.logsumexp(zz.masked_fill(~pos, -1e30), 1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                net.eval()
                fsel = predict(net, xs_, gs_, Vs["LP"], kind, mu, sd, dev)
                msel = metr(fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"]))
                hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                if msel["MRR"] > best[0]: best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                elif ep - best[2] >= a.patience: break
            net.load_state_dict(best[1]); net.eval()
            torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_, model=kind), f"{a.out}/models/{arm}_s{sd_}.pt")
            fsel = predict(net, xs_, gs_, Vs["LP"], kind, mu, sd, dev); fte = predict(net, xt_, gt_, Vt["LP"], kind, mu, sd, dev)
            srec = dict(best_epoch=best[2], history=hist)
            rs_a = fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"]); rt_a = fused_ranks_float(fte, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rb"])
            srec.update(alone_select=metr(rs_a), alone_test=metr(rt_a))
            ranks_out[f"{arm}_alone_s{sd_}"] = rt_a.astype(np.float32); sel_out[f"{arm}_alone_s{sd_}"] = rs_a.astype(np.float32)
            if kind != "qblend":
                al, be = calibrate_ab(Vs["LP"], Vs["CA"], Vs["FI"], Vs["gp"], fsel); srec.update(cal_alpha=al, cal_beta=be)
                ssel = al * np.where(np.isfinite(Vs["LP"]), Vs["LP"], -1e9) + be * fsel; ste = al * np.where(np.isfinite(Vt["LP"]), Vt["LP"], -1e9) + be * fte
                rs = fused_ranks_float(ssel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"]); rt = fused_ranks_float(ste, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rb"])
                srec.update(fused_select=metr(rs), fused_test=metr(rt))
                ranks_out[f"{arm}_fused_s{sd_}"] = rt.astype(np.float32); sel_out[f"{arm}_fused_s{sd_}"] = rs.astype(np.float32)
                np.save(f"{a.out}/promlist_{arm}_fused_s{sd_}.npy", prom_lists(Vt["CA"], ste))
            else:
                np.save(f"{a.out}/promlist_{arm}_alone_s{sd_}.npy", prom_lists(Vt["CA"], fte))
            if arm in ALONE_LISTS:
                np.save(f"{a.out}/promlist_{arm}_alone_s{sd_}.npy", prom_lists(Vt["CA"], fte))
            rec["seeds"][sd_] = srec
            log(arm, "seed", sd_, "best_ep", best[2], "alone sel/test", round(srec["alone_select"]["MRR"], 5), round(srec["alone_test"]["MRR"], 5),
                "fused sel/test", (round(srec["fused_select"]["MRR"], 5), round(srec["fused_test"]["MRR"], 5)) if "fused_test" in srec else None)
        rec["n_params"] = int(sum(q.numel() for q in net.parameters()))
        out["arms"][arm] = rec
        del Xfit, lpfit, valid_c, fi, yfit, xs_, xt_, gs_, gt_
        np.savez(f"{a.out}/ranks_test.npz", **ranks_out); np.savez(f"{a.out}/ranks_select.npz", **sel_out); jdump(out, f"{a.out}/train_v7.json")
    out["elapsed_s"] = round(time.time() - t0, 1)
    np.savez(f"{a.out}/ranks_test.npz", **ranks_out); np.savez(f"{a.out}/ranks_select.npz", **sel_out)
    jdump(out, f"{a.out}/train_v7.json"); jdump(out, f"train_v7_{a.label}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
