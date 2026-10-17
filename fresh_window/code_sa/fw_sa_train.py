#!/usr/bin/env python3
"""phase2_fresh_window / fw_sa_train.py -- v78_train.py (v78_rule_amendment_1 SA arm) split into a validation-only fit stage and
a test apply stage (v79 scoring discipline).  Model, features, loss, optimiser, fit rows, epoch selection and ranks are the
v78_train.py code: helpers are imported from the byte-identical v78_train.py copy in this dir and the per-arm training block
below is copied verbatim (only the test-split lines moved to --stage apply).  Arm SA_W100 = (SA50 pool, SA16 columns, mlp64,
no fusion, history tail), seeds 0-2.
  python3 fw_sa_train.py --stage fit   --ds gdeltfw --pools P --arms SA_W100 --out O       (validation only; models frozen)
  python3 fw_sa_train.py --stage apply --ds gdeltfw --pools P --arms SA_W100 --out O       (test: ranks + promoted lists)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v78_common as C
from v78_train import arms_for, metr, fused_ranks_float, prom_lists, MLP, RelLin, load_v78, feats, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "apply"])
    ap.add_argument("--ds", required=True, choices=list(C.DS)); ap.add_argument("--pools", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_max", type=int, default=120000)
    a = ap.parse_args(); t0 = time.time(); log = C.log_fn(t0)
    cfg = C.DS[a.ds]; dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(int(os.environ.get("V7_THREADS", "8")))
    ARMS = arms_for(list(cfg["bases"])); arms = a.arms.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(f"{a.out}/models", exist_ok=True)
    jf = f"{a.out}/train_fw_sa.json"
    out = json.load(open(jf)) if os.path.exists(jf) else dict(ds=a.ds, rule="v79 + v78_rule_amendment_1 (SA)", arms={})
    out[f"host_{a.stage}"] = f"{os.environ.get('HD_HOST', '')} ({dev}, torch {torch.__version__})"; out[f"cfg_{a.stage}"] = vars(a)
    for arm in arms:
        pool, cols, model, fuse, tail = ARMS[arm]
        assert model == "mlp64" and not fuse and pool.startswith("SA"), "fw_sa_train covers the SA arms only"
        if a.stage == "fit":
            Dv = load_v78(f"{a.pools}/{pool}", "valid", tail, 0); K = Dv["K"]; W = 2 * K
            ts = Dv["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
            fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
            gpv = Dv["gp"][fit_idx]; fit_tr = fit_idx[gpv >= 0]; n_ret = len(fit_tr)
            if len(fit_tr) > a.fit_max: fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
            V = lambda Dd, idx: dict(LP=Dd["LP"][idx], CA=Dd["CA"][idx], FI=Dd["FI"][idx], BA=Dd["BA"][idx], gp=Dd["gp"][idx], rt=Dd["rt"][idx])
            Vf, Vs = V(Dv, fit_tr), V(Dv, sel_idx)
            relv = Dv["sro"][:, 1]
            x = feats(Dv, fit_tr, cols, W)
            m = Vf["CA"] >= 0; Z = x[m]; subz = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 3000000), replace=False)]
            mu = subz.mean(0).astype(np.float32); sd = (subz.std(0) + 1e-4).astype(np.float32); del Z, subz
            Xfit = torch.tensor((x - mu) / sd, device=dev); del x
            mfit = torch.tensor(m & ~Vf["FI"], device=dev); yfit = torch.tensor(Vf["gp"], device=dev); rfit = torch.as_tensor(relv[fit_tr], device=dev)
            ident_valid = float(np.mean(np.abs(fused_ranks_float(-np.tile(np.arange(W, dtype=np.float64), (len(sel_idx), 1)), Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K) - Vs["rt"]) < 1e-9))
            rec = dict(pool=pool, cols=cols, model=model, fuse=fuse, tail=tail, K=K, W=W, n_features=len(cols), valid_cut_ts=int(cut),
                       n_fit_retained=int(n_ret), n_fit_queries=int(len(fit_tr)), n_select=int(len(sel_idx)),
                       identity_tail_order_select=ident_valid, recall_select=float((Vs["gp"] >= 0).mean()), hist_select=metr(Vs["rt"]), seeds={})
            log(arm, pool, "K", K, "W", W, "fit", len(fit_tr), "sel", len(sel_idx), "identity(select)", ident_valid)
            sel_out = {}
            for sd_ in seeds:
                torch.manual_seed(sd_); rng = np.random.default_rng(sd_)
                net = MLP(len(cols)).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5); hist = []; best = (-1.0, None, -1); nfit = len(fit_tr)
                for ep in range(a.epochs):
                    net.train(); perm = torch.from_numpy(rng.permutation(nfit)).to(dev); tl = 0.0; nb = 0
                    for b0 in range(0, nfit, a.bs):
                        idx = perm[b0:b0 + a.bs]; z = net(Xfit[idx], rfit[idx]).masked_fill(~mfit[idx], -1e30)
                        loss = Fn.cross_entropy(z, yfit[idx])
                        opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
                    net.eval(); fsel = predict(net, Dv, sel_idx, cols, W, mu, sd, relv, dev)
                    msel = metr(fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K))
                    hist.append(dict(ep=ep, loss=round(tl / nb, 5), sel_MRR=msel["MRR"]))
                    if msel["MRR"] > best[0]: best = (msel["MRR"], {k: v.clone() for k, v in net.state_dict().items()}, ep)
                    elif ep - best[2] >= a.patience: break
                net.load_state_dict(best[1]); net.eval()
                fsel = predict(net, Dv, sel_idx, cols, W, mu, sd, relv, dev)
                rs = fused_ranks_float(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rt"], K)
                srec = dict(best_epoch=best[2], history=hist, lam=None, select=metr(rs))
                torch.save(dict(state_dict=best[1], mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_, model=model, lam=None, K=K, W=W), f"{a.out}/models/{arm}_s{sd_}.pt")
                sel_out[f"{arm}_s{sd_}"] = rs.astype(np.float32); rec["seeds"][sd_] = srec
                log(arm, "seed", sd_, "best_ep", best[2], "select MRR", round(srec["select"]["MRR"], 5))
            rec["n_params"] = int(sum(p.numel() for p in net.parameters()))
            out["arms"][arm] = rec
            np.savez(f"{a.out}/ranks_select_{arm}.npz", rows=sel_idx, **sel_out)
            del Xfit, mfit, yfit, rfit
        else:
            Dt = load_v78(f"{a.pools}/{pool}", "test", tail, 0); K = Dt["K"]; W = 2 * K
            te_idx = np.arange(len(Dt["gp"])); relt = Dt["sro"][:, 1]
            Vt = dict(LP=Dt["LP"], CA=Dt["CA"], FI=Dt["FI"], BA=Dt["BA"], gp=Dt["gp"], rt=Dt["rt"])
            ident = float(np.mean(np.abs(fused_ranks_float(-np.tile(np.arange(W, dtype=np.float64), (len(te_idx), 1)), Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K) - Vt["rt"]) < 1e-9))
            rk_out = {}; rec = out["arms"][arm]; rec["identity_tail_order_test"] = ident; rec["n_test"] = int(len(te_idx))
            for sd_ in seeds:
                ck = torch.load(f"{a.out}/models/{arm}_s{sd_}.pt", map_location=dev, weights_only=False)
                net = MLP(len(ck["cols"])).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()
                fte = predict(net, Dt, te_idx, ck["cols"], W, ck["mu"], ck["sd"], relt, dev)
                rt = fused_ranks_float(fte, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rt"], K)
                np.save(f"{a.out}/promlist_{arm}_s{sd_}.npy", prom_lists(Vt["CA"], fte, K))
                rk_out[f"{arm}_s{sd_}"] = rt.astype(np.float32)
                vte = np.where(Vt["CA"] >= 0, fte, -np.inf); mint = vte[:, :K].min(1)
                promc = (vte[:, K:] > mint[:, None]) & (Vt["CA"][:, K:] >= 0)
                rec["seeds"][str(sd_)].update(test_harness=metr(rt), test_promoted_block_per_query=float(promc.sum(1).mean()),
                                              test_frac_with_promotion=float(promc.any(1).mean()))
            np.savez(f"{a.out}/ranks_test_{arm}.npz", **rk_out)
            log(arm, "test applied (metrics written, not printed)")
        C.jdump(out, jf)
    out[f"elapsed_{a.stage}_s"] = round(time.time() - t0, 1); C.jdump(out, jf)
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
