#!/usr/bin/env python3
"""phase2_fresh_window / fh_train.py -- STAGE 1 (fit + freeze, no test split) of the G2 re-ranker recipe on GDELTFW.

The training loop below is presub/g2_train.py main() with every test-split statement removed; all helpers (Net, view,
ranks_of, predict, calibrate_ab, prom_lists, ARMS, COLS, metrics) are imported from the byte-identical copy of g2_train.py
in this directory.  Recipe (a_presub_rule.json common.reranker_recipe): MLP d-64-64-1 (zero-init last layer), listwise CE
over the arm's candidate set, Adam lr 1e-3 wd 1e-5, batch 512, <= 30 epochs, early stopping on valid-select MRR with
patience 5, final score s = alpha*log p + beta*f with (alpha, beta) = argmin valid-select listwise NLL (LBFGS); seeds 0-2;
valid timestamps < 70% quantile = fit, rest = select; fit rows = fit queries whose gold is in the arm's candidate set,
capped at 120,000 (rng 0).  Arms: HR (W = 50, 19 features) and HRCOV_PAT (W = 100, 25 features).
Frozen per (arm, seed): models/<arm>_s<seed>.pt = {state_dict, mu, sd, cols, arm, seed, W, kind, cal_alpha, cal_beta,
best_epoch} + select ranks (ranks_select.npz) used by fh_apply.py as a reproduction check before any test score.
  python3 fh_train.py --feat <fh_features valid out> --out <dir>
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_train import (KC, SEEDS, ARMS, COLS, log_fn, jdump, Net, load_split, subset, view, ranks_of, metr,  # noqa: E402
                      predict, calibrate_ab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="HR,HRCOV_PAT")
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
    torch.set_num_threads(int(os.environ.get("G2_THREADS", "8")))
    sfx = "_smoke" if a.smoke else ""
    fd = a.feat; rd = a.out; os.makedirs(f"{rd}/models", exist_ok=True)
    assert not os.path.exists(f"{fd}/test_X.npy") or os.environ.get("FH_ALLOW_TEST_FEATS") == "1", \
        "Stage 1: test features must not exist yet"
    Vall = load_split(fd, "valid")
    ts = Vall["Q"]["sro"][:, 3]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]
    if a.smoke:                                   # smoke rows span ~1 timestamp: alternate rows instead
        fit_idx = np.arange(0, len(ts), 2); sel_idx = np.arange(1, len(ts), 2)
    Dsel = subset(Vall, sel_idx)
    log("loaded valid", len(ts), "fit rows", len(fit_idx), "select", len(sel_idx), "device", dev)
    meta = json.load(open(f"{a.feat}/meta.json"))
    out = dict(base=meta["base"], ds=meta["ds"], host=f"{os.environ.get('HD_HOST', '')} ({dev}, torch {torch.__version__})", valid_cut_ts=int(cut),
               fit_ts=[int(ts[fit_idx].min()), int(ts[fit_idx].max())], select_ts=[int(ts[sel_idx].min()), int(ts[sel_idx].max())],
               n_fit_rows=int(len(fit_idx)), n_select=int(len(sel_idx)), cfg=vars(a), arms={})
    WS = sorted({ARMS[x][1] for x in a.arms.split(",")} | {KC})
    Vsel = {W: view(Dsel, W) for W in WS}
    out["base_select"] = metr(Vsel[KC]["rb"])
    out["identity_select"] = {W: float(np.mean(np.abs(ranks_of(Vsel[W], Vsel[W]["LP"]) - Vsel[W]["rb"]) < 1e-9)) for W in WS}
    log("identity (select)", out["identity_select"])
    sel_out = dict(base=Vsel[KC]["rb"].astype(np.float32))
    seeds = [int(x) for x in a.seeds.split(",")]
    fit_cache = {}
    for arm in a.arms.split(","):
        kind, W, cs = ARMS[arm]; cols = COLS[cs]
        gpv = Vall["Q"]["gold_pos"][fit_idx].astype(np.int64)
        fit_tr = fit_idx[(gpv >= 0) & (gpv < W)]
        n_ret = int(len(fit_tr))                   # training-selection accounting (PI review section 4)
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
        rec = dict(kind=kind, W=W, features=cs, n_features=len(cols), n_fit_queries=int(nfit),
                   fit_selection=dict(n_fit_period_rows=int(len(fit_idx)), n_gold_in_candidate_set=n_ret,
                                      retained_share=float(n_ret / max(len(fit_idx), 1)), capped_at=int(a.fit_max)), seeds={})
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
            fsel = predict(net, Dsel["X"], mu, sd, cols, W, dev)
            srec = dict(best_epoch=best[2], history=hist)
            if kind == "resid":
                al, be = None, None
                ssel = Vsel[W]["LP"] + fsel
            else:
                srec.update(alone_select=metr(ranks_of(Vsel[W], fsel)))
                al, be = calibrate_ab(Vsel[W], fsel)
                srec.update(cal_alpha=al, cal_beta=be)
                ssel = al * Vsel[W]["LP"] + be * fsel
            rs = ranks_of(Vsel[W], ssel)
            srec.update(select=metr(rs))
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in best[1].items()}, mu=mu, sd=sd, cols=cols, arm=arm, seed=sd_,
                            W=W, kind=kind, cal_alpha=al, cal_beta=be, best_epoch=best[2], train_device=dev),
                       f"{rd}/models/{arm}_s{sd_}.pt")
            np.save(f"{rd}/models/{arm}_s{sd_}_fsel.npy", fsel.astype(np.float32))
            sel_out[f"{arm}_s{sd_}"] = rs.astype(np.float32)
            rec["seeds"][sd_] = srec
            log(arm, "seed", sd_, "best_ep", best[2], "select", srec["select"]["MRR"])
        rec["n_params"] = int(sum(p.numel() for p in net.parameters()))
        out["arms"][arm] = rec
    np.savez(f"{rd}/ranks_select.npz", rows=sel_idx, **sel_out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"train_fh_{meta['ds']}_{meta['base']}{sfx}.json"); jdump(out, f"{rd}/train_fh.json")
    os.system(f"chmod -R a+rX {rd}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
