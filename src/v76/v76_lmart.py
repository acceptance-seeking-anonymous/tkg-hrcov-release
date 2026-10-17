#!/usr/bin/env python3
"""v76 / v76_lmart.py (v2, amendment 1: tie-exact fused-rank statistic) -- M2b LambdaMART arm (v76_rule.json): LightGBM lambdarank on the 25 HRCOV_PAT features of the PT pool
(W = 100), the same fit / selection rows as v7_btrain (fit: valid ts < 70% cut, gold in pool, cap 120,000 with rng(0);
selection: valid ts >= cut), early stopping + grid selection on the selection MRR of the exact fused-rank statistic of the
LMART score alone (= the MLP's epoch criterion), calibrated alpha/beta fusion with the base log-prob (v7 calibrate_ab).
Groups = one per fit query: its valid (CA >= 0), unfiltered (not another true answer) candidates; label 1 = gold.
Grid point chosen with seed 0; seeds 1, 2 refit with it.  Writes ranks_test/ranks_select npz, promoted lists, models,
train_lmart.json (grid, rounds, timings) and a latency profile (LMART CPU vs MLP CPU / GPU on the same test tensor).
  python3 v76_lmart.py --layout pc --feat F --lists L --out O --label polecat_tirgn [--mlp_model O2/models/MLP64_PAT_s0.pt]
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v7_btrain as VB
import torch

PAT25 = VB.PAT25
FIXED = dict(objective="lambdarank", metric="None", lambdarank_truncation_level=100, label_gain=[0, 1], min_data_in_leaf=200,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, deterministic=True,
             force_row_wise=True, verbosity=-1)
GRID = [(nl, lr) for nl in (15, 31, 63) for lr in (0.03, 0.1)]


def fused_ranks_tie(s, cand, filt, babove, gold_pos, r_base, kc=VB.KC):
    """v7 fused_ranks_float with TIES resolved exactly as the promoted list resolves them (prom_lists: stable descending sort,
    equal scores in column order), i.e. the rank the official evaluator gives the lifted list.  Identical to
    fused_ranks_float when scores have no ties (MLP); required for tree scores, which tie heavily (amendment 1)."""
    n, W = s.shape; valid = cand >= 0; s = np.where(valid, s, -np.inf); mint = s[:, :kc].min(1)
    jj = np.arange(W)[None, :]; prom = valid & ((jj < kc) | (s > mint[:, None])); unf = prom & ~filt
    ar = np.arange(n); g = np.maximum(gold_pos, 0); sg = s[ar, g]; gin = (gold_pos >= 0) & prom[ar, g]
    above = unf & ((s > sg[:, None]) | ((s == sg[:, None]) & (jj < g[:, None]))); above[ar, g] = False
    r_in = 1 + above.sum(1); add = (unf & (jj >= kc) & ~babove).sum(1)
    return np.where(gin, r_in, r_base + add).astype(np.float64)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True); ap.add_argument("--lists", default=""); ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--label", required=True); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fit_max", type=int, default=120000); ap.add_argument("--rounds", type=int, default=3000); ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--mlp_model", default=""); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); log = VB.log_fn(t0)
    os.makedirs(f"{a.out}/models", exist_ok=True)
    Dv = VB.load_pt(a.feat, a.lists, a.layout, "", "valid"); Dt = VB.load_pt(a.feat, a.lists, a.layout, "", "test")
    ts = Dv["ts"]; uts = np.unique(ts); cut = uts[int(0.7 * len(uts))]
    fit_idx = np.where(ts < cut)[0]; sel_idx = np.where(ts >= cut)[0]; te_idx = np.arange(len(Dt["gp"]))
    if a.smoke:
        fit_idx = fit_idx[:a.smoke]; sel_idx = sel_idx[:a.smoke // 2]; te_idx = te_idx[:a.smoke]
    gpv = Dv["gp"][fit_idx]; fit_tr = fit_idx[gpv >= 0]; n_ret = len(fit_tr)
    if len(fit_tr) > a.fit_max: fit_tr = np.sort(np.random.default_rng(0).choice(fit_tr, a.fit_max, replace=False))
    view = lambda D, idx: dict(LP=D["LP"][idx], CA=D["CA"][idx], FI=D["FI"][idx], BA=D["BA"][idx], gp=D["gp"][idx], rb=D["rb"][idx])
    Vf, Vs, Vt = view(Dv, fit_tr), view(Dv, sel_idx), view(Dt, te_idx)
    x, _ = VB.feats(Dv, fit_tr, PAT25, "mlp"); xs, _ = VB.feats(Dv, sel_idx, PAT25, "mlp"); xt, _ = VB.feats(Dt, te_idx, PAT25, "mlp")
    log("loaded", dict(cut=int(cut), fit_pop=len(fit_idx), fit_retained=n_ret, fit=len(fit_tr), sel=len(sel_idx), test=len(te_idx)))
    # training groups (masked-loss analogue): valid, unfiltered candidates
    m = (Vf["CA"] >= 0) & ~Vf["FI"]; jj = np.arange(100)[None, :]
    assert m[np.arange(len(fit_tr)), Vf["gp"]].all(), "gold must be a valid unfiltered candidate"
    Xtr = x[m]; ytr = (jj == Vf["gp"][:, None])[m].astype(np.float32); grp = m.sum(1)
    del x
    nsel = len(sel_idx); Xsel = xs.reshape(-1, xs.shape[-1]); ysel = (jj == np.maximum(Vs["gp"], 0)[:, None]).astype(np.float32).reshape(-1)
    state = dict(n=0, t=0.0)

    def feval(preds, data):
        t1 = time.time()
        rk = fused_ranks_tie(preds.reshape(nsel, 100).astype(np.float64), Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"])
        state["n"] += 1; state["t"] += time.time() - t1
        return "selMRR", float(np.mean(1.0 / rk)), True

    def fit(nl, lr, seed):
        p = dict(FIXED, num_leaves=nl, learning_rate=lr, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
                 num_threads=a.threads)
        dtr = lgb.Dataset(Xtr, label=ytr, group=grp, free_raw_data=False)
        dsel = lgb.Dataset(Xsel, label=ysel, group=np.full(nsel, 100), reference=dtr, free_raw_data=False)
        ev = {}; t1 = time.time(); state.update(n=0, t=0.0)
        bst = lgb.train(p, dtr, num_boost_round=a.rounds if not a.smoke else 30, valid_sets=[dsel], valid_names=["sel"], feval=feval,
                        callbacks=[lgb.early_stopping(a.patience, first_metric_only=True, verbose=False), lgb.record_evaluation(ev)])
        wall = time.time() - t1
        curve = ev["sel"]["selMRR"]
        return bst, dict(num_leaves=nl, learning_rate=lr, seed=seed, best_round=int(bst.best_iteration), best_sel_MRR=float(max(curve)),
                         rounds_run=len(curve), wall_s=round(wall, 1), eval_s=round(state["t"], 1), train_s_excl_eval=round(wall - state["t"], 1),
                         curve_every50=[round(c, 6) for c in curve[::50]])

    out = dict(label=a.label, rule="v76_rule.json M2b + amendment 1 (tie-exact selection statistic)", host=os.environ.get("HD_HOST", ""), lightgbm=lgb.__version__,
               valid_cut_ts=int(cut), n_fit_population=int(len(fit_idx)), n_fit_retained=int(n_ret), n_fit_queries=int(len(fit_tr)),
               n_fit_rows=int(len(ytr)), n_select=int(nsel), n_test=int(len(te_idx)), fixed=FIXED, grid=GRID, cfg=vars(a), grid_results=[], seeds={})
    # ---- grid on seed 0
    best = None
    for nl, lr in GRID:
        bst, rec = fit(nl, lr, 0); out["grid_results"].append(rec)
        log("grid", rec["num_leaves"], rec["learning_rate"], "best_round", rec["best_round"], "selMRR", round(rec["best_sel_MRR"], 6), "wall", rec["wall_s"])
        if best is None or rec["best_sel_MRR"] > best[1]["best_sel_MRR"]:
            best = (bst, rec)
        VB.jdump(out, f"{a.out}/train_lmart.json")
    nl, lr = best[1]["num_leaves"], best[1]["learning_rate"]; out["selected"] = dict(num_leaves=nl, learning_rate=lr)
    seeds = [int(s) for s in a.seeds.split(",")]
    ranks_out = {"base_PT": Vt["rb"].astype(np.float32)}; sel_out = {}
    lat = {}
    for sd in seeds:
        if sd == 0:
            bst, rec = best
        else:
            bst, rec = fit(nl, lr, sd)
        bst.save_model(f"{a.out}/models/LMART_PAT_s{sd}.txt", num_iteration=bst.best_iteration)
        fsel = bst.predict(Xsel, num_iteration=bst.best_iteration, num_threads=a.threads).reshape(nsel, 100)
        t1 = time.time(); fte = bst.predict(xt.reshape(-1, xt.shape[-1]), num_iteration=bst.best_iteration, num_threads=a.threads).reshape(len(te_idx), 100)
        lat_s = time.time() - t1
        rs_a = fused_ranks_tie(fsel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"]); rt_a = fused_ranks_tie(fte, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rb"])
        al, be = VB.calibrate_ab(Vs["LP"], Vs["CA"], Vs["FI"], Vs["gp"], fsel)
        ssel = al * np.where(np.isfinite(Vs["LP"]), Vs["LP"], -1e9) + be * fsel; ste = al * np.where(np.isfinite(Vt["LP"]), Vt["LP"], -1e9) + be * fte
        rs = fused_ranks_tie(ssel, Vs["CA"], Vs["FI"], Vs["BA"], Vs["gp"], Vs["rb"]); rt = fused_ranks_tie(ste, Vt["CA"], Vt["FI"], Vt["BA"], Vt["gp"], Vt["rb"])
        rec.update(cal_alpha=al, cal_beta=be, alone_select=VB.metr(rs_a), alone_test=VB.metr(rt_a), fused_select=VB.metr(rs), fused_test=VB.metr(rt),
                   test_predict_s_cpu=round(lat_s, 2), n_trees=int(bst.best_iteration), finite_scores=bool(np.isfinite(fte).all() and np.isfinite(fsel).all()))
        assert rec["finite_scores"], "non-finite LMART score"
        ranks_out[f"LMART_PAT_alone_s{sd}"] = rt_a.astype(np.float32); ranks_out[f"LMART_PAT_fused_s{sd}"] = rt.astype(np.float32)
        sel_out[f"LMART_PAT_alone_s{sd}"] = rs_a.astype(np.float32); sel_out[f"LMART_PAT_fused_s{sd}"] = rs.astype(np.float32)
        np.save(f"{a.out}/promlist_LMART_PAT_fused_s{sd}.npy", VB.prom_lists(Vt["CA"], ste))
        np.save(f"{a.out}/promlist_LMART_PAT_alone_s{sd}.npy", VB.prom_lists(Vt["CA"], fte))
        out["seeds"][sd] = rec
        log("seed", sd, "best_round", rec["best_round"], "alone sel/test", round(rec["alone_select"]["MRR"], 5), round(rec["alone_test"]["MRR"], 5),
            "fused sel/test", round(rec["fused_select"]["MRR"], 5), round(rec["fused_test"]["MRR"], 5))
        np.savez(f"{a.out}/ranks_test.npz", **ranks_out); np.savez(f"{a.out}/ranks_select.npz", **sel_out); VB.jdump(out, f"{a.out}/train_lmart.json")
    # ---- latency profile on the full test tensor (features precomputed; identical inputs)
    nq = len(te_idx); X2 = xt.reshape(-1, xt.shape[-1])
    bst0 = lgb.Booster(model_file=f"{a.out}/models/LMART_PAT_s0.txt")
    t1 = time.time(); _ = bst0.predict(X2, num_threads=a.threads); lat["LMART_cpu_ms_per_1000q"] = 1000 * (time.time() - t1) / nq * 1000
    lat["LMART_n_trees_seed0"] = int(bst0.num_trees()); lat["threads"] = a.threads
    if a.mlp_model and os.path.exists(a.mlp_model):
        ck = torch.load(a.mlp_model, map_location="cpu", weights_only=False)
        net = VB.make_net(ck["model"], len(ck["cols"])); net.load_state_dict(ck["state_dict"]); net.eval()
        torch.set_num_threads(a.threads)
        t1 = time.time(); _ = VB.predict(net, xt, None, Vt["LP"], ck["model"], ck["mu"], ck["sd"], "cpu"); lat["MLP_cpu_ms_per_1000q"] = 1000 * (time.time() - t1) / nq * 1000
        if torch.cuda.is_available():
            netg = net.to("cuda"); _ = VB.predict(netg, xt[:20000], None, Vt["LP"][:20000], ck["model"], ck["mu"], ck["sd"], "cuda"); torch.cuda.synchronize()
            t1 = time.time(); _ = VB.predict(netg, xt, None, Vt["LP"], ck["model"], ck["mu"], ck["sd"], "cuda"); torch.cuda.synchronize()
            lat["MLP_gpu_ms_per_1000q"] = 1000 * (time.time() - t1) / nq * 1000; lat["gpu"] = torch.cuda.get_device_name(0)
        lat["MLP_params"] = int(sum(q.numel() for q in net.parameters()))
    out["latency"] = lat; out["elapsed_s"] = round(time.time() - t0, 1)
    VB.jdump(out, f"{a.out}/train_lmart.json"); VB.jdump(out, f"train_lmart_{a.label}{'_smoke' if a.smoke else ''}.json")
    os.system(f"chmod -R a+rX {a.out}")
    log("ALL DONE", json.dumps(lat))


if __name__ == "__main__":
    main()
