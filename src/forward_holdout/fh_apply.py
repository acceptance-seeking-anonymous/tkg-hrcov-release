#!/usr/bin/env python3
"""phase2_forward_holdout / fh_apply.py -- STAGE 2: apply the FROZEN re-rankers (fh_train.py models) to the test features.

For every (arm, seed) model file: (i) reproduction check -- recompute the valid-select scores and fused ranks with the
frozen model and require them to equal the Stage-1 stored ranks (ranks_select.npz) on every row (else abort before any
test score); (ii) test scores s = cal_alpha * log p + cal_beta * f exactly as g2_train.py; fused ranks (ranks_of), promoted
lists (prom_lists) -> promlist_<arm>_s<seed>.npy for fh_blend.py --stage test.  No parameter is fitted here.
  python3 fh_apply.py --feat <fh_features dir with valid + test> --train <fh_train out dir>
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_train import KC, ARMS, COLS, log_fn, jdump, Net, load_split, subset, view, ranks_of, metr, predict, prom_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--arms", default="HR,HRCOV_PAT"); ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    dev = "cpu"
    torch.set_num_threads(int(os.environ.get("G2_THREADS", "8")))
    rd = a.train
    tr = json.load(open(f"{rd}/train_fh.json")); RS = np.load(f"{rd}/ranks_select.npz")
    Vall = load_split(a.feat, "valid"); sel_idx = RS["rows"]
    assert len(sel_idx) == tr["n_select"]
    Dsel = subset(Vall, sel_idx); Dte = load_split(a.feat, "test")
    WS = sorted({ARMS[x][1] for x in a.arms.split(",")} | {KC})
    Vsel = {W: view(Dsel, W) for W in WS}; Vte = {W: view(Dte, W) for W in WS}
    assert np.array_equal(Vsel[KC]["rb"].astype(np.float32), RS["base"]), "select base ranks differ from Stage 1"
    out = dict(n_test=int(len(Dte["LP"])), arms={}, identity_test={W: float(np.mean(np.abs(ranks_of(Vte[W], Vte[W]["LP"]) - Vte[W]["rb"]) < 1e-9)) for W in WS})
    ranks_out = dict(base=Vte[KC]["rb"].astype(np.float32))
    for arm in a.arms.split(","):
        kind, W, cs = ARMS[arm]; rec = {}
        for sd_ in [int(x) for x in a.seeds.split(",")]:
            M = torch.load(f"{rd}/models/{arm}_s{sd_}.pt", map_location="cpu", weights_only=False)
            assert M["arm"] == arm and M["seed"] == sd_ and M["W"] == W and list(M["cols"]) == COLS[cs]
            net = Net(len(M["cols"])); net.load_state_dict(M["state_dict"]); net.eval()
            fsel = predict(net, Dsel["X"], M["mu"], M["sd"], M["cols"], W, dev)
            ssel = M["cal_alpha"] * Vsel[W]["LP"] + M["cal_beta"] * fsel
            rs = ranks_of(Vsel[W], ssel)
            agree = float(np.mean(rs.astype(np.float32) == RS[f"{arm}_s{sd_}"]))
            fdev = float(np.abs(fsel - np.load(f"{rd}/models/{arm}_s{sd_}_fsel.npy")).max())
            assert agree >= 0.9999, f"{arm} s{sd_}: frozen model does not reproduce Stage-1 select ranks ({agree})"
            fte = predict(net, Dte["X"], M["mu"], M["sd"], M["cols"], W, dev)
            ste = M["cal_alpha"] * Vte[W]["LP"] + M["cal_beta"] * fte
            rt = ranks_of(Vte[W], ste)
            np.save(f"{rd}/promlist_{arm}_s{sd_}.npy", prom_lists(Vte[W], ste))
            ranks_out[f"{arm}_s{sd_}"] = rt.astype(np.float32)
            rec[sd_] = dict(select_rank_agreement=agree, select_score_max_abs_dev=fdev, test=metr(rt))
            if W > KC:
                vte = Vte[W]; s_ = np.where(vte["CA"] >= 0, ste, -np.inf); mint = s_[:, :KC].min(1)
                promc = (s_[:, KC:] > mint[:, None]) & (vte["CA"][:, KC:] >= 0)
                rec[sd_]["test_promoted_cov_per_query"] = float(promc.sum(1).mean()); rec[sd_]["test_frac_queries_with_promotion"] = float(promc.any(1).mean())
            log(arm, sd_, json.dumps(rec[sd_]))
        out["arms"][arm] = rec
    np.savez(f"{rd}/ranks_test.npz", **ranks_out)
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(out, f"{rd}/apply_fh.json"); jdump(out, "apply_fh.json")
    os.system(f"chmod -R a+rX {rd}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
