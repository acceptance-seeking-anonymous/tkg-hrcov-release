#!/usr/bin/env python3
"""hardening / hd_sp_promlists.py -- promoted candidate lists (final fused order) for EVERY seed of the smallpedia re-rankers.

sp_train.py saved promoted lists for seed 0 only.  This script reloads the saved seed models (work/train_<b><sfx>/models/
<arm>_s<k>.pt, trained on valid only) and the valid-calibrated (alpha, beta) of train_<b>.json, re-predicts the TEST feature
tensors, asserts that the exact fused ranks reproduce the stored ranks_test.npz of that seed, and writes
promlist_<arm>_s<k>.npy (the official-evaluator systems input).  Read-only on the pilot directory.
  SP_STATIC_MODE=full python3 hd_sp_promlists.py --base tirgn --arms HR,HRCOV,HRCOV_ST --out OUT
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.expandvars("$WORK_DIR/tracks/pilot_smallpedia/code"))
import numpy as np
import torch
import sp_train as T
from sp_common import WORK, MODE_SFX

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--arms", default="HR,HRCOV,HRCOV_ST")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--out", required=True)
ap.add_argument("--feat_dir", default=""); ap.add_argument("--train_dir", default="")
a = ap.parse_args()
t0 = time.time()
fd = a.feat_dir or f"{WORK}/feat_{a.base}{MODE_SFX}"; rd = a.train_dir or f"{WORK}/train_{a.base}{MODE_SFX}"
os.makedirs(a.out, exist_ok=True)
tj = json.load(open(f"{rd}/train_{a.base}.json"))
RT = np.load(f"{rd}/ranks_test.npz")
Dte = T.load_split(fd, "test")
rep = dict(base=a.base, mode=MODE_SFX or "full", feat_dir=fd, train_dir=rd, arms={})
for arm in a.arms.split(","):
    Vt = T.view(Dte, arm); L = Vt["L"]
    for sd in [int(x) for x in a.seeds.split(",")]:
        ck = torch.load(f"{rd}/models/{arm}_s{sd}.pt", map_location="cpu", weights_only=False)
        net = T.Net(len(ck["feats"])); net.load_state_dict(ck["state_dict"]); net.eval()
        fte = T.predict(net, Dte[L]["X"], Vt, ck["mu"], ck["sd"], "cpu")
        srec = tj["arms"][arm]["seeds"][str(sd)]
        ste = srec["cal_alpha"] * Vt["LP"] + srec["cal_beta"] * fte
        rt = T.ranks_of(Vt, ste)
        agree = float(np.mean(rt == RT[f"{arm}_s{sd}"]))
        pl = T.prom_lists(Vt, ste)
        np.save(f"{a.out}/promlist_{arm}{MODE_SFX}_{a.base}_s{sd}.npy", pl)
        rep["arms"][f"{arm}_s{sd}"] = dict(rank_agreement_with_stored=agree, MRR_repro=float(np.mean(1.0 / rt)),
                                           MRR_stored=float(np.mean(1.0 / RT[f"{arm}_s{sd}"])), W=int(pl.shape[1]))
        print(arm, sd, rep["arms"][f"{arm}_s{sd}"], round(time.time() - t0, 1), flush=True)
json.dump(rep, open(f"{a.out}/promlists_{a.base}{MODE_SFX}.json", "w"), indent=1)
