#!/usr/bin/env python3
"""pilot_icews / pi_relnorm_data.py -- (1) writes the relation-normalised harness data dir $P/stage/tkgl-icews-relnorm
(train/valid/test with relation ids mapped by $P/stage/relmap.npy; entity / timestamp ids, row order, stat.txt unchanged;
relation2id.txt keeps all 391 ids, the 117 merged ids simply no longer occur); (2) POST-HOC control (not pre-registered):
full-history RecB-lite on that data dir (same patched datasets-track profiler as pi_prep, lambda re-selected on full
valid), saved in harness test-sample order -> work/recbRN_test_sample_rank.npy.  The time-aware filter is unchanged by
the merge for every test key (no merged code pair is used on the same day: overlap_days = 0 for all 117 pairs).
  python3 pi_relnorm_data.py --nproc 16
"""
import os, sys, json, time, argparse, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import P, DD, DS, WORK, Splits, jdump, log_fn
from pi_prep import profiler_to_split_order

ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=16); a = ap.parse_args()
t0 = time.time(); log = log_fn(t0)
RD = f"{P}/stage/tkgl-icews-relnorm"; os.makedirs(RD, exist_ok=True)
relmap = np.load(f"{P}/stage/relmap.npy")
for fn in ("stat.txt", "entity2id.txt", "relation2id.txt", "timestamp2id.txt"):
    shutil.copy(f"{DD}/{fn}", f"{RD}/{fn}")
cnt = {}
for sp in ("train", "valid", "test"):
    q = np.loadtxt(f"{DD}/{sp}.txt", dtype=np.int64).reshape(-1, 4)
    cnt[sp] = dict(n=int(len(q)), n_remapped=int((relmap[q[:, 1]] != q[:, 1]).sum()))
    q[:, 1] = relmap[q[:, 1]]
    np.savetxt(f"{RD}/{sp}.txt", q, fmt="%d", delimiter="\t")
json.dump(dict(source=DD, relmap=f"{P}/stage/relmap.npy", counts=cnt, n_relation_ids_in_use=int(len(np.unique(relmap)))),
          open(f"{RD}/conversion_log.json", "w"), indent=1)
log("relnorm data dir", cnt)
env = dict(os.environ, RECB_PREFIX=f"{WORK}/recbRN")
subprocess.run([sys.executable, f"{WORK}/profile_tkg_v2_saveranks.py", "--name", "tkgl-icews-relnorm", "--dir", RD,
                "--out", f"{WORK}/recbRN_profile.json", "--nproc", str(a.nproc)], check=True, env=env)
prof = json.load(open(f"{WORK}/recbRN_profile.json")); rb = prof["recb_lite"]; j = rb["lambda_grid"].index(rb["lambda_selected"])
S = Splits(DS)
trk = np.load(f"{WORK}/recbRN_test_rank_prof.npy")[:, j][profiler_to_split_order(S, "test")]
tidx = np.load(f"{WORK}/test_sample_idx.npy")
np.save(f"{WORK}/recbRN_test_sample_rank.npy", trk[tidx]); np.save("recbRN_test_sample_rank.npy", trk[tidx])
res = dict(host="serverA (CPU, tkg-gpu:latest)", post_hoc=True, lambda_selected=rb["lambda_selected"],
           valid_MRR_by_lambda=rb["valid_MRR_by_lambda"], valid_full=rb["valid"], counts=cnt, elapsed_s=round(time.time() - t0, 1))
jdump(res, "recbRN_summary.json"); jdump(res, f"{WORK}/recbRN_summary.json")
os.system(f"chmod -R a+rX {RD} {WORK}")
log("ALL DONE")
