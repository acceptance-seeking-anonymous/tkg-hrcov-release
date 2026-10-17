#!/usr/bin/env python3
"""pilot_icews / pi_prepB.py -- Stage B base-independent preparation (CPU).  Run with PI_STAGE=stageB PI_TEST_FRAC=1.0.

Re-uses the Stage A base-independent products (work/) and writes the Stage B work dir (work_stageB/):
  * valid_sample_idx.npy  : the Stage A 200,000-query valid sample (copied, identical rows)
  * test_sample_idx.npy   : arange(4,651,378) -- the full test
  * te_TKGL_ICEWS.npz     : pattern target-encoding tables (early 70% of train days; base independent; copied)
  * recb_*                : RecB-lite full valid/test ranks (split order; computed on FULL valid/test in Stage A) ->
                            recb_test_sample_rank.npy = full test, recb_valid_sample_rank.npy = valid sample
  * recbRN_test_sample_rank.npy : RecB-lite on relation-normalised data (post-hoc row of Stage A) on the full test
  * prep_summary.json     : Stage A summary + Stage B note (RecB-lite lambda unchanged)
  * strata_valid.npz      : copied (same valid sample);  strata_test.npz : recomputed on the FULL test (pi_prep recipe)
  * tgb_ns/<test_ns.pkl>  : copied
"""
import os, sys, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import P, DS, WORK, STAGE, TEST_FRAC, Splits, split_queries, test_sample_idx, jdump, log_fn
from rs_features import Hist
from pi_prep import profiler_to_split_order
assert STAGE == "stageB" and TEST_FRAC >= 1.0 and WORK.endswith("work_stageB"), (STAGE, TEST_FRAC, WORK)
t0 = time.time(); log = log_fn(t0)
A = f"{P}/work"; os.makedirs(WORK, exist_ok=True); os.makedirs(f"{WORK}/tgb_ns", exist_ok=True)
S = Splits(DS); N, R = S.n_ent, S.R
nt = 2 * len(S.q["test"]); tidx = test_sample_idx(nt); assert (tidx == np.arange(nt)).all()
np.save(f"{WORK}/test_sample_idx.npy", tidx)
for f in ("valid_sample_idx.npy", "te_TKGL_ICEWS.npz", "recb_test_rank.npy", "recb_valid_rank.npy", "recb_test_rank_allLam.npy",
          "recb_valid_sample_rank.npy", "recb_profile.json", "recbRN_profile.json", "strata_valid.npz", "profile_tkg_v2_saveranks.py"):
    shutil.copy2(f"{A}/{f}", f"{WORK}/{f}")
for f in os.listdir(f"{A}/tgb_ns"):
    shutil.copy2(f"{A}/tgb_ns/{f}", f"{WORK}/tgb_ns/{f}")
rt = np.load(f"{WORK}/recb_test_rank.npy"); assert len(rt) == nt
np.save(f"{WORK}/recb_test_sample_rank.npy", rt)
prof = json.load(open(f"{WORK}/recbRN_profile.json")); rb = prof["recb_lite"]; j = rb["lambda_grid"].index(rb["lambda_selected"])
trk = np.load(f"{A}/recbRN_test_rank_prof.npy")[:, j][profiler_to_split_order(S, "test")]
np.save(f"{WORK}/recbRN_test_sample_rank.npy", trk)
summ = json.load(open(f"{A}/prep_summary.json"))
summ["stageB"] = dict(note="Stage B: test sample = full test; valid sample, TE tables, RecB-lite ranks reused from Stage A (base independent)",
                      n_test=int(nt), recbRN_lambda=rb["lambda_selected"])
summ["n_queries"]["test_sample"] = int(nt)
# strata on the full test (pi_prep recipe)
relmap = np.load(f"{P}/stage/relmap.npy"); tr_rels = np.unique(S.q["train"][:, 1])
Hh = Hist(S); Q = split_queries(S, "test")[tidx]
t1 = time.time()
s, r, o, t = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
st = {}
for lo in range(0, len(Q), 200000):
    hi = min(len(Q), lo + 200000)
    hf = Hh.feats(s[lo:hi], r[lo:hi], t[lo:hi], o[lo:hi, None])
    for k in ("ntrip", "ndyad", "nent", "nquery", "nsubj"):
        st.setdefault(k, []).append(np.asarray(hf[k]).reshape(hi - lo, -1)[:, 0])
    if (lo // 200000) % 5 == 0: log("strata", hi)
st = {k: np.concatenate(v).astype(np.int64) for k, v in st.items()}
out = dict(sro=Q, trip=st["ntrip"] > 0, dyad=st["ndyad"] > 0, nev_s=st["nsubj"], nev_o=st["nent"], n_sr=st["nquery"])
out["first_seen"] = (out["nev_s"] == 0) | (out["nev_o"] == 0)
out["self_loop"] = s == o
rb_ = r % R
out["rel_unseen_train"] = ~np.isin(rb_, tr_rels)
out["rel_merged_code"] = relmap[rb_] != rb_
out["freq_stratum"] = S.stratum(S.freq[o]); out["in_hist"] = S.in_hist(s, o)
np.savez(f"{WORK}/strata_test.npz", **out)
# consistency with the Stage A sample strata
sa = np.load(f"{A}/strata_test.npz"); tA = np.load(f"{A}/test_sample_idx.npy")
agree = {k: bool((np.asarray(sa[k]) == np.asarray(out[k])[tA]).all()) for k in ("sro", "trip", "dyad", "first_seen", "rel_unseen_train", "freq_stratum", "in_hist")}
summ["strata_test"] = dict(n=int(len(Q)), seconds=round(time.time() - t1, 1), stageA_sample_agreement=agree,
                           shares={k: float(np.mean(out[k])) for k in ("trip", "dyad", "first_seen", "self_loop", "rel_unseen_train", "rel_merged_code", "in_hist")},
                           head_share=float(np.mean(out["freq_stratum"] == 2)), cold_share=float(np.mean(out["freq_stratum"] == 0)))
log(json.dumps(summ["strata_test"]))
assert all(agree.values()), agree
jdump(summ, "prep_summary.json"); jdump(summ, f"{WORK}/prep_summary.json")
os.system(f"chmod -R a+rX {WORK}")
log("ALL DONE")
