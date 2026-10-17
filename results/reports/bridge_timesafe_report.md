# Time-respecting bridge cohort index — polecat re-measurement (pilot 1 follow-up)

**Question.** In pilot 1 the stage-2 bridge ranker was trained on 8,000 coverage-hard TRAIN queries whose bridge
features came from a `BridgeIndex` built once over *all* train facts, so a training query at time t saw its own gold
fact and every later train fact. Does training the ranker on snapshot-before-t features change HRCOV,
HRCOV-with-bridge (`HRCOV_BR`), or the polecat headline (HRCOV − HR +0.0375 / +0.0360; HRCOV 0.272 vs the best
published 0.256)?

**Answer.** No, the headline does not change. With the time-safe ranker, HRCOV − HR is +0.0384 [+0.0378, +0.0389] (RE-GCN) and +0.0360 [+0.0355, +0.0365] (LogCL), against +0.0375 / +0.0360 in pilot 1; HRCOV reaches 0.2727 / 0.2721 (official evaluator, seed 0: 0.2729 / 0.2722), still above the best published 0.256, and the pre-registered rule passes on both bases with no head damage. HRCOV itself moves by +0.0009 [+0.0008, +0.0011] (RE-GCN; below the 0.001 materiality threshold) and +0.0001 [-0.0001, +0.0002] (LogCL; not significant). HRCOV-with-bridge (HRCOV_BR) moves by +0.0004 [+0.0002, +0.0005] / +0.0001 [+0.0000, +0.0003]. What does change is the bridge's own picture: the leak had inflated cohort reach (0.959 → 0.644), and the time-safe ranker ranks better at short list lengths (pool-absent gold in the top-50 bridge list 0.074 → 0.094 RE-GCN, 0.066 → 0.087 LogCL), now above the popularity control (0.075 / 0.063). That improvement is worth about +0.001 MRR end-to-end at most.

Verdict file: `bridge_timesafe_verdict.json`. Pilot-1 verdict being revised: `polecat_pilot_verdict.json` and `polecat_pilot_report.md`.

---

## 1. What changed and what did not

* **Changed (only this):** the stage-2 bridge ranker's training cohort. Same 30,000 sampled coverage-hard train queries
  (rng 0), same "first 8,000 reached" rule, same ranker recipe (X16 + cascade features, MLP 64-64, pairwise logistic
  loss, 64 negatives, 40 fixed epochs, 3-seed ensemble; there is no valid-side selection in this stage, as before).
  Each cohort query at time t now uses an index built from train facts with timestamp < t — **one exact snapshot per
  distinct cohort timestamp** (no bucketing; staleness t − cut = 0 for every query), asserted per snapshot
  (max fact timestamp < min query timestamp).
* **Unchanged:** valid/test bridge features (they already used only train facts, all earlier than every valid/test
  timestamp), the pattern-prior block, the base models, F_hist, HR (not retrained; its inputs were verified identical,
  §3), the HRCOV recipe and its valid-only selection (fit/select split at the 70% valid-timestamp quantile).
* **Re-measured:** C_cov bridge block → exact base log-p lookup → features → HRCOV and HRCOVFEAT (3 seeds) →
  post-hoc HRCOV_BR (top-50 + bridge block) and HRCOV_PAT (top-50 + pattern block) (3 seeds) → official py-tgb
  Evaluator on HRCOV and HRCOV_BR seed 0 → paired bootstrap (Poisson(1), B = 2,000) old vs new on all 532,636 test
  queries.

## 2. Effect on the training cohort

| base | cohort index | gold reached by natural bridge set | median |C| (share of N) | ranker training queries | snapshots |
|---|---|---|---|---|---|
| regcn | static all-train (old) | 0.9594 | 27713 (18.4%) | 8000 | — |
| regcn | snapshot-before-t, exact (new) | 0.6444 | 13216 (8.8%) | 8000 | 1190 snapshots, staleness median 0.0 |
| logcl | static all-train (old) | 0.9483 | 27094 (18.0%) | 8000 | — |
| logcl | snapshot-before-t, exact (new) | 0.6362 | 13058 (8.7%) | 8000 | 1188 snapshots, staleness median 0.0 |

The transductive index made the gold reachable for 0.959 (RE-GCN) / 0.948 (LogCL) of the cohort; with snapshot-before-t indices
it is 0.644 / 0.636, with natural candidate sets about half as large. The static index contained two kinds of information a training query could not have at time t: its own (s, r, o) fact and every later train fact (including later s–o contacts). The 8,000 training queries are still the first 8,000 reached in sample order, so the new cohort is a different, harder subset (30,000 sampled → 19,333 / 19,086 reached). Each base needed 1,190 / 1,188 snapshot indices (summed build time 14,358 s / 14,233 s, 1,291 s / 1,278 s wall at 12 workers).

## 3. Input-identity checks

* Base ranks identical old vs new (both bases): True / True.
* F_hist of the base top-50 identical for 100% of valid-sample and 100% of test queries (RE-GCN), 100% / 100% (LogCL) ⇒ HR is unaffected and was not retrained.
* Pattern-prior list identical for 100% / 100% of test queries.
* C_cov columns 0–99 (top-50 + pattern block) identical for 90.9% / 94.8% of test queries; the rest have a pattern block shorter than 50, so bridge candidates start before column 100 (this also means HRCOV_PAT contains some bridge candidates, as in pilot 1).
* Exact base log-p lookup vs dump (same check as pilot 1): test rank mismatches 1,506 (RE-GCN) / 1,543 (LogCL) of 532,636 (agreement 0.9972 / 0.9971), max |Δ log p(gold)| 1.5e-5 / 1.0e-5.

## 4. Effect on the bridge lists (test)

| base | Lbr top-10 overlap old/new | top-50 | top-200 | C_cov bridge block overlap | gold in Lbr@10 old → new | gold in Lbr@50 | gold in Lbr@50 (pool-absent) | gold in bridge block | gold in C_cov |
|---|---|---|---|---|---|---|---|---|---|
| regcn | 0.366 | 0.552 | 0.686 | 0.459 | 0.0870 → 0.0946 | 0.2228 → 0.2368 | 0.0740 → 0.0943 | 0.0278 → 0.0337 | 0.6001 → 0.6069 |
| logcl | 0.359 | 0.505 | 0.657 | 0.374 | 0.0775 → 0.1036 | 0.1991 → 0.2339 | 0.0659 → 0.0869 | 0.0236 → 0.0308 | 0.5975 → 0.6051 |

Overlap = mean share of the old top-k list that is also in the new top-k list (100k uniform test sample). The new ranker re-orders the bridge candidates substantially (about half of the top-50 changes) and places the gold higher: among pool-absent test queries the gold is in the top-50 bridge list for 0.0943 (RE-GCN) / 0.0869 (LogCL) of queries, against 0.0740 / 0.0659 before and 0.0753 / 0.0627 for the pilot-1 popularity control (degree-matched 0.0229 / 0.0169); at B = 10: 0.0283 / 0.0263 vs popularity 0.0236 / 0.0182. The controls come from pilot 1's 20k test sample and were sized to the old list, so this comparison is approximate. The natural bridge set (valid/test index) is unchanged, as expected.

## 5. Full-test results (n = 532,636, both directions, 3-seed means; `_old` = pilot 1, `_new` = time-safe)

| base | system | MRR (3-seed mean) | seed s.d. | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|
| regcn | HR_old | 0.2343 | 0.0000 | 0.1682 | 0.2694 | 0.3631 |
| regcn | HRCOV_old | 0.2718 | 0.0004 | 0.1884 | 0.3102 | 0.4390 |
| regcn | HRCOV_new | 0.2727 | 0.0003 | 0.1892 | 0.3110 | 0.4399 |
| regcn | HRCOV_BR_old | 0.2393 | 0.0001 | 0.1720 | 0.2752 | 0.3704 |
| regcn | HRCOV_BR_new | 0.2396 | 0.0002 | 0.1722 | 0.2755 | 0.3708 |
| regcn | HRCOVFEAT_old | 0.2387 | 0.0002 | 0.1722 | 0.2749 | 0.3678 |
| regcn | HRCOVFEAT_new | 0.2385 | 0.0001 | 0.1719 | 0.2749 | 0.3678 |
| regcn | HRCOV_PAT_old | 0.2717 | 0.0003 | 0.1889 | 0.3105 | 0.4385 |
| regcn | HRCOV_PAT_new | 0.2717 | 0.0002 | 0.1888 | 0.3104 | 0.4386 |
| logcl | HR_old | 0.2361 | 0.0001 | 0.1689 | 0.2709 | 0.3683 |
| logcl | HRCOV_old | 0.2721 | 0.0003 | 0.1888 | 0.3104 | 0.4391 |
| logcl | HRCOV_new | 0.2721 | 0.0002 | 0.1888 | 0.3102 | 0.4390 |
| logcl | HRCOV_BR_old | 0.2414 | 0.0002 | 0.1731 | 0.2775 | 0.3757 |
| logcl | HRCOV_BR_new | 0.2416 | 0.0001 | 0.1730 | 0.2775 | 0.3761 |
| logcl | HRCOVFEAT_old | 0.2404 | 0.0002 | 0.1727 | 0.2768 | 0.3733 |
| logcl | HRCOVFEAT_new | 0.2405 | 0.0004 | 0.1728 | 0.2768 | 0.3732 |
| logcl | HRCOV_PAT_old | 0.2715 | 0.0001 | 0.1888 | 0.3100 | 0.4378 |
| logcl | HRCOV_PAT_new | 0.2715 | 0.0002 | 0.1888 | 0.3100 | 0.4379 |

| base | contrast | Δ MRR [95% CI] | p (one-sided) |
|---|---|---|---|
| regcn | HRCOV_new - HR_old | +0.0384 [+0.0378, +0.0389] | 0.0 |
| regcn | HRCOV_old - HR_old (pilot-1 verdict) | +0.0375 [+0.0369, +0.0380] | 0.0 |
| regcn | HRCOV_new - HRCOV_old | +0.0009 [+0.0008, +0.0011] | 0.0 |
| regcn | HRCOV_BR_new - HR_old | +0.0053 [+0.0050, +0.0056] | 0.0 |
| regcn | HRCOV_BR_new - HRCOV_BR_old | +0.0004 [+0.0002, +0.0005] | 0.0 |
| regcn | HRCOV_BR_new - HRCOVFEAT_new | +0.0011 [+0.0010, +0.0013] | 0.0 |
| regcn | HRCOV_BR_old - HRCOVFEAT_old | +0.0006 [+0.0005, +0.0007] | 0.0 |
| regcn | HRCOV_new - HRCOV_PAT_new | +0.0011 [+0.0009, +0.0012] | 0.0 |
| regcn | HRCOV_old - HRCOV_PAT_old | +0.0001 [-0.0000, +0.0002] | 0.056 |
| regcn | HRCOVFEAT_new - HRCOVFEAT_old | -0.0002 [-0.0003, -0.0000] | 0.997 |
| regcn | HRCOV_PAT_new - HRCOV_PAT_old | -0.0000 [-0.0002, +0.0001] | 0.7395 |
| logcl | HRCOV_new - HR_old | +0.0360 [+0.0355, +0.0365] | 0.0 |
| logcl | HRCOV_old - HR_old (pilot-1 verdict) | +0.0360 [+0.0355, +0.0365] | 0.0 |
| logcl | HRCOV_new - HRCOV_old | +0.0001 [-0.0001, +0.0002] | 0.1905 |
| logcl | HRCOV_BR_new - HR_old | +0.0055 [+0.0052, +0.0057] | 0.0 |
| logcl | HRCOV_BR_new - HRCOV_BR_old | +0.0001 [+0.0000, +0.0003] | 0.013 |
| logcl | HRCOV_BR_new - HRCOVFEAT_new | +0.0011 [+0.0009, +0.0012] | 0.0 |
| logcl | HRCOV_BR_old - HRCOVFEAT_old | +0.0010 [+0.0009, +0.0011] | 0.0 |
| logcl | HRCOV_new - HRCOV_PAT_new | +0.0006 [+0.0005, +0.0007] | 0.0 |
| logcl | HRCOV_old - HRCOV_PAT_old | +0.0006 [+0.0005, +0.0007] | 0.0 |
| logcl | HRCOVFEAT_new - HRCOVFEAT_old | +0.0001 [-0.0000, +0.0002] | 0.139 |
| logcl | HRCOV_PAT_new - HRCOV_PAT_old | +0.0000 [-0.0001, +0.0002] | 0.314 |

Pre-registered rule re-evaluated with the time-safe HRCOV (HRCOV beats HR with one-sided p < 0.05 and Δ ≥ 0.001, no
head damage): **PASS on both bases.** RE-GCN +0.0384 [+0.0378, +0.0389], p < 5e-4, head stratum +0.0240 [+0.0233, +0.0248]; LogCL +0.0360 [+0.0355, +0.0365], p < 5e-4, head +0.0206 [+0.0199, +0.0214]; no damage.

### Official py-tgb Evaluator on the new rows (seed 0)

| base | system (new, seed 0) | official py-tgb MRR | official H@10 | harness MRR, same ranks |
|---|---|---|---|---|
| regcn | HRCOV_s0 | 0.272883 | 0.4402 | 0.272883 |
| regcn | HRCOV_BR_s0 | 0.239368 | 0.3709 | 0.239368 |
| logcl | HRCOV_s0 | 0.272240 | 0.4391 | 0.272240 |
| logcl | HRCOV_BR_s0 | 0.241474 | 0.3759 | 0.241474 |

## 6. Strata (HRCOV new vs old)

| base | stratum | share | HR | HRCOV old | HRCOV new | HRCOV new − old [95% CI] | HRCOV_BR new − old [95% CI] |
|---|---|---|---|---|---|---|---|
| regcn | pool-absent | 0.579 | 0.0027 | 0.0756 | 0.0768 | +0.0012 [+0.0011, +0.0014] | +0.0003 [+0.0002, +0.0004] |
| regcn | cold (gold train freq<50) | 0.333 | 0.0678 | 0.1226 | 0.1240 | +0.0014 [+0.0012, +0.0016] | +0.0009 [+0.0008, +0.0011] |
| regcn | cold & pair history | 0.117 | 0.1875 | 0.3371 | 0.3412 | +0.0042 [+0.0036, +0.0048] | +0.0019 [+0.0014, +0.0023] |
| regcn | cold & no pair history | 0.216 | 0.0032 | 0.0068 | 0.0067 | -0.0002 [-0.0002, -0.0001] | +0.0004 [+0.0003, +0.0005] |
| regcn | no pair history | 0.386 | 0.0124 | 0.0208 | 0.0214 | +0.0006 [+0.0005, +0.0007] | +0.0007 [+0.0006, +0.0009] |
| regcn | no pair & 2-hop unreachable | 0.172 | 0.0039 | 0.0038 | 0.0045 | +0.0007 [+0.0006, +0.0009] | -0.0001 [-0.0003, +0.0000] |
| regcn | first-seen entity | 0.108 | 0.0039 | 0.0048 | 0.0059 | +0.0010 [+0.0008, +0.0013] | -0.0002 [-0.0004, +0.0000] |
| regcn | head (>=500) | 0.455 | 0.3777 | 0.4010 | 0.4017 | +0.0007 [+0.0004, +0.0010] | -0.0009 [-0.0011, -0.0006] |
| regcn | gold in NEW bridge block | 0.034 | 0.0075 | 0.0280 | 0.0416 | +0.0135 [+0.0128, +0.0143] | +0.0210 [+0.0200, +0.0222] |
| logcl | pool-absent | 0.569 | 0.0025 | 0.0714 | 0.0715 | +0.0002 [+0.0000, +0.0003] | +0.0003 [+0.0002, +0.0004] |
| logcl | cold (gold train freq<50) | 0.333 | 0.0682 | 0.1250 | 0.1246 | -0.0003 [-0.0005, -0.0002] | +0.0003 [+0.0002, +0.0005] |
| logcl | cold & pair history | 0.117 | 0.1893 | 0.3435 | 0.3427 | -0.0008 [-0.0014, -0.0003] | +0.0003 [-0.0001, +0.0007] |
| logcl | cold & no pair history | 0.216 | 0.0028 | 0.0070 | 0.0069 | -0.0001 [-0.0002, -0.0000] | +0.0004 [+0.0003, +0.0004] |
| logcl | no pair history | 0.386 | 0.0106 | 0.0203 | 0.0203 | +0.0000 [-0.0001, +0.0001] | +0.0002 [+0.0001, +0.0003] |
| logcl | no pair & 2-hop unreachable | 0.172 | 0.0015 | 0.0011 | 0.0013 | +0.0002 [+0.0001, +0.0003] | +0.0000 [-0.0001, +0.0001] |
| logcl | first-seen entity | 0.108 | 0.0007 | 0.0010 | 0.0011 | +0.0002 [+0.0001, +0.0003] | -0.0001 [-0.0001, -0.0000] |
| logcl | head (>=500) | 0.455 | 0.3789 | 0.3993 | 0.3996 | +0.0003 [+0.0001, +0.0005] | -0.0003 [-0.0005, -0.0000] |
| logcl | gold in NEW bridge block | 0.031 | 0.0071 | 0.0282 | 0.0402 | +0.0120 [+0.0113, +0.0127] | +0.0208 [+0.0197, +0.0218] |

## 7. Does the polecat headline change?

| quantity | pilot 1 (transductive cohort) | time-safe cohort | change |
|---|---|---|---|
| HRCOV − HR, RE-GCN | +0.0375 [0.0369, 0.0380] | +0.0384 [+0.0378, +0.0389] | +0.0009 [+0.0008, +0.0011] |
| HRCOV − HR, LogCL | +0.0360 [0.0355, 0.0365] | +0.0360 [+0.0355, +0.0365] | +0.0001 [-0.0001, +0.0002] |
| HRCOV MRR, RE-GCN / LogCL (3-seed) | 0.2718 / 0.2721 | 0.2727 / 0.2721 | — |
| HRCOV seed 0, official py-tgb | 0.2719 / 0.2717 | 0.2729 / 0.2722 | — |
| vs best published (CountTRuCoLa 0.256) | above | above | — |
| pre-registered rule | PASS / PASS | PASS / PASS | — |

Criterion for "the headline changes" (coded in `pc_ts_report.py` before any time-safe test number was read): |HRCOV new − old| ≥ 0.001 on a base, or the rule's PASS status flips, or HRCOV falls below 0.256. None holds (largest |Δ| = 0.0009, RE-GCN, whose CI upper bound 0.0011 just reaches the threshold), so `headline_changes = false` on both bases.

Attribution after the fix: the pattern block still carries the gain (HRCOV_PAT − HR +0.0373 [+0.0368, +0.0379] / +0.0354 [+0.0349, +0.0359]); the bridge block on top of the pattern block adds +0.0011 [+0.0009, +0.0012] / +0.0006 [+0.0005, +0.0007] (pilot 1: +0.0001 [-0.0000, +0.0002] / +0.0006 [+0.0005, +0.0007]); bridge candidates beyond their features add +0.0011 [+0.0010, +0.0013] / +0.0011 [+0.0009, +0.0012] (pilot 1: +0.0006 [+0.0005, +0.0007] / +0.0010 [+0.0009, +0.0011]). On RE-GCN, the stratum results show where the +0.0009 comes from: queries whose gold sits in the new bridge block (3.4% of test) gain +0.0135, and queries whose gold was in the old bridge block lose −0.0084; cold golds without pair history still gain nothing (−0.0002). The pilot-1 conclusion — the HRCOV gain is history-aware pool widening, not cold-entity coverage — stands; the pilot-1 statements about the bridge itself that need revising are listed in the verdict (`pilot1_statements_revised`).

## 8. Deviations

1. HR (and HR_R) not retrained: their inputs (base top-50 + F_hist) do not depend on the bridge and were verified identical (F_hist top-50 identical for 100% of valid-sample and 100% of test queries, both bases); pilot-1 HR ranks are the comparator.
2. Re-measured arms: HRCOV and HRCOVFEAT (pre-registered recipe, 3 seeds) and the post-hoc diagnostics HRCOV_BR and HRCOV_PAT (3 seeds); HRCOV_R, HRCOV_DYAD, HRCOV_NODYAD, PAT_BUMP, BR_CS and the augmentation arms were not re-run.
3. Official py-tgb Evaluator on seed 0 only (HRCOV_s0, HRCOV_BR_s0); 3-seed means use the harness evaluator, which matches the official one to 1e-6 on all four rows.
4. G5 density controls (degree-matched / popularity sets at matched |C|) were not recomputed for the time-safe cohort; the pool-absent list comparison uses the pilot-1 controls (20k test sample, sized to the old lists).
5. HRCOV_PAT (columns 0-99) includes bridge candidates for queries whose pattern block has fewer than 50 entries (C_cov columns 0-99 identical old vs new for 90.9% RE-GCN / 94.8% LogCL of test queries); same definition as pilot 1.
6. Compute: CPU work stayed on serverA (serverC has ~10 GB free RAM, too little for the lists stage); on request the running containers were capped at 4 CPUs each (8 total) instead of being restarted, which stretched the target-list stage to ~4.6 h per base.
7. Valid/test bridge features still use the all-train index (train facts only, all earlier than every valid/test query), exactly as in pilot 1; only the stage-2 training cohort changed.

## 9. Artifacts

* `bridge_timesafe_verdict.json` — all numbers (both bases), input checks, cohort/list changes, comparisons, strata, official rows, compute, deviations
* Code (changed for this rerun): `pc_common.py`, `pc_lists.py`, `pc_train_diag.py`, `pc_ts_agg.py`, `pc_ts_chain.sh`, `pc_ts_check.py`, `pc_ts_report.py`
* Code (unchanged, reused): `pc_lookup.py`, `pc_features.py`, `pc_train.py`
* Jobs on serverA: RE-GCN chain redacted-id, LogCL chain redacted-id; remote outputs under `tracks/pilot_polecat/work_ts/` (pilot-1 `work/` untouched).
