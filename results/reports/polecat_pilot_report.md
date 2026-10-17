# Pilot 1 of 4 — the coverage stack on TGB 2.0 `tkgl-polecat`

**Question.** Do the coverage / cold-entity mechanisms (GraphBridge typed bridge + stage-2 ranker, 6.1
pattern-reachable evidence, 6.5 gated augmentation) add measurable full-test MRR *on top of* the history
re-ranker (6.2 arm B) on a benchmark where no-pair-history and first-seen mass is large?

**Answer (pre-registered rule, both bases).** Yes for the coverage *candidate pool* (`HRCOV` vs `HR`):
RE-GCN +0.0375 MRR [95% CI 0.0369, 0.0380], LogCL +0.0360 [0.0355, 0.0365], one-sided p < 5e-4 (0 of 2,000
bootstrap replicates ≤ 0), head guardrail passes (head stratum +0.023 / +0.020). `useful_on_polecat = true`.
The mechanism that carries the gain is **not** cold-entity coverage: post-hoc attribution shows it is the
pattern-prior block re-admitting *historically linked* partners that the base model ranks outside its top-50
(history-aware pool widening). The bridge block adds +0.0006 / +0.0010 beyond its features, most of the bridge's reach is reproduced by
density / popularity controls at matched set size, the no-pair-and-2-hop-unreachable and first-seen strata gain ≈ 0, and
6.5 gated augmentation adds only +0.0011 / +0.0019 on the frozen base, located in the head stratum, with a loss on the cold stratum (the 6.5 rule fails on both bases).

Verdict file: `polecat_pilot_verdict.json`; pre-registered rule:
`polecat_decision_rule.json` (saved before any polecat test number existed);
pipeline: `pilot_pipeline_README.md`.

---

## 1. Setup

**Data and protocol.** `tkgl-polecat` (TGB 2.0): 150,931 entities, 16 relations (+16 inverse), 1,826 daily
timestamps; train / valid / test facts 1,246,556 / 266,736 / 266,318, TGB 70/15/15 split by timestamp. Queries in
both directions: 533,472 valid, **532,636 test** (all scored). Time-aware filtered, 1-vs-all ranking over all
entities, ground-truth online history before t (the TGB setting).

**TGB identity checks (job).** The TGB → harness entity map is the identity (`h2t == arange`); TGB edge
order equals harness CSV order; the TGB negative/conflict sets equal the harness time-aware filter for all 454,516
valid and 447,006 test (s, r, t) keys; the TGB evaluator rank is 1 + #greater + 0.5·#ties. The harness exact
filtered MRR therefore *is* the TGB statistic, and this was confirmed numerically with the official py-tgb 2.3.0
`Evaluator` on both bases and on `HR`/`HRCOV` seed 0 (Table 1, Table 2).

**Bases** (harness `base_dump.py`, native torch, h = 200, 2 layers, history 10, 12 epochs, seed 42; full score rows
dumped): RE-GCN and LogCL (cl_weight 0.1). TiRGN was not run (per-timestamp history cache over 1,826 days × 151k
entities, estimated > 4 h).

**Systems** (all on frozen bases; every hyperparameter, epoch, calibration weight and threshold chosen on valid
only — valid timestamps < 70% quantile = fit, rest = select; test scored once):

| system | candidates | features | notes |
|---|---|---|---|
| `HR` (history re-ranker, 6.2 arm B) | base top-50 | F_hist (8 base + 11 online evidence) | MLP 64-64, listwise CE, fused α·log p + β·f, 3 seeds |
| `HR_R` (secondary) | base top-50 | F_hist | residual R0 |
| `HRCOV` (coverage stack) | top-50 ∪ 50 pattern-prior ∪ 50 stage-2 bridge (W = 150) | F_hist + F_cov (11) | same recipe; exact base log p for every candidate by lookup |
| `HRCOV_R` (secondary) | as HRCOV | as HRCOV | residual R0 |
| `HRCOVFEAT` (diagnostic) | base top-50 | F_hist + F_cov | coverage features without coverage candidates |
| `PAT_BUMP` (6.1, non-learned) | full row | pattern prior | s' = log p + δ·1[c ∈ top-B pattern list]; valid-selected B = 10, δ = 2 |
| `BR_CS` (bridge backfill) | full row | stage-2 bridge | ConfidenceSwitch; valid-selected q = 0.3, M = 200, B = 200 |
| `AUG` (6.5) | frozen base, injected evidence | — | gold-injection oracle, then utility-gated arms (stop rule fired) |

Stage-2 bridge ranker: G12 arm-5 features without the event-LM column (X16 + cascade), 3-seed ensemble trained on
8,000 coverage-hard train queries. Pattern prior: hierarchical target-encoded pattern confidence fitted on the first
70% of train timestamps.

**Decision rule (pre-registered).** Useful iff on both bases `HRCOV` beats `HR` on full-test MRR with one-sided
paired-bootstrap p < 0.05 and ΔMRR ≥ 0.001, and the head-damage guardrail (gold train frequency ≥ 500; DAMAGE iff
the 95% CI lies below 0 and |Δ| ≥ 0.001) passes. Statistic: per-query reciprocal ranks averaged over 3 seeds;
Poisson(1) paired bootstrap over queries, B = 2,000 (harness `paired_bootstrap`).

## 2. Base verification (Table 1)

| row | test MRR | H@1 | H@3 | H@10 | source |
|---|---|---|---|---|---|
| RE-GCN (ours), harness exact filtered | 0.16501 | 0.1042 | 0.1858 | 0.2855 | job |
| RE-GCN (ours), official py-tgb Evaluator | 0.16501 | — | — | 0.2855 | job / lookup |
| LogCL (ours), harness | 0.17846 | 0.1161 | 0.2016 | 0.3023 | job |
| LogCL (ours), official py-tgb Evaluator | 0.17846 | — | — | 0.3023 | lookup |
| RecB-lite (λ = 0.99, valid-selected) | 0.19626 | 0.1356 | 0.2200 | 0.3146 | pc_prep |
| RecB, official TGB script, default (λ 0.1, α 0.99) | 0.16709 | — | — | 0.2638 | job |
| published: RE-GCN (TGB, DGL, history 3) | 0.175 | | | | TGB 2.0 leaderboard |
| published: CEN | 0.184 | | | | TGB 2.0 leaderboard |
| published: TGB RecB | 0.198 | | | | TGB 2.0 leaderboard |
| published: CountTRuCoLa (best published) | 0.256 | | | | TGB 2.0 leaderboard |

Harness vs official evaluator: per-query rank identical for 99.56% (RE-GCN) / 99.45% (LogCL) of queries, the rest
are ties only (max |Δrank| = 1.5), MRR equal to 1e-7. Our RE-GCN is 0.010 below the TGB row; the configurations
differ (TGB: train history 3, shuffled snapshots, layer norm, basis decomposition, valid-best checkpoint; ours: history
10, fixed 12 epochs, no valid selection) and we did not tune toward the TGB row. The official RecB in default mode
(0.167) is below both the TGB RecB row (0.198, train mode, per-relation λ/α on valid) and our RecB-lite (0.196);
train mode was not run (effectively serial, projected > 8 h).

Lookup vs stored dump (the coverage stack needs exact base log p for candidates outside the dumped top list):
rank agreement RE-GCN 0.99767 valid / 0.99711 test, LogCL 0.99763 / 0.99705, max |Δrank| = 2, max |Δ log p(gold)|
≈ 1e-5 (fp32 near-ties). The pre-set 0.999 assert was relaxed to 0.99; reference systems scored with lookup ranks
are compared with the lookup base (`base_lookup − base` = 0.0000, p = 0.34 / 0.07).

## 3. Full-test results (Table 2; n = 532,636, both directions, 3-seed means)

| base | system | seeds | MRR | H@1 | H@3 | H@10 | Δ vs base | Δ vs HR | host |
|---|---|---|---|---|---|---|---|---|---|
| - | RecB-lite (lambda=0.99, valid-selected) | - | 0.1963 | 0.1356 | 0.2200 | 0.3146 |  |  | serverA CPU |
| - | RecB official TGB script (default lambda=0.1, alpha=0.99; train-mode search not run) | - | 0.1671 | — | — | 0.2638 |  |  | serverA CPU |
| regcn | regcn (frozen base) | - | 0.1650 | 0.1042 | 0.1858 | 0.2855 |  |  | serverA GPU |
| regcn | regcn (frozen base) [official py-tgb Evaluator] | - | 0.1650 | — | — | 0.2855 |  |  | serverA GPU |
| regcn | HR | mean(3) | 0.2343 | 0.1682 | 0.2694 | 0.3631 | +0.0693 |  | serverA GPU |
| regcn | HR_R | mean(3) | 0.2340 | 0.1678 | 0.2691 | 0.3629 | +0.0690 | -0.0004 | serverA GPU |
| regcn | HRCOV | mean(3) | 0.2718 | 0.1884 | 0.3102 | 0.4390 | +0.1068 | +0.0375 | serverA GPU |
| regcn | HRCOV_R | mean(3) | 0.2708 | 0.1874 | 0.3090 | 0.4381 | +0.1058 | +0.0364 | serverA GPU |
| regcn | HRCOVFEAT | mean(3) | 0.2387 | 0.1722 | 0.2749 | 0.3678 | +0.0737 | +0.0043 | serverA GPU |
| regcn | PAT_BUMP | - | 0.1811 | 0.1131 | 0.2071 | 0.3173 | +0.0161 |  | serverA CPU |
| regcn | BR_CS | - | 0.1650 | 0.1042 | 0.1858 | 0.2855 | +0.0000 |  | serverA CPU |
| regcn | HRCOV_BR (post-hoc diagnostic) | mean(3) | 0.2393 | 0.1720 | 0.2752 | 0.3704 | +0.0743 | +0.0049 | serverA GPU |
| regcn | HRCOV_DYAD (post-hoc diagnostic) | mean(3) | 0.2697 | 0.1887 | 0.3089 | 0.4326 | +0.1047 | +0.0354 | serverA GPU |
| regcn | HRCOV_NODYAD (post-hoc diagnostic) | mean(3) | 0.2426 | 0.1733 | 0.2784 | 0.3784 | +0.0776 | +0.0082 | serverA GPU |
| regcn | HRCOV_PAT (post-hoc diagnostic) | mean(3) | 0.2717 | 0.1889 | 0.3105 | 0.4385 | +0.1067 | +0.0373 | serverA GPU |
| regcn | AUG util_ll | mean(3) | 0.1661 | 0.1045 | 0.1867 | 0.2881 | +0.0011 |  | serverA GPU |
| regcn | AUG random_matched_s0 | - | 0.1644 | 0.1034 | 0.1849 | 0.2858 | -0.0006 |  | serverA GPU |
| regcn | AUG aa_matched_s0 | - | 0.1647 | 0.1038 | 0.1855 | 0.2859 | -0.0003 |  | serverA GPU |
| regcn | AUG unc_matched_s0 | - | 0.1647 | 0.1036 | 0.1855 | 0.2864 | -0.0003 |  | serverA GPU |
| regcn | AUG perturb_s0 | - | 0.1641 | 0.1034 | 0.1844 | 0.2841 | -0.0010 |  | serverA GPU |
| regcn | AUG ungated_sameBm | - | 0.1654 | 0.1036 | 0.1860 | 0.2884 | +0.0004 |  | serverA GPU |
| regcn | AUG util_rr | - | 0.1680 | 0.1060 | 0.1892 | 0.2907 | +0.0030 |  | serverA GPU |
| logcl | logcl (frozen base) | - | 0.1785 | 0.1161 | 0.2016 | 0.3023 |  |  | serverA GPU |
| logcl | logcl (frozen base) [official py-tgb Evaluator] | - | 0.1785 | — | — | 0.3023 |  |  | serverA GPU |
| logcl | HR | mean(3) | 0.2361 | 0.1689 | 0.2709 | 0.3683 | +0.0576 |  | serverA GPU |
| logcl | HR_R | mean(3) | 0.2354 | 0.1679 | 0.2704 | 0.3681 | +0.0569 | -0.0007 | serverA GPU |
| logcl | HRCOV | mean(3) | 0.2721 | 0.1888 | 0.3104 | 0.4391 | +0.0936 | +0.0360 | serverA GPU |
| logcl | HRCOV_R | mean(3) | 0.2710 | 0.1879 | 0.3093 | 0.4380 | +0.0926 | +0.0349 | serverA GPU |
| logcl | HRCOVFEAT | mean(3) | 0.2404 | 0.1727 | 0.2768 | 0.3733 | +0.0620 | +0.0043 | serverA GPU |
| logcl | PAT_BUMP | - | 0.1937 | 0.1254 | 0.2209 | 0.3315 | +0.0152 |  | serverA CPU |
| logcl | BR_CS | - | 0.1785 | 0.1161 | 0.2016 | 0.3023 | +0.0000 |  | serverA CPU |
| logcl | HRCOV_BR (post-hoc diagnostic) | mean(3) | 0.2414 | 0.1731 | 0.2775 | 0.3757 | +0.0630 | +0.0053 | serverA GPU |
| logcl | HRCOV_DYAD (post-hoc diagnostic) | mean(3) | 0.2694 | 0.1887 | 0.3084 | 0.4316 | +0.0909 | +0.0333 | serverA GPU |
| logcl | HRCOV_NODYAD (post-hoc diagnostic) | mean(3) | 0.2448 | 0.1744 | 0.2805 | 0.3838 | +0.0663 | +0.0086 | serverA GPU |
| logcl | HRCOV_PAT (post-hoc diagnostic) | mean(3) | 0.2715 | 0.1888 | 0.3100 | 0.4378 | +0.0930 | +0.0354 | serverA GPU |
| logcl | AUG util_ll | mean(3) | 0.1804 | 0.1168 | 0.2034 | 0.3067 | +0.0019 |  | serverA GPU |
| logcl | AUG random_matched_s0 | - | 0.1774 | 0.1144 | 0.2003 | 0.3026 | -0.0011 |  | serverA GPU |
| logcl | AUG aa_matched_s0 | - | 0.1782 | 0.1155 | 0.2013 | 0.3034 | -0.0002 |  | serverA GPU |
| logcl | AUG unc_matched_s0 | - | 0.1783 | 0.1151 | 0.2015 | 0.3044 | -0.0002 |  | serverA GPU |
| logcl | AUG perturb_s0 | - | 0.1761 | 0.1144 | 0.1985 | 0.2985 | -0.0024 |  | serverA GPU |
| logcl | AUG ungated_sameBm | - | 0.1799 | 0.1154 | 0.2035 | 0.3086 | +0.0015 |  | serverA GPU |
| logcl | AUG util_rr | - | 0.1855 | 0.1221 | 0.2089 | 0.3111 | +0.0070 |  | serverA GPU |

!Full-test MRR

## 4. Decision

| base | ΔMRR HRCOV − HR | 95% CI | p (one-sided) | head stratum Δ [CI] | PASS | secondary HRCOV_R − HR_R |
|---|---|---|---|---|---|---|
| RE-GCN | +0.03745 | [0.03689, 0.03800] | < 5e-4 | +0.0233 [0.0226, 0.0241] | yes | +0.0368 [0.0363, 0.0373] |
| LogCL | +0.03596 | [0.03549, 0.03648] | < 5e-4 | +0.0203 [0.0196, 0.0210] | yes | +0.0357 [0.0352, 0.0362] |

Seed s.d. of HRCOV MRR: 0.0004 (RE-GCN), 0.0003 (LogCL). `HRCOV` seed-0 rows reproduce under the official py-tgb
Evaluator: RE-GCN 0.271851 (H@10 0.4390), LogCL 0.271673 (H@10 0.4385); `HR` seed 0: 0.234356 / 0.236190.

HRCOV (0.272) is above the best published TGB 2.0 row for this dataset (CountTRuCoLa 0.256) under an evaluator
verified to be identical to the official one. This is a statement about this two-stage system on this split, not
a leaderboard submission; §6 shows what the gain consists of.

Training-budget caveat: `HR` reached its best select-split MRR at epoch 26–29 of a 30-epoch cap (select MRR still
rising by ≈ 0.0004 over the last 10 epochs) while `HRCOV` peaked at epochs 4–23. Additional HR epochs would plausibly
add < 0.001, two orders of magnitude below the measured Δ.

## 5. Where the gain lands (RE-GCN; LogCL in the verdict, same pattern)

!Where the gain lands

| stratum | share | base | RecB-lite | HR | HRCOV | Δ HRCOV − HR [95% CI] |
|---|---|---|---|---|---|---|
| cold (gold train freq<50) | 0.333 | 0.0423 | 0.0938 | 0.0678 | 0.1226 | +0.0548 [+0.0539, +0.0558] |
| mid | 0.212 | 0.1248 | 0.1469 | 0.1883 | 0.2288 | +0.0405 [+0.0394, +0.0415] |
| head (>=500) | 0.455 | 0.2735 | 0.2943 | 0.3777 | 0.4010 | +0.0233 [+0.0226, +0.0241] |
| no pair history (no (s,o) event before t) | 0.386 | 0.0107 | 0.0109 | 0.0124 | 0.0208 | +0.0084 [+0.0081, +0.0087] |
| pair history | 0.614 | 0.2620 | 0.3127 | 0.3738 | 0.4295 | +0.0557 [+0.0549, +0.0565] |
| first-seen entity (s or o has no event before t) | 0.108 | 0.0011 | 0.0133 | 0.0039 | 0.0048 | +0.0010 [+0.0007, +0.0012] |
| pool-absent (gold outside base top-50) | 0.579 | 0.0027 | 0.0646 | 0.0027 | 0.0756 | +0.0729 [+0.0721, +0.0737] |
| gold in base top-50 | 0.421 | 0.3886 | 0.3776 | 0.5535 | 0.5421 | -0.0114 [-0.0120, -0.0107] |
| hist=no (harness: no train+valid co-occurrence) | 0.499 | 0.0324 | 0.0675 | 0.0657 | 0.1154 | +0.0498 [+0.0491, +0.0505] |
| exact recurrence | 0.425 | 0.3267 | 0.4459 | 0.4555 | 0.5022 | +0.0466 [+0.0456, +0.0476] |
| non-recurrent | 0.575 | 0.0455 | 0.0116 | 0.0708 | 0.1014 | +0.0307 [+0.0301, +0.0312] |
| no pair & 2-hop reachable | 0.214 | 0.0181 | 0.0114 | 0.0192 | 0.0345 | +0.0152 [+0.0148, +0.0157] |
| no pair & 2-hop unreachable | 0.172 | 0.0015 | 0.0104 | 0.0039 | 0.0038 | -0.0001 [-0.0003, +0.0002] |
| self-loop (s==o) | 0.104 | 0.5003 | 0.5128 | 0.5589 | 0.5342 | -0.0248 [-0.0261, -0.0235] |
| E_emerging | 0.108 | 0.0011 | 0.0133 | 0.0039 | 0.0048 | +0.0010 [+0.0007, +0.0012] |
| R_rec_in | 0.311 | 0.4439 | 0.5017 | 0.6198 | 0.6077 | -0.0121 [-0.0129, -0.0113] |
| R_rec_out | 0.114 | 0.0056 | 0.2930 | 0.0056 | 0.2131 | +0.2075 [+0.2046, +0.2106] |
| D_dyad_in | 0.081 | 0.2678 | 0.0208 | 0.4402 | 0.4295 | -0.0107 [-0.0123, -0.0090] |
| D_dyad_out | 0.108 | 0.0041 | 0.0073 | 0.0041 | 0.1446 | +0.1405 [+0.1379, +0.1431] |
| P_pat_in | 0.018 | 0.1187 | 0.0330 | 0.1509 | 0.1722 | +0.0214 [+0.0186, +0.0243] |
| P_pat_out | 0.064 | 0.0034 | 0.0175 | 0.0034 | 0.0518 | +0.0484 [+0.0471, +0.0497] |
| U_deep_in | 0.010 | 0.1444 | 0.0153 | 0.1247 | 0.0803 | -0.0444 [-0.0484, -0.0402] |
| U_deep_out | 0.187 | 0.0013 | 0.0051 | 0.0013 | 0.0020 | +0.0007 [+0.0007, +0.0008] |

* The gain is concentrated where the gold is **outside the base top-50** (pool-absent, 58% of queries:
  +0.073). Inside the top-50 HRCOV *loses* to HR (−0.011 RE-GCN / −0.007 LogCL): 100 extra candidates dilute the
  listwise re-ranking.
* By the 9-class error taxonomy: recurrence-out-of-pool (R_rec_out, 11%) +0.208, dyad-out-of-pool (D_dyad_out, 11%)
  +0.141, pattern-out (P_pat_out, 6%) +0.048; deep-unreachable (U_deep_out, 19%) +0.0007; emerging/first-seen
  (E_emerging, 11%) +0.001 (LogCL +0.0002).
* By gold train frequency the largest gain is on **cold golds** (< 50 train events, 33% of queries: +0.055 / +0.057),
  but it sits almost entirely in cold golds *with* pair history (11.7% of queries: +0.150 / +0.154, contributing
  +0.017 / +0.018 of the full-test Δ); cold golds without pair history (21.6%) gain +0.004 (post-hoc cross-tab,
  `strata.post_hoc_cold_x_pair_history` in the verdict).
* The cold-entity target strata are essentially untouched: no-pair history (39%) +0.008, of which the
  2-hop-reachable part +0.015 and the **2-hop-unreachable part (17%) −0.0001 [−0.0003, 0.0002]** (LogCL −0.0004,
  significant loss); first-seen entity +0.001.
* Losses: self-loop queries (s = o, 10%) −0.025 (LogCL −0.011); R_rec_in −0.012; D_dyad_in −0.011; U_deep_in
  −0.044 (1%).
* RecB-lite beats HRCOV on R_rec_out (0.293 vs 0.213) and on first-seen (0.013 vs 0.005): a recurrency candidate
  block is a cheaper and stronger source for part of the out-of-pool recurrence mass.

## 6. What the coverage gain consists of (post-hoc attribution, not pre-registered)

| arm (3 seeds, RE-GCN / LogCL) | test MRR | Δ vs HR |
|---|---|---|
| HR | 0.2343 / 0.2361 | — |
| HRCOVFEAT (coverage features, no coverage candidates) | 0.2387 / 0.2404 | +0.0043 / +0.0043 |
| HRCOV_BR (top-50 + bridge block only) | 0.2393 / 0.2414 | +0.0049 / +0.0053 |
| HRCOV_NODYAD (coverage candidates without pair history only) | 0.2426 / 0.2448 | +0.0082 / +0.0087 |
| HRCOV_DYAD (coverage candidates with pair history only) | 0.2697 / 0.2694 | +0.0354 / +0.0333 |
| HRCOV_PAT (top-50 + pattern block only) | 0.2717 / 0.2715 | +0.0374 / +0.0354 |
| HRCOV (full) | 0.2718 / 0.2721 | +0.0375 / +0.0360 |

* The pattern block alone reproduces the full gain (HRCOV − HRCOV_PAT = +0.0001 [−0.00002, 0.0002], p = 0.056
  RE-GCN; +0.0006 LogCL). The bridge block adds +0.0049 / +0.0053, i.e. about what its features alone add
  (HRCOVFEAT +0.0043) — +0.0006 / +0.0010 from bridge *candidates*.
* The stage-2 bridge ranker was fitted on train queries whose bridge features come from that same static train graph
  (own gold edge and future train facts present — inherited from the harness `BridgeIndex`, which does not exclude the
  query's own fact), while its valid/test features use past facts only. Test scores are leak-free, but the ranker is
  trained under a feature shift; "bridge candidates ≈ +0.001" is a result for this training recipe, not for typed
  bridges in general. A time-respecting (snapshot-before-t) cohort index is the fix to test next.
* Restricting coverage candidates to those with a prior (s, o) event recovers 94% / 93% of the gain; new-partner
  candidates add ≈ +0.004 beyond the features. The gain is history-aware pool widening — design option 4 of the error
  analysis — not reach to unseen pairs.

**Most of the bridge's reach is reproduced by density controls (G5, `pc_density.py`; post-hoc).** Caveat: cohort numbers
use the static train-graph `BridgeIndex` (harness design), which contains each training query's own gold edge and later
train facts, so they are transductive and optimistic; the test-sample numbers use only train facts, all earlier than every
test timestamp, and are the leak-free comparison. On the 30,000 coverage-hard train cohort, the
natural bridge set reaches the gold for 0.959 of queries but has median |C| = 27,184 entities = 18.0% of N (mean
13.9%). At the same |C|: uniform random 0.139, degree-matched random 0.630, popularity top-|C| 0.753. On a uniform
20k test sample: bridge 0.726 at 14.6% of N vs degree-matched 0.588 and popularity 0.678. The pattern union, by
contrast, reaches 0.534 with |C| = 0.13% of N (degree-matched 0.111, popularity 0.313). At a usable list length
(pool-absent queries, B = 50): bridge list 0.074 ≈ popularity 0.075 (degree-matched 0.023), pattern list 0.249
(popularity 0.073). LogCL: cohort 0.948 at 17.4% of N, degree-matched 0.620. The natural bridge set does exceed the controls at matched
|C| (cohort +0.33 over degree-matched, +0.21 over popularity; test sample +0.14 / +0.05), but it does so with sets of
~15–18% of all entities; once cut to a usable list length the stage-2 bridge ranking is no better than popularity.

!Reach and ceilings

**Ceilings (RE-GCN; LogCL in verdict).** Gold in base top-50: 0.420; in C_cov: 0.600 (coverage block given
pool-absent 0.310). Perfect placement within top-50: MRR 0.4215; within C_cov: 0.6007; HRCOV realises 0.2718 —
45% of the C_cov placement ceiling. Placement-at-1 ceilings of the lists alone: pattern @10 0.312 / @50 0.428,
bridge @10 0.222 / @50 0.327.

**Coverage-only references.** PAT_BUMP +0.016 / +0.015 over the base (valid-selected B = 10, δ = 2) — the pattern
prior carries signal on its own. BR_CS +0.00001 / +0.00002 over the base: the confidence switch, selected on valid,
barely ever backfills (q = 0.3, M = 200, B = 200 safe mode; the unsafe M = 0 variant is −0.0001).

## 7. 6.5 augmentation (oracle → gated arms)

**Oracle and stop rule** (uniform 20k test sample, frozen base rescored with the harness `rescore_lib` mechanism).
The structural pool (stage-2 bridge top-20 ∪ pattern-prior top-20) proposes the gold for 31.4% (RE-GCN) / 29.7%
(LogCL) of sampled queries. Injecting only the gold quad: m = 1 +0.0495 [0.0473, 0.0519] / +0.0683 [0.0655, 0.0711];
m = 3 +0.0607 [0.0583, 0.0634] / +0.0950 [0.0915, 0.0985] full-test-extrapolated ΔMRR. Both exceed the 0.002 stop
threshold, so the gated arms were run on both bases.

**Gated arms** (`pc_augarms.py`; 6,387 s / 6,978 s on one GPU each). Target queries: base top-1 probability
p1 ≤ τ, τ = 0.3-quantile of valid p1 (0.137 / 0.127) → 187,879 (35.3%) / 176,319 (33.1%) test queries; the structural
pool proposes the gold for 24.6% / 23.9% of them (gold = 0.78% / 0.86% of the 5.95 M / 4.91 M candidate pairs).
S1 (V1 = 5,000 valid queries, every pair injected alone): mean Δlog p(gold) is +0.257 / +0.279 for **non-gold**
evidence and +2.36 / +2.93 for the gold — most injected non-gold facts already raise the gold's likelihood. S2 gate
AUC for gold identity on V2: Δlog p gates 0.447–0.475 / 0.488–0.490 (below chance), Δ(1/rank) gates 0.544–0.554 /
0.582–0.584. S3 (V2 = 4,000 valid queries) selected θ 0.2–0.4, per-query cap B = 3, m = 1 (LogCL seed 0: m = 3),
V2 ΔMRR +0.0037 to +0.0058.

| arm (test pool only; non-pool queries keep the base rank) | RE-GCN MRR | Δ vs base | gold share of admitted pairs | LogCL MRR | Δ vs base | gold share |
|---|---|---|---|---|---|---|
| utility gate, Δlog p target (primary, 3 seeds) | 0.1661 | +0.0011 | 0.73% | 0.1804 | +0.0019 | 0.96% |
| random gate, matched #pairs/B/m | 0.1644 | -0.0006 | 0.83% | 0.1774 | -0.0011 | 0.87% |
| triadic-closure (Adamic-Adar) gate, matched | 0.1647 | -0.0003 | 2.04% | 0.1782 | -0.0002 | 2.11% |
| uncertainty 4u(1−u) gate, matched | 0.1647 | -0.0003 | 2.74% | 0.1783 | -0.0002 | 2.54% |
| ungated completion, same B, m | 0.1654 | +0.0004 | 2.59% | 0.1799 | +0.0015 | 2.83% |
| perturbation control (admitted slots, random entities) | 0.1641 | -0.0010 | — | 0.1761 | -0.0024 | — |
| utility gate, Δ(1/rank) target (sensitivity, 1 seed) | 0.1680 | +0.0030 | 3.63% | 0.1855 | +0.0070 | 5.85% |

Pre-registered 6.5 rule (utility gate beats the base on the cold stratum, gold train frequency < 50, with p < 0.05,
beats every matched comparator there, and no head damage): **fails on both bases.**

| test (utility gate Δlog p, 3-seed mean) | RE-GCN Δ [95% CI] | LogCL Δ [95% CI] |
|---|---|---|
| vs base, full test | +0.0011 [+0.0010, +0.0012] | +0.0019 [+0.0017, +0.0021] |
| vs base, cold stratum | -0.0004 [-0.0005, -0.0003] | -0.0004 [-0.0006, -0.0002] |
| vs base, head stratum | +0.0031 [+0.0029, +0.0033] | +0.0048 [+0.0044, +0.0051] |
| vs random matched, cold | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0002, +0.0001] |
| vs triadic-closure matched, cold | -0.0001 [-0.0002, +0.0000] | +0.0004 [+0.0002, +0.0006] |
| vs uncertainty matched, cold | -0.0001 [-0.0003, +0.0000] | -0.0004 [-0.0006, -0.0002] |
| vs ungated, cold | -0.0003 [-0.0004, -0.0001] | -0.0012 [-0.0014, -0.0010] |
| vs random matched, full | +0.0017 [+0.0015, +0.0018] | +0.0030 [+0.0028, +0.0032] |
| vs perturbation control, full | +0.0021 [+0.0020, +0.0022] | +0.0043 [+0.0041, +0.0044] |

Reading: the utility gate does beat every matched comparator on the full test (+0.0004 to +0.0043), but its gain
sits in the head stratum and it *loses* on the cold stratum it was meant for. It admits fewer gold pairs than the
triadic-closure and uncertainty gates (0.6–1.0% vs 2.0–2.7%), so the measured gain is the base's response to
plausible non-gold evidence, not recovery of the missing fact — the perturbation-response pattern already seen in
6.5. The realised gain is about 2% of the m = 3 oracle on either base (+0.0011 of +0.0607; +0.0019 of +0.0950).

Scope limits: AUG was measured on the frozen base, not stacked under HR / HRCOV (HR alone adds +0.069 / +0.058 over
the base). No no-injection test arm was run; the valid-sample no-injection rescoring reproduces the dump rank for
99.3–99.5% of queries (differences ≤ 1 rank), so arm-vs-base deltas carry a small reproduction term, whereas
arm-vs-arm contrasts share the same rescoring path. Matched comparators were run against the seed-0 primary gate
only; the Δ(1/rank) sensitivity gate was run with one seed on test.

## 8. Negative and null results kept in view

1. Cold-entity coverage, the stated motivation, is not delivered: cold golds without pair history +0.004; no-pair & 2-hop-unreachable (17%) Δ ≈ 0 (LogCL
   significantly negative), first-seen (11%) +0.001 / +0.0002, U_deep_out (19%) +0.0007.
2. In-pool damage: gold-in-top-50 −0.011 / −0.007; self-loop −0.025 / −0.011; U_deep_in −0.044 / −0.039.
3. Bridge candidates add +0.0006 / +0.0010 beyond their features; at B = 50 the bridge list reaches pool-absent golds no better than popularity (0.074 vs 0.075).
4. BR_CS ≈ 0 over the base.
5. 6.5 gated augmentation fails its rule on both bases: cold-stratum Δ −0.0004 vs base, Δlog p gate gold-identity AUC < 0.5, fewer gold pairs admitted than triadic-closure / uncertainty gates, ≈ 2% of the m = 3 oracle realised.
6. Our RE-GCN is 0.010 below the TGB RE-GCN row; official RecB in default mode 0.167 vs RecB-lite 0.196.
7. HR beats RecB-lite by +0.038 / +0.040 but the base alone is −0.031 / −0.018 below RecB-lite; recurrence is the
   dominant signal on polecat.

## 9. Deviations from the task specification

1. TiRGN not run (pure-torch shim estimated > 4 h: per-timestamp history cache over 1,826 days × 151k entities); two bases (RE-GCN, LogCL) instead of three.
2. The 6 u_pred (event-LM) features of the residual track are dropped (no event-LM vocabulary for polecat); the stage-2 bridge ranker is G12 arm 5 without its LM column.
3. Self-loop-free variant (`tkgl-polecat-noself`) not trained; a self-loop stratum (s = o, 10.4% of test) of the official run is reported instead.
4. Official TGB RecB run in default mode only (λ 0.1, α 0.99; test 0.167); train mode (per-relation λ/α search) not run (effectively serial, projected > 8 h); RecB-lite (λ = 0.99, valid-selected, 0.196) is the recurrence reference used for strata.
5. Bases not tuned toward the TGB configuration (history 10 vs 3, fixed 12 epochs, no valid checkpoint selection): our RE-GCN 0.165 vs TGB row 0.175.
6. Re-ranker training and all valid-side selection used a 200,000-query uniform valid sample (rng 12345) of the 533,472 valid queries; test is always the full 532,636.
7. Lookup-vs-dump rank agreement threshold relaxed from 0.999 to 0.99 (measured 0.9971–0.9977, max |Δrank| = 2); the measured values are in the verdict.
8. 6.5 augmentation: generators = stage-2 bridge + pattern prior only (no peer-import, no LLM generator); oracle on a 20k uniform test sample (full-test-extrapolated); gate fitting / selection on 5,000 / 4,000 valid queries; matched comparators vs the seed-0 primary gate only; Δ(1/rank) sensitivity gate with 1 seed on test; AUG measured on the frozen base, not on top of HR.
9. Official py-tgb evaluation of the re-rankers for seed 0 only (HR_s0, HRCOV_s0); 3-seed means use the harness evaluator, verified identical.
10. Post-hoc block-attribution arms (HRCOV_PAT, HRCOV_BR, HRCOV_DYAD, HRCOV_NODYAD) and the G5 density controls were added after the test numbers were read; they are labelled post-hoc and do not enter the decision.
11. The first gated-augmentation submission was cancelled after S3 (quadratic per-query grouping in the test-arm build) and rerun from scratch with the fix; results are from the rerun (jobs / 50416019).

## 10. Implications for pilots 2–4 and the project

* On polecat, the large no-pair / first-seen mass is real (39% / 11% of test queries) but the coverage machinery does
  not reach it; the measured gain comes from re-admitting historical partners the base ranks below 50. Pilots 2–3
  (`tkgl-smallpedia`, `tkgl-icews`) should report HRCOV_PAT and HRCOV_DYAD as first-class arms so the same
  attribution is visible there.
* A recurrency candidate block (RecB top-k) is the obvious cheap competitor to the pattern block for R_rec_out,
  where RecB-lite is 0.08 MRR above HRCOV.
* Bridge reach claims should always be reported with |C|/N and degree-matched / popularity controls at matched |C|.
* For Task 3 (contextual TKG reasoning with LLMs), the useful structure on polecat is entity-pair history beyond the
  base's top-50; context supplied to an LLM re-ranker should prioritise pair-history evidence over 2-hop bridges.

## 11. Artifacts

* `polecat_pilot_verdict.json` — all numbers, per-base comparisons, strata, ceilings, density controls, rank agreement, AUG tests, hosts
* `polecat_decision_rule.json` — pre-registered rule (saved before any test number)
* `pilot_pipeline_README.md` — paths, commands, run times, pitfalls, reuse notes
* Figures: `fig1_fulltest_mrr.png`, `fig2_where_gain_lands.png`, `fig3_reach_and_ceilings.png`
* Code: `pc_common.py`, `pc_tgb_verify.py`, `pc_prep.py`, `pc_lists.py`, `pc_lookup.py`, `pc_features.py`, `pc_train.py`, `pc_refs.py`, `pc_aug.py`, `pc_augarms.py`, `pc_density.py`, `pc_train_diag.py`, `pc_agg.py`, `pc_figs.py`, `pc_report_fill.py`, `pc_chain.sh`, `pc_recb_official.sh`
