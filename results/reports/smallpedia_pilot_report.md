# Pilot 2 of 4 — static-KG fusion on TGB 2.0 `tkgl-smallpedia` ( Task 2)

**Question.** Does fusing the shipped static Wikidata KG with the temporal graph reach and correctly rank golds that temporal history cannot (no-pair-history / first-seen entities), *on top of* the strongest history-based system (HRCOV, the pilot-1 coverage stack)?

**Answer (pre-registered rule).** Yes, on all 3 bases: `HRCOV_ST` (HRCOV + 50 static-KG candidates + 12 static features) beats HRCOV on full-test MRR by RE-GCN +0.0077, LogCL +0.0078, TiRGN +0.0080 (one-sided p < 5e-4 each), the no-pair-history stratum improves by +0.0268, +0.0265, +0.0283 (p < 5e-4), and no head-stratum CI lies below zero. `useful_on_smallpedia = true` (3 of 3 bases pass; ≥ 2 required). The comparator is HRCOV, not HR as in the plan's wording — deliberately stricter.

**Direct-static-edge control (pre-registered addendum).** Retraining HRCOV_ST with every direct static s–candidate edge removed (`HRCOV_ST_NODIRECT`; 2-hop shared-attribute, static→temporal and temporal→static paths kept) still passes the same rule on 3 of 3 bases: ΔMRR vs HRCOV RE-GCN +0.0030 [+0.0025, +0.0035], LogCL +0.0035 [+0.0029, +0.0040], TiRGN +0.0051 [+0.0046, +0.0056], no-pair RE-GCN +0.0164, LogCL +0.0158, TiRGN +0.0172 (p < 5e-4), no head damage. **PASS survives without direct static edges** (`PASS_survives_without_direct_static_edges = true`); the strict variant, in which directly linked candidates get no static evidence at all, also passes on 3 of 3 (RE-GCN +0.0042, LogCL +0.0031, TiRGN +0.0031). The surviving gain is 39–64% of the pilot-2 HRCOV_ST gain. The removed part (HRCOV_ST − HRCOV_ST_NODIRECT = RE-GCN +0.0047, LogCL +0.0043, TiRGN +0.0029) is **time-less background knowledge, possible future leakage — not forecasting gain**: the static KG is a 2024 snapshot, and in 53% of the 6,762 direct-edge test queries the static property restates, inverts or implies the forecast relation (Wikidata rules + LLM judge, §4b). The estimate of the forecasting-attributable gain of static fusion on smallpedia is therefore the NODIRECT / STRICT range, +0.0030 to +0.0051 MRR, not +0.008 — and it is an upper bound, because 2-hop paths through the same 2024 snapshot (e.g. `P3842` present-day territory) are not controlled.

**Base-strength caveat.** Our RE-GCN base (test MRR 0.422, 3 valid-selected epochs) is 0.17 below the TGB 2.0 RE-GCN row (0.594); deltas measured relative to the base overstate what a tuned base would leave to re-ranking. The absolute HRCOV / HRCOV_ST / NODIRECT MRRs stand as measured, and the static-fusion deltas above are against HRCOV, not against the base.

**What the gain consists of (post-hoc, not in the decision).** (i) About three quarters of it needs the static *candidates*: a degree-matched random block of the same size, carrying the same static features, adds +0.0020, +0.0017, +0.0019, and static features without static candidates add +0.0017, +0.0021, +0.0017. (ii) It sits almost entirely on the 1.8% of queries whose gold enters through the static block (MRR ≈ 0.003 → ≈ 0.40). (iii) About half comes from the 4.1% of test queries whose (s, o) pair is already a static edge (time-less background knowledge, possible future leakage; see the addendum); the other half from 2-hop shared-attribute paths (e.g. birthplace → headquarters location, citizenship → country). (iv) First-seen *objects* (no event before t) go from MRR 0.000 to ≈ 0.07. (v) No-pair queries the union graph cannot reach (18% of test) lose ≈ 0.004–0.005.

Verdict: `smallpedia_pilot_verdict.json`; pre-registered rule: `smallpedia_decision_rule.json` and addendum `smallpedia_decision_rule_addendum_nodirect.json`; leakage pairs: `smallpedia_leakage_relation_pairs.csv`; pipeline: `pilot_pipeline_README_v2.md`.

---

## 1. Setup

**Data and protocol.** `tkgl-smallpedia` (TGB 2.0): 47,433 entities, 283 relations (+283 inverse), 125 yearly snapshots (1900–2024); train / valid / test facts 387,757 / 81,033 / 81,586; both query directions; **all 163,172 test queries** and all 162,066 valid queries (no subsampling). Time-aware filtered 1-vs-all ranking over the 47,433 temporal entities, ground-truth online history before t.

**TGB identity checks (job, py-tgb 2.3.0).** Entity and relation maps are identities; TGB rows equal the harness split files in order; the TGB conflict sets equal the harness time-aware filter for all 101,243 valid and 103,430 test keys; TGB negatives are exactly the temporal entities; evaluator rank = 1 + #greater + 0.5·#ties.

**Static KG.** `tkgl-smallpedia_static_edgelist.csv` through py-tgb's own `static_data` processing: 978,022 edges, 753 relation types, 262,361 static-only nodes (attribute values; never ranked, only intermediates), 89.3% of temporal entities have ≥ 1 static edge, max static degree 39,525 (122 nodes > 1,000). 4.14% of test facts have their (s, o) pair as a static edge (TGB removed only exact (head, tail, property) duplicates).

**Bases** (all valid-selected; test never used): RE-GCN and LogCL with the unmodified harness `base_dump.py` (h = 200, 2 layers, history 10), epoch budget chosen from {3, 6, 12, 30, 60} by full-valid MRR; TiRGN (authors' model code via the project's pure-torch DGL shim, history 10, history rate 0.3, no word graph), epoch with the best full-valid MRR (≤ 30, patience 5).

| base | valid MRR by budget | selected |
|---|---|---|
| RE-GCN | 3 ep 0.4900, 6 ep 0.4843, 12 ep 0.4781, 30 ep 0.4202, 60 ep 0.3963 | 3 epochs |
| LogCL | 3 ep 0.5091, 6 ep 0.5086, 12 ep 0.4806, 30 ep 0.4152, 60 ep 0.3895 | 3 epochs |
| TiRGN | best of 26 evaluated epochs: 0.6541 | epoch 20 |

RE-GCN and LogCL overfit fast on smallpedia (48.6% of test golds are unseen in train); both selected the smallest budget in the grid.

**Systems** (every hyper-parameter, epoch, calibration weight chosen on valid; valid timestamps < 70% quantile = fit (105,606 queries), rest = select (56,460); 3 seeds; pilot-1 recipe: MLP 64-64, listwise CE, s = α·log p + β·f calibrated on select):

| arm | candidates | features | role |
|---|---|---|---|
| `HR` | base top-50 | F_hist (19) | history re-ranker (6.2 arm B) |
| `HRCOV` | top-50 ++ pattern-prior ++ stage-2 bridge (W = 150) | F_hist + F_cov (30) | **reference** (pilot-1 PASS) |
| `HRCOV_ST` | HRCOV ++ 50 static-KG candidates (W = 200) | + F_static (42) | **primary** |
| `HRCOV_DM` | HRCOV ++ 50 degree-matched random candidates (W = 200) | 42 | density control |
| `HRCOV_STFEAT` | HRCOV candidates (W = 150) | 42 | static features without static candidates |
| `HRCOV_PAT` | top-50 ++ pattern block | 30 | attribution (pilot-1 recommendation) |
| `HR_ST` | top-50 ++ 50 static (W = 100) | F_hist + F_static (31) | secondary vs HR |
| `STONLY` | top-50 ++ 100 static (W = 150) | 31 | static-only candidates at HRCOV's set size |

Static candidates: typed paths over the union of the static graph and the temporal history before t — S (s–o static edge), SS (shared static neighbour, intermediate degree ≤ 1,000), ST (static neighbour's temporal partner), TS (temporal partner's static neighbour) — ranked by a path-type precision prior (target encoding P(c = gold | c reached by type), relation-specific with back-off, fitted on 100,000 early-train queries, ts < 1968); the static block is the first 50 of this list outside C_cov. Static features: 1-hop flag, path counts per shape, best path-type precision, shared-attribute count, candidate and subject static degree, list position, block flag.

**Decision rule (pre-registered before any system was trained).** Useful iff on ≥ 2 bases: (a) HRCOV_ST − HRCOV full-test ΔMRR ≥ 0.001 with one-sided paired Poisson bootstrap p < 0.05 (B = 2,000; per-query RR averaged over 3 seeds); (b) head guardrail (gold train frequency ≥ 500; DAMAGE iff the 95% CI lies below 0 and |Δ| ≥ 0.001) passes; (c) no-pair-history stratum Δ > 0 with p < 0.05. The registration note was later amended (provenance only) to list a TiRGN smoke-run base row that had also been printed; criteria unchanged.

## 2. Base verification (Table 1)

| row | test MRR | H@1 | H@10 | source / host |
|---|---|---|---|---|
| RecB-lite (λ = 0.01, valid-selected) | 0.6038 | 0.5244 | 0.7148 | serverA CPU |
| regcn (frozen base, full-row exact ranks) | 0.4218 | 0.3625 | 0.5345 | serverA (GPU, tkg-gpu:latest) |
| regcn (frozen base, harness dump gold_rank_filt) | 0.4218 | 0.3625 | 0.5345 | serverA (GPU, tkg-gpu:latest) |
| regcn (frozen base) [official py-tgb Evaluator] | 0.4218 | — | 0.5345 | serverA (CPU, tkg-gpu:latest) |
| logcl (frozen base, full-row exact ranks) | 0.4664 | 0.4059 | 0.5786 | serverA (GPU, tkg-gpu:latest) |
| logcl (frozen base, harness dump gold_rank_filt) | 0.4664 | 0.4059 | 0.5786 | serverA (GPU, tkg-gpu:latest) |
| logcl (frozen base) [official py-tgb Evaluator] | 0.4664 | — | 0.5786 | serverA (CPU, tkg-gpu:latest) |
| tirgn (frozen base, full-row exact ranks) | 0.6181 | 0.5523 | 0.7239 | serverA (GPU, tkg-gpu:latest) |
| tirgn (frozen base, harness dump gold_rank_filt) | 0.6181 | 0.5523 | 0.7239 | serverA (GPU, tkg-gpu:latest) |
| tirgn (frozen base) [official py-tgb Evaluator] | 0.6181 | — | 0.7239 | serverA (CPU, tkg-gpu:latest) |
| published: CountTRuCoLa (best published) | 0.644 | | | TGB 2.0 leaderboard (as quoted in dataset_shortlist.md) |
| published: CEN | 0.612 | | | TGB 2.0 leaderboard (as quoted in dataset_shortlist.md) |
| published: RE-GCN (TGB) | 0.594 | | | TGB 2.0 leaderboard (as quoted in dataset_shortlist.md) |
| published: TGB RecB (train mode) | 0.604 | | | TGB 2.0 leaderboard (as quoted in dataset_shortlist.md) |

* RE-GCN: harness vs official py-tgb per-query rank identical for 99.86% of queries (rest ties only), MRR 0.421759 vs 0.421759; full-row replay vs stored dump rank agreement 99.89% (max |Δrank| 2).
* LogCL: harness vs official py-tgb per-query rank identical for 99.86% of queries (rest ties only), MRR 0.466358 vs 0.466358; full-row replay vs stored dump rank agreement 99.91% (max |Δrank| 2).
* TiRGN: harness vs official py-tgb per-query rank identical for 98.72% of queries (rest ties only), MRR 0.618058 vs 0.618058; full-row replay vs stored dump rank agreement 100.00% (max |Δrank| 0).

Our RE-GCN (0.422) and LogCL (0.466) are far below the TGB RE-GCN row (0.594): with fixed budgets and no layer norm / valid-best checkpoint they overfit within a few epochs. We did not tune toward the TGB row. TiRGN (0.618) is above RecB-lite (0.604) and the TGB RE-GCN / CEN rows.

## 3. Full-test results (Table 2; n = 163,172, both directions, 3-seed means; host serverA GPU for training, harness exact ranks)

| base | system | MRR | H@1 | H@3 | H@10 | seed s.d. | Δ vs HRCOV [95% CI] |
|---|---|---|---|---|---|---|---|
| RE-GCN | `HR` | 0.5693 | 0.5349 | 0.5935 | 0.6180 | 0.0003 | -0.0902 [-0.0917, -0.0887] |
| RE-GCN | `HRCOV` | 0.6595 | 0.6081 | 0.7008 | 0.7367 | 0.0003 | — |
| RE-GCN | `HRCOV_ST` | 0.6672 | 0.6135 | 0.7096 | 0.7493 | 0.0002 | +0.0077 [+0.0071, +0.0083] |
| RE-GCN | `HRCOV_DM` | 0.6615 | 0.6102 | 0.7027 | 0.7384 | 0.0002 | +0.0020 [+0.0016, +0.0025] |
| RE-GCN | `HRCOV_STFEAT` | 0.6612 | 0.6095 | 0.7030 | 0.7383 | 0.0007 | +0.0017 [+0.0012, +0.0021] |
| RE-GCN | `HRCOV_PAT` | 0.6597 | 0.6085 | 0.7010 | 0.7364 | 0.0002 | +0.0002 [-0.0000, +0.0004] |
| RE-GCN | `HR_ST` | 0.5842 | 0.5464 | 0.6104 | 0.6400 | 0.0003 | (vs HR) +0.0149 [+0.0141, +0.0156] |
| RE-GCN | `STONLY` | 0.5862 | 0.5483 | 0.6124 | 0.6423 | 0.0006 | -0.0734 [-0.0749, -0.0718] |
| LogCL | `HR` | 0.5884 | 0.5498 | 0.6171 | 0.6435 | 0.0002 | -0.0706 [-0.0720, -0.0691] |
| LogCL | `HRCOV` | 0.6590 | 0.6072 | 0.7005 | 0.7371 | 0.0003 | — |
| LogCL | `HRCOV_ST` | 0.6668 | 0.6130 | 0.7090 | 0.7491 | 0.0002 | +0.0078 [+0.0072, +0.0083] |
| LogCL | `HRCOV_DM` | 0.6607 | 0.6087 | 0.7028 | 0.7387 | 0.0007 | +0.0017 [+0.0012, +0.0021] |
| LogCL | `HRCOV_STFEAT` | 0.6611 | 0.6092 | 0.7029 | 0.7385 | 0.0005 | +0.0021 [+0.0016, +0.0025] |
| LogCL | `HRCOV_PAT` | 0.6590 | 0.6075 | 0.7003 | 0.7364 | 0.0004 | -0.0000 [-0.0002, +0.0002] |
| LogCL | `HR_ST` | 0.6021 | 0.5603 | 0.6325 | 0.6641 | 0.0002 | (vs HR) +0.0137 [+0.0130, +0.0144] |
| LogCL | `STONLY` | 0.6034 | 0.5614 | 0.6342 | 0.6650 | 0.0003 | -0.0556 [-0.0571, -0.0542] |
| TiRGN | `HR` | 0.6572 | 0.6067 | 0.6989 | 0.7301 | 0.0001 | -0.0029 [-0.0033, -0.0025] |
| TiRGN | `HRCOV` | 0.6601 | 0.6089 | 0.7019 | 0.7353 | 0.0002 | — |
| TiRGN | `HRCOV_ST` | 0.6682 | 0.6146 | 0.7111 | 0.7480 | 0.0006 | +0.0080 [+0.0075, +0.0086] |
| TiRGN | `HRCOV_DM` | 0.6620 | 0.6107 | 0.7042 | 0.7370 | 0.0005 | +0.0019 [+0.0014, +0.0023] |
| TiRGN | `HRCOV_STFEAT` | 0.6618 | 0.6105 | 0.7043 | 0.7371 | 0.0003 | +0.0017 [+0.0013, +0.0021] |
| TiRGN | `HRCOV_PAT` | 0.6596 | 0.6090 | 0.7015 | 0.7337 | 0.0002 | -0.0005 [-0.0007, -0.0003] |
| TiRGN | `HR_ST` | 0.6650 | 0.6122 | 0.7078 | 0.7431 | 0.0005 | (vs HR) +0.0078 [+0.0072, +0.0084] |
| TiRGN | `STONLY` | 0.6659 | 0.6131 | 0.7086 | 0.7437 | 0.0002 | +0.0057 [+0.0051, +0.0064] |

Official py-tgb Evaluator, seed 0 (CPU, host serverA):

| base | base | HR | HRCOV | HRCOV_ST | HR_ST |
|---|---|---|---|---|---|
| RE-GCN | 0.4218 | 0.5696 | 0.6593 | 0.6669 | 0.5844 |
| LogCL | 0.4664 | 0.5887 | 0.6589 | 0.6665 | 0.6019 |
| TiRGN | 0.6181 | 0.6574 | 0.6599 | 0.6680 | 0.6643 |

!Full-test MRR

*Figure 1.* (a) Full-test MRR per base; the dashed line is the best published TGB 2.0 row (CountTRuCoLa 0.644). (b) MRR difference against HRCOV with 95% paired-bootstrap CIs; dotted line = materiality 0.001. n = 163,172 test queries, 3 seeds.

HRCOV (0.659–0.660) and HRCOV_ST (0.667–0.668) exceed the best published TGB 2.0 row (0.644) under an evaluator verified to be identical to the official one. These are statements about two-stage systems on this split, not leaderboard submissions.

## 4. Decision (Table 3)

| base | (a) ΔMRR HRCOV_ST − HRCOV [95% CI] | p | (b) head Δ [CI] | DAMAGE | (c) no-pair Δ [CI] | p | PASS |
|---|---|---|---|---|---|---|---|
| RE-GCN | +0.0077 [+0.0071, +0.0083] | < 5e-4 | +0.0005 [-0.0014, +0.0024] | no | +0.0268 [+0.0253, +0.0282] | < 5e-4 | **yes** |
| LogCL | +0.0078 [+0.0072, +0.0083] | < 5e-4 | -0.0014 [-0.0033, +0.0006] | no | +0.0265 [+0.0251, +0.0279] | < 5e-4 | **yes** |
| TiRGN | +0.0080 [+0.0075, +0.0086] | < 5e-4 | +0.0015 [-0.0005, +0.0036] | no | +0.0283 [+0.0268, +0.0297] | < 5e-4 | **yes** |

`useful_on_smallpedia = true`. Secondary (pre-registered, not in the decision):

| base | HR_ST − HR | first-seen stratum HRCOV_ST − HRCOV | HRCOV_ST − HRCOV_DM | STONLY − HRCOV |
|---|---|---|---|---|
| RE-GCN | +0.0149 [+0.0141, +0.0156] | +0.0639 [+0.0600, +0.0680] | +0.0057 [+0.0052, +0.0061] | -0.0734 [-0.0749, -0.0718] |
| LogCL | +0.0137 [+0.0130, +0.0144] | +0.0620 [+0.0579, +0.0663] | +0.0061 [+0.0057, +0.0066] | -0.0556 [-0.0571, -0.0542] |
| TiRGN | +0.0078 [+0.0072, +0.0084] | +0.0660 [+0.0620, +0.0699] | +0.0062 [+0.0058, +0.0066] | +0.0057 [+0.0051, +0.0064] |

Training-budget caveat: HR, HRCOV and HRCOV_ST often reach their best select MRR at epochs 25–29 of the 30-epoch cap (select MRR still rising by ≈ 0.0004–0.0024 over the last 10 epochs, for reference and primary arms alike).

## 4b. Direct-static-edge leakage control (pre-registered addendum; Tables 3b–3e)

**Arms.** `HRCOV_ST_NODIRECT` = HRCOV_ST with every static path of shape S (s –p→ c or c –p→ s, any of the 753 properties) removed before candidate ranking and feature computation (static block built from SS / ST / TS paths only; `st_1hop`, `st_log_n_S` ≡ 0). `HRCOV_ST_NODIRECT_STRICT` additionally gives every candidate with a direct static edge to s the 'no static evidence' feature values and keeps it out of the static list (its static degree is computed without the s–c edges), so a directly linked candidate is indistinguishable from one without static evidence. Same re-ranker, epochs, seeds and valid-only selection as HRCOV_ST; the HRCOV comparator is the pilot-2 run (C_cov candidate ids and F_cov values asserted identical on all valid and test rows; bridge rankers loaded, not retrained). Registered before any addendum list, model or number existed.

| arm | base | (a) Δ vs HRCOV [95% CI] | p | (b) head Δ [CI] | DAMAGE | (c) no-pair Δ [CI] | p | PASS |
|---|---|---|---|---|---|---|---|---|
| `HRCOV_ST_NODIRECT` | RE-GCN | +0.0030 [+0.0025, +0.0035] | < 5e-4 | -0.0006 [-0.0025, +0.0012] | no | +0.0164 [+0.0153, +0.0175] | < 5e-4 | **yes** |
| `HRCOV_ST_NODIRECT` | LogCL | +0.0035 [+0.0029, +0.0040] | < 5e-4 | -0.0001 [-0.0020, +0.0018] | no | +0.0158 [+0.0146, +0.0169] | < 5e-4 | **yes** |
| `HRCOV_ST_NODIRECT` | TiRGN | +0.0051 [+0.0046, +0.0056] | < 5e-4 | -0.0008 [-0.0027, +0.0010] | no | +0.0172 [+0.0161, +0.0183] | < 5e-4 | **yes** |
| `HRCOV_ST_NODIRECT_STRICT` | RE-GCN | +0.0042 [+0.0037, +0.0047] | < 5e-4 | -0.0014 [-0.0033, +0.0003] | no | +0.0141 [+0.0131, +0.0152] | < 5e-4 | **yes** |
| `HRCOV_ST_NODIRECT_STRICT` | LogCL | +0.0031 [+0.0026, +0.0036] | < 5e-4 | -0.0009 [-0.0027, +0.0009] | no | +0.0142 [+0.0132, +0.0152] | < 5e-4 | **yes** |
| `HRCOV_ST_NODIRECT_STRICT` | TiRGN | +0.0031 [+0.0026, +0.0036] | < 5e-4 | -0.0006 [-0.0026, +0.0013] | no | +0.0150 [+0.0139, +0.0160] | < 5e-4 | **yes** |

`PASS_survives_without_direct_static_edges = true` (3 of 3 bases; ≥ 2 required); strict sensitivity: 3 of 3. Pre-stated interpretation that applies: *multi-hop static structure carries a forecasting-relevant gain independent of direct s-o background facts*. The statement covers direct edges only; multi-hop paths through the same 2024 snapshot can still carry post-t facts.

!Direct-edge control

*Figure 4.* (a) Full-test MRR difference against HRCOV with 95% paired-bootstrap CIs (n = 163,172 test queries, 3 seeds) for HRCOV_ST with direct static edges (pilot 2), without them, and without them in the strict variant; dotted line = materiality 0.001. (b) The same differences split into the contribution of queries whose (s, o) pair is a direct static edge (hatched, 4.1% of test) and of all other queries (solid). RE = RE-GCN, Lo = LogCL, Ti = TiRGN.

*Table 3c.* Full-test MRR (3-seed means; seed s.d.; official py-tgb Evaluator for seed 0).

| base | system | MRR | H@1 | H@3 | H@10 | seed s.d. | official seed 0 | Δ vs HRCOV_ST [95% CI] |
|---|---|---|---|---|---|---|---|---|
| RE-GCN | `HRCOV` | 0.6595 | 0.6081 | 0.7008 | 0.7367 | 0.0003 | 0.6593 | -0.0077 [-0.0083, -0.0071] |
| RE-GCN | `HRCOV_ST` | 0.6672 | 0.6135 | 0.7096 | 0.7493 | 0.0002 | 0.6669 | — |
| RE-GCN | `HRCOV_ST_NODIRECT` | 0.6625 | 0.6081 | 0.7054 | 0.7460 | 0.0026 | 0.6588 | -0.0047 [-0.0051, -0.0043] |
| RE-GCN | `HRCOV_ST_NODIRECT_STRICT` | 0.6637 | 0.6105 | 0.7055 | 0.7448 | 0.0006 | 0.6631 | -0.0035 [-0.0039, -0.0031] |
| LogCL | `HRCOV` | 0.6590 | 0.6072 | 0.7005 | 0.7371 | 0.0003 | 0.6589 | -0.0078 [-0.0083, -0.0072] |
| LogCL | `HRCOV_ST` | 0.6668 | 0.6130 | 0.7090 | 0.7491 | 0.0002 | 0.6665 | — |
| LogCL | `HRCOV_ST_NODIRECT` | 0.6625 | 0.6079 | 0.7055 | 0.7461 | 0.0022 | 0.6593 | -0.0043 [-0.0047, -0.0039] |
| LogCL | `HRCOV_ST_NODIRECT_STRICT` | 0.6621 | 0.6078 | 0.7049 | 0.7452 | 0.0024 | 0.6588 | -0.0047 [-0.0052, -0.0043] |
| TiRGN | `HRCOV` | 0.6601 | 0.6089 | 0.7019 | 0.7353 | 0.0002 | 0.6599 | -0.0080 [-0.0086, -0.0075] |
| TiRGN | `HRCOV_ST` | 0.6682 | 0.6146 | 0.7111 | 0.7480 | 0.0006 | 0.6680 | — |
| TiRGN | `HRCOV_ST_NODIRECT` | 0.6653 | 0.6120 | 0.7079 | 0.7448 | 0.0007 | 0.6653 | -0.0029 [-0.0033, -0.0026] |
| TiRGN | `HRCOV_ST_NODIRECT_STRICT` | 0.6632 | 0.6093 | 0.7066 | 0.7436 | 0.0027 | 0.6594 | -0.0050 [-0.0054, -0.0046] |

The official Evaluator reproduces the harness MRR of every addendum seed-0 system to 1e-6 (per-query rank agreement RE-GCN 99.88%, LogCL 99.88%, TiRGN 98.83%; the rest are ties).

*Table 3d.* Where the gain lands without direct edges (Δ vs HRCOV, 3-seed means; CI shown for NODIRECT).

| stratum | share | RE-GCN: MRR HRCOV; Δ ST / **ND** [CI] / NS | LogCL: MRR HRCOV; Δ ST / **ND** [CI] / NS | TiRGN: MRR HRCOV; Δ ST / **ND** [CI] / NS |
|---|---|---|---|---|
| direct s–o static edge | 0.041 | 0.825; +0.0971 / **+0.0112 [+0.0086, +0.0138]** / -0.0056 | 0.824; +0.0987 / **+0.0126 [+0.0099, +0.0154]** / -0.0044 | 0.826; +0.0937 / **+0.0116 [+0.0092, +0.0141]** / -0.0043 |
| no direct s–o static edge | 0.959 | 0.652; +0.0038 / **+0.0027 [+0.0021, +0.0032]** / +0.0046 | 0.652; +0.0039 / **+0.0031 [+0.0026, +0.0036]** / +0.0034 | 0.653; +0.0043 / **+0.0048 [+0.0043, +0.0054]** / +0.0034 |
| no pair history | 0.280 | 0.039; +0.0268 / **+0.0164 [+0.0153, +0.0175]** / +0.0141 | 0.039; +0.0265 / **+0.0158 [+0.0146, +0.0169]** / +0.0142 | 0.040; +0.0283 / **+0.0172 [+0.0161, +0.0183]** / +0.0150 |
| no pair & no direct edge | 0.274 | 0.040; +0.0136 / **+0.0148 [+0.0137, +0.0159]** / +0.0145 | 0.039; +0.0132 / **+0.0139 [+0.0128, +0.0150]** / +0.0146 | 0.040; +0.0149 / **+0.0158 [+0.0147, +0.0169]** / +0.0155 |
| first-seen object | 0.049 | 0.000; +0.0653 / **+0.0267 [+0.0240, +0.0295]** / +0.0172 | 0.000; +0.0655 / **+0.0272 [+0.0243, +0.0300]** / +0.0196 | 0.000; +0.0675 / **+0.0255 [+0.0228, +0.0283]** / +0.0174 |
| first-seen object & no direct edge | 0.045 | 0.000; +0.0184 / **+0.0211 [+0.0185, +0.0236]** / +0.0186 | 0.000; +0.0181 / **+0.0210 [+0.0186, +0.0235]** / +0.0211 | 0.000; +0.0187 / **+0.0204 [+0.0179, +0.0229]** / +0.0188 |
| first-seen entity (s or o) | 0.083 | 0.037; +0.0639 / **+0.0337 [+0.0311, +0.0367]** / +0.0262 | 0.034; +0.0620 / **+0.0309 [+0.0280, +0.0338]** / +0.0253 | 0.044; +0.0660 / **+0.0318 [+0.0291, +0.0346]** / +0.0252 |
| gold in pilot-2 static block * | 0.018 | 0.003; +0.4076 / **+0.2528 [+0.2388, +0.2670]** / +0.2181 | 0.003; +0.3987 / **+0.2561 [+0.2420, +0.2703]** / +0.2189 | 0.003; +0.4019 / **+0.2490 [+0.2342, +0.2634]** / +0.2250 |
| head (≥ 500) | 0.025 | 0.955; +0.0005 / **-0.0006 [-0.0025, +0.0012]** / -0.0014 | 0.955; -0.0014 / **-0.0001 [-0.0020, +0.0018]** / -0.0009 | 0.954; +0.0015 / **-0.0008 [-0.0027, +0.0010]** / -0.0006 |

\* base-dependent stratum (share shown for RE-GCN). ST = HRCOV_ST, ND = HRCOV_ST_NODIRECT, NS = HRCOV_ST_NODIRECT_STRICT.

Contribution to the full-test Δ vs HRCOV, split by whether (s, o) is a direct static edge (4.1% of test): HRCOV_ST RE-GCN +0.0040 / +0.0037, LogCL +0.0041 / +0.0037, TiRGN +0.0039 / +0.0042; NODIRECT RE-GCN +0.0005 / +0.0025, LogCL +0.0005 / +0.0029, TiRGN +0.0005 / +0.0046; STRICT RE-GCN -0.0002 / +0.0044, LogCL -0.0002 / +0.0033, TiRGN -0.0002 / +0.0033 (direct-edge queries / all other queries). On queries *without* a direct edge, NODIRECT gains about what HRCOV_ST gained there; on no-pair queries without a direct edge it gains as much or slightly more. What disappears is the large lift on direct-edge queries (HRCOV_ST +0.09–0.10 MRR on that 4.1%, NODIRECT ≈ +0.01, STRICT ≈ −0.005) and about 60% of the first-seen-object gain (MRR 0.065 → 0.027, 0.066 → 0.027, 0.068 → 0.026 for HRCOV_ST → NODIRECT; HRCOV ≈ 0.000).

Caveats: NODIRECT has larger seed variation than HRCOV_ST (seed s.d. up to 0.0026 vs ≤ 0.0006; one seed of RE-GCN and LogCL stops early), and STRICT − NODIRECT changes sign across bases (RE-GCN +0.0012, LogCL -0.0004, TiRGN -0.0020), so the two no-direct estimates are not separable at this precision.

**Leakage audit (Table 3e).** 6,762 test queries (4.1%) have a direct static edge between s and the gold o, over 258 distinct (static property, direction, temporal relation) pairs (113 static properties × 86 temporal relations). Classes: identical property (SAME), inverse property (INVERSE), static fact entails / restates the temporal fact (IMPLIES), correlated only (RELATED), none (UNRELATED); query class = strongest over its static edges. 23 pairs are labelled by Wikidata metadata (identity, P1696 inverse property, symmetric property); the rest by an LLM judge given Wikidata labels and descriptions (judge self-agreement 0.945 query-weighted, 0.965 on restating-vs-not; agreement with a smaller model 0.655 / 0.767).

| class | share of direct-edge queries (primary judge) | judge pass 2 | smaller model | RE-GCN contribution to HRCOV_ST − HRCOV | LogCL contribution to HRCOV_ST − HRCOV | TiRGN contribution to HRCOV_ST − HRCOV |
|---|---|---|---|---|---|---|
| SAME | 0.026 | 0.034 | 0.020 | +0.0002 | +0.0002 | +0.0002 |
| INVERSE | 0.213 | 0.207 | 0.240 | +0.0006 | +0.0006 | +0.0006 |
| IMPLIES | 0.293 | 0.300 | 0.485 | +0.0006 | +0.0006 | +0.0004 |
| RELATED | 0.434 | 0.425 | 0.182 | +0.0017 | +0.0018 | +0.0018 |
| UNRELATED | 0.034 | 0.034 | 0.072 | +0.0010 | +0.0010 | +0.0010 |
| **SAME + INVERSE + IMPLIES** | **0.532** | 0.541 | 0.746 | +0.0013 | +0.0013 | +0.0011 |
| outcome-recording pairs (post-hoc) | 0.048 | | | +0.0013 | +0.0012 | +0.0014 |

Top-20 relation pairs by test-query count (s′, o′ = subject / object of the forward temporal fact; MRR on these queries, TiRGN base). Classes for human check; the full 258-pair table is in the CSV.

| n | static property | direction | temporal relation | class (source) | pass 2 / smaller | MRR HRCOV → HRCOV_ST |
|---|---|---|---|---|---|---|
| 756 | `P150` contains the administrative territorial entity | o' -p-> s' | `P131` located in the administrative territorial entity | INVERSE (rule) | INVERSE / INVERSE | 0.997 → 0.999 |
| 522 | `P1532` country for sport | s' -p-> o' | `P27` country of citizenship | IMPLIES (LLM) | IMPLIES / IMPLIES | 0.825 → 0.814 |
| 398 | `P1376` capital of | s' -p-> o' | `P131` located in the administrative territorial entity | IMPLIES (LLM) | IMPLIES / IMPLIES | 0.995 → 1.000 |
| 244 | `P647` drafted by | s' -p-> o' | `P54` member of sports team | IMPLIES (LLM) | IMPLIES / IMPLIES | 0.839 → 0.920 |
| 224 | `P27` country of citizenship | o' -p-> s' | `P35` head of state | RELATED (LLM) | RELATED / IMPLIES | 0.926 → 0.957 |
| 192 | `P36` capital | o' -p-> s' | `P131` located in the administrative territorial entity | IMPLIES (LLM) | IMPLIES / IMPLIES | 1.000 → 1.000 |
| 182 | `P27` country of citizenship | o' -p-> s' | `P6` head of government | RELATED (LLM) | RELATED / IMPLIES | 0.886 → 0.913 |
| 176 | `P6087` coach of sports team | s' -p-> o' | `P54` member of sports team | RELATED (LLM) | RELATED / RELATED | 0.670 → 0.840 |
| 168 | `P27` country of citizenship | s' -p-> o' | `P1532` country for sport | IMPLIES (LLM) | IMPLIES / IMPLIES | 0.974 → 0.978 |
| 166 | `P1830` owner of | o' -p-> s' | `P127` owned by | INVERSE (rule) | INVERSE / INVERSE | 0.819 → 0.924 |
| 132 | `P131` located in the administrative territorial entity | s' -p-> o' | `P1366` replaced by | UNRELATED (LLM) | UNRELATED / UNRELATED | 0.031 → 0.900 |
| 130 | `P20` place of death | s' -p-> o' | `P551` residence | RELATED (LLM) | IMPLIES / IMPLIES | 0.953 → 0.979 |
| 114 | `P102` member of political party | o' -p-> s' | `P488` chairperson | RELATED (LLM) | RELATED / UNRELATED | 0.847 → 0.922 |
| 112 | `P54` member of sports team | s' -p-> o' | `P647` drafted by | RELATED (LLM) | RELATED / IMPLIES | 0.045 → 0.596 |
| 104 | `P20` place of death | s' -p-> o' | `P937` work location | RELATED (LLM) | RELATED / IMPLIES | 0.933 → 0.955 |
| 90 | `P112` founder | s' -p-> o' | `P488` chairperson | RELATED (LLM) | RELATED / RELATED | 0.933 → 0.993 |
| 80 | `P937` work location | o' -p-> s' | `P6` head of government | RELATED (LLM) | RELATED / RELATED | 0.975 → 0.983 |
| 74 | `P19` place of birth | s' -p-> o' | `P937` work location | RELATED (LLM) | RELATED / IMPLIES | 0.978 → 0.972 |
| 70 | `P361` part of | s' -p-> o' | `P463` member of | IMPLIES (LLM) | RELATED / IMPLIES | 0.935 → 0.983 |
| 66 | `P6087` coach of sports team | o' -p-> s' | `P286` head coach | INVERSE (rule) | INVERSE / INVERSE | 0.582 → 0.689 |

Reading the audit. (1) About half of the direct-edge queries are answer-restating (inverse containment such as `P150` contains ↔ `P131` located in, `P1376` capital of → located in, `P1532` country for sport → citizenship); here HRCOV already ranks the gold highly (MRR ≈ 0.87), so their share of the gain is modest. (2) The largest per-query lifts are on pairs a semantic judge calls RELATED or UNRELATED but whose static value is the *present-day outcome* of the event being forecast: a former municipality 'located in' the municipality it was merged into vs the temporal `P1366` replaced by, and a player 'member of' the team vs `P647` drafted by. Per pair (TiRGN base, HRCOV → HRCOV_ST): `P131` s' -p-> o' `P1366` n = 132, 0.031 → 0.900; `P3842` s' -p-> o' `P1366` n = 34, 0.000 → 0.974; `P131` o' -p-> s' `P1365` n = 36, 0.569 → 0.920; `P3842` o' -p-> s' `P1365` n = 12, 0.005 → 0.807; `P54` s' -p-> o' `P647` n = 112, 0.045 → 0.596. Pooled over these 5 outcome-recording pairs (326 queries; P131 / P3842 vs P1366 replaced by and P1365 replaces, P54 vs P647), MRR HRCOV → HRCOV_ST is RE-GCN 0.095 → 0.736, LogCL 0.094 → 0.718, TiRGN 0.091 → 0.802. This is the clearest instance of possible future leakage, and it is identified post-hoc. (3) Semantic class alone therefore undercounts leakage; the operational control is the retrained NODIRECT arm, not the class split.

## 5. Where the gain lands (Table 4; HRCOV_ST − HRCOV, 3-seed means, B = 1,000)

| stratum | share (RE-GCN) | RE-GCN HRCOV → ST, Δ [CI] | LogCL HRCOV → ST, Δ [CI] | TiRGN HRCOV → ST, Δ [CI] |
|---|---|---|---|---|
| no pair history | 0.280 | 0.039 → 0.066, +0.0268 [+0.0253, +0.0282] | 0.039 → 0.065, +0.0265 [+0.0250, +0.0280] | 0.040 → 0.068, +0.0283 [+0.0268, +0.0296] |
| pair history | 0.720 | 0.901 → 0.901, +0.0003 [-0.0003, +0.0009] | 0.900 → 0.901, +0.0005 [-0.0000, +0.0010] | 0.901 → 0.901, +0.0002 [-0.0004, +0.0007] |
| first-seen entity | 0.083 | 0.037 → 0.101, +0.0639 [+0.0599, +0.0683] | 0.034 → 0.096, +0.0620 [+0.0582, +0.0662] | 0.044 → 0.110, +0.0660 [+0.0618, +0.0701] |
| first-seen object | 0.049 | 0.000 → 0.065, +0.0653 [+0.0605, +0.0703] | 0.000 → 0.066, +0.0655 [+0.0608, +0.0711] | 0.000 → 0.068, +0.0675 [+0.0622, +0.0727] |
| first-seen subject | 0.049 | 0.064 → 0.153, +0.0890 [+0.0824, +0.0955] | 0.058 → 0.144, +0.0856 [+0.0791, +0.0929] | 0.076 → 0.166, +0.0901 [+0.0839, +0.0964] |
| cold gold (< 50) | 0.789 | 0.621 → 0.632, +0.0107 [+0.0100, +0.0114] | 0.622 → 0.632, +0.0104 [+0.0098, +0.0110] | 0.623 → 0.633, +0.0105 [+0.0098, +0.0111] |
| mid | 0.186 | 0.783 → 0.778, -0.0042 [-0.0053, -0.0031] | 0.778 → 0.776, -0.0022 [-0.0032, -0.0011] | 0.781 → 0.779, -0.0014 [-0.0025, -0.0003] |
| head (≥ 500) | 0.025 | 0.955 → 0.956, +0.0005 [-0.0014, +0.0024] | 0.955 → 0.954, -0.0014 [-0.0034, +0.0006] | 0.954 → 0.956, +0.0015 [-0.0005, +0.0037] |
| gold outside base top-50 * | 0.344 | 0.293 → 0.315, +0.0224 [+0.0211, +0.0236] | 0.251 → 0.275, +0.0240 [+0.0227, +0.0252] | 0.010 → 0.040, +0.0307 [+0.0292, +0.0324] |
| gold in base top-50 * | 0.656 | 0.852 → 0.852, -0.0000 [-0.0007, +0.0006] | 0.848 → 0.849, +0.0003 [-0.0003, +0.0008] | 0.859 → 0.860, +0.0011 [+0.0005, +0.0017] |
| non-recurrent | 0.284 | 0.047 → 0.073, +0.0262 [+0.0246, +0.0277] | 0.046 → 0.072, +0.0262 [+0.0248, +0.0276] | 0.048 → 0.075, +0.0277 [+0.0263, +0.0291] |
| no pair, union-reachable (uncapped) | 0.098 | 0.038 → 0.124, +0.0854 [+0.0813, +0.0893] | 0.038 → 0.123, +0.0855 [+0.0815, +0.0896] | 0.038 → 0.126, +0.0878 [+0.0841, +0.0923] |
| no pair, union-unreachable | 0.182 | 0.040 → 0.035, -0.0050 [-0.0055, -0.0045] | 0.039 → 0.034, -0.0054 [-0.0061, -0.0048] | 0.041 → 0.037, -0.0039 [-0.0045, -0.0034] |
| no pair, gold in static candidate set | 0.031 | 0.050 → 0.331, +0.2813 [+0.2697, +0.2937] | 0.050 → 0.328, +0.2782 [+0.2659, +0.2910] | 0.053 → 0.340, +0.2868 [+0.2741, +0.2994] |
| no pair, gold outside static set | 0.248 | 0.038 → 0.033, -0.0054 [-0.0059, -0.0049] | 0.037 → 0.032, -0.0053 [-0.0058, -0.0048] | 0.038 → 0.034, -0.0044 [-0.0048, -0.0040] |
| gold in admitted static block * | 0.018 | 0.003 → 0.410, +0.4076 [+0.3869, +0.4275] | 0.003 → 0.401, +0.3987 [+0.3800, +0.4184] | 0.003 → 0.404, +0.4019 [+0.3815, +0.4217] |
| gold in C_cov * | 0.773 | 0.851 → 0.852, +0.0006 [+0.0000, +0.0012] | 0.850 → 0.850, +0.0009 [+0.0004, +0.0015] | 0.857 → 0.858, +0.0012 [+0.0006, +0.0018] |
| gold outside all candidates * | 0.208 | 0.005 → 0.005, -0.0003 [-0.0003, -0.0003] | 0.005 → 0.005, -0.0004 [-0.0004, -0.0004] | 0.005 → 0.004, -0.0003 [-0.0003, -0.0002] |

\* base-dependent stratum (share shown for RE-GCN).

!Where the static gain lands

*Figure 3.* HRCOV_ST − HRCOV per stratum with 95% CIs, three bases. (a) population strata (share of test in brackets); (b) strata conditional on the gold being reachable through static paths.

## 6. Reach and density (Table 5; test, TiRGN base for base-dependent rows)

| reach of the gold | all | no pair | first-seen | cold | pool-absent | no pair & pool-absent |
|---|---|---|---|---|---|---|
| share of test | 1.000 | 0.280 | 0.083 | 0.789 | 0.234 | 0.231 |
| temporal 1-hop (pair history) | 0.7201 | 0.0000 | 0.0000 | 0.6916 | 0.0130 | 0.0000 |
| temporal 2-hop | 0.0065 | 0.0015 | 0.0000 | 0.0031 | 0.0015 | 0.0013 |
| static 1-hop | 0.0414 | 0.0223 | 0.0609 | 0.0395 | 0.0233 | 0.0231 |
| static 2-hop, uncapped | 0.3907 | 0.3298 | 0.2904 | 0.3661 | 0.3360 | 0.3347 |
| static 2-hop, intermediate degree ≤ 1,000 | 0.1650 | 0.0810 | 0.1055 | 0.1443 | 0.0837 | 0.0828 |
| mixed static/temporal 2-hop | 0.0671 | 0.0211 | 0.0254 | 0.0382 | 0.0183 | 0.0167 |
| union graph ≤ 2-hop, uncapped | 0.8184 | 0.3513 | 0.3320 | 0.7979 | 0.3618 | 0.3534 |
| static candidate set used by the system (capped) | 0.2129 | 0.1123 | 0.1650 | 0.1838 | 0.1123 | 0.1100 |
| … its median size | 179 | 147 | 105 | 139 | 132 | 131 |
| … uniform random at that size | 0.0104 | 0.0073 | 0.0061 | 0.0074 | 0.0067 | 0.0067 |
| … degree-matched random at that size | 0.0185 | 0.0072 | 0.0061 | 0.0072 | 0.0066 | 0.0066 |
| … popularity top-k at that size | 0.1084 | 0.0536 | 0.0394 | 0.0006 | 0.0290 | 0.0288 |
| static list L_st @ 50 | 0.1640 | 0.0789 | 0.1268 | 0.1392 | 0.0783 | 0.0767 |
| static block admitted (≤ 50, outside C_cov) | 0.0179 | 0.0632 | 0.1142 | 0.0209 | 0.0755 | 0.0762 |
| degree-matched random block (same size) | 0.0002 | 0.0006 | 0.0005 | 0.0001 | 0.0007 | 0.0007 |
| C_cov (top-50 + pattern + bridge) | 0.7695 | 0.1989 | 0.1270 | 0.7359 | 0.0440 | 0.0339 |

Uncapped 2-hop *sets* on a uniform 20,000-query test sample (rng 7): temporal-only reach 0.720 with median |C| = 68 (0.29% of N; degree-matched 0.015); static-only 0.402 with median |C| = 6978 (13.6% of N; degree-matched 0.140, popularity 0.271); union 0.816 at 13.9% of N (degree-matched 0.150). The uncapped static 2-hop reach is almost entirely set size: at matched |C| a popularity list reaches 0.27 vs 0.40. The usable signal is in the capped, precision-ranked list: the admitted static block reaches the gold for 6.3% of no-pair and 11.4% of first-seen queries, against 0.06% / 0.05% for the degree-matched block of the same size.

!Reach by stratum

*Figure 2.* Share of test golds reached by each candidate source, per stratum (TiRGN base for the pool-absent stratum and the admitted blocks). Uncapped rows are membership tests through any intermediate; the capped static set is what the system can draw from.

**Ceilings** (placing the gold at rank 1 whenever it is in the set):

| base | base MRR | top-50 @1 | C_cov @1 | C_cov + static @1 | gold in static block | HRCOV realised | HRCOV_ST realised |
|---|---|---|---|---|---|---|---|
| RE-GCN | 0.4218 | 0.6485 | 0.7747 | 0.7925 | 0.0179 | 0.6595 | 0.6672 |
| LogCL | 0.4664 | 0.6758 | 0.7757 | 0.7935 | 0.0179 | 0.6590 | 0.6668 |
| TiRGN | 0.6181 | 0.7589 | 0.7706 | 0.7884 | 0.0179 | 0.6601 | 0.6682 |

The static block raises the placement ceiling by ≈ 0.018; HRCOV_ST realises ≈ 0.008 of it (≈ 43%).

## 7. What the static gain consists of (post-hoc attribution, not pre-registered)

| base | total Δ | gold in static block | gold in C_cov | elsewhere | queries with an s–o static edge (4.1%) | queries without (Δ masked, 95% CI) | density arm HRCOV_DM − HRCOV |
|---|---|---|---|---|---|---|---|
| RE-GCN | +0.0077 | +0.0073 | +0.0005 | -0.0001 | +0.0040 | +0.0037 [+0.0031, +0.0042] | +0.0020 (gold in DM block +0.0000) |
| LogCL | +0.0078 | +0.0071 | +0.0007 | -0.0001 | +0.0041 | +0.0037 [+0.0032, +0.0041] | +0.0017 (gold in DM block +0.0000) |
| TiRGN | +0.0080 | +0.0072 | +0.0009 | -0.0001 | +0.0039 | +0.0042 [+0.0036, +0.0046] | +0.0019 (gold in DM block +0.0000) |

* **Density.** HRCOV_DM's gain comes from C_cov golds (its random block almost never holds the gold), i.e. from the static *features*; HRCOV_STFEAT gives the same size of gain. Static candidates add ≈ +0.006 beyond features and density on every base.
* **Direct static edges.** About half of the gain is on the 4.1% of queries whose (s, o) pair is already a static edge under a *different* property (same property, reversed pair: 0.06% of queries, ≈ 0 contribution). The static KG is a time-less 2024 snapshot, so such an edge can encode the test pair itself (e.g. `P131` located-in, `P47` shares-border, `P3842` present-day territory, `P527`/`P361` has-part/part-of). This part is time-less background knowledge and possible future leakage, not forecasting gain. The masked decomposition here (+0.0037 to +0.0042 on the remaining 95.9%) is superseded by the retrained no-direct-edge arms in §4b.

| base | best path shape of gold in static block: n, contribution, mean Δ |
|---|---|
| RE-GCN | S: 664, +0.0030, +0.741; SS: 1831, +0.0036, +0.325; ST: 200, +0.0005, +0.377; TS: 220, +0.0002, +0.120 |
| LogCL | S: 672, +0.0031, +0.750; SS: 1837, +0.0035, +0.309; ST: 193, +0.0004, +0.352; TS: 222, +0.0002, +0.118 |
| TiRGN | S: 665, +0.0030, +0.736; SS: 1897, +0.0037, +0.318; ST: 162, +0.0003, +0.312; TS: 191, +0.0002, +0.154 |

Top path types by contribution (TiRGN): `SS:P19/P159^-1` (n = 182, +0.0007); `S:P131` (n = 58, +0.0003); `SS:P17/P1532^-1` (n = 173, +0.0003); `SS:P1532/P17^-1` (n = 129, +0.0003); `S:P131^-1` (n = 56, +0.0003); `SS:P27/P17^-1` (n = 132, +0.0003); `SS:P17/P27^-1` (n = 168, +0.0002); `SS:P159/P19^-1` (n = 97, +0.0002).
Top first-hop static relations: `P131` (+0.0008), `P19` (+0.0007), `P17` (+0.0006), `P27` (+0.0004), `P1532` (+0.0004), `P47` (+0.0003), `P3842` (+0.0003), `P159` (+0.0002).

SS paths (shared-attribute explanations: `P19` place of birth / `P159` headquarters location, `P27` citizenship / `P17` country, `P1532` country for sport) carry the largest share; S paths have the largest per-query effect (mean Δ ≈ 0.74). ST/TS (mixed static/temporal) paths add little.

**First-seen entities.** Static edges do help entities with no temporal history: first-seen objects go from MRR 0.000 → 0.065, 0.000 → 0.066, 0.000 → 0.068; when the gold is in the admitted static block, first-seen queries reach MRR ≈ 0.52–0.54. These queries are 4.9% (object) / 4.9% (subject) of test.

## 8. Negative and null results kept in view

1. Losses where static candidates cannot help: no-pair queries unreachable in the union graph (18.2%): -0.0050 [-0.0055, -0.0045], -0.0054 [-0.0061, -0.0048], -0.0039 [-0.0045, -0.0034]; mid-frequency golds: -0.0042, -0.0022, -0.0014; LogCL head stratum -0.0014 (CI includes 0).
2. Reach is mostly hub-driven when uncapped: static 2-hop reaches 33% of no-pair golds but through sets of ≈ 14% of all entities, where degree-matched random admission reaches 0.14 of all golds and popularity 0.27 (20k sample). Only 11% of no-pair golds are in the capped static candidate set, 6.3% in the admitted block.
3. Static candidates cannot replace history-aware pool widening on weak bases: STONLY − HRCOV = RE-GCN -0.0734, LogCL -0.0556, TiRGN +0.0057 (positive only on TiRGN, whose top-50 already holds the historical partners: HRCOV − HR = +0.0029).
4. About half of the HRCOV_ST gain rests on direct s–o static edges of a time-less 2024 snapshot (time-less background knowledge, possible future leakage); without them (§4b) the PASS survives but the gain shrinks to RE-GCN +0.0030, LogCL +0.0035, TiRGN +0.0051 and the first-seen-object gain by ≈ 60%. 2-hop static paths can still carry post-t snapshot facts; no dated static snapshot ships with TGB 2.0.
5. HRCOV_ST realises ≈ 43% of the static-block placement ceiling; 21% of test golds are in no candidate set at all (MRR ≈ 0.005).
6. Our RE-GCN / LogCL are 0.13–0.17 below the TGB RE-GCN row; HR minus RecB-lite is RE-GCN -0.0345, LogCL -0.0153, TiRGN +0.0535 (HR on the two weak bases is below the recurrency rule).
7. HRCOV_PAT ≈ HRCOV (RE-GCN +0.0002, LogCL -0.0000, TiRGN -0.0005): as on polecat, the pattern block alone carries the coverage stack; the bridge block adds ≈ 0.
8. 6.5 utility-gated augmentation with static evidence was not run (not cheap here; pilot-1 gate AUC < 0.5).

## 9. Deviations from the task specification

1. TiRGN was trained from scratch (no released smallpedia checkpoint) with the authors' model code via the project's pure-torch DGL shim and the project's ICEWS18 TiRGN hyper-parameters, without the word graph; TiRGN's sparse history files were replaced by an on-the-fly first-occurrence index (identical 0/1 vocabulary) and history_mode's float64 CPU copy by vocab.float(); valid-selected epoch 20 (early-stopped at epoch 25 of 30, patience 5).
2. RE-GCN / LogCL epoch budget selected on full valid from {3, 6, 12, 30, 60}; both selected 3 epochs (grid edge; 1-2 epochs not tried). The first 3/6-epoch attempts died of GPU OOM on the shared card and were rerun unchanged.
3. Pilot-1's GPU per-list lookup was replaced by full float32 log-prob rows of every valid/test query (62 GB per base); valid/test top-K, the HR pool and the bridge anchors come from the full rows (agreement with the dump 99.89% / 99.91% / 100%); pattern-prior base mass is exact instead of top-200 + uniform tail.
4. Static path enumerator caps: intermediates with static degree > 1,000 do not expand (122 hub nodes), ST hop expands <= 128 static neighbours, temporal hops use the 64 most recent distinct partners; uncapped reach is reported as membership tests on all test queries, uncapped 2-hop SET sizes only on a uniform 20,000-query test sample (rng 7).
5. Static path-type precision table fitted on 100,000 of 465,134 early-train queries (ts < 1968); pattern TE tables on 400,000 of 465,134 (pilot-1 recipe).
6. Optional 6.5 utility gate with static evidence not run.
7. Official py-tgb evaluation of re-rankers for seed 0 only (HR, HRCOV, HRCOV_ST, HR_ST); 3-seed means use the harness evaluator (MRR identical to 1e-6 for every seed-0 system).
8. rs_features 30-snapshot recency feature not rescaled for yearly snapshots (30 years here).
9. Re-ranker training capped at 30 epochs; several arms (reference and primary alike) peak at epochs 25-29.
10. Attribution (density split, direct-static-edge split, path shapes / relations, first-seen) is post-hoc. The direct-edge question was subsequently tested with retrained arms (addendum, registered before their results): HRCOV_ST_NODIRECT and HRCOV_ST_NODIRECT_STRICT, 3 bases x 3 seeds, reusing the pilot-2 HRCOV ranks and bridge rankers (C_cov asserted identical to pilot 2 on all valid/test rows).
11. Decision-rule registration note amended after registration (provenance only: a TiRGN smoke-run base row had also been printed); criteria unchanged.
12. Stage-2 bridge cohort: 8,000 of 45,884-53,774 coverage-hard train rows, built on the transductive static BridgeIndex (pilot-1 pitfall 6).
13. Pilot-1 arms HR_R / HRCOV_R, HRCOV_DYAD and the PAT_BUMP / BR_CS references were not run; HRCOV_PAT was.
14. Leakage audit: static-vs-temporal relation pairs classified by Wikidata metadata rules (identity, P1696 inverse, symmetric) for 23 of 258 pairs and by an LLM judge (reasoning model, labels + descriptions from wikidata.org) for the rest, not by human annotators; judge self-agreement 0.945 (query-weighted), agreement with a smaller model 0.655; top-20 pairs listed for human check. The 'outcome-recording' pair group is post-hoc.
15. Addendum arms evaluated with the official py-tgb Evaluator for seed 0 only; 3-seed means use the harness evaluator (MRR identical to 1e-6).

## 10. Mapping to  Task 2 and implications for pilots 3–4

* **Task 2 (temporal knowledge fusion).** This is the first positive Task-2 result in the project: earlier fusion priors (ECCKG relation prior, actor-KG prior, two-channel fusion) were null on ICEWS18/GDELT. Here the fused knowledge source is entity-attribute knowledge (Wikidata), not eventuality commonsense, and it works as a *candidate generator plus features* on top of the strongest history system rather than as a score prior. The shared-attribute paths (birthplace ↔ headquarters, citizenship ↔ country) are human-readable explanations of why two entities become linked, which is the Task-2 premise; the direct-edge half is retrieval of a pair already known to the static KG (background knowledge / possible leakage). The defensible Task-2 effect size is the no-direct-edge estimate (+0.003 to +0.005 MRR over HRCOV, PASS on 3/3 bases), concentrated on no-pair and first-seen queries.
* **Where it matters.** The gain is confined to queries without temporal pair history and to first-seen entities — exactly the strata that temporal history cannot reach (temporal 2-hop reaches 0.15% of no-pair golds). On pair-history queries Δ ≈ 0.
* **Pilot 3 (tkgl-icews).** No static KG ships with ICEWS; the transferable part is the candidate-block-plus-density-control design. Full log-prob rows are infeasible there (≈ 1.6 TB); use pilot-1's list lookup.
* **Pilot 4 (story-context channel, Task 3).** Static-path explanations (typed SS paths with precision) are compact context for an LLM re-ranker on no-history queries; pair-history evidence remains the context to prioritise elsewhere.

## 11. Hosts

All training, dumps, lists, features, re-rankers and official evaluation: `serverA` (GPU: RTX PRO 6000 Blackwell Max-Q, docker `tkg-gpu:latest`, torch 2.8.0+cu128; CPU stages in the same image; py-tgb 2.3.0). Aggregation, bootstrap and figures: local laptop (CPU). Jobs: verification 1320ca69; prep 117ffe9a; base grids dcaa493c, dfa41d7e, 542e9f3b, 66787db1; TiRGN 486a8a57; chains 4cbaace8 (RE-GCN), cbe24163 (LogCL), b3e2e72f (TiRGN); direct-edge control c0010ec1 (RE-GCN), c676ce84 (LogCL), 6de5f663 (TiRGN). Wikidata property labels for the leakage audit: wikidata.org API (local).

## 12. Artifacts

* `smallpedia_pilot_verdict.json`
* `smallpedia_decision_rule.json`
* `smallpedia_decision_rule_addendum_nodirect.json`
* `smallpedia_leakage_relation_pairs.csv`
* `pilot_pipeline_README_v2.md`
* `base_selection.json`
* `fig1_fulltest_mrr.png`
* `fig2_reach_by_stratum.png`
* `fig3_where_static_lands.png`
* `fig4_nodirect_control.png`
* `sp_common.py`
* `sp_verify.py`
* `sp_static.py`
* `sp_prep.py`
* `sp_tirgn.py`
* `sp_fullrows.py`
* `sp_lists.py`
* `sp_features.py`
* `sp_train.py`
* `sp_official.py`
* `sp_listctl.py`
* `sp_agg.py`
* `sp_figs.py`
* `sp_chain.sh`
* `sp_base_grid.sh`
* `leakage_audit.json`
* `wikidata_property_labels.json`
* `leakage_pair_judgements.json`
* `sp_chain_nd.sh`
* `sp_leakage_audit.py`
* `sp_agg_nd.py`
