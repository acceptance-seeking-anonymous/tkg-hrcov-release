# G8 — CognTKE base on tkgl-polecat: verdict under the registered rule

**Outcome: PRACTICAL_ADVANCE_ON_COGNTKE** (ts-block sensitivity: PRACTICAL_ADVANCE_ON_COGNTKE). Rule: g8_rule.json (registered 2026-10-13T18:28:20Z, sha256 273be7b9…), amended by g8_deviation_B.json (2026-10-13T18:43:00Z); both registered before any CognTKE MRR existed. No configuration was changed after a test number was read.

## Summary (3 lines)
1. HRCOV on CognTKE reaches official test MRR 0.3040 against 0.2941 for the better control (CognTKE⊕RecB blend): Δ = +0.0099, 95% CI [+0.0096, +0.0103], one-sided p < 0.0005 (B = 2000); against RecB-POOL Δ = +0.0104. Both clear the 0.005 margin; the head guardrail shows no damage (Δ_head = +0.0123, CI [+0.0117, +0.0129]).
2. Base fidelity is within tolerance: our CognTKE gives MRR 0.2914 / H@10 0.4603 against the published 0.287 / 0.455 (|ΔMRR| = 0.0044 ≤ 0.010).
3. The margin is about one fifth of the margin on TiRGN, and it comes from queries whose gold has history. When the gold lies outside CognTKE's reasoning subgraph (unreached, 24.1% of test queries), HRCOV is -0.0019 below the blend.

## Main-table rows (tkgl-polecat, official py-tgb 2.3.0 evaluator, n = 532,636 test queries)
| System | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| CognTKE (README configuration, valid-best epoch 17) | 0.2914 | 0.2049 | 0.3316 | 0.4603 |
| CognTKE ⊕ RecB blend (valid-selected; better control) | 0.2941 | 0.2070 | 0.3336 | 0.4649 |
| CognTKE + RecB-POOL (valid-selected) | 0.2936 | 0.2070 | 0.3336 | 0.4649 |
| CognTKE + HR (seeds 0–2) | 0.3011 | 0.2164 | 0.3405 | 0.4683 |
| CognTKE + HRCOV (HRCOV_PAT, bridge-free, seeds 0–2) | 0.3040 | 0.2178 | 0.3438 | 0.4741 |
| CognTKE + HRCOV with bridge (secondary) | 0.3042 | – | – | – |

Per-seed MRR: HR 0.3011, 0.3015, 0.3007; HRCOV_PAT 0.3042, 0.3042, 0.3036; HRCOV (bridge) 0.3043, 0.3041, 0.3042.

## Decision statistics (paired; query bootstrap B = 2000, timestamp-block bootstrap over 304 test timestamps as sensitivity)
| Contrast | Δ MRR | 95% CI (query) | 95% CI (ts-block) | Pass |
|---|---|---|---|---|
| C1: HRCOV − BLEND | +0.0099 | [+0.0096, +0.0103] | [+0.0094, +0.0105] | True |
| C1: HRCOV − POOL | +0.0104 | [+0.0101, +0.0107] | [+0.0099, +0.0110] | True |
| C2: head (gold train freq ≥ 500, n = 242,300) HRCOV − BLEND | +0.0123 | [+0.0117, +0.0129] | – | no damage |

Secondary contrasts (reported; not part of the decision):
| Contrast | Δ MRR | 95% CI (query) | 95% CI (ts-block) |
|---|---|---|---|
| HRCOV − CognTKE | +0.0126 | [+0.0122, +0.0129] | [+0.0120, +0.0131] |
| HR − CognTKE | +0.0097 | [+0.0093, +0.0100] | [+0.0091, +0.0102] |
| HR − better control | +0.0070 | [+0.0067, +0.0073] | [+0.0065, +0.0075] |
| HRCOV − HR | +0.0029 | [+0.0027, +0.0031] | [+0.0026, +0.0032] |
| HRCOV (bridge) − HRCOV (bridge-free) | +0.00024 | [+0.0001, +0.0004] | [+0.0000, +0.0004] |
| HRCOV on CognTKE − HRCOV on TiRGN (0.2824) | +0.0216 | [+0.0211, +0.0221] | [+0.0205, +0.0227] |
| CognTKE − TiRGN base (0.2205) | +0.0709 | [+0.0702, +0.0717] | [+0.0692, +0.0726] |

## Strata (HRCOV − better control, query bootstrap B = 1000)
| Stratum | Share | MRR base | MRR BLEND | MRR HRCOV | Δ | 95% CI |
|---|---|---|---|---|---|---|
| head (gold train freq >= 500) | 0.455 | 0.3832 | 0.3881 | 0.4004 | +0.0123 | [+0.0117, +0.0129] |
| mid (50-499) | 0.212 | 0.2190 | 0.2201 | 0.2375 | +0.0174 | [+0.0168, +0.0181] |
| cold (< 50) | 0.333 | 0.2122 | 0.2127 | 0.2146 | +0.0019 | [+0.0015, +0.0023] |
| pool-absent (gold outside CognTKE top-50) | 0.401 | 0.0029 | 0.0069 | 0.0063 | -0.0006 | [-0.0007, -0.0004] |
| first-seen | 0.108 | 0.1591 | 0.1702 | 0.1609 | -0.0093 | [-0.0097, -0.0088] |
| no-pair-history | 0.386 | 0.0675 | 0.0711 | 0.0687 | -0.0024 | [-0.0027, -0.0022] |
| unreached | 0.241 | 0.0000 | 0.0066 | 0.0047 | -0.0019 | [-0.0022, -0.0016] |

Reached queries (gold inside CognTKE's subgraph, n = 404,455): HRCOV − BLEND = +0.0137, CI [+0.0132, +0.0141]. The released forward pass gives unreached entities logit 0, so in that stratum the base gold MRR is 1.3e-05. Scoring unreached entities at −inf instead (the registered secondary) leaves the base MRR unchanged at 0.2914 (Δ = 0.0e+00).

## Fidelity and evaluation checks
- Base vs published (CountTRuCoLa, arXiv:2509.09474 Table 1): MRR 0.2914 vs 0.287 (|Δ| = 0.0044, tolerance 0.010, BASE_FIDELITY_OK); H@10 0.4603 vs 0.455.
- Training: deviation B (README configuration), 20 epochs, best full-valid MRR 0.3012 at epoch 17 (v_n_nan = 0).
- Official evaluator vs closed-form rank: max per-batch deviation 0.0 (official_vs_formula_max_batch_dev).
- Finiteness, checked before g8_agg.py ran: the CognTKE dumps (valid + test sparse rows, log-sum-exp, top-200, gold scores; 2.62 × 10⁹ values), the official outputs and the training state hold 0 NaN and 0 ±inf. The pc pipeline holds 0 NaN and 0 +inf. Its 8,056,559 −inf values all sit at padded candidate slots of feat_cogn/{valid,test}_logp.npy (5,970,720 test + 2,085,839 valid); none sits at a real candidate. HR / HRCOV / HRCOV_PAT scores (seeds 0–2) were recomputed from the saved models over every valid and test slot, and they and the fused scores are all finite. The official CognTKE rows are read from the scanned g8_dump store, not replayed.

## Caveats
- The harness metric used for validation selection is inflated for the CognTKE base: test harness MRR is 0.3230 against 0.2914 official. For the controls the two metrics are close (BLEND 0.2950 vs 0.2941). The inflation fits the large block of entities tied at logit 0, which the two rank conventions handle differently; we did not investigate it further. As a result the controls scored below the base on validation, although on the official test evaluator both are above it (+0.0027 / +0.0022). Blend configurations are compared with each other on the same metric, so control selection is not obviously distorted. The selected blend is tr MIX 0.05.
- The gain is smaller than on TiRGN. On TiRGN, HRCOV_PAT − BLEND is about +0.048 (v7 seed-42 values 0.2824 vs 0.2348); on CognTKE it is +0.0099. HRCOV is below the blend on first-seen, no-pair-history, pool-absent and unreached queries. The advantage comes from head and mid-frequency golds whose pair history the re-ranker can use.
- Single CognTKE training run (one seed). Re-ranker seeds 0–2 are averaged per query.
