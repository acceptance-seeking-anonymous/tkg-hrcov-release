# C4 — base-seed variation, TiRGN on tkgl-polecat (pre-specified later analysis)

Protocol: c4_protocol.json (registered 2026-10-14T18:54:49Z, sha256 171774ea…, before any seed-43 number), under tois_v7_rule.json. Second TiRGN seed 43 trained with the seed-42 recipe on <HOST> GPU 0 after G8 released it. Pipeline: pc lists, exact lookups and features; v7 MLP64_PAT and EV_PT fits (seeds 0–2); controls selected on the valid grid; one official test pass (py-tgb 2.3.0; official vs closed-form max deviation 0.0). All chain stages rc = 0. Label: **pre-specified later analysis (C4)**; descriptive.

## Summary
1. **The primary contrast holds on the second base seed.** HRCOV_PAT − better control (POOL) = +0.0454 [+0.0443, +0.0466] (ts-block 95% CI) on seed 43, against +0.0475 on seed 42. HRCOV_PAT − BLEND = +0.0455. The margin changes by -0.0021 [-0.0027, -0.0014] between seeds, about 4% of its size, and that change lies inside ±0.005 in both directions.
2. **The B3 base contribution reproduces.** FULL − EV_NF_PT = +0.0087 on seed 43 and +0.0086 on seed 42 (difference +0.0001 [-0.0003, +0.0004]).
3. **Base seed variation is small in aggregate but large per query.** Base MRR is 0.2218 vs 0.2205 (+0.0013 [+0.0004, +0.0021]). Per-query RR differs on 81% of test queries (Pearson r = 0.874). HRCOV_PAT moves the other way (-0.0009 [-0.0014, -0.0005]), so the margin over the controls shrinks by about 0.002.

## Per-system MRR / H@10 (official evaluator, n = 532,636)
| System | MRR s42 | H@10 s42 | MRR s43 | H@10 s43 | Δ MRR (s43 − s42) [ts CI] |
|---|---|---|---|---|---|
| TiRGN base | 0.2205 | 0.3750 | 0.2218 | 0.3758 | +0.0013 [+0.0004, +0.0021] |
| BASE⊕RecB blend | 0.2348 | 0.3879 | 0.2359 | 0.3886 | +0.0011 [+0.0004, +0.0018] |
| RecB-POOL (better control on both seeds) | 0.2349 | 0.3879 | 0.2360 | 0.3886 | +0.0011 [+0.0004, +0.0018] |
| EV_NF_PT (evidence features, score alone) | 0.2737 | 0.4382 | 0.2727 | 0.4372 | -0.0010 (point) |
| HRCOV_PAT (= FULL; seeds 0–2) | 0.2823 | 0.4527 | 0.2814 | 0.4517 | -0.0009 [-0.0014, -0.0005] |

Re-ranker seeds (per-seed MRR): HRCOV_PAT s42 0.2820, 0.2825, 0.2824; s43 0.2814, 0.2818, 0.2810. EV_NF_PT s42 0.2738, 0.2735, 0.2737; s43 0.2727, 0.2727, 0.2728.

## Contrasts (paired; Δ MRR with 95% CI from the timestamp-block bootstrap; query-bootstrap CIs are in c4_results.json)
| Contrast | seed 42 | seed 43 | seed 43 − seed 42 |
|---|---|---|---|
| HRCOV_PAT − better control (POOL) | +0.0475 [+0.0463, +0.0487] | +0.0454 [+0.0443, +0.0466] | -0.0021 [-0.0027, -0.0014] |
| HRCOV_PAT − BLEND | +0.0475 [+0.0464, +0.0487] | +0.0455 [+0.0444, +0.0467] | -0.0020 [-0.0026, -0.0014] |
| FULL − EV_NF_PT (B3 base contribution, common PT pool) | +0.0086 [+0.0080, +0.0093] | +0.0087 [+0.0080, +0.0094] | +0.0001 [-0.0003, +0.0004] |
| HRCOV_PAT − base | +0.0618 [+0.0605, +0.0633] | +0.0596 [+0.0583, +0.0610] | -0.0022 (point) |

All seed-42 and seed-43 contrasts are present (one-sided L > 0 under both bootstraps). Seed-42 HRCOV_PAT here is the v7 re-fit (MLP64_PAT_fused, 0.2823), the same code path as seed 43. Using the stored claim-system HRCOV_PAT (0.2824) instead gives a seed change of -0.0021 for (HRCOV_PAT − BLEND).

## Base training and selection
- Seed 43: best epoch 5 (patience stop after epoch 8), best full-valid filtered MRR 0.2418, 6,107 s training. Seed 42: best epoch 4, 0.2404.
- Valid selection picked the same configuration on both seeds: blend LOGLIN (0.25, 0.001); POOL K = 100 with the same score. Seed-43 valid MRR (200k sample): base 0.2424, blend 0.2541, pool 0.2542.

## Validity
Finiteness (same check as G8): the seed-43 checkpoint and dumps, pc pipeline, v7 fits and official outputs hold 0 NaN and 0 +inf. The 6,319,351 −inf values all sit at padded candidate slots of feat_tirgn2/{valid,test}_logp.npy (4,866,386 test + 1,452,965 valid); none sits at a real candidate. As for seed 42, TiRGN test rows are replayed live inside the official pass rather than stored. Re-ranker scores were not recomputed.

## Caveats
- The CIs resample test queries (and timestamps) for two fixed trained bases. They do not estimate training-seed variance: with two seeds, that variance is limited to the observed range. The seed-to-seed change in the margin (0.0020) is of the same order as the re-ranker seed range (s43 0.0007, s42 0.0005), and both are far below the margin (0.045).
- Seed-42 official scoring ran on <HOST> (RTX 4090); seed 43 ran on <HOST> (RTX PRO 6000).

Suggested manuscript wording: *"A second TiRGN training seed (pre-specified later analysis) gives HRCOV_PAT − better control = +0.0454 (95% CI [+0.0443, +0.0466]) against +0.0475 on the original seed. The base contribution on the common pool (FULL − EV_NF_PT) is +0.0087 against +0.0086."*

## Note on the seed-42 replay (added 2026-10-15, investigator request)
The seed-42 arms in this comparison come from two official scoring runs of the same checkpoint. TiRGN polecat rows are re-computed from the checkpoint inside each official pass (live replay); they are not stored.
- The controls file (`off_polecat_tirgn_ctrl.npz`: base, BLEND, POOL) was scored on <HOST> (RTX 4090).
- The v7 files were scored on <HOST> GPU 1 (RTX PRO 6000): `official_test_polecat_v7_polecat_tirgn_00.npz` holds HRCOV_PAT = FULL and `_01.npz` holds EV_NF_PT.

The two GPUs give float-level differences in the replayed scores. This creates or breaks exact ties for a few queries, and the official evaluator counts a tie as half a position.

**How many queries differ** (base reciprocal rank, threshold 1e-6; n = 532,636):

| File compared with the controls file | Queries that differ | max abs dRR | mean abs dRR | mean signed dRR |
|---|---|---|---|---|
| `_00` (HRCOV_PAT = FULL) | 57 (0.011%) | 0.0022 | 6.3e-9 | -5.1e-9 |
| `_01` (EV_NF_PT) | 51 (0.010%) | 0.0048 | 1.6e-8 | -1.5e-8 |
| stored HRCOV_PAT run (sensitivity row) | 44 (0.008%) | 0.0014 | 4.5e-9 | -3.3e-9 |

Union of `_00` and `_01`: 75 queries.
- The typical change is half a rank position (median |d rank| = 0.5), the signature of a tie appearing or disappearing.
- The affected golds are mostly deep in the base ranking: median base rank about 450, best rank 10.
- A few rank shifts are large (up to 45,037 positions) at very deep ranks, where the reciprocal rank is close to zero.

**Effect on the seed-42 contrasts**
- Worst-case bound on the replay effect: 57 x 0.0022 / n = 2.3e-7 for `_00`, and 51 x 0.0048 / n = 4.6e-7 for `_01`.
- Recomputing with the 75 affected queries excluded changes the contrasts by at most 4.2e-6:
  - HRCOV_PAT - POOL: +0.047462 -> +0.047465
  - FULL - EV_NF_PT: +0.008630 -> +0.008631
  - seed-43 - seed-42 margin change: -0.002051 -> -0.002051
  - base seed difference: +0.001301 -> +0.001299

The worst-case bound (at most 4.6e-7) is about 200 times smaller than the reported precision (0.0001), roughly two orders of magnitude, and the largest change from excluding the affected queries (4.2e-6) is about 24 times smaller; both are far below the width of every CI. The C4 conclusions are unchanged.

The seed-43 arms all come from a single official pass on <HOST> GPU 0 (base, BLEND, POOL and the promoted lists together), so they are not affected.
