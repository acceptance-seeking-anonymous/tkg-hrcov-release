# Claims-hardening controls: is HRCOV a practical advance or "recurrency done well"? (TGB 2.0 `tkgl-polecat`, `tkgl-smallpedia`)

**Question.** HRCOV is the pilot-1 coverage stack: a history re-ranker over the base model's top-50 plus 50 pattern-prior candidates and 50 stage-2 bridge candidates, 30 features, 3 seeds. Does it beat recurrency used *well*? That means the TGB recurrency baseline RecB (Gastinger et al., IJCAI 2024) in its trained mode, blended into the base model's scores (BASE(+)RecB), or used to widen the base model's candidate pool (RecB-POOL), with every weight and pool size chosen on validation. The test is run on the strongest bases we could train. Epochs were chosen by validation for TiRGN (both datasets) and for the re-tuned smallpedia RE-GCN / LogCL. The polecat RE-GCN / LogCL checkpoints keep the pilot-1 fixed 12-epoch budget, and polecat LogCL is one of the two decision bases (§3, §8).

**Answer (pre-registered rule, official TGB evaluator, every test query).** The claim survives on both datasets: the outcome is **PRACTICAL_ADVANCE** on `tkgl-polecat` and on `tkgl-smallpedia`. On each dataset's two strongest bases (by validation MRR), HRCOV exceeds the better recurrency control on that base by +0.033 to +0.059 MRR (criterion C1, margin 0.005). It also exceeds the strongest BASE(+)RecB blend of any base by +0.031 to +0.048 (C2). Every one-sided bootstrap p is < 0.0005 (B = 2000), and the head stratum gains rather than loses (C3: +0.010 to +0.049).

| dataset | base (valid rank) | HRCOV (3 seeds) | best recurrency control on that base | C1 Δ [95% CI] | C2 Δ vs strongest blend | head Δ (C3) | outcome |
|---|---|---|---|---|---|---|---|
| tkgl-polecat | TiRGN (1) | 0.2824 ± 0.0002 | RecB-POOL 0.2349 | +0.0475 [+0.0469, +0.0481] | +0.0476 (TiRGN blend 0.2348) | +0.0476 | pass |
| tkgl-polecat | LogCL (2) | 0.2721 ± 0.0003 | RecB-POOL 0.2131 | +0.0590 [+0.0583, +0.0596] | +0.0373 | +0.0490 | pass |
| tkgl-smallpedia | TiRGN (1) | 0.6601 ± 0.0002 | BASE(+)RecB 0.6275 | +0.0327 [+0.0318, +0.0336] | +0.0327 (TiRGN blend 0.6275) | +0.0194 | pass |
| tkgl-smallpedia | RE-GCN re-tuned (2) | 0.6587 ± 0.0003 | BASE(+)RecB 0.6108 | +0.0480 [+0.0468, +0.0492] | +0.0313 | +0.0105 | pass |

(C1 Δ is against the better of blend and pool on that base. Both are listed in §5; they differ by ≤ 0.0002.)

**Qualification: what the gain is made of.** The question was whether the gain is *history-aware pool widening*. On the strongest bases, pool widening is only a minor part of HRCOV's margin. Most of it comes from the history re-ranker over the base top-50 (HR). HRCOV − HR is +0.0129 [+0.0126, +0.0133] on polecat TiRGN and only +0.0029 [+0.0025, +0.0033] on smallpedia TiRGN. By comparison, HR − best control is +0.0346 and +0.0298 on those two bases. On weaker bases pool widening is large (+0.036 to +0.090), because the pool then adds the recurrent golds that the base ranks outside its top-50. "History-aware pool widening" therefore describes the mechanism on weak bases; on strong bases the re-ranker does most of the work. Re-tuning the smallpedia bases raised their valid MRR (RE-GCN 0.490 → 0.546, LogCL 0.509 → 0.533) but left HRCOV unchanged (0.6585–0.6601 on every smallpedia base).

**Against published rows (not re-run; cited for scale).** polecat HRCOV 0.2824 vs CountTRuCoLa 0.256 (arXiv 2509.09474), NHCTE 0.245, TLogic 0.228, RecB-trained 0.198 (TGB 2.0). smallpedia HRCOV 0.6601 (HRCOV_ST 0.6682) vs CountTRuCoLa 0.644, NHCTE 0.618, CEN 0.612, RecB-trained 0.605. The comparison inherits our protocol (official evaluator and negatives). The published systems were not re-run here.

!Absolute test MRR per base

*Figure 1.* Test MRR (official py-tgb evaluator, all test queries; HRCOV = mean of seeds 0–2 with across-seed SD bar) per base, ordered by validation MRR (strongest on top). Grey: base model alone. Blue: the better of BASE(+)RecB and RecB-POOL on that base (valid-selected). Light orange: history re-ranker HR. Dark orange: HRCOV. Dashed line: RecB trained mode (faithful reproduction of the TGB script; polecat 0.198, smallpedia 0.605). Dotted line: best published row (CountTRuCoLa, not re-run). On smallpedia TiRGN, the HR marker (0.6572) is mostly hidden behind HRCOV (0.6601). n = 532,636 (polecat) and 163,172 (smallpedia) test queries.

## 1. Pre-registered decision rule

`hardening_decision_rule.json` was registered at 2026-10-12T18:43:20Z, before any number for any control existed. Summary:

1. **Comparators per base b.** BASE(+)RecB: full-row blend of the base probabilities with the normalised RecB score; MIX β ∈ 13 values or LOGLIN (w, ε) ∈ 8 values; RecB default or trained; 42 configurations. RecB-POOL: base top-K ∪ RecB top-K, K ∈ {20, 50, 100}, ranked by the blend, with no learned re-ranker; 126 configurations. Each comparator is selected by validation MRR (polecat: the pilots' 200k valid sample; smallpedia: full valid), and test is scored once.
2. **Two strongest bases.** Per architecture, the valid-best configuration and epoch; then the two architectures with the highest valid MRR among bases on which HRCOV (3 seeds) was completed.
3. **C1.** On each of the two strongest bases, HRCOV − b(+)RecB ≥ 0.005 and HRCOV − RecB-POOL_b ≥ 0.005, each with one-sided p < 0.05.
4. **C2.** On each of them, HRCOV_b exceeds the strongest b'(+)RecB of any trained base of the dataset (p < 0.05).
5. **C3.** No head damage. On the head stratum (gold train frequency ≥ 500), the 95% CI of HRCOV − better control must not lie entirely below 0 with |Δ| ≥ 0.001.
6. **Outcome.** PASS on both bases = PRACTICAL_ADVANCE; C1 or C2 failing = RECURRENCY_DONE_WELL; C1 and C2 passing with C3 failing = HEAD_DAMAGE. The claim survives overall iff both datasets are PRACTICAL_ADVANCE.
7. **Statistic.** Official py-tgb 2.3.0 Evaluator per test query with the shipped negatives, re-rankers averaged over seeds 0–2, Poisson(1) paired bootstrap (B = 2000; strata B = 1000). No configuration was changed after any test number was read.


## 2. Recurrency baseline (RecB) reproduction

RecB is the combined recurrency baseline ψΔξ of Gastinger et al. (IJCAI 2024, arXiv:2404.16726) as implemented in TGB 2.0
(`modules/recurrencybaseline_predictor.py`): score(o) = 1000·α_r·ψ(o) + 1000·(1−α_r)·ξ(o), with ψ the exponentially
time-decayed strict-recurrency score (decay 2^{λ_r(k−t)}, normalised by the relation's time span) and ξ the relaxed
object frequency of relation r before t. Default mode uses λ = 0.1, α = 0.99 for every relation; train mode selects
λ_r and then α_r per relation on that relation's validation queries over the TGB grids. The re-implementation
(`hd_recb.py`, vectorised CSR form) runs on serverA CPU (16 workers; polecat valid+test in about 21 s,
smallpedia in a similar time).

Verification against the official TGB script (pilot-1 run, 8 processes, polecat): 533,450 of 533,472 per-query
validation reciprocal ranks match the official per-query CSV (22 differ, 0.004%); validation MRR 0.1698590 vs
official 0.16985896; test MRR 0.16709 vs official 0.16709. The official script was not re-run in train mode
(projected > 8 h on polecat); instead the re-implemented train mode is checked against the TGB 2.0 leaderboard
rows (below), which it reproduces to the third decimal.

Official-code quirk (documented, not corrected in the "faithful" rows): `predictions_psi` is not reset between
queries inside a worker's chunk, so a query without strict-recurrency candidates inherits the previous query's ψ
row. "Faithful" rows emulate this (with the official 8-process chunking); "clean" rows reset ψ per query. All
blends and pools in this study use the clean variant; the difference is ≤ 0.0008 MRR on either dataset.

| dataset | split | RecB default faithful | RecB default clean | RecB trained faithful | RecB trained clean | leaderboard (default / trained) |
|---|---|---|---|---|---|---|
| tkgl-polecat | valid | 0.16986 | 0.17009 | 0.20265 | 0.20312 | - |
| tkgl-polecat | test | 0.16709 | 0.16769 | 0.19843 | 0.19923 | 0.167 / 0.198 |
| tkgl-smallpedia | valid | 0.54179 | 0.54165 | 0.63928 | 0.63907 | - |
| tkgl-smallpedia | test | 0.48625 | 0.48610 | 0.60489 | 0.60464 | 0.486 / 0.605 |

Source: `recb_{valid,test}_meta_tkgl-*.json` (job, serverA). Trained test H@10: polecat 0.3197,
smallpedia 0.7162 (clean). Selected decay rates: polecat mostly λ_r = 0.005 (21 of 32 directed relations) with
α_r ≥ 0.999; smallpedia mostly λ_r ∈ {0, 1.0001} (i.e. no decay or keep only the last step), with α_r = 0 for 86
directed relations (pure relaxed frequency ξ).

## 3. Bases: verification, selected epochs, and strength relative to published rows

| dataset | base | architecture config | selected epoch | selection valid MRR | full-valid MRR | test MRR (official) | pilot test MRR | TGB 2.0 row (same architecture) | host |
|---|---|---|---|---|---|---|---|---|---|
| tkgl-polecat | RE-GCN (pilot ckpt) | RE-GCN, h 200, 2 layers, history 10, lr 1e-3, 12 epochs (pilot-1) | 12 (pilot-1 fixed budget) | 0.2022 (200k valid sample) | 0.2016 | 0.1650 | 0.1650 | RE-GCN 0.175 (valid 0.191) | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| tkgl-polecat | LogCL (pilot ckpt) | LogCL, h 200, 2 layers, history 10, lr 1e-3, 12 epochs (pilot-1) | 12 (pilot-1 fixed budget) | 0.2134 (200k valid sample) | 0.2133 | 0.1785 | 0.1785 | - | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| tkgl-polecat | TiRGN | TiRGN, authors' code, history 10, history rate 0.3, lr 1e-3, ≤ 12 epochs, patience 3 | 4 (0-indexed, i.e. 5th epoch; valid-best, patience stop after epoch 7) | 0.2402 (200k valid sample) | 0.2404 | 0.2205 | - | - | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| tkgl-smallpedia | RE-GCN (pilot ckpt) | RE-GCN, history 10, lr 1e-3, 3-epoch budget (pilot-2) | 3 (pilot-2 fixed budget) | 0.4900 (full valid) | 0.4900 | 0.4218 | 0.4218 | RE-GCN 0.594 (valid 0.631) | CPU serverA (controls); base/re-rankers pilot-2 |
| tkgl-smallpedia | LogCL (pilot ckpt) | LogCL, history 10, lr 1e-3, 3-epoch budget (pilot-2) | 3 (pilot-2 fixed budget) | 0.5091 (full valid) | 0.5091 | 0.4664 | 0.4664 | - | CPU serverA (controls); base/re-rankers pilot-2 |
| tkgl-smallpedia | TiRGN | TiRGN, authors' code, history 10, history rate 0.3 (pilot-2) | 20 (pilot-2 valid-best) | 0.6541 (full valid) | 0.6541 | 0.6181 | 0.6181 | - | CPU serverA (controls); base/re-rankers pilot-2 |
| tkgl-smallpedia | RE-GCN (re-tuned) | RE-GCN re-tuned: history 3, lr 1e-3, shuffled snapshots | 5 (grid L3/lr1e-3/shuffled, valid-best) | 0.5457 (full valid) | 0.5457 | 0.4825 | - | RE-GCN 0.594 (valid 0.631) | GPU serverB (RTX 4090) |
| tkgl-smallpedia | LogCL (re-tuned) | LogCL re-tuned: history 3, lr 3e-4, time-ordered | 12 (grid L3/lr3e-4/time-ordered, valid-best = last epoch of the 12-epoch cap) | 0.5332 (full valid) | 0.5332 | 0.4663 | - | - | GPU serverB (RTX 4090) |

Pilot rows reproduce exactly: the official-evaluator base rows above equal the pilot reports' base rows (polecat RE-GCN 0.1650 / LogCL 0.1785; smallpedia RE-GCN 0.4218 / LogCL 0.4664 / TiRGN 0.6181), and the 3-seed HRCOV means on the pilot bases equal the pilot harness means (polecat 0.2718 / 0.2721; smallpedia 0.6595 / 0.6590 / 0.6601). Polecat selection-set MRRs are on the pilots' 200k valid sample (rng 12345, used for every polecat base and control); full-valid MRRs are from the base training logs.

Smallpedia RE-GCN re-tuning grid (full-valid filtered MRR, valid-best epoch per configuration; host serverB_(GPU_RTX_4090)):

| history L | lr | snapshot order | best epoch | best valid MRR | train s |
|---|---|---|---|---|---|
| 10 | 0.001 | time-ordered | 6 | 0.4975 | 113 |
| 3 | 0.001 | time-ordered | 5 | 0.5325 | 66 |
| 10 | 0.0003 | time-ordered | 10 | 0.4812 | 147 |
| 3 | 0.0003 | time-ordered | 12 | 0.5299 | 88 |
| 10 | 0.001 | shuffled | 5 | 0.4996 | 99 |
| 3 | 0.001 | shuffled | 5 | 0.5457 (selected) | 59 |
| 10 | 0.0003 | shuffled | 10 | 0.4900 | 149 |
| 3 | 0.0003 | shuffled | 9 | 0.5177 | 88 |

Smallpedia LogCL re-tuning grid (time-ordered only) (full-valid filtered MRR, valid-best epoch per configuration; host serverB_(GPU_RTX_4090)):

| history L | lr | snapshot order | best epoch | best valid MRR | train s |
|---|---|---|---|---|---|
| 10 | 0.001 | time-ordered | 5 | 0.5063 | 346 |
| 3 | 0.001 | time-ordered | 4 | 0.5310 | 260 |
| 10 | 0.0003 | time-ordered | 9 | 0.4906 | 517 |
| 3 | 0.0003 | time-ordered | 12 | 0.5332 (selected) | 446 |

TiRGN on tkgl-polecat (history L = 10, authors' hyper-parameters, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition on serverA GPU 0; full-valid filtered MRR per epoch, epochs indexed from 0):

| epoch | train loss | valid MRR | train s |
|---|---|---|---|
| 0 | 5.4310 | 0.1940 | 639 |
| 1 | 4.9784 | 0.2223 | 761 |
| 2 | 4.6059 | 0.2356 | 752 |
| 3 | 4.2389 | 0.2333 | 915 |
| 4 (selected) | 3.8988 | 0.2404 | 948 |
| 5 | 3.6492 | 0.2310 | 1099 |
| 6 | 3.4651 | 0.2275 | 1364 |
| 7 | 3.3101 | 0.2235 | 1432 |

Stop reason: patience; selected epoch 4 (valid MRR 0.2404, H@10 0.3985).

Polecat RE-GCN / LogCL valid-best-epoch diagnostic: not obtained. The 12-epoch per-epoch-valid re-run of the pilot-1 configuration ran out of GPU memory on the 24 GB RTX 4090 at history 10 (job), and the queued serverA GPU 0 run (job) aborted by design because the card was occupied by other users when the job started. The pilot-1 training logs show the training loss still falling at epoch 12 (RE-GCN 4.08 → 3.98, LogCL 4.62 → 4.53 over epochs 10-12); whether valid MRR peaked earlier is unknown.

## 4. Recurrency controls: valid-selected configurations

| dataset | base | BASE(+)RecB selected (variant, form, weight) | valid MRR | RecB-POOL selected (K; variant, form, weight) | valid MRR | valid MRR base | max batch deviation official vs per-query formula |
|---|---|---|---|---|---|---|---|
| tkgl-polecat | RE-GCN (pilot ckpt) | def, LOGLIN, (0.5, 1e-05) | 0.2367 | 100; def, LOGLIN, (0.5, 1e-05) | 0.2367 | 0.2022 | 0.0 |
| tkgl-polecat | LogCL (pilot ckpt) | def, LOGLIN, (0.5, 1e-05) | 0.2387 | 100; def, LOGLIN, (0.5, 1e-05) | 0.2386 | 0.2134 | 0.0 |
| tkgl-polecat | TiRGN | def, LOGLIN, (0.25, 0.001) | 0.2525 | 100; def, LOGLIN, (0.25, 0.001) | 0.2525 | 0.2402 | 0.0 |
| tkgl-smallpedia | RE-GCN (pilot ckpt) | tr, MIX, 0.9 | 0.6462 | 100; tr, MIX, 0.9 | 0.6459 | 0.4900 | 0.0 |
| tkgl-smallpedia | LogCL (pilot ckpt) | tr, MIX, 0.95 | 0.6455 | 100; tr, MIX, 0.98 | 0.6450 | 0.5091 | 0.0 |
| tkgl-smallpedia | TiRGN | tr, LOGLIN, (0.25, 1e-05) | 0.6615 | 100; tr, LOGLIN, (0.25, 1e-05) | 0.6611 | 0.6541 | 0.0 |
| tkgl-smallpedia | RE-GCN (re-tuned) | tr, MIX, 0.9 | 0.6459 | 100; tr, MIX, 0.9 | 0.6458 | 0.5457 | 0.0 |
| tkgl-smallpedia | LogCL (re-tuned) | tr, MIX, 0.98 | 0.6471 | 100; tr, MIX, 0.98 | 0.6465 | 0.5332 | 0.0 |

Variant 'def' = RecB default (λ = 0.1, α = 0.99), 'tr' = RecB trained (per-relation λ_r, α_r); MIX weight = β; LOGLIN weight = (w, ε).

## 5. Full-test results and the pre-registered decision

All rows: every test query of the dataset, both directions, time-aware filtered, official py-tgb 2.3.0 Evaluator with the shipped negatives (per-query reciprocal rank; re-rankers averaged over seeds 0-2, `sd` = across-seed SD of the seed MRRs). `harness MRR` = the harness exact filtered evaluator on the same score rows. Published rows are copied from the cited sources, not re-run.


### tkgl-polecat (n = 532,636 test queries)

| base | arm | evaluator | seeds | MRR | sd | H@1 | H@3 | H@10 | harness MRR | host |
|---|---|---|---|---|---|---|---|---|---|---|
| - | RecB default (TGB script, faithful) | official formula (exact, hd_recb.py) | - | 0.1671 | - | 0.1162 | 0.1843 | 0.2638 | - | serverA (CPU) |
| - | RecB trained (TGB train mode, faithful) | official formula (exact, hd_recb.py) | - | 0.1984 | - | 0.1377 | 0.2219 | 0.3169 | - | serverA (CPU) |
| - | RecB default (clean psi reset) | official formula (exact, hd_recb.py) | - | 0.1677 | - | 0.1165 | 0.1850 | 0.2655 | - | serverA (CPU) |
| - | RecB trained (clean psi reset) | official formula (exact, hd_recb.py) | - | 0.1992 | - | 0.1378 | 0.2226 | 0.3197 | - | serverA (CPU) |
| RE-GCN (pilot ckpt) | base alone | official py-tgb Evaluator | - | 0.1650 | - | 0.1042 | 0.1858 | 0.2855 | 0.1650 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | base (+) RecB blend | official py-tgb Evaluator | - | 0.2087 | - | 0.1401 | 0.2353 | 0.3445 | 0.2087 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | RecB-POOL | official py-tgb Evaluator | - | 0.2088 | - | 0.1401 | 0.2353 | 0.3445 | 0.2088 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | HR | official py-tgb Evaluator | 3 | 0.2343 | 0.0001 | 0.1682 | 0.2694 | 0.3631 | 0.2343 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.2718 | 0.0005 | 0.1884 | 0.3102 | 0.4390 | 0.2718 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | HR_RB | harness exact filtered (secondary arm) | 3 | 0.2337 | 0.0007 | 0.1675 | 0.2688 | 0.3628 | - | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| RE-GCN (pilot ckpt) | HRCOV_RB | harness exact filtered (secondary arm) | 3 | 0.2723 | 0.0005 | 0.1889 | 0.3106 | 0.4398 | - | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | base alone | official py-tgb Evaluator | - | 0.1785 | - | 0.1161 | 0.2016 | 0.3023 | 0.1785 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | base (+) RecB blend | official py-tgb Evaluator | - | 0.2130 | - | 0.1440 | 0.2403 | 0.3499 | 0.2130 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | RecB-POOL | official py-tgb Evaluator | - | 0.2131 | - | 0.1440 | 0.2403 | 0.3499 | 0.2131 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | HR | official py-tgb Evaluator | 3 | 0.2361 | 0.0001 | 0.1689 | 0.2709 | 0.3683 | 0.2361 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.2721 | 0.0003 | 0.1888 | 0.3104 | 0.4391 | 0.2721 | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | HR_RB | harness exact filtered (secondary arm) | 3 | 0.2359 | 0.0005 | 0.1686 | 0.2707 | 0.3682 | - | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| LogCL (pilot ckpt) | HRCOV_RB | harness exact filtered (secondary arm) | 3 | 0.2722 | 0.0004 | 0.1891 | 0.3104 | 0.4391 | - | GPU serverB (RTX 4090); re-rankers pilot-1 serverA |
| TiRGN | base alone | official py-tgb Evaluator | - | 0.2205 | - | 0.1429 | 0.2498 | 0.3750 | 0.2205 | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| TiRGN | base (+) RecB blend | official py-tgb Evaluator | - | 0.2348 | - | 0.1576 | 0.2649 | 0.3879 | 0.2348 | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| TiRGN | RecB-POOL | official py-tgb Evaluator | - | 0.2349 | - | 0.1576 | 0.2649 | 0.3879 | 0.2349 | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| TiRGN | HR | official py-tgb Evaluator | 3 | 0.2694 | 0.0004 | 0.1895 | 0.3076 | 0.4269 | 0.2694 | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| TiRGN | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.2824 | 0.0002 | 0.1961 | 0.3215 | 0.4537 | 0.2824 | training serverA GPU 0 (RTX PRO 6000); lookup + controls + official test serverB (RTX 4090); features + re-rankers serverA CPU |
| - | published: CountTRuCoLa (arXiv 2509.09474) | published row (not re-run) | - | 0.2560 | - | - | - | - | - | cited source (see arm label) |
| - | published: NHCTE (KBS 2026) | published row (not re-run) | - | 0.2450 | - | - | - | - | - | cited source (see arm label) |
| - | published: TLogic (TGB 2.0) | published row (not re-run) | - | 0.2280 | - | - | - | - | - | cited source (see arm label) |
| - | published: RecB trained (TGB 2.0) | published row (not re-run) | - | 0.1980 | - | - | - | - | - | cited source (see arm label) |
| - | published: CEN (TGB 2.0) | published row (not re-run) | - | 0.1840 | - | - | - | - | - | cited source (see arm label) |
| - | published: RE-GCN (TGB 2.0) | published row (not re-run) | - | 0.1750 | - | - | - | - | - | cited source (see arm label) |

Architecture selection (valid MRR): regcn: RE-GCN (pilot ckpt) 0.2022; logcl: LogCL (pilot ckpt) 0.2134; tirgn: TiRGN 0.2402

Two strongest bases: TiRGN, LogCL (pilot ckpt). Strongest blend overall (C2 comparator): TiRGN (+) RecB, MRR 0.2348.

| base | contrast | delta MRR | 95% CI | one-sided p | p(delta <= 0.005) | criterion |
|---|---|---|---|---|---|---|
| TiRGN | HRCOV - base (+) RecB blend | +0.0476 | [+0.0470, +0.0481] | <0.0005 | <0.0005 | C1 (pass jointly) |
| TiRGN | HRCOV - RecB-POOL | +0.0475 | [+0.0469, +0.0481] | <0.0005 | <0.0005 | C1 (pass jointly) |
| TiRGN | HRCOV - tirgn(+)RecB | +0.0476 | [+0.0470, +0.0481] | <0.0005 | <0.0005 | C2 (pass) |
| TiRGN | head stratum HRCOV - RecB-POOL | +0.0476 | [+0.0467, +0.0485] | <0.0005 | - | C3 (no damage) |
| TiRGN | HR_minus_POOL (secondary) | +0.0346 | [+0.0340, +0.0351] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_minus_RecB_trained_clean (secondary) | +0.0831 | [+0.0824, +0.0840] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_minus_HR (pool-widening gain) (secondary) | +0.0129 | [+0.0126, +0.0133] | <0.0005 | <0.0005 | - |
| TiRGN | BLEND_minus_base (secondary) | +0.0143 | [+0.0140, +0.0147] | <0.0005 | <0.0005 | - |
| LogCL (pilot ckpt) | HRCOV - base (+) RecB blend | +0.0590 | [+0.0584, +0.0597] | <0.0005 | <0.0005 | C1 (pass jointly) |
| LogCL (pilot ckpt) | HRCOV - RecB-POOL | +0.0590 | [+0.0583, +0.0596] | <0.0005 | <0.0005 | C1 (pass jointly) |
| LogCL (pilot ckpt) | HRCOV - tirgn(+)RecB | +0.0373 | [+0.0366, +0.0379] | <0.0005 | <0.0005 | C2 (pass) |
| LogCL (pilot ckpt) | head stratum HRCOV - RecB-POOL | +0.0490 | [+0.0480, +0.0499] | <0.0005 | - | C3 (no damage) |
| LogCL (pilot ckpt) | HR_minus_POOL (secondary) | +0.0230 | [+0.0225, +0.0235] | <0.0005 | <0.0005 | - |
| LogCL (pilot ckpt) | HRCOV_minus_RecB_trained_clean (secondary) | +0.0728 | [+0.0720, +0.0736] | <0.0005 | <0.0005 | - |
| LogCL (pilot ckpt) | HRCOV_minus_HR (pool-widening gain) (secondary) | +0.0360 | [+0.0354, +0.0365] | <0.0005 | <0.0005 | - |
| LogCL (pilot ckpt) | BLEND_minus_base (secondary) | +0.0346 | [+0.0341, +0.0350] | <0.0005 | <0.0005 | - |

**Outcome (tkgl-polecat): PRACTICAL_ADVANCE**


Strata, TiRGN: HRCOV minus RecB-POOL (B = 1000)

| stratum | share | MRR base | MRR RecB-tr | MRR blend | MRR pool | MRR HR | MRR HRCOV | delta | 95% CI | contribution to full delta |
|---|---|---|---|---|---|---|---|---|---|---|
| head (gold train freq >= 500) | 0.455 | 0.3312 | 0.3040 | 0.3535 | 0.3538 | 0.3948 | 0.4014 | +0.0476 | [+0.0467, +0.0485] | +0.0217 |
| mid (50-499) | 0.212 | 0.1730 | 0.1450 | 0.1823 | 0.1821 | 0.2213 | 0.2374 | +0.0552 | [+0.0541, +0.0564] | +0.0117 |
| cold (< 50) | 0.333 | 0.0994 | 0.0907 | 0.1060 | 0.1060 | 0.1287 | 0.1484 | +0.0424 | [+0.0416, +0.0433] | +0.0141 |
| no-pair-history | 0.386 | 0.0225 | 0.0130 | 0.0234 | 0.0236 | 0.0212 | 0.0263 | +0.0027 | [+0.0024, +0.0031] | +0.0011 |
| pair-history | 0.614 | 0.3449 | 0.3163 | 0.3676 | 0.3676 | 0.4254 | 0.4433 | +0.0757 | [+0.0748, +0.0765] | +0.0465 |
| first-seen (s or o) | 0.108 | 0.0093 | 0.0157 | 0.0117 | 0.0119 | 0.0156 | 0.0155 | +0.0036 | [+0.0030, +0.0041] | +0.0004 |
| exact recurrence (s,r,o) before t | 0.425 | 0.4437 | 0.4500 | 0.4789 | 0.4789 | 0.5121 | 0.5197 | +0.0408 | [+0.0398, +0.0418] | +0.0173 |
| pool-absent (gold outside base top-50) | 0.467 | 0.0032 | 0.0168 | 0.0037 | 0.0038 | 0.0032 | 0.0275 | +0.0237 | [+0.0233, +0.0241] | +0.0111 |
| gold in base top-50 | 0.533 | 0.4111 | 0.3593 | 0.4376 | 0.4376 | 0.5030 | 0.5060 | +0.0684 | [+0.0674, +0.0694] | +0.0364 |
| R_rec_out (recurrence, gold outside base top-50) | 0.047 | 0.0083 | 0.1116 | 0.0116 | 0.0122 | 0.0083 | 0.0552 | +0.0430 | [+0.0412, +0.0448] | +0.0020 |
| R_rec_in (recurrence, gold in base top-50) | 0.378 | 0.4982 | 0.4923 | 0.5373 | 0.5373 | 0.5751 | 0.5779 | +0.0405 | [+0.0394, +0.0417] | +0.0153 |

Oracle ceilings (gold-in-set fraction): base_top10 0.375; base_top20 0.447; base_top50 0.533; base_top100 0.591; RecB_trained_top20 0.374; RecB_trained_top50 0.443; RecB_trained_top100 0.493; base_top20_union_RecB_top20 0.498; base_top50_union_RecB_top50 0.581; base_top100_union_RecB_top100 0.638

Strata, LogCL (pilot ckpt): HRCOV minus RecB-POOL (B = 1000)

| stratum | share | MRR base | MRR RecB-tr | MRR blend | MRR pool | MRR HR | MRR HRCOV | delta | 95% CI | contribution to full delta |
|---|---|---|---|---|---|---|---|---|---|---|
| head (gold train freq >= 500) | 0.455 | 0.2966 | 0.3040 | 0.3501 | 0.3503 | 0.3789 | 0.3993 | +0.0490 | [+0.0480, +0.0499] | +0.0223 |
| mid (50-499) | 0.212 | 0.1373 | 0.1450 | 0.1582 | 0.1578 | 0.1932 | 0.2301 | +0.0722 | [+0.0709, +0.0736] | +0.0153 |
| cold (< 50) | 0.333 | 0.0432 | 0.0907 | 0.0606 | 0.0608 | 0.0682 | 0.1250 | +0.0642 | [+0.0632, +0.0653] | +0.0214 |
| no-pair-history | 0.386 | 0.0104 | 0.0130 | 0.0152 | 0.0153 | 0.0106 | 0.0203 | +0.0050 | [+0.0046, +0.0053] | +0.0019 |
| pair-history | 0.614 | 0.2841 | 0.3163 | 0.3373 | 0.3373 | 0.3778 | 0.4303 | +0.0929 | [+0.0919, +0.0940] | +0.0571 |
| first-seen (s or o) | 0.108 | 0.0002 | 0.0157 | 0.0033 | 0.0037 | 0.0007 | 0.0010 | -0.0027 | [-0.0030, -0.0024] | -0.0003 |
| exact recurrence (s,r,o) before t | 0.425 | 0.3630 | 0.4500 | 0.4390 | 0.4392 | 0.4645 | 0.5036 | +0.0645 | [+0.0633, +0.0656] | +0.0274 |
| pool-absent (gold outside base top-50) | 0.569 | 0.0025 | 0.0570 | 0.0146 | 0.0147 | 0.0025 | 0.0714 | +0.0567 | [+0.0560, +0.0574] | +0.0323 |
| gold in base top-50 | 0.431 | 0.4107 | 0.3870 | 0.4749 | 0.4749 | 0.5444 | 0.5369 | +0.0620 | [+0.0609, +0.0631] | +0.0267 |
| R_rec_out (recurrence, gold outside base top-50) | 0.097 | 0.0056 | 0.2802 | 0.0628 | 0.0634 | 0.0056 | 0.2080 | +0.1445 | [+0.1416, +0.1473] | +0.0140 |
| R_rec_in (recurrence, gold in base top-50) | 0.328 | 0.4688 | 0.5002 | 0.5504 | 0.5504 | 0.6004 | 0.5912 | +0.0408 | [+0.0395, +0.0421] | +0.0134 |

Oracle ceilings (gold-in-set fraction): base_top10 0.302; base_top20 0.359; base_top50 0.431; base_top100 0.484; RecB_trained_top20 0.374; RecB_trained_top50 0.443; RecB_trained_top100 0.493; base_top20_union_RecB_top20 0.461; base_top50_union_RecB_top50 0.539; base_top100_union_RecB_top100 0.594

### tkgl-smallpedia (n = 163,172 test queries)

| base | arm | evaluator | seeds | MRR | sd | H@1 | H@3 | H@10 | harness MRR | host |
|---|---|---|---|---|---|---|---|---|---|---|
| - | RecB default (TGB script, faithful) | official formula (exact, hd_recb.py) | - | 0.4863 | - | 0.4004 | 0.5358 | 0.6510 | - | serverA (CPU) |
| - | RecB trained (TGB train mode, faithful) | official formula (exact, hd_recb.py) | - | 0.6049 | - | 0.5295 | 0.6699 | 0.7160 | - | serverA (CPU) |
| - | RecB default (clean psi reset) | official formula (exact, hd_recb.py) | - | 0.4861 | - | 0.3998 | 0.5358 | 0.6513 | - | serverA (CPU) |
| - | RecB trained (clean psi reset) | official formula (exact, hd_recb.py) | - | 0.6046 | - | 0.5290 | 0.6697 | 0.7162 | - | serverA (CPU) |
| RE-GCN (pilot ckpt) | base alone | official py-tgb Evaluator | - | 0.4218 | - | 0.3625 | 0.4455 | 0.5345 | 0.4218 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | base (+) RecB blend | official py-tgb Evaluator | - | 0.6096 | - | 0.5321 | 0.6734 | 0.7285 | 0.6096 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | RecB-POOL | official py-tgb Evaluator | - | 0.6095 | - | 0.5321 | 0.6734 | 0.7280 | 0.6095 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | HR | official py-tgb Evaluator | 3 | 0.5693 | 0.0003 | 0.5349 | 0.5935 | 0.6180 | 0.5693 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.6595 | 0.0004 | 0.6081 | 0.7008 | 0.7367 | 0.6595 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | HRCOV_ST | official py-tgb Evaluator | 3 | 0.6672 | 0.0003 | 0.6135 | 0.7096 | 0.7493 | 0.6672 | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (pilot ckpt) | HRCOV_ST_NODIRECT | official py-tgb Evaluator | 3 | 0.6625 | 0.0032 | 0.6081 | 0.7054 | 0.7460 | 0.6625 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | base alone | official py-tgb Evaluator | - | 0.4664 | - | 0.4059 | 0.4955 | 0.5786 | 0.4664 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | base (+) RecB blend | official py-tgb Evaluator | - | 0.6091 | - | 0.5316 | 0.6735 | 0.7268 | 0.6091 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | RecB-POOL | official py-tgb Evaluator | - | 0.6093 | - | 0.5321 | 0.6736 | 0.7258 | 0.6093 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | HR | official py-tgb Evaluator | 3 | 0.5884 | 0.0003 | 0.5498 | 0.6171 | 0.6435 | 0.5884 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.6590 | 0.0003 | 0.6072 | 0.7005 | 0.7371 | 0.6590 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | HRCOV_ST | official py-tgb Evaluator | 3 | 0.6668 | 0.0003 | 0.6130 | 0.7090 | 0.7491 | 0.6668 | CPU serverA (controls); base/re-rankers pilot-2 |
| LogCL (pilot ckpt) | HRCOV_ST_NODIRECT | official py-tgb Evaluator | 3 | 0.6625 | 0.0027 | 0.6079 | 0.7055 | 0.7461 | 0.6625 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | base alone | official py-tgb Evaluator | - | 0.6181 | - | 0.5523 | 0.6691 | 0.7239 | 0.6181 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | base (+) RecB blend | official py-tgb Evaluator | - | 0.6275 | - | 0.5613 | 0.6819 | 0.7276 | 0.6275 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | RecB-POOL | official py-tgb Evaluator | - | 0.6274 | - | 0.5613 | 0.6819 | 0.7277 | 0.6274 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HR | official py-tgb Evaluator | 3 | 0.6572 | 0.0001 | 0.6067 | 0.6989 | 0.7301 | 0.6572 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.6601 | 0.0002 | 0.6089 | 0.7019 | 0.7353 | 0.6601 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HRCOV_ST | official py-tgb Evaluator | 3 | 0.6682 | 0.0007 | 0.6146 | 0.7111 | 0.7480 | 0.6682 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HRCOV_ST_NODIRECT | official py-tgb Evaluator | 3 | 0.6653 | 0.0009 | 0.6120 | 0.7079 | 0.7448 | 0.6653 | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HR_RB | harness exact filtered (secondary arm) | 3 | 0.6303 | 0.0029 | 0.5697 | 0.6798 | 0.7212 | - | CPU serverA (controls); base/re-rankers pilot-2 |
| TiRGN | HRCOV_RB | harness exact filtered (secondary arm) | 3 | 0.6482 | 0.0052 | 0.5943 | 0.6916 | 0.7282 | - | CPU serverA (controls); base/re-rankers pilot-2 |
| RE-GCN (re-tuned) | base alone | official py-tgb Evaluator | - | 0.4825 | - | 0.4205 | 0.5150 | 0.5971 | 0.4825 | GPU serverB (RTX 4090) |
| RE-GCN (re-tuned) | base (+) RecB blend | official py-tgb Evaluator | - | 0.6108 | - | 0.5334 | 0.6761 | 0.7284 | 0.6108 | GPU serverB (RTX 4090) |
| RE-GCN (re-tuned) | RecB-POOL | official py-tgb Evaluator | - | 0.6107 | - | 0.5334 | 0.6761 | 0.7281 | 0.6107 | GPU serverB (RTX 4090) |
| RE-GCN (re-tuned) | HR | official py-tgb Evaluator | 3 | 0.6014 | 0.0001 | 0.5602 | 0.6333 | 0.6602 | 0.6014 | GPU serverB (RTX 4090) |
| RE-GCN (re-tuned) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.6587 | 0.0003 | 0.6076 | 0.7001 | 0.7351 | 0.6587 | GPU serverB (RTX 4090) |
| RE-GCN (re-tuned) | HRCOV_ST | official py-tgb Evaluator | 3 | 0.6652 | 0.0037 | 0.6109 | 0.7088 | 0.7466 | 0.6652 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | base alone | official py-tgb Evaluator | - | 0.4663 | - | 0.4020 | 0.4981 | 0.5867 | 0.4663 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | base (+) RecB blend | official py-tgb Evaluator | - | 0.6101 | - | 0.5335 | 0.6744 | 0.7265 | 0.6101 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | RecB-POOL | official py-tgb Evaluator | - | 0.6101 | - | 0.5335 | 0.6744 | 0.7263 | 0.6101 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | HR | official py-tgb Evaluator | 3 | 0.5964 | 0.0004 | 0.5546 | 0.6284 | 0.6570 | 0.5964 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | HRCOV (claim) | official py-tgb Evaluator | 3 | 0.6585 | 0.0009 | 0.6070 | 0.7003 | 0.7355 | 0.6585 | GPU serverB (RTX 4090) |
| LogCL (re-tuned) | HRCOV_ST | official py-tgb Evaluator | 3 | 0.6664 | 0.0009 | 0.6130 | 0.7093 | 0.7476 | 0.6664 | GPU serverB (RTX 4090) |
| - | published: CountTRuCoLa (arXiv 2509.09474) | published row (not re-run) | - | 0.6440 | - | - | - | - | - | cited source (see arm label) |
| - | published: NHCTE (KBS 2026) | published row (not re-run) | - | 0.6180 | - | - | - | - | - | cited source (see arm label) |
| - | published: CEN (TGB 2.0) | published row (not re-run) | - | 0.6120 | - | - | - | - | - | cited source (see arm label) |
| - | published: RecB trained (TGB 2.0 leaderboard) | published row (not re-run) | - | 0.6050 | - | - | - | - | - | cited source (see arm label) |
| - | published: TLogic (TGB 2.0) | published row (not re-run) | - | 0.5950 | - | - | - | - | - | cited source (see arm label) |
| - | published: RE-GCN (TGB 2.0) | published row (not re-run) | - | 0.5940 | - | - | - | - | - | cited source (see arm label) |

Architecture selection (valid MRR): regcn: RE-GCN (re-tuned) 0.5457; logcl: LogCL (re-tuned) 0.5332; tirgn: TiRGN 0.6541

Two strongest bases: TiRGN, RE-GCN (re-tuned). Strongest blend overall (C2 comparator): TiRGN (+) RecB, MRR 0.6275.

| base | contrast | delta MRR | 95% CI | one-sided p | p(delta <= 0.005) | criterion |
|---|---|---|---|---|---|---|
| TiRGN | HRCOV - base (+) RecB blend | +0.0327 | [+0.0318, +0.0336] | <0.0005 | <0.0005 | C1 (pass jointly) |
| TiRGN | HRCOV - RecB-POOL | +0.0327 | [+0.0318, +0.0336] | <0.0005 | <0.0005 | C1 (pass jointly) |
| TiRGN | HRCOV - tirgn(+)RecB | +0.0327 | [+0.0318, +0.0336] | <0.0005 | <0.0005 | C2 (pass) |
| TiRGN | head stratum HRCOV - base (+) RecB blend | +0.0194 | [+0.0160, +0.0229] | <0.0005 | - | C3 (no damage) |
| TiRGN | HR_minus_BLEND (secondary) | +0.0298 | [+0.0289, +0.0307] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_ST_minus_BLEND (secondary) | +0.0407 | [+0.0397, +0.0417] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_ST_ND_minus_BLEND (secondary) | +0.0378 | [+0.0368, +0.0387] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_minus_RecB_trained_clean (secondary) | +0.0555 | [+0.0543, +0.0566] | <0.0005 | <0.0005 | - |
| TiRGN | HRCOV_minus_HR (pool-widening gain) (secondary) | +0.0029 | [+0.0025, +0.0033] | <0.0005 | 1.0000 | - |
| TiRGN | BLEND_minus_base (secondary) | +0.0094 | [+0.0089, +0.0098] | <0.0005 | <0.0005 | - |
| RE-GCN (re-tuned) | HRCOV - base (+) RecB blend | +0.0480 | [+0.0468, +0.0492] | <0.0005 | <0.0005 | C1 (pass jointly) |
| RE-GCN (re-tuned) | HRCOV - RecB-POOL | +0.0480 | [+0.0468, +0.0492] | <0.0005 | <0.0005 | C1 (pass jointly) |
| RE-GCN (re-tuned) | HRCOV - tirgn(+)RecB | +0.0313 | [+0.0303, +0.0323] | <0.0005 | <0.0005 | C2 (pass) |
| RE-GCN (re-tuned) | head stratum HRCOV - base (+) RecB blend | +0.0105 | [+0.0072, +0.0143] | <0.0005 | - | C3 (no damage) |
| RE-GCN (re-tuned) | HR_minus_BLEND (secondary) | -0.0094 | [-0.0106, -0.0081] | 1.0000 | 1.0000 | - |
| RE-GCN (re-tuned) | HRCOV_ST_minus_BLEND (secondary) | +0.0544 | [+0.0532, +0.0556] | <0.0005 | <0.0005 | - |
| RE-GCN (re-tuned) | HRCOV_minus_RecB_trained_clean (secondary) | +0.0541 | [+0.0529, +0.0554] | <0.0005 | <0.0005 | - |
| RE-GCN (re-tuned) | HRCOV_minus_HR (pool-widening gain) (secondary) | +0.0573 | [+0.0560, +0.0586] | <0.0005 | <0.0005 | - |
| RE-GCN (re-tuned) | BLEND_minus_base (secondary) | +0.1283 | [+0.1268, +0.1299] | <0.0005 | <0.0005 | - |

**Outcome (tkgl-smallpedia): PRACTICAL_ADVANCE**


Strata, TiRGN: HRCOV minus base (+) RecB blend (B = 1000)

| stratum | share | MRR base | MRR RecB-tr | MRR blend | MRR pool | MRR HR | MRR HRCOV | delta | 95% CI | contribution to full delta |
|---|---|---|---|---|---|---|---|---|---|---|
| head (gold train freq >= 500) | 0.025 | 0.9246 | 0.9424 | 0.9347 | 0.9348 | 0.9506 | 0.9542 | +0.0194 | [+0.0160, +0.0229] | +0.0005 |
| mid (50-499) | 0.186 | 0.7268 | 0.7274 | 0.7381 | 0.7381 | 0.7725 | 0.7806 | +0.0424 | [+0.0404, +0.0445] | +0.0079 |
| cold (< 50) | 0.789 | 0.5828 | 0.5651 | 0.5917 | 0.5917 | 0.6209 | 0.6225 | +0.0308 | [+0.0298, +0.0318] | +0.0243 |
| no-pair-history | 0.280 | 0.0316 | 0.0148 | 0.0317 | 0.0315 | 0.0336 | 0.0400 | +0.0083 | [+0.0075, +0.0090] | +0.0023 |
| pair-history | 0.720 | 0.8460 | 0.8339 | 0.8591 | 0.8591 | 0.8997 | 0.9012 | +0.0422 | [+0.0409, +0.0433] | +0.0304 |
| first-seen (s or o) | 0.083 | 0.0414 | 0.0287 | 0.0437 | 0.0436 | 0.0443 | 0.0442 | +0.0006 | [-0.0003, +0.0014] | +0.0000 |
| exact recurrence (s,r,o) before t | 0.716 | 0.8495 | 0.8388 | 0.8629 | 0.8629 | 0.9023 | 0.9033 | +0.0404 | [+0.0391, +0.0417] | +0.0289 |
| pool-absent (gold outside base top-50) | 0.234 | 0.0028 | 0.0061 | 0.0037 | 0.0035 | 0.0028 | 0.0097 | +0.0059 | [+0.0054, +0.0065] | +0.0014 |
| gold in base top-50 | 0.766 | 0.8065 | 0.7879 | 0.8185 | 0.8185 | 0.8577 | 0.8593 | +0.0408 | [+0.0397, +0.0420] | +0.0313 |
| R_rec_out (recurrence, gold outside base top-50) | 0.001 | 0.0091 | 0.3289 | 0.0916 | 0.0920 | 0.0091 | 0.2911 | +0.1995 | [+0.1493, +0.2487] | +0.0003 |
| R_rec_in (recurrence, gold in base top-50) | 0.714 | 0.8512 | 0.8398 | 0.8644 | 0.8644 | 0.9040 | 0.9045 | +0.0400 | [+0.0387, +0.0412] | +0.0286 |

Oracle ceilings (gold-in-set fraction): base_top10 0.724; base_top20 0.743; base_top50 0.766; base_top100 0.786; RecB_trained_top20 0.727; RecB_trained_top50 0.739; RecB_trained_top100 0.751; base_top20_union_RecB_top20 0.750; base_top50_union_RecB_top50 0.774; base_top100_union_RecB_top100 0.796

Strata, RE-GCN (re-tuned): HRCOV minus base (+) RecB blend (B = 1000)

| stratum | share | MRR base | MRR RecB-tr | MRR blend | MRR pool | MRR HR | MRR HRCOV | delta | 95% CI | contribution to full delta |
|---|---|---|---|---|---|---|---|---|---|---|
| head (gold train freq >= 500) | 0.025 | 0.9379 | 0.9424 | 0.9455 | 0.9455 | 0.9524 | 0.9560 | +0.0105 | [+0.0072, +0.0143] | +0.0003 |
| mid (50-499) | 0.186 | 0.6700 | 0.7274 | 0.7372 | 0.7371 | 0.7616 | 0.7814 | +0.0442 | [+0.0419, +0.0465] | +0.0082 |
| cold (< 50) | 0.789 | 0.4240 | 0.5651 | 0.5705 | 0.5705 | 0.5527 | 0.6205 | +0.0500 | [+0.0488, +0.0513] | +0.0395 |
| no-pair-history | 0.280 | 0.0324 | 0.0148 | 0.0281 | 0.0279 | 0.0366 | 0.0398 | +0.0117 | [+0.0108, +0.0127] | +0.0033 |
| pair-history | 0.720 | 0.6575 | 0.8339 | 0.8373 | 0.8373 | 0.8210 | 0.8994 | +0.0620 | [+0.0604, +0.0636] | +0.0447 |
| first-seen (s or o) | 0.083 | 0.0212 | 0.0287 | 0.0300 | 0.0298 | 0.0342 | 0.0344 | +0.0044 | [+0.0025, +0.0063] | +0.0004 |
| exact recurrence (s,r,o) before t | 0.716 | 0.6599 | 0.8388 | 0.8416 | 0.8416 | 0.8240 | 0.9015 | +0.0599 | [+0.0584, +0.0615] | +0.0429 |
| pool-absent (gold outside base top-50) | 0.304 | 0.0036 | 0.1422 | 0.1372 | 0.1370 | 0.0036 | 0.2148 | +0.0777 | [+0.0755, +0.0799] | +0.0236 |
| gold in base top-50 | 0.696 | 0.6917 | 0.8067 | 0.8177 | 0.8177 | 0.8626 | 0.8527 | +0.0350 | [+0.0336, +0.0363] | +0.0243 |
| R_rec_out (recurrence, gold outside base top-50) | 0.072 | 0.0073 | 0.5730 | 0.5499 | 0.5499 | 0.0073 | 0.8770 | +0.3270 | [+0.3185, +0.3363] | +0.0237 |
| R_rec_in (recurrence, gold in base top-50) | 0.643 | 0.7334 | 0.8687 | 0.8744 | 0.8744 | 0.9160 | 0.9043 | +0.0299 | [+0.0284, +0.0312] | +0.0192 |

Oracle ceilings (gold-in-set fraction): base_top10 0.597; base_top20 0.641; base_top50 0.696; base_top100 0.736; RecB_trained_top20 0.727; RecB_trained_top50 0.739; RecB_trained_top100 0.751; base_top20_union_RecB_top20 0.754; base_top50_union_RecB_top50 0.778; base_top100_union_RecB_top100 0.800

## 6. What the gain is made of

### 6.1 Pool-widening gain (HRCOV minus HR, same base, official evaluator, B = 2000) and margin over the better recurrency control

| dataset | base | valid MRR base | HR | HRCOV | HRCOV - HR | 95% CI | HRCOV - better control | 95% CI |
|---|---|---|---|---|---|---|---|---|
| tkgl-polecat | TiRGN | 0.2402 | 0.2694 | 0.2824 | +0.0129 | [+0.0126, +0.0133] | +0.0475 | [+0.0469, +0.0481] |
| tkgl-polecat | LogCL (pilot ckpt) | 0.2134 | 0.2361 | 0.2721 | +0.0360 | [+0.0354, +0.0365] | +0.0590 | [+0.0583, +0.0596] |
| tkgl-polecat | RE-GCN (pilot ckpt) | 0.2022 | 0.2343 | 0.2718 | +0.0375 | [+0.0369, +0.0380] | +0.0630 | [+0.0624, +0.0637] |
| tkgl-smallpedia | TiRGN | 0.6541 | 0.6572 | 0.6601 | +0.0029 | [+0.0025, +0.0033] | +0.0327 | [+0.0318, +0.0336] |
| tkgl-smallpedia | RE-GCN (re-tuned) | 0.5457 | 0.6014 | 0.6587 | +0.0573 | [+0.0560, +0.0586] | +0.0480 | [+0.0468, +0.0492] |
| tkgl-smallpedia | LogCL (re-tuned) | 0.5332 | 0.5964 | 0.6585 | +0.0621 | [+0.0608, +0.0634] | +0.0483 | [+0.0472, +0.0494] |
| tkgl-smallpedia | LogCL (pilot ckpt) | 0.5091 | 0.5884 | 0.6590 | +0.0706 | [+0.0692, +0.0719] | +0.0497 | [+0.0486, +0.0509] |
| tkgl-smallpedia | RE-GCN (pilot ckpt) | 0.4900 | 0.5693 | 0.6595 | +0.0902 | [+0.0887, +0.0918] | +0.0499 | [+0.0488, +0.0511] |

### 6.2 RecB score as an extra re-ranker feature (secondary arms HR_RB / HRCOV_RB; harness evaluator, B = 2000)

| dataset | base | HR | HR_RB | HR_RB - HR | 95% CI | HRCOV | HRCOV_RB | HRCOV_RB - HRCOV | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| tkgl-polecat | RE-GCN (pilot ckpt) | 0.2343 | 0.2337 | -0.0006 | [-0.0008, -0.0005] | 0.2718 | 0.2723 | +0.0006 | [+0.0004, +0.0007] |
| tkgl-polecat | LogCL (pilot ckpt) | 0.2361 | 0.2359 | -0.0002 | [-0.0003, -0.0001] | 0.2721 | 0.2722 | +0.0002 | [+0.0000, +0.0003] |
| tkgl-smallpedia | TiRGN | 0.6572 | 0.6303 | -0.0269 | [-0.0276, -0.0263] | 0.6601 | 0.6482 | -0.0119 | [-0.0125, -0.0114] |

### 6.3 Oracle ceilings (fraction of test queries whose gold is inside the candidate set = MRR of a perfect re-ranker over that set)

| dataset | base | base top-50 | RecB-trained top-50 | base top-50 ∪ RecB top-50 | base top-100 ∪ RecB top-100 | HRCOV pool C_cov (pilot report; TiRGN polecat: this study's feature summary) | HRCOV MRR |
|---|---|---|---|---|---|---|---|
| tkgl-polecat | TiRGN | 0.533 | 0.443 | 0.581 | 0.638 | 0.630 | 0.2824 |
| tkgl-polecat | LogCL (pilot ckpt) | 0.431 | 0.443 | 0.539 | 0.594 | 0.598 | 0.2721 |
| tkgl-smallpedia | TiRGN | 0.766 | 0.739 | 0.774 | 0.796 | 0.770 | 0.6601 |
| tkgl-smallpedia | RE-GCN (re-tuned) | 0.696 | 0.739 | 0.778 | 0.800 | - | 0.6587 |

### 6.4 Where the margin over the recurrency controls lands

!Per-stratum delta

*Figure 2.* HRCOV minus the better recurrency control on the same base, per query stratum, with 95% bootstrap CIs (B = 1000), for the two strongest bases of each dataset (official evaluator, seeds averaged). Strata: recurrence = the exact (s, r, o) occurred before t; "gold outside base top-50" = pool-absent for the base; no (s, o) history = no event between s and o before t; first-seen = s or o without any earlier event; head / mid / cold = gold train frequency ≥ 500 / 50–499 / < 50. Shares are listed in the §5 strata tables. The smallpedia "recurrence, gold outside base top-50" stratum on TiRGN holds 0.1% of queries, hence its wide CI. The x-axes differ between panels.

1. **The margin is broad, not confined to recurrent or pool-absent queries.** On polecat TiRGN, HRCOV beats the better control in every stratum. Of the full-test Δ of +0.0475, queries whose gold is inside the base top-50 contribute +0.0364 (within-stratum Δ +0.0684) and pool-absent queries +0.0111. The head stratum alone contributes +0.0217. On smallpedia TiRGN, 0.0313 of the +0.0327 comes from queries whose gold is already in the base top-50. Pool-absent queries add only +0.0014 there, because TiRGN already places most recurrent golds in its top-50 (only 0.1% of queries are recurrent with the gold outside the top-50).
2. **Recurrency controls are strong, and our implementation of them is faithful.** RecB-trained alone reaches 0.1984 (polecat) and 0.6049 (smallpedia), matching the leaderboard. Blending RecB into a weak base recovers almost all of RecB: smallpedia blends land at 0.609–0.611 for every RE-GCN / LogCL base, against 0.6046 for clean RecB alone. Blending RecB into TiRGN adds +0.0143 (polecat) and +0.0094 (smallpedia). Pooling and blending are indistinguishable (≤ 0.0002 apart) because the selected pool (K = 100) is ranked by the same blended score.
3. **The history re-ranker over the base top-50 is the main source of the margin on strong bases.** HR − best control is +0.0346 (polecat TiRGN), +0.0298 (smallpedia TiRGN), +0.0230 (polecat LogCL). On weak smallpedia bases it is negative: −0.0094 on re-tuned RE-GCN, −0.0138 on re-tuned LogCL, −0.0403 on pilot RE-GCN. There the base's top-50 misses recurrent golds that RecB finds, and HRCOV's pool widening is what puts HRCOV ahead.
4. **Oracle ceilings.** A perfect re-ranker over base top-50 ∪ RecB top-50 would reach 0.581 (polecat TiRGN) and 0.774 (smallpedia TiRGN). HRCOV's own pool C_cov reaches 0.630 and 0.770. HRCOV realises 45% (0.2824 / 0.630) and 86% (0.6601 / 0.770) of its pool's ceiling. On polecat, the remaining headroom is placement within the pool, not coverage.

## 7. Negative and null results (kept intact)

1. **First-seen entities, polecat LogCL:** HRCOV is below the RecB-POOL control, −0.0027 [−0.0030, −0.0024] (10.8% of queries; HRCOV 0.001 vs control 0.004). On RE-GCN (pilot), outside the decision set, the same stratum is −0.0008 [−0.0012, −0.0005]. On TiRGN it is +0.0036. Recurrency's relaxed frequency term ξ still scores the object of a query with an unseen subject, while the re-ranker's history features are empty there.
2. **RecB score as a re-ranker feature does not help** (secondary arms, harness evaluator). On polecat, HR_RB − HR is −0.0006 / −0.0002 and HRCOV_RB − HRCOV is +0.0006 / +0.0002 (RE-GCN / LogCL), all within ±0.001. On smallpedia TiRGN the feature hurts: HR_RB − HR −0.0269 [−0.0276, −0.0263], HRCOV_RB − HRCOV −0.0119 [−0.0125, −0.0114]. A plausible cause, not tested here: λ_r / α_r are selected on the same validation split on which the re-ranker is fitted, so the feature is optimistically calibrated in training. The 11 evidence features already contain RecB's ingredients (exact-triple count / recency ≈ ψ, (r, o) frequency ≈ ξ).
3. **HR alone loses to the blend on weak smallpedia bases** (item 3 of §6.4). HR is not a practical advance by itself on those bases.
4. **Re-tuning does not move the system.** Re-tuned smallpedia RE-GCN gained +0.056 valid and +0.061 test MRR as a base (0.4218 → 0.4825). Re-tuned LogCL gained +0.024 valid but nothing on test (0.4664 → 0.4663). HRCOV on all five smallpedia bases is 0.6585–0.6601, and the blends are 0.609–0.628. Our RE-GCN stays far below the TGB 2.0 RE-GCN row (0.4825 vs 0.594). Base-relative gains on weak bases should therefore not be read as what a well-tuned RE-GCN would leave.
5. **Pool widening is small on strong bases** (§6.1): +0.0029 on smallpedia TiRGN and +0.0129 on polecat TiRGN, against +0.036 to +0.090 on weak bases. The pilot-1 attribution of the HRCOV gain to pool widening holds for weak bases only.
6. **Polecat RE-GCN / LogCL valid-best epochs were not established.** The diagnostic could not run (GPU memory on the 4090, and the serverA card was occupied). The polecat decision rests on TiRGN (valid-best epoch 4 of a patience-stopped run) and LogCL (fixed 12 epochs).
7. **HisRES was not compared** (no public code found).

## 8. Deviations from the plan

1. HisRES (arXiv:2405.10621) was not run: no public code was found (arXiv PDF has no code link; GitHub search for the name / full title returned nothing; paperswithcode is outside the network allowlist and was not requested). The base pool is RE-GCN, LogCL, TiRGN.
2. Official TGB RecB script was not re-run in train mode (projected > 8 h on polecat); the vectorised re-implementation was verified per query against the official default-mode CSV (22 of 533,472 valid queries differ) and reproduces the TGB 2.0 leaderboard train-mode rows to the third decimal (polecat 0.198, smallpedia 0.605).
3. The rule's reference row for smallpedia RecB (0.604, quoted from the pilot report) is 0.605 on the TGB 2.0 leaderboard; the report uses 0.605. Not decision-relevant.
4. Polecat RE-GCN and LogCL are the pilot-1 checkpoints with the fixed 12-epoch budget (no valid-best epoch selection). A per-epoch valid-curve diagnostic could not run: the 24 GB RTX 4090 runs out of memory at history 10, and the 96 GB card on serverA (GPU 0) was occupied by other users when the job checked it (it aborted by design). Polecat valid-best epoch selection is applied to TiRGN only.
5. TiRGN on polecat uses history L = 10 (authors' value) on serverA GPU 0 (96 GB), not the L in {6, 4, 3} fallback planned for the 24 GB card. The planned fallback did not produce a model: L = 6 passed a 40-step probe on serverB but ran out of memory in its first full epoch (job). The serverA L = 10 run (job) is therefore the only TiRGN polecat model, and it is the reported base. This removes, rather than adds, a deviation from the authors' configuration.
6. TiRGN is trained with entity prediction only (relation_prediction = False, no static graph), as in pilot 2. In that configuration TiRGN's own get_loss objective (0.7 * entity NLL + 0.3 * relation NLL + static loss) reduces to 0.7 * entity NLL, because the relation and static terms are identically zero. The chunked decoder loss used on polecat reproduces that objective exactly.
7. TiRGN polecat pipeline stages after training ran on mixed hosts because serverA GPU 0 became occupied by other users: exact list lookup on the serverB RTX 4090 (193 s; a CPU lookup on serverA was cancelled after 1.7 h), features and HR/HRCOV training on serverA CPU (16 threads), controls and the official test pass on the serverB RTX 4090.
8. TiRGN polecat: the pilot's lookup-vs-dump exact-rank agreement check (> 0.99, calibrated for same-GPU runs) failed across GPUs (dump on the RTX PRO 6000, lookup on the RTX 4090: 0.980 valid / 0.978 test). All disagreements are deep near-tie blocks (median gold rank about 121k of 150,931; gold log p equal to 1.4e-5; MRR equal to 1e-8; top-1 identical), so the threshold was lowered to 0.97 for this base only (diagnostic hardening/diag/rank_agree.py). Not tuned on any outcome.
9. LogCL re-tuning grid on smallpedia is time-ordered only (no shuffled-snapshot arm): LogCL's global copy-history is order-dependent.
10. LogCL re-tuned smallpedia selected the last epoch of the 12-epoch cap (valid still rising by 0.0002 at epoch 12).
11. Official evaluator applied batch-wise: per-query reciprocal ranks computed with the evaluator's formula on the official negatives; each batch's mean compared with Evaluator.eval (largest recorded deviation 0.0 on every pass; recorded, not asserted).
12. HR_RB / HRCOV_RB (RecB score features) evaluated with the harness evaluator only (declared secondary, not in the decision).
13. Polecat blend / pool selection used the pilots' 200k valid sample (rng 12345), as pre-registered; smallpedia used the full valid split.
14. HR / HRCOV on the pilot bases are the pilot re-ranker models (smallpedia promoted lists of seeds 1-2 regenerated from the saved seed models, rank agreement >= 0.99999); new bases (smallpedia re-tuned RE-GCN / LogCL, polecat TiRGN) were trained with the unchanged pilot recipe.
15. About 9 h of SSH outage (both hosts) delayed collection; jobs kept running and failed harvests were recovered from the job directories on disk.

## 9. Hosts and jobs

| work | host | job(s) |
|---|---|---|
| RecB re-implementation, both datasets, valid + test | serverA CPU (16 workers) | a1b75700 |
| smallpedia promoted lists seeds 0–2 (pilot bases) | serverA CPU | b154ba1c |
| polecat controls, RE-GCN / LogCL (valid grid, selection, official test) | serverB RTX 4090 | 32a69b57 |
| smallpedia controls, pilot bases (valid grid) | serverA CPU 3 × 5 threads | 9c89a983 |
| smallpedia controls, pilot bases (selection, official test) | serverA CPU | 0077f61d |
| smallpedia RE-GCN re-tuning grid + dump | serverB RTX 4090 | dbb94d67 |
| smallpedia re-tuned RE-GCN chain (lists, features, HR/HRCOV/HRCOV_ST, controls, test) | serverB RTX 4090 | 628b95b1 |
| smallpedia LogCL re-tuning grid + dump; TiRGN polecat L = 6 probe (passed 40 steps, OOM in full epoch) | serverB RTX 4090 | fc0463f8 |
| smallpedia re-tuned LogCL chain | serverB RTX 4090 | 697d8daa |
| HR_RB / HRCOV_RB (secondary) | serverA CPU 8 threads | 9a61750c |
| TiRGN polecat training (L = 10) + dump + lists | serverA GPU 0 (RTX PRO 6000) / CPU | 71b010cd |
| TiRGN polecat lookup | serverB RTX 4090 | df47deea (CPU attempt de3797ff cancelled) |
| TiRGN polecat features + HR/HRCOV × 3 seeds | serverA CPU 16 threads | b8a2389f (first attempt e182c81f stopped at the cross-GPU rank-agreement assert) |
| TiRGN polecat controls + official test | serverB RTX 4090 | f9d8e291 |
| polecat RE-GCN / LogCL valid-curve diagnostic (not obtained) | serverB (OOM) / serverA GPU 0 (busy at start, aborted) | 221281d8, 01f95c0d |
| aggregation, bootstraps, figures, report | local | `hd_agg.py`, `hd_figs.py`, `hd_report.py` |

GPU policy on serverA: GPU 0 only. Every serverA GPU job checked that GPU 0 was idle at dispatch, and GPU 1 was never used.

## 10. Reproduction

`python3 hd_agg.py --cfg agg_cfg.json --rule hardening_decision_rule.json --out hardening_verdict_raw.json` (local, about 10 min). Then `hd_report.py`, and `exec(open('hd_figs.py').read())` with `VERDICT_PATH` / `OUT_PREFIX` set. The inputs are the harvested job outputs listed in `agg_cfg.json` (official-evaluator per-query RR / rank arrays per base, blend selection JSONs, RecB rank arrays, strata). Stage commands and run times are in the pipeline README, section 10.
