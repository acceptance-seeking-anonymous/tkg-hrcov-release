# Track 6.2: residual / Δ-log-likelihood objective against a frozen TKG base — methods and results

**Verdict: FAIL.** The residual objective beats A, C and D on every dataset x base, but ties (3/6) or loses (2/6) to the valid-calibrated standalone ranker B; all conditions hold only on ICEWS18/TiRGN, so the >=2-bases-on-both-datasets rule is not met.

All numbers below were produced on **serverA** (CPU only, docker `tkg-gpu:latest`, torch 2.8.0). No GPU was used, and serverB was not used. Full records: `residual_verdict.json` (every arm × dataset × base × seed row, strata, bootstrap, guardrail, oracle) and `residual_aggregate.json`.

## 1. Question and pre-registered decision rule

**Question.** Does training a module *through* the frozen base's logits beat the alternatives at identical module, evidence and capacity? The residual objective is

s(c) = log p_base(c) + g·f(q,c),

trained with listwise cross-entropy of softmax over the candidate set C(q). Maximising this equals maximising log p(gold | base ⊕ evidence) − log p(gold | base), restricted to C. The alternatives are:

- **A**: the same module trained base-agnostically, then λ-mixed (BridgeMem style).
- **B**: post-hoc valid-calibrated mixing of the standalone module (SNAM/FreshCast style).
- **C**: the G10/G11 utility-as-feature ranker (arm4) used as a backfill.
- **D**: a BridgeMem-mechanism control.

**Rule.** R passes iff, on at least 2 frozen bases and on **both** datasets, all of the following hold:

1. R improves full-test MRR over the base (one-sided paired bootstrap p < 0.05).
2. R beats each of A, B, C and D (p < 0.05).
3. R passes the head-damage guardrail.

The R gate variant (none / reusability / confidence) is chosen per dataset × base by valid-select MRR. Every λ, gate, (α, β), epoch and variant choice was made on valid only.

## 2. Methods

**Harness and protocol.** The harness is `harness`: datasets, splits, dumps and filter come from `oih_common`, and base ranks from `fulltest_eval.BaseView`. Evaluation is on the full test set, both directions, time-aware filtered, RE-GCN split (ICEWS18 n = 99,090; GDELT n = 610,482). Metrics are MRR and H@1/3/10/50.

**Bases.** All six frozen-base MRRs reproduce the harness exactly: TiRGN 0.337006 / 0.216701, LogCL 0.287839 / 0.211389, RE-GCN 0.286905 / 0.197346 (ICEWS18 / GDELT). With f = 0, the fused rank equals the harness base rank on 100% of rows of every split.

**Candidate set.**

- **Base candidates.** C(q) contains the base top-50 for every query.
- **Bridge candidates (TiRGN trigger-pool queries only).** C(q) also takes ≤50 bridge candidates: the first entries of the shared G12-arm5 backfill top-200 list that lie outside the top-50. The bridge caches exist only for TiRGN.
- **Base log-prob for bridge candidates.** A bridge candidate uses its dump score if it is visible in the dump top-K; otherwise it uses a gold-agnostic tail value, log((1 − Σ top-K p)/(N − K)).
- **Final list.** C_promoted is sorted by fused score, followed by the base remainder. A bridge candidate is promoted only if its fused score exceeds the lowest fused score in the top-50 (a gold-free rule). Golds outside C_promoted keep the harness-conservative rank: the base filtered rank plus the number of promoted, unfiltered bridge candidates that were not already visibly above the gold.

**Features: 29 per (q, c).**

- **8 base:** log p, log p − log p₁, log base rank, in-top-50 flag, p₁, top-50 entropy, margin₁₂, log tail mass.
- **6 u_pred:** the shared gold-free utility predictor applied to each candidate's own role / country / role_country pattern. My vectorised re-implementation is bit-exact against `UtilityEstimator.goldfree_features`.
- **11 evidence:** all computed from facts with ts < t_q (train+valid+test history, the standard online setting):
  - exact-triple count, dyad count, recency of each, and triple count within the last 30 snapshots;
  - (r′, c) popularity and its recency, entity frequency;
  - query recurrence and subject frequency;
  - the BridgeMem dyadic likelihood-ratio score A(q, c).
- **4 bridge:** in-backfill flag, arm5 score, log backfill position, query has a bridge.
- **Evidence-only set:** the 11 evidence features plus, for TiRGN only, the 4 bridge features.

**Module.** The module is the G10–G12 stage-2 MLP (d → 64 → 64 → 1) with a zero-initialised last layer. It is identical for R and A within a feature set: 6,145 parameters (TiRGN full), 5,889 (LogCL / RE-GCN full), 5,249 / 4,993 (evidence-only). Training uses Adam (lr 1e-3, weight decay 1e-5), batches of 512 queries, ≤ 30 epochs, early stopping on valid-select MRR with patience 5, and seeds 0, 1, 2.

**Valid split.** Valid timestamps below the 70% quantile form valid-fit; the rest form valid-select. ICEWS18 uses all 91,990 valid rows. GDELT uses 200k valid rows (the 20k harness sample plus 180k uniform), with ≤ 120k fit queries whose gold lies in C.

**Arms.**

- **R0 / Rconf / Rreuse** — residual-in-loss with gate none, g(q) = 1 − p₁, or g(c) = 1 − 1/rolecount(c). Rreuse is degenerate on GDELT, where roles are unique and g ≡ 0.
- **A** — standalone listwise CE (base logit not in the loss), then s = log p_base + λ·g·f. λ ∈ {0.1, …, 4} and g ∈ {1, 4u(1−u)} are chosen by BridgeMem's rule: maximise valid MRR subject to no decrease in MRR, H@1, H@3 or H@10.
- **B** — the same standalone f, with s = α·log p_base + β·f and (α, β) = argmin of the valid-select listwise NLL (LBFGS).
- **C** — arm4 trained on the g12 cohort (3 seeds, harness recipe), used as a backfill through the harness ConfidenceSwitch with (q, M, B) chosen on the valid sample. For LogCL and RE-GCN, C reuses the TiRGN-anchored backfill lists (row-matched) with the base's own trigger; it is a cross-base reference row.
- **D** — BridgeMem-mechanism control. It is an entity-anonymised count reader over ≤8 events of the ordered pair (a, c) within 65 snapshots, with context (relation and direction, log-lag bin). It is fitted on train facts only and backs off P(r | r_e, lb) → P(r | r_e) → P(r) with τ = 100. The score is A = log mean_j P(r_q | ctx_j) − log P(r_q), fused as s = log p_base + λ·g·A under the same rule as A. The uncertainty u is the base's top-50 probability mass on candidates with dyadic history (a CENET-oracle proxy).
- **COPY** (diagnostic) — s = log p_base + λ·log(1 + n(s, r′, c)).

**Statistics.** Per-query reciprocal ranks are averaged over the 3 seeds. The test is a paired Poisson bootstrap (B = 2000; harness convention Δ = Σ wᵢdᵢ / n), with one-sided p = P_boot(Δ ≤ 0). A head stratum is flagged DAMAGE when its CI lies entirely below 0 **and** |ΔMRR| ≥ 0.001.

**Leakage audit.** A brute-force `ts < t_q` recount over 382k (query, candidate) pairs matches the cached features to within float16 storage error (max |Δ| ≤ 0.0039 in log space). No test query has a duplicate of its own fact at the same timestamp. The 34 GDELT pairs that fall closer to the inclusive count are float16 rounding at large counts: a material inclusive/strict difference would show |Δ| ≥ 0.69 (0→1) or > 0.008 (n < ~125), above the observed maximum of 0.0039.

## 3. Oracle ceiling (computed first; full test)

| dataset · base | base MRR | in-pool@1 | C@1 | bridge-only@1 | bridge block (realised) | gold in top-50 | best R (realised) | headroom used |
|---|---|---|---|---|---|---|---|---|
| ICEWS18 · TiRGN | 0.3370 | 0.7430 | 0.7485 | 0.3426 | 0.3370 | 0.741 | 0.3672 | 7.3% |
| ICEWS18 · LogCL | 0.2878 | 0.6624 | 0.6624 | 0.2878 | — | 0.661 | 0.3401 | 14.0% |
| ICEWS18 · RE-GCN | 0.2869 | 0.6705 | 0.6705 | 0.2869 | — | 0.669 | 0.3414 | 14.2% |
| GDELT · TiRGN | 0.2167 | 0.6242 | 0.6245 | 0.2170 | 0.2167 | 0.621 | 0.2654 | 11.9% |
| GDELT · LogCL | 0.2114 | 0.6046 | 0.6046 | 0.2114 | — | 0.602 | 0.2528 | 10.5% |
| GDELT · RE-GCN | 0.1973 | 0.5786 | 0.5786 | 0.1973 | — | 0.576 | 0.2428 | 11.9% |

"Headroom used" = (R − base) / (C@1 − base). The bridge contributes almost nothing even at the oracle. Placing every bridge-only gold at rank 1 adds +0.0056 MRR on ICEWS18 and +0.00025 on GDELT. The realised arm5 block (all bridge candidates after the top-50, in arm5 order) adds +3×10⁻⁵ and +3×10⁻⁶. The re-ranking headroom inside the top-50 is large (+0.37 to +0.41 at rank 1), and the best realised R uses 7–14% of it.

## 4. Results — full test

**Full-test MRR (mean ± sd over 3 seeds; D, COPY and the base are deterministic; C is 3 seeds with sd < 1e-6):**

| arm | ICEWS18 · TiRGN | ICEWS18 · LogCL | ICEWS18 · RE-GCN | GDELT · TiRGN | GDELT · LogCL | GDELT · RE-GCN |
|---|---|---|---|---|---|---|
| frozen base | 0.3370 | 0.2878 | 0.2869 | 0.2167 | 0.2114 | 0.1973 |
| R residual, no gate | 0.3674 ± 0.0002 | 0.3401 ± 0.0003 | 0.3414 ± 0.0003 | 0.2654 ± 0.0002 | 0.2528 ± 0.0002 | 0.2428 ± 0.0003 |
| R residual, conf. gate | 0.3672 ± 0.0004 | 0.3401 ± 0.0004 | 0.3410 ± 0.0006 | 0.2655 ± 0.0002 | 0.2531 ± 0.0003 | 0.2429 ± 0.0002 |
| R residual, reuse gate | 0.3442 ± 0.0005 | 0.3023 ± 0.0003 | 0.3020 ± 0.0003 | 0.2167 ± 0.0000 | 0.2114 ± 0.0000 | 0.1973 ± 0.0000 |
| B standalone, calibrated | 0.3664 ± 0.0006 | 0.3401 ± 0.0006 | 0.3414 ± 0.0004 | 0.2656 ± 0.0004 | 0.2534 ± 0.0006 | 0.2429 ± 0.0005 |
| standalone alone | 0.3664 ± 0.0006 | 0.3401 ± 0.0007 | 0.3412 ± 0.0003 | 0.2655 ± 0.0004 | 0.2533 ± 0.0006 | 0.2427 ± 0.0006 |
| A standalone, λ-mix | 0.3652 ± 0.0008 | 0.3379 ± 0.0009 | 0.3380 ± 0.0007 | 0.2634 ± 0.0006 | 0.2508 ± 0.0009 | 0.2404 ± 0.0009 |
| D BridgeMem dyadic | 0.3425 | 0.2966 | 0.2964 | 0.2348 | 0.2257 | 0.2128 |
| copy prior | 0.3412 | 0.3053 | 0.3073 | 0.2256 | 0.2197 | 0.2167 |
| C arm4 backfill | 0.3370 ± 0.0000 | 0.2878 ± 0.0000 | 0.2869 ± 0.0000 | 0.2167 ± 0.0000 | 0.2114 ± 0.0000 | 0.1973 ± 0.0000 |
| R evidence-only | 0.3616 ± 0.0003 | 0.3223 ± 0.0004 | 0.3222 ± 0.0007 | 0.2600 ± 0.0003 | 0.2475 ± 0.0005 | 0.2365 ± 0.0002 |
| B evidence-only | 0.3618 ± 0.0007 | 0.3378 ± 0.0004 | 0.3383 ± 0.0004 | 0.2651 ± 0.0000 | 0.2532 ± 0.0003 | 0.2430 ± 0.0000 |

**H@1 / H@3 / H@10 (seed means):**

| arm | ICEWS18 · TiRGN | ICEWS18 · LogCL | ICEWS18 · RE-GCN | GDELT · TiRGN | GDELT · LogCL | GDELT · RE-GCN |
|---|---|---|---|---|---|---|
| frozen base | 0.2316 / 0.3817 / 0.5425 | 0.1949 / 0.3243 / 0.4711 | 0.1933 / 0.3219 / 0.4717 | 0.1363 / 0.2327 / 0.3760 | 0.1336 / 0.2272 / 0.3639 | 0.1247 / 0.2097 / 0.3391 |
| R residual, no gate | 0.2633 / 0.4151 / 0.5699 | 0.2437 / 0.3863 / 0.5271 | 0.2444 / 0.3869 / 0.5296 | 0.1793 / 0.2937 / 0.4336 | 0.1702 / 0.2786 / 0.4142 | 0.1639 / 0.2670 / 0.3965 |
| B standalone, calibrated | 0.2621 / 0.4137 / 0.5690 | 0.2438 / 0.3859 / 0.5271 | 0.2443 / 0.3870 / 0.5298 | 0.1796 / 0.2938 / 0.4338 | 0.1709 / 0.2791 / 0.4144 | 0.1639 / 0.2671 / 0.3965 |

!Full-test ΔMRR vs frozen base, all arms

**Decision rule (R − comparator, ×10⁻³ MRR; p = P_boot(Δ ≤ 0); R − C and R − D all have p = 0.000):**

| dataset · base | R selected (valid) | R − base [95% CI] | R − A (λ-mix) | R − B (calibrated) | R − C | R − D | head guardrail | all conditions |
|---|---|---|---|---|---|---|---|---|
| ICEWS18 · TiRGN | Rconf | +30.2 [+28.9, +31.4] | +1.99 [+1.46, +2.49] p=0.000 | +0.81 [+0.39, +1.21] p=0.000 | +30.2 | +24.7 | HELP | PASS |
| ICEWS18 · LogCL | Rconf | +52.2 [+50.8, +53.7] | +2.14 [+1.66, +2.63] p=0.000 | +0.00 [-0.37, +0.36] p=0.506 | +52.2 | +43.5 | HELP | fail |
| ICEWS18 · RE-GCN | R0 | +54.5 [+53.0, +56.0] | +3.38 [+2.92, +3.86] p=0.000 | -0.00 [-0.33, +0.32] p=0.490 | +54.5 | +45.0 | HELP | fail |
| GDELT · TiRGN | R0 | +48.7 [+48.1, +49.3] | +2.02 [+1.85, +2.19] p=0.000 | -0.22 [-0.34, -0.11] p=1.000 | +48.7 | +30.6 | HELP | fail |
| GDELT · LogCL | R0 | +41.4 [+40.9, +42.0] | +2.03 [+1.86, +2.19] p=0.000 | -0.55 [-0.65, -0.44] p=1.000 | +41.4 | +27.1 | HELP | fail |
| GDELT · RE-GCN | R0 | +45.5 [+44.9, +46.1] | +2.43 [+2.27, +2.57] p=0.000 | -0.04 [-0.14, +0.06] p=0.785 | +45.5 | +30.0 | HELP | fail |

!Pairwise R − comparator

**Reading.**

1. **The gain over every base is large.** Every learned re-ranker over the top-50 lifts its base by +0.030 to +0.054 MRR (ICEWS18 TiRGN 0.337 → 0.367; GDELT TiRGN 0.217 → 0.265), and the gain is large on the head stratum too (guardrail HELP everywhere).
2. **That gain does not depend on the residual objective.** The valid-calibrated standalone ranker B matches R within ±0.0008 MRR. R is better only on ICEWS18 TiRGN (+0.0008, CI [+0.0004, +0.0012]). It ties on ICEWS18 LogCL, ICEWS18 RE-GCN and GDELT RE-GCN, and is significantly worse on GDELT TiRGN (−0.0002) and GDELT LogCL (−0.0005).
3. **R beats A everywhere (+0.002 to +0.003), but this reflects the mixing rule, not the objective.** BridgeMem's no-decrease λ-mix adds the base logit to a ranker that already uses it as a feature; its selected λ sits at the grid edge (4) on most seeds, and the λ-mix stays below the standalone ranker alone.
4. **R beats D (the BridgeMem dyadic residual, +0.005 to +0.018 over base) and C (the arm4 backfill, ΔMRR between −1.3e-5 and +1e-6) everywhere.**
5. **The reusability gate costs 0.023–0.039 MRR relative to R0 on ICEWS18 and is null on GDELT.** The confidence gate is equivalent to no gate (valid selection picks it on 2/6 combinations).

**Evidence-only features: pinning the base logit hurts.**

| dataset · base | R_ev − B_ev ×10⁻³ [95% CI] | B weights (α on log p_base, β on f), 3 seeds | B_ev weights |
|---|---|---|---|
| ICEWS18 · TiRGN | -0.23 [-0.98, +0.56] | 0.01/0.97; 0.06/0.97; 0.00/0.97 | 0.49/0.63; 0.49/0.61; 0.49/0.60 |
| ICEWS18 · LogCL | -15.55 [-16.57, -14.45] | -0.00/1.01; 0.01/0.98; 0.02/1.03 | 0.23/0.85; 0.24/0.83; 0.24/0.81 |
| ICEWS18 · RE-GCN | -16.03 [-17.13, -15.02] | -0.01/0.96; -0.07/1.05; -0.01/1.03 | 0.26/0.78; 0.27/0.81; 0.26/0.81 |
| GDELT · TiRGN | -5.11 [-5.38, -4.82] | -0.10/1.05; -0.03/1.07; -0.08/1.11 | 0.07/0.98; 0.07/1.00; 0.04/1.01 |
| GDELT · LogCL | -5.73 [-5.99, -5.45] | -0.05/1.09; -0.07/1.10; -0.04/1.06 | 0.01/0.98; 0.05/0.97; 0.03/1.02 |
| GDELT · RE-GCN | -6.41 [-6.68, -6.12] | -0.02/1.01; -0.07/1.07; -0.06/1.06 | 0.09/0.96; 0.06/0.96; 0.06/0.96 |

With evidence-only features the residual objective is significantly *worse* than the calibrated standalone ranker on 5/6 combinations (−4 to −16 ×10⁻³). On GDELT the calibrated weight on log p_base is ≈ 0 even when f sees no base features. There, TiRGN, LogCL and RE-GCN log-probs carry little information beyond their top-50 candidate set once history evidence is available. A fixed coefficient of 1 on the base logit is therefore a handicap rather than a useful inductive bias, which contradicts the premise of idea 6.2.

## 5. Where the gains land (descriptive; step 4 skipped)

The decision rule failed in step 3, so the confirmatory strikingness check (step 4) was skipped per protocol. The numbers below were computed by the step-3 evaluator. No choice used them.

**Strikingness.** Strikingness follows the authors' definition (Huang, Zhang & Wei, arXiv:2605.13153 / IJCAI'26, RSMF):

- **Rules and expectation score:** length-1 TLogic rules; the expectation score is Σ conf(rule)·e^(−0.1·Δt) over a 200-snapshot window.
- **Strikingness score:** the strikingness of subject, object and relation is computed against L2-normalised peer events and combined with weights α_s = 0.4, α_o = 0.4, α_r = 0.2.
- **Weights used:** the authors' released per-quadruple weights (100% of test quadruples matched; the same weight is applied to both directions, as in their `evaluate.py`).
- **Metric:** WMRR = Σ(sk + b)/rank / Σ(sk + b), with b = 0.1.

**ΔMRR ×10⁻³ of the selected R vs the base, by stratum; last columns WMRR:**

| dataset · base | cold | mid | head | hist=no | non-repeat | rec. top-1 miss | no dyad hist. (65) | WMRR (b=0.1) base → R | R − B WMRR ×10⁻³ |
|---|---|---|---|---|---|---|---|---|---|
| ICEWS18 · TiRGN | +13.7 | +21.2 | +41.4 | +22.2 | +21.0 | +14.5 | -8.2 | 0.2387 → 0.2504 | +0.94 [+0.57, +1.29] |
| ICEWS18 · LogCL | +23.3 | +39.5 | +70.3 | +37.6 | +17.3 | +25.9 | +3.7 | 0.1973 → 0.2233 | -0.01 [-0.36, +0.35] |
| ICEWS18 · RE-GCN | +18.2 | +43.9 | +74.2 | +36.5 | +8.9 | +26.1 | +3.7 | 0.2006 → 0.2269 | +0.01 [-0.29, +0.29] |
| GDELT · TiRGN | +33.6 | +45.5 | +49.3 | +50.3 | +13.0 | +14.6 | +8.9 | 0.1577 → 0.1860 | -0.00 [-0.11, +0.11] |
| GDELT · LogCL | +6.7 | +20.9 | +43.8 | +30.2 | +5.1 | +10.4 | +9.0 | 0.1542 → 0.1774 | -0.25 [-0.35, -0.16] |
| GDELT · RE-GCN | +4.0 | +14.1 | +48.9 | +19.3 | -5.3 | +14.3 | +13.3 | 0.1434 → 0.1716 | +0.16 [+0.07, +0.25] |

!Gains by stratum and R − B by stratum

**Reading.**

1. **Gains sit mostly on repetitive events.** In the lowest strikingness band (sk < 0.1), R gains +0.107 to +0.164 MRR. In the highest band (sk ≥ 0.5) it gains only +0.003 to +0.013.
2. **The strikingness-weighted gain is smaller than the unweighted one.** It is +0.012 vs +0.030 on ICEWS18 TiRGN, and +0.023 to +0.028 vs +0.041 to +0.054 elsewhere.
3. **Some strata get worse.** On ICEWS18 TiRGN, queries whose gold has no dyadic history in the 65-snapshot window lose 0.0082 MRR (CI entirely < 0): the re-ranker promotes historical candidates over a non-historical gold. On GDELT RE-GCN, non-repeat queries lose 0.0053.
4. **R − B per stratum is within ±2.2 × 10⁻³ and changes sign across bases.** On WMRR it is positive only on ICEWS18 TiRGN (+0.0009) and GDELT RE-GCN (+0.00016), and negative on GDELT LogCL (−0.00025). The objective therefore does not move gains toward non-trivial events.

## 6. Negatives and caveats (kept intact)

- **Decision rule.** The rule fails. Only ICEWS18 / TiRGN satisfies every condition, and the margin there over B is +0.0008.
- **GDELT valid scope.** GDELT training and selection used 200k of the 477,530 valid rows, with ≤ 120k fit queries. The test set was full.
- **Bridge component.** It is TiRGN-only and capped at 50 candidates from the shared arm5 lists, and its valid training signal covers only the harness 20k valid sample (6.6% / 2.0% of valid rows). The oracle shows this cap barely binds: bridge-only@1 is worth at most +0.0056.
- **C on LogCL / RE-GCN.** It reuses TiRGN-anchored bridge lists (a cross-base reference row). It is ≈ 0 on every base, as is the harness arm5 row b.
- **u_pred.** The predictor was trained against TiRGN on forward queries and is applied to LogCL and RE-GCN distributions and to inverse queries. It is a feature, not an estimator claim.
- **D is a simplified BridgeMem.** It uses a count reader without the neural reader, a proxy uncertainty and a mean mixture over events. It is a mechanism control under our protocol, not a reproduction of BridgeMem's numbers.
- **Superseded runs.** A first pair of 12-epoch runs was cancelled because most arms selected their final epoch on valid-select. The rerun gives every arm ≤ 30 epochs with patience 5. Test numbers printed in the cancelled logs were seen but not used for any choice.
- **No SOTA claim.** GDELT R/B (0.265) and ICEWS18 R (0.367) approach published taf rows (HisRES 26.58 / 37.69). The harness's conservative filtered rank beyond the top-K and the top-50 candidate cap make the protocols non-identical, so no claim is made.

## 7. Implications (Task 3, and the novelty addendum)

- **Idea 6.2 as an objective claim is not supported.** At matched features and capacity, training through the frozen base's logits is equivalent to, or worse than, a standalone ranker with valid-calibrated mixing. Recommend changing the addendum's 6.2 verdict from NARROWED to **KILLED** as a stand-alone contribution. BridgeMem already holds the architecture, and the objective adds nothing measurable.
- **The learned history-feature re-ranker is a strong, cheap plug-in.** It adds +0.03 to +0.05 full-test MRR on three bases and two datasets, with head gains. It belongs in Task 3 as a baseline and reference row, not as a novelty. The 6.1 and 6.5 tracks should report it next to the frozen bases, because "beats TiRGN" is now a weak bar.
- **The gains are on low-strikingness events.** Any future claim about "non-trivial" event reasoning must be made on WMRR or strikingness bins, where this re-ranker's gains shrink by about 2–3×.

## 8. Reproduction

All code is in `$WORK_DIR/tracks/residual/code/` on serverA . Features are under `…/tracks/residual/feat/<DS>_<base>/`, and per-query test ranks under `…/results/<DS>_<base>/ranks_test.npz` and `…/results/C/`.

- `rs_features.py --dataset D --base B` builds the features, gates and oracle.
- `rs_train.py --dataset D --base B` trains R, A, B, D and COPY over 3 seeds and writes the test ranks.
- `rs_armC.py --dataset D` runs arm C.
- `rs_aggregate.py` runs the bootstrap, strata, WMRR and decision rule.
- `rs_leak_audit.py` runs the leakage audit.

Code files: `rs_features.py`, `rs_train.py`, `rs_armC.py`, `rs_aggregate.py`, `rs_leak_audit.py`. Seed-level table: `residual_fulltest_table.csv`.
