# GDELT entity-resolution-aware re-scoring

Evaluation-hygiene follow-up to the error analysis. Question: are the GDELT system ordering and the history re-ranker (HR) gain artefacts of alias fragmentation in the entity vocabulary (e.g. `AMERICAN (USA@…)` and `UNITED STATES (USA@…)` being different entities)?

## 1. Verdict

- **H1 (ordering), primary space b: SURVIVES.** Kendall τ_b = 1.0; all 2000 bootstrap replicates have τ_b = 1. The order S4 > S6 > S5 > S1 > S3 > S2 is identical in spaces a, b, b2 and b_noR3.
- **H2 (HR gap over TiRGN), space b: SURVIVES; DIMINISHED.** ΔMRR(S4−S1) = 0.0488 in a → 0.0478 in b (95% CI 0.0473–0.0484; one-sided p < 0.0005). The change Δ_b−Δ_a = -0.0010 (CI -0.0012 to -0.0009) lies entirely below 0, so the pre-registered rule returns DIMINISHED; the gain retained is 97.9%.
- **Post-hoc sensitivity without demonym merges (b_names, not pre-registered):** τ_b = 1.0, ordering SURVIVES; ΔMRR(S4−S1) = 0.0485, change -0.0003 (CI -0.0004 to -0.0002), 99.4% of the gain retained. Demonym merges (AMERICAN→UNITED STATES etc.) account for about three quarters of the MRR shift in b (TiRGN +0.0047 in b_names vs +0.0192 in b).
- Validation passed before space b was read: space-a MRR of all six systems within 0.002 of the track reports (largest deviation 0.000001); ranks of golds inside C_promoted equal the stored 6.2 ranks for 100% of queries (3 bases × 3 seeds).
- Reading: alias resolution moves every system up by about the same amount (+0.018 to +0.020 MRR in b), so it changes absolute GDELT numbers but none of the comparative conclusions. The HR loses about 2% of its gain; a plausible mechanism (not tested here) is that resolution credits the bases for alias hits the HR had partly recovered by promoting the exact surface form. The formal DIMINISHED label reflects n = 610,482 making a 0.001 shift detectable; no practical-equivalence margin was pre-registered.

!Re-scoring results

*Figure. (a) MRR of the six systems in nested spaces of increasing coarseness: original (a), country-name variants only (b_names, post-hoc), alias classes incl. demonyms (b, primary), plus type-level generic aggregates (b2); solid = with HR, dashed = frozen base. (b) HR gain over its base with 95% paired-bootstrap CI (B = 2000) in each space; 'alias, no R3' = b without country-alias merges. (c) Contribution of each gold-entity stratum to TiRGN's MRR shift a→b (share × ΔMRR); percentages are the stratum's share of the 610,482 test queries.*

## 2. Alias and aggregate-actor map

GDELT entity names have the form `CORE (CC@ ROLES)`: CORE is the actor text, CC the CAMEO country code, ROLES the CAMEO role codes observed for that actor. No two entities share a core string; fragmentation arises from demonyms, foreign-language or variant names, plural/singular forms and punctuation.

| Rule | Definition | Unions | Entities touched |
|---|---|---|---|
| R1 normalised string | upper-case, strip punctuation, drop leading THE; merge only if CC compatible and attached role sets overlap | 8 | 16 |
| R2 plural→singular | same CC, same attachedness, overlapping roles | 22 | 44 |
| R3 country-actor alias | demonyms, foreign-language and misspelled country names merged into the country actor CAMEO codes them to (LLM proposals per CC, analyst-reviewed; 43 rejections) | 408 | 583 |
| R4 surname-only | surname entity merged with the full-name entity when unambiguous (OBAMA, MUSHARRAF, GERDAU) | 3 | 6 |

Result: 7,255 alias classes from 7,691 entities; 649 entities sit in multi-member classes. 46.2% of test queries have a gold in a multi-member class, dominated by country actors (the UNITED STATES class has 13 members and is the gold for 59,486 queries).

**Exclusions (R3).** Cities, capitals, territories, constituent countries and dependent areas are not merged into the sovereign country (e.g. PUERTO RICO, AMERICAN SAMOA, WELSH/SCOT/ENGLISHMAN/MANX, DEN HAAG, BERNE/GENF, PRAG/PRAHA, DACCA, TEHERAN, VILNIUS, LISBOA, MALE, BARBUDA, GRENADINES, TOBAGO, FRENCH POLYNESIAN/GUIANESE, SANTIAGO DE CHILE), nor are false friends (AMERICAN HUMANIST ASSOCIATION, DEUTSCHE, FENNO, HELVETICA, LUSITANIA). Historical or sub-state labels are kept separate (WEST/EAST GERMANY, CHECHENIA, NETHERLANDS ANTILLES, CZECHO, VENDA, CORSICAN, SCOTTISH, AFAR, COMMONWEALTH OF NATIONS, U.S. GOVERNMENT).

**Generic aggregates.** Country-less role actors (GOVERNMENT, POLICE, CITIZEN, COMPANY, …; 1,997 entities) have no fixed referent and are **not** merged in b. In b2 they are additionally merged by their first role code (e.g. all `@GOV` generics form one class), which scores type-level prediction; b2 is a sensitivity space only.

| Entity kind | Entities |
|---|---|
| named_actor_with_country | 2,201 |
| generic_aggregate | 1,997 |
| place_geo_actor | 1,889 |
| named_actor_no_country | 972 |
| country_actor | 613 |
| other | 19 |

**Spot-check (100 random merges, rng seed 0; 90 R3, 8 R2, 2 R1).** The 100 merges were printed with names, frequencies and target class and read; verdicts were recorded by exception (5 flagged, the rest recorded as correct). Result: 95 correct, 5 questionable, 0 wrong. Questionable: CONGOLESE and CENTRAL AFRICAN (ambiguous between two states; merged per CAMEO code), SPANIARD, FILIPINA, IRISHWOMAN (individual demonyms at person granularity). All five were kept in b; all five are demonyms, and the post-hoc space b_names (§4) removes every demonym merge (211 entities, 7,427 classes). Generic/named label check: 40/40 GENERIC and 20/20 NAMED labels plausible. Recall of the map was not estimated.

## 3. Protocol (pre-registered)

Pre-registration `gdelt_er_preregistration.json` written 2026-10-13T05:49:00.262094Z before any resolved-space score was computed; class maps frozen (sha256 `8212395118895d03…`).

- Systems: S1 TiRGN, S2 RE-GCN, S3 LogCL (frozen bases); S4–S6 the 6.2 history re-ranker on each base (valid-selected arm: TiRGN Acal, RE-GCN R0, LogCL Acal), 3 seeds, per-query RR averaged over seeds.
- Rank: 1 + #classes ≠ class(gold), not filtered, whose max member score exceeds the gold class's max member score (strict >). Time-aware filter in the resolved space: classes of every o′ with (s″, r′, o′, t) true for any s″ in class(s). Space a = singleton classes (the harness filter).
- Re-ranker list: C_promoted (base top-50 plus promoted bridge candidates) ordered by the fused score, then all remaining entities by base score. Identical to the 6.2 evaluator inside C_promoted; exact (not conservative) outside it, using the stored full score rows (610,482 × 7,691 log-softmax per base).
- Tests: H1 Kendall τ_b between the six MRRs in a and b; H2 Δ = MRR(S4) − MRR(S1), Poisson(1) paired bootstrap, B = 2000. Secondary: S5−S2, S6−S3 and all tests in b2; b_noR3 descriptive.

## 4. Results

| System | MRR a | MRR b_names* | MRR b | MRR b2 | MRR noR3 | H@1 a | H@1 b | H@10 a | H@10 b |
|---|---|---|---|---|---|---|---|---|---|
| S1 TiRGN | 0.2167 | 0.2214 | 0.2359 | 0.2879 | 0.2172 | 0.1363 | 0.1503 | 0.3760 | 0.4056 |
| S2 RE-GCN | 0.1973 | 0.2023 | 0.2172 | 0.2694 | 0.1979 | 0.1247 | 0.1392 | 0.3391 | 0.3705 |
| S3 LogCL | 0.2114 | 0.2164 | 0.2305 | 0.2854 | 0.2119 | 0.1336 | 0.1478 | 0.3639 | 0.3937 |
| S4 TiRGN+HR | 0.2656 | 0.2700 | 0.2838 | 0.3357 | 0.2661 | 0.1796 | 0.1936 | 0.4338 | 0.4603 |
| S5 RE-GCN+HR | 0.2428 | 0.2475 | 0.2615 | 0.3152 | 0.2433 | 0.1639 | 0.1782 | 0.3965 | 0.4234 |
| S6 LogCL+HR | 0.2534 | 0.2579 | 0.2717 | 0.3244 | 0.2538 | 0.1709 | 0.1851 | 0.4144 | 0.4407 |

*b_names: post-hoc sensitivity (country-name variants merged, demonyms not), not pre-registered.*

**H1 ordering.**

| Space | τ_b | P_boot(τ_b = 1) | Decision |
|---|---|---|---|
| b | 1.0 | 1.0 | SURVIVES |
| b2 | 1.0 | 1.0 | SURVIVES |
| noR3 | 1.0 | 1.0 | SURVIVES |
| b_names* | 1.0 | 1.0 | SURVIVES |

**H2 HR gain.**

| Space | Pair | Δ a | Δ space (95% CI) | Δ space − Δ a (95% CI) | Ratio | Decision |
|---|---|---|---|---|---|---|
| b | S4 TiRGN+HR - S1 TiRGN | 0.0488 | 0.0478 (0.0473–0.0484) | -0.0010 (-0.0012 to -0.0009) | 0.979 | SURVIVES; DIMINISHED |
| b | S5 RE-GCN+HR - S2 RE-GCN | 0.0454 | 0.0441 (0.0437–0.0449) | -0.0013 (-0.0014 to -0.0010) | 0.973 | SURVIVES; DIMINISHED |
| b | S6 LogCL+HR - S3 LogCL | 0.0418 | 0.0411 (0.0406–0.0417) | -0.0008 (-0.0010 to -0.0006) | 0.981 | SURVIVES; DIMINISHED |
| b2 | S4 TiRGN+HR - S1 TiRGN | 0.0488 | 0.0478 (0.0473–0.0484) | -0.0010 (-0.0014 to -0.0008) | 0.979 | SURVIVES; DIMINISHED |
| b2 | S5 RE-GCN+HR - S2 RE-GCN | 0.0454 | 0.0457 (0.0452–0.0464) | +0.0003 (-0.0000 to +0.0005) | 1.006 | SURVIVES; UNDIMINISHED |
| b2 | S6 LogCL+HR - S3 LogCL | 0.0418 | 0.0389 (0.0384–0.0395) | -0.0030 (-0.0033 to -0.0027) | 0.929 | SURVIVES; DIMINISHED |
| noR3 | S4 TiRGN+HR - S1 TiRGN | 0.0488 | 0.0487 (0.0483–0.0495) | -0.0000 (-0.0001 to -0.0000) | 0.999 | SURVIVES; DIMINISHED |
| noR3 | S5 RE-GCN+HR - S2 RE-GCN | 0.0454 | 0.0453 (0.0449–0.0460) | -0.0001 (-0.0001 to -0.0000) | 0.999 | SURVIVES; DIMINISHED |
| noR3 | S6 LogCL+HR - S3 LogCL | 0.0418 | 0.0418 (0.0414–0.0425) | -0.0000 (-0.0001 to -0.0000) | 0.999 | SURVIVES; DIMINISHED |
| b_names* | S4 TiRGN+HR - S1 TiRGN | 0.0488 | 0.0485 (0.0481–0.0492) | -0.0003 (-0.0004 to -0.0002) | 0.994 | SURVIVES; DIMINISHED |
| b_names* | S5 RE-GCN+HR - S2 RE-GCN | 0.0454 | 0.0450 (0.0446–0.0457) | -0.0004 (-0.0005 to -0.0003) | 0.991 | SURVIVES; DIMINISHED |
| b_names* | S6 LogCL+HR - S3 LogCL | 0.0418 | 0.0414 (0.0409–0.0421) | -0.0005 (-0.0005 to -0.0004) | 0.989 | SURVIVES; DIMINISHED |

In noR3 (alias classes without country aliases) the gain changes by < 0.0001 MRR; the shrinkage in b therefore comes from the R3 country-alias merges. In b2 the LogCL re-ranker loses 7% of its gain (−0.0030), RE-GCN's is unchanged, TiRGN's loses 2%.

## 5. Where the shift comes from (descriptive)

| Gold kind | Gold class aliased (b) | Share of queries | TiRGN MRR a | TiRGN MRR b | TiRGN+HR MRR a | TiRGN+HR MRR b | HR gain a | HR gain b | Contribution to Δ change |
|---|---|---|---|---|---|---|---|---|---|
| country_actor | yes | 42.0% | 0.3470 | 0.3881 | 0.3896 | 0.4287 | 0.0426 | 0.0406 | -0.00086 |
| generic_aggregate | no | 36.1% | 0.1367 | 0.1395 | 0.1933 | 0.1959 | 0.0565 | 0.0563 | -0.00007 |
| place_geo_actor | no | 12.6% | 0.0675 | 0.0702 | 0.1152 | 0.1174 | 0.0477 | 0.0472 | -0.00007 |
| named_actor_with_country | no | 3.8% | 0.0995 | 0.1027 | 0.1503 | 0.1530 | 0.0507 | 0.0503 | -0.00002 |
| generic_aggregate | yes | 3.6% | 0.2038 | 0.2153 | 0.2514 | 0.2619 | 0.0476 | 0.0466 | -0.00004 |
| named_actor_no_country | no | 1.1% | 0.1138 | 0.1166 | 0.1690 | 0.1710 | 0.0552 | 0.0545 | -0.00001 |
| named_actor_with_country | yes | 0.6% | 0.1014 | 0.1111 | 0.1548 | 0.1660 | 0.0534 | 0.0550 | +0.00001 |
| country_actor | no | 0.1% | 0.0820 | 0.0838 | 0.1457 | 0.1467 | 0.0637 | 0.0630 | -0.00000 |
| other | no | 0.1% | 0.1482 | 0.1490 | 0.1217 | 0.1228 | -0.0265 | -0.0262 | +0.00000 |
| place_geo_actor | yes | 0.0% | 0.0072 | 0.0082 | 0.0199 | 0.0223 | 0.0128 | 0.0141 | +0.00000 |

Country-actor golds with aliases (42% of queries) account for +0.0173 of TiRGN's +0.0192 MRR shift and for −0.0009 of the −0.0010 change in the HR gain. Ten alias classes carry 70% of the shift; UNITED STATES alone contributes +0.0071. Part of the shift for singleton golds comes from competitor collapse (several aliases of one country scoring above the gold now count as one class) and from the larger resolved filter.

| Alias class (canonical) | Members | Test queries with this gold | TiRGN ΔMRR contribution | TiRGN+HR ΔMRR contribution |
|---|---|---|---|---|
| UNITED STATES | 13 | 59,486 | +0.00713 | +0.00641 |
| UNITED KINGDOM | 7 | 12,065 | +0.00105 | +0.00110 |
| SYRIA | 2 | 7,511 | +0.00102 | +0.00089 |
| RUSSIA | 4 | 11,096 | +0.00096 | +0.00088 |
| TURKEY | 5 | 6,273 | +0.00093 | +0.00074 |
| ISRAEL | 7 | 8,319 | +0.00076 | +0.00083 |
| CHINA | 3 | 9,371 | +0.00058 | +0.00066 |
| AFGHANISTAN | 5 | 3,748 | +0.00037 | +0.00031 |
| NIGERIA | 4 | 6,268 | +0.00035 | +0.00030 |
| COMPANY | 2 | 7,378 | +0.00033 | +0.00030 |

Duplicate queries after resolution (kept, not deduplicated): b 15,176; b2 27,708; noR3 96.

## 6. Implications for the project

1. The GDELT comparative claims in the 6.2 report (re-ranker beats every base; TiRGN-based system strongest) do not depend on alias fragmentation. Error-analysis redesign 2 (alias/aggregate-aware evaluation) is therefore a reporting addition, not a results-changing correction: report space-b MRR as a secondary metric next to the standard one.
2. Absolute GDELT MRRs understate entity-level correctness by 0.005 MRR (country-name variants only) to 0.019 MRR (including demonyms); H@1 by 0.003–0.014. Whether a demonym actor (AMERICAN) and the country actor (UNITED STATES) are the same referent is the deciding modelling choice; CAMEO assigns both the same country code. The realised alias correction for TiRGN+HR (H@1 0.180 → 0.194) is about a third of the geo-alias upper bound estimated in the error analysis (≤ 0.224), which counted any same-country top-1.
3. Type-level scoring (b2) adds about 0.07 MRR to every system, concentrated on generic aggregates. This is a ceiling for a role-type predictor (redesign 3, role × country factorisation) under a type-level metric, not evidence that such a model would improve entity-level MRR.
4. If the mechanism suggested in §1 holds (the re-ranker promotes exact surface forms the base already covered through an alias), an alias-aware training target for the re-ranker (credit any class member) is a low-cost variant worth one run if space-b MRR is adopted as a reported metric.

## 7. Deviations and limitations

- 6.1 IB plug-in and 6.5 utility-gated augmentation not re-scored (per-query outputs not stored; 6.5 needs GPU re-injection); declared in the pre-registration.
- Alias recall not estimated; precision favoured (cities, territories, constituent countries, capitals excluded).
- R3 country-alias candidates proposed by an LLM (209 prompts) and accepted/rejected by the analyst (43 exclusions); generic-vs-named labels for 3149 country-less entities by LLM, 60 spot-checked.
- Spot-check of 100 random merges: 95 correct, 5 questionable, 0 wrong; verdicts were recorded by exception (the 100 printed merges were read and the 5 flagged; unflagged merges were recorded as correct in code). Questionable merges kept in b.
- Post-hoc (not pre-registered) sensitivity space b_names without demonym merges, run after a review concern about demonym granularity; ordering and HR-gain conclusions unchanged.
- b2 (type-level) is a coarse sensitivity space, not a recommended metric.
- No deduplication: queries that become duplicates after resolution are kept (b: 15,176; b2: 27,708).
- No smallest-effect-of-interest was pre-registered for H2, so 'DIMINISHED' triggers on any shrink the bootstrap can detect (here 0.001 MRR at n=610,482).
- Bootstrap MRR replicates use the ratio estimator sum(w*rr)/sum(w) (Poisson(1) weights, seed 0, B=2000).
- TiRGN space-a ranks outside C_promoted differ from the harness's conservative ranks for 1.5% of queries (exact ranks from full score rows are used; MRR equal to 6 decimals; ranks equal the retriever's exact TiRGN ranks for 100%).
- GDELT only, as tasked; ICEWS18 not re-scored.

## 8. Files

- `gdelt_alias_map.csv` — one row per entity: parsed core/CC/roles, entity kind, alias class (b), type class (b2), merge rule, spot-check verdict.
- `gdelt_class_maps.npz` — frozen class vectors for spaces a, b, b2, noR3.
- `gdelt_er_preregistration.json` — frozen hypotheses and decision rules.
- `gdelt_er_verdict.json` — decisions, all test statistics, validation, deviations.
- `gdelt_er_stats.json` — full remote output (metrics, tests, checks).
- `gdelt_er_stats_sens_bnames.json` and `gdelt_class_maps_names.npz` — post-hoc demonym-free sensitivity (class map sha256 086c761b…).
- `er_rescore.py` (remote scorer), `er_local.py` (strata), `build_er_report.py` (this report). Per-query ranks on serverA: tracks/error_analysis/entity_resolution/.