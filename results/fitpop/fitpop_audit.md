# Fit-population audit of the v7.8 ladder and HRcov (review item 2)

**Label: audit plus sensitivity analysis, run after the v7.8 verdict. The common-intersection refit does not replace the registered verdict (F4_CONDITIONAL, results version `ff0dcaea`).**

## 1. Fit populations

Every arm starts from the same validation fit part (timestamp id below the 70% cut) and keeps only the fit queries whose gold is in its own candidate pool (v7.8 trainer and HRcov trainer alike; the 120,000 cap never binds). +C, +S and +T share one pool, so their fit queries are identical; SA (history pool) and HRcov (pattern pool) keep different sets.

| config | common start | SA | +C = +S = +T | HRcov | intersection of all five | Jaccard SA vs +C | Jaccard SA vs HRcov | Jaccard +C vs HRcov |
|---|---|---|---|---|---|---|---|---|
| S1 (tkgl-smallpedia, tirgn) | 105,606 | 79,357 (75.1%) | 82,138 (77.8%) | 81,874 (77.5%) | 78,047 (73.9%) | 0.9410 | 0.9382 | 0.9966 |
| S2 (tkgl-polecat, tirgn) | 139,236 | 81,715 (58.7%) | 86,847 (62.4%) | 86,236 (61.9%) | 77,065 (55.3%) | 0.8647 | 0.8493 | 0.9797 |
| S3 (tkgl-polecat, cogn) | 139,236 | 81,715 (58.7%) | 88,652 (63.7%) | 88,299 (63.4%) | 77,481 (55.6%) | 0.8478 | 0.8380 | 0.9843 |

| config | only SA (not +C) | only +C (not SA) | only +C (not HRcov) | only HRcov (not +C) | HRcov count = stored HRcov training count |
|---|---|---|---|---|---|
| S1 | 1,062 | 3,843 | 272 | 8 | 81,874 (yes) |
| S2 | 3,547 | 8,679 | 1,192 | 581 | 86,236 (yes) |
| S3 | 3,546 | 10,483 | 876 | 523 | 88,299 (yes) |

The retained identities are therefore not identical across arms: the registered ladder step +C − SA changes the pool and the fit population together (SA → +C adds base candidates; queries whose gold is only in the base top-50 enter the fit set, and some history-only golds leave it), and HRcov − +T (+P) also changes both. +S − +C and +T − +S use identical fit queries.

## 2. Common-intersection sensitivity refit

All five arms retrained on the intersection of their fit sets (same seeds, selection rows, early stopping, fusion and official scoring as registered). HRcov is refitted in the v7.8 trainer (arm HRCOV_RE: stored HRcov pool and features, the HRcov recipe).

| config | system | registered MRR (official evaluator) | common-intersection refit MRR (official evaluator) | refit − registered |
|---|---|---|---|---|
| S1 | SA | 0.6544 | 0.6535 | -0.0009 |
| S1 | +C | 0.6553 | 0.6549 | -0.0004 |
| S1 | +S | 0.6582 | 0.6555 | -0.0027 |
| S1 | +T | 0.6585 | 0.6558 | -0.0027 |
| S1 | HRcov | 0.6584 | 0.6558 | -0.0026 |
| S2 | SA | 0.2752 | 0.2753 | +0.0001 |
| S2 | +C | 0.2807 | 0.2763 | -0.0044 |
| S2 | +S | 0.2825 | 0.2730 | -0.0094 |
| S2 | +T | 0.2825 | 0.2730 | -0.0094 |
| S2 | HRcov | 0.2824 | 0.2740 | -0.0084 |
| S3 | SA | 0.2752 | 0.2755 | +0.0003 |
| S3 | +C | 0.2987 | 0.2756 | -0.0231 |
| S3 | +S | 0.3046 | 0.2992 | -0.0054 |
| S3 | +T | 0.3046 | 0.2991 | -0.0055 |
| S3 | HRcov | 0.3040 | 0.2993 | -0.0047 |

### Ladder and HRcov − SA (MRR; L, U = 5th / 95th bootstrap percentiles)

| config | step | registered Δ | query [L, U] | ts-block [L, U] | common-intersection Δ | query [L, U] | ts-block [L, U] |
|---|---|---|---|---|---|---|---|
| S1 | +C - SA | +0.0008 | [+0.0005, +0.0011] | [-0.0014, +0.0021] | +0.0014 | [+0.0012, +0.0016] | [+0.0002, +0.0023] |
| S1 | +S - +C | +0.0029 | [+0.0026, +0.0033] | [+0.0023, +0.0034] | +0.0006 | [+0.0001, +0.0011] | [-0.0004, +0.0013] |
| S1 | +T - +S | +0.0003 | [+0.0003, +0.0003] | [+0.0002, +0.0003] | +0.0003 | [+0.0003, +0.0003] | [+0.0002, +0.0003] |
| S1 | HRcov - +T (+P) | -0.0000 | [-0.0003, +0.0002] | [-0.0003, +0.0002] | +0.0000 | [-0.0002, +0.0002] | [-0.0002, +0.0002] |
| S1 | HRcov - SA | +0.0040 | [+0.0035, +0.0044] | [+0.0016, +0.0055] | +0.0023 | [+0.0018, +0.0028] | [+0.0007, +0.0036] |
| S2 | +C - SA | +0.0055 | [+0.0053, +0.0057] | [+0.0051, +0.0059] | +0.0010 | [+0.0008, +0.0012] | [+0.0008, +0.0012] |
| S2 | +S - +C | +0.0017 | [+0.0015, +0.0019] | [+0.0014, +0.0021] | -0.0033 | [-0.0036, -0.0030] | [-0.0038, -0.0028] |
| S2 | +T - +S | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] |
| S2 | HRcov - +T (+P) | -0.0001 | [-0.0002, +0.0000] | [-0.0003, +0.0000] | +0.0009 | [+0.0008, +0.0010] | [+0.0008, +0.0010] |
| S2 | HRcov - SA | +0.0072 | [+0.0069, +0.0075] | [+0.0066, +0.0078] | -0.0013 | [-0.0017, -0.0010] | [-0.0020, -0.0007] |
| S3 | +C - SA | +0.0235 | [+0.0231, +0.0239] | [+0.0227, +0.0243] | +0.0001 | [-0.0001, +0.0003] | [-0.0001, +0.0003] |
| S3 | +S - +C | +0.0059 | [+0.0056, +0.0061] | [+0.0055, +0.0063] | +0.0236 | [+0.0232, +0.0240] | [+0.0227, +0.0245] |
| S3 | +T - +S | -0.0000 | [-0.0000, -0.0000] | [-0.0001, -0.0000] | -0.0001 | [-0.0001, -0.0000] | [-0.0001, -0.0000] |
| S3 | HRcov - +T (+P) | -0.0006 | [-0.0007, -0.0005] | [-0.0007, -0.0005] | +0.0002 | [+0.0001, +0.0003] | [+0.0001, +0.0003] |
| S3 | HRcov - SA | +0.0288 | [+0.0283, +0.0292] | [+0.0279, +0.0297] | +0.0238 | [+0.0234, +0.0243] | [+0.0229, +0.0247] |

| config | HRcov − SA call, registered | HRcov − SA call, common intersection |
|---|---|---|
| S1 | SMALL | MATCHES |
| S2 | BEATS | MATCHES |
| S3 | BEATS | BEATS |

## 3. Checks

- S1: HRcov re-run in the v7.8 trainer on its own population (n = 81,874) vs stored HRcov: Δ MRR +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000]; refit fit-query counts {"SA": 78047, "+C": 78047, "+S": 78047, "+T": 78047, "HRcov": 78047}; base reciprocal ranks equal to the registered base (mean |Δrr| 0.0e+00).
- S2: HRcov re-run in the v7.8 trainer on its own population (n = 86,236) vs stored HRcov: Δ MRR -0.0001 | [-0.0001, +0.0000] | [-0.0001, +0.0000]; refit fit-query counts {"SA": 77065, "+C": 77065, "+S": 77065, "+T": 77065, "HRcov": 77065}; base reciprocal ranks equal to the registered base (mean |Δrr| 1.1e-08).
- S3: HRcov re-run in the v7.8 trainer on its own population (n = 88,299) vs stored HRcov: Δ MRR +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000]; refit fit-query counts {"SA": 77481, "+C": 77481, "+S": 77481, "+T": 77481, "HRcov": 77481}; base reciprocal ranks equal to the registered base (mean |Δrr| 0.0e+00).

## 4. Diagnostic: size-matched random subsamples (harness ranks)

To separate the size of the fit set from its composition, each arm was also retrained on a random subsample of its own fit population with exactly the size of the common intersection (training only; harness exact ranks from the trainer, compared with the registered arms in the same harness scale). This table is not comparable in level with the official-evaluator table of section 2: in S3 the trainer harness ranks of the base-tail arms (+T, HRcov) are about 0.027 MRR higher than official.

| config | system | registered (trainer harness ranks) | size-matched refit (trainer harness ranks) | size-matched − registered | common-intersection refit (trainer harness ranks) |
|---|---|---|---|---|---|
| S1 | SA | 0.6545 | 0.6540 | -0.0005 | 0.6535 |
| S1 | +C | 0.6564 | 0.6565 | +0.0001 | 0.6558 |
| S1 | +S | 0.6592 | 0.6590 | -0.0001 | 0.6565 |
| S1 | +T | 0.6585 | 0.6583 | -0.0002 | 0.6558 |
| S1 | HRcov | 0.6584 | 0.6587 | +0.0003 | 0.6558 |
| S2 | SA | 0.2752 | 0.2750 | -0.0002 | 0.2753 |
| S2 | +C | 0.2817 | 0.2819 | +0.0001 | 0.2772 |
| S2 | +S | 0.2835 | 0.2836 | +0.0001 | 0.2739 |
| S2 | +T | 0.2825 | 0.2826 | +0.0001 | 0.2730 |
| S2 | HRcov | 0.2823 | 0.2823 | -0.0000 | 0.2740 |
| S3 | SA | 0.2752 | 0.2752 | -0.0000 | 0.2755 |
| S3 | +C | 0.3005 | 0.3002 | -0.0003 | 0.2773 |
| S3 | +S | 0.3065 | 0.3059 | -0.0006 | 0.3010 |
| S3 | +T | 0.3317 | 0.3311 | -0.0006 | 0.3263 |
| S3 | HRcov | 0.3311 | 0.3311 | -0.0000 | 0.3265 |

| config | HRcov − SA (harness): registered | size-matched | common intersection |
|---|---|---|---|
| S1 | +0.0040 (SMALL) | +0.0047 (SMALL) | +0.0023 (MATCHES) |
| S2 | +0.0071 (BEATS) | +0.0073 (BEATS) | -0.0013 (MATCHES) |
| S3 | +0.0559 (BEATS) | +0.0559 (BEATS) | +0.0510 (BEATS) |

In S3 the trainer's harness rank of the base-tail arms (+T, HRcov) is about 0.027 MRR above the official rank because CognTKE's tail has many tied scores; S1 and S2 agree with the official evaluator to within 0.001. The size-matched comparison stays within one scale and is unaffected.

## 5. Diagnostic: where the intersection losses occur (official ranks, test queries)

| config | test stratum (gold location) | share | SA reg → int | +C reg → int | HRcov reg → int | contribution to HRcov int − reg |
|---|---|---|---|---|---|---|
| S1 | gold in SA and HRcov pools | 0.7283 | 0.8940 → 0.8935 | 0.8942 → 0.8949 | 0.8968 → 0.8949 | -0.0014 |
| S1 | gold in HRcov pool only (base / pattern candidates) | 0.0331 | 0.0061 → 0.0060 | 0.0851 → 0.0535 | 0.1235 → 0.0857 | -0.0012 |
| S1 | gold in SA pool only | 0.0117 | 0.2263 → 0.1791 | 0.0651 → 0.0764 | 0.0288 → 0.0287 | -0.0000 |
| S1 | gold in neither pool | 0.2269 | 0.0019 → 0.0019 | 0.0018 → 0.0018 | 0.0037 → 0.0037 | -0.0000 |
| S2 | gold in SA and HRcov pools | 0.5541 | 0.4910 → 0.4914 | 0.4902 → 0.4924 | 0.4925 → 0.4862 | -0.0035 |
| S2 | gold in HRcov pool only (base / pattern candidates) | 0.0567 | 0.0033 → 0.0032 | 0.1413 → 0.0416 | 0.1531 → 0.0670 | -0.0049 |
| S2 | gold in SA pool only | 0.0373 | 0.0675 → 0.0642 | 0.0186 → 0.0181 | 0.0056 → 0.0050 | -0.0000 |
| S2 | gold in neither pool | 0.3520 | 0.0013 → 0.0013 | 0.0012 → 0.0012 | 0.0018 → 0.0016 | -0.0001 |
| S3 | gold in SA and HRcov pools | 0.5600 | 0.4859 → 0.4863 | 0.4822 → 0.4850 | 0.4871 → 0.4855 | -0.0009 |
| S3 | gold in HRcov pool only (base / pattern candidates) | 0.0747 | 0.0022 → 0.0021 | 0.3721 → 0.0412 | 0.4069 → 0.3570 | -0.0037 |
| S3 | gold in SA pool only | 0.0313 | 0.0785 → 0.0800 | 0.0143 → 0.0141 | 0.0058 → 0.0057 | -0.0000 |
| S3 | gold in neither pool | 0.3339 | 0.0015 → 0.0014 | 0.0014 → 0.0013 | 0.0018 → 0.0018 | -0.0000 |

## 6. Reading

1. Fit populations are not identical across the ladder. All arms start from the same validation fit part, but each keeps the queries whose gold is in its own pool: SA keeps 75.1% (S1) and 58.7% (S2, S3), the base-candidate pool of +C/+S/+T keeps 77.8%, 62.4% and 63.7%, and HRcov keeps 77.5%, 61.9% and 63.4%. +C, +S and +T use identical fit queries, so the steps +S − +C and +T − +S compare like with like; the steps +C − SA and HRcov − +T change the pool and the fit population together.
2. Fewer fit queries alone does not explain it: random subsamples of each arm's own population, of the intersection's size, move every arm by at most 0.0006 MRR and HRcov − SA by at most 0.0007 (harness: S1 +0.0040 → +0.0047, S2 +0.0071 → +0.0073, S3 +0.0559 → +0.0559; calls unchanged).
3. The common-intersection refit removes a specific kind of fit query: those whose gold is reachable only through the base (or pattern) candidates. Without them, the arms that use base candidates lose most of their gain on test queries whose gold is base-only (S2: +C 0.141 → 0.042, HRcov 0.153 → 0.067; S3: +C 0.372 → 0.041, HRcov 0.407 → 0.357), while SA is unchanged; in S2 HRcov also loses on queries whose gold both pools reach (−0.0035 of its −0.0084). Under this restriction HRcov − SA becomes MATCHES in S2 (−0.0013, SA slightly ahead) and in S1 (+0.0023) and stays BEATS in S3 (+0.0238), where base-score features recover most base-only golds even without such examples (+S on base-only golds: 0.409 → 0.355 with CognTKE, against 0.155 → 0.060 with TiRGN in S2).
4. Consequence for the attribution: the base-candidate effect in the ladder (+C − SA) cannot be separated from the training-population change, because the population change is part of the mechanism: admitting base candidates is what admits training queries with base-only golds. The ladder should be described as "base candidates together with the training queries they make usable". The end-to-end comparisons (HRcov vs SA, HRcov vs the external comparator) remain valid as comparisons of complete systems, each trained on the queries its own pool makes usable. The intersection is not a neutral control: by construction it removes the examples that only base-candidate arms can learn from. When every arm is restricted to fit queries whose gold SA can also reach, HRcov no longer beats SA on polecat–TiRGN (S2), while the CognTKE advantage (S3) survives.

## Deviations

1. HRcov was refitted in the v7.8 trainer (arm HRCOV_RE: stored HRcov pool and features, the hcpat_train.py recipe) instead of hcpat_train.py, which has no fit-row option; its own-population re-run reproduces the stored HRcov (official MRR within 0.0001 in all three configurations).
2. Feature standardisation (mean and SD) is recomputed on each refit's own fit rows, as in the registered trainer; the intersection therefore also changes the standardisation slightly.
3. The size-matched control and its comparisons use the trainer's harness ranks, not the official evaluator (to keep GPU use small); in S3 the harness level of base-tail arms is about 0.027 above official (tied CognTKE tail), which affects levels but not within-scale differences.
4. The test strata use the HRcov pool for all base-candidate arms; the +C/+S/+T pool differs from it for at most 2% of fit queries (Jaccard 0.980-0.997).
5. GPU 0 only (never GPU 1), one process at a time, with a free-memory guard before each stage; GPU 0 was otherwise idle for most of the run (a second process of about 0.7 GB appeared briefly).
