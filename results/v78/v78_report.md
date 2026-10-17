# v7.8 standalone ladder — report

Rule `v78_framing_rule.json` (sha256 8e05c67b7ce8…, registered 2026-10-15T17:34:21Z) with `v78_rule_amendment_1.json` (sha256 8fffffcaad75…, registered 2026-10-15T18:54:33Z, before any v7.8 test number). Official py-tgb 2.3.0 evaluator with shipped negatives; learned systems = seed-mean per-query RR over ranker seeds 0–2; paired query Poisson and timestamp-block bootstraps, B = 2000, shared draws, one-sided 95% bounds (L = 5th, U = 95th percentile). BEATS: L > 0 under both and Δ ≥ 0.005; MATCHES: U < 0.005 under both; otherwise SMALL.

## Verdict

**Registered outcome label: F4_CONDITIONAL.** Results differ between the datasets. On tkgl-polecat, HRcov BEATS both comparators. With TiRGN (S2) it beats the matched standalone ranker SA by +0.0072 and the CountTRuCoLa+base blend by +0.0169; with CognTKE (S3) by +0.0288 and +0.0074. On tkgl-smallpedia–TiRGN (S1) HRcov − SA is SMALL (+0.0040; timestamp-block upper bound 0.0055). The CountTRuCoLa+TiRGN blend MATCHES HRcov (+0.0037; upper bounds 0.0042/0.0045), and SA is level with that blend (−0.0003). F3 needs an external MATCHES in ≥ 2 configurations (here only S1). F2 needs a standalone MATCHES in ≥ 2 (none). F1 needs BEATS against both comparators everywhere, which fails on S1. Rule consequence: the conditions under which HRcov helps (dataset, base strength) become the central result, and the general claims are narrowed. The ladder attributes most of the base's contribution on polecat to the base's candidates (+C: S2 +0.0055 of +0.0072, S3 +0.0235 of +0.0288), with base-score features and fusion adding +0.0017/+0.0059 (S1 +0.0029); the base tail order and the utility ordering of the pattern list contribute ≈ 0.

| configuration | standalone comparator | HRcov − standalone: call | external comparator | HRcov − external: call |
|---|---|---|---|---|
| S1 (smallpedia, tirgn) | SA | SMALL | EXT_blend_counttrucola | MATCHES |
| S2 (polecat, tirgn) | SA | BEATS | EXT_blend_counttrucola | BEATS |
| S3 (polecat, cogn) | SA | BEATS | EXT_blend_counttrucola | BEATS |

Sensitivity (registered, not decision-relevant): with the standalone comparator chosen from {SA, SA_GEN} on the selection part, the label is **F4_CONDITIONAL** (calls: S1: SMALL, S2: BEATS, S3: BEATS).

## S1: smallpedia, tirgn (n = 163,172 test queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR | valid-select MRR |
|---|---|---|---|---|---|---|
| frozen base | 0.6181 | 0.5523 | 0.6691 | 0.7239 |  | – |
| HIST (RecB_xf history order alone) | 0.6055 | 0.5308 | 0.6713 | 0.7174 |  | – |
| EdgeReMIND order alone (history tie-break; SA_ERM tail) | 0.6157 | 0.5426 | 0.6838 | 0.7216 |  | – |
| SA (W100, 16 base-free features) | 0.6544 | 0.6060 | 0.6967 | 0.7243 | 0.6541, 0.6544, 0.6548 | 0.6579 |
| SA_ERM (EdgeReMIND top-50 + base-free block, 16 features) | 0.6534 | 0.6047 | 0.6954 | 0.7241 | 0.6535, 0.6534, 0.6531 | 0.6576 |
| SA_GEN (sensitivity: + RecB generator slots) | 0.6541 | 0.6055 | 0.6967 | 0.7244 | 0.6547, 0.6530, 0.6545 | 0.6579 |
| +C (base top-50 candidates) | 0.6553 | 0.6047 | 0.6967 | 0.7291 | 0.6555, 0.6555, 0.6548 | 0.6606 |
| +S (+ base-score features, fusion) | 0.6582 | 0.6079 | 0.6998 | 0.7323 | 0.6578, 0.6583, 0.6584 | 0.6641 |
| +T (+ base tail order) | 0.6585 | 0.6079 | 0.6998 | 0.7322 | 0.6581, 0.6586, 0.6587 | 0.6639 |
| HRcov (stored v7.7, W100) | 0.6584 | 0.6080 | 0.6999 | 0.7318 | 0.6585, 0.6581, 0.6587 | – |
| SA W50 | 0.6535 | 0.6048 | 0.6964 | 0.7229 | 0.6534, 0.6534, 0.6538 | 0.6571 |
| SA W150 | 0.6542 | 0.6055 | 0.6966 | 0.7245 | 0.6540, 0.6541, 0.6545 | 0.6584 |
| HRcov-recipe W50 (refit) | 0.6588 | 0.6086 | 0.7004 | 0.7317 | 0.6586, 0.6586, 0.6593 | 0.6617 |
| HRcov-recipe W150 (refit) | 0.6573 | 0.6061 | 0.6996 | 0.7320 | 0.6548, 0.6585, 0.6586 | 0.6615 |
| REL_LIN (relation offsets) | 0.6516 | 0.5962 | 0.6983 | 0.7317 | 0.6516, 0.6517, 0.6516 | 0.6565 |
| global linear (refit) | 0.6432 | 0.5835 | 0.6939 | 0.7293 | 0.6430, 0.6433, 0.6433 | 0.6488 |
| EXT_alone EdgeReMIND (faithful, s1337) | 0.6130 | 0.5387 | 0.6811 | 0.7190 |  | – |
| EXT_blend EdgeReMIND + base | 0.6354 | 0.5710 | 0.6906 | 0.7286 |  | – |
| EXT_alone CountTRuCoLa (faithful) | 0.6365 | 0.5747 | 0.6812 | 0.7102 |  | – |
| EXT_blend CountTRuCoLa + base | 0.6547 | 0.6033 | 0.6977 | 0.7288 |  | – |

Decision and context comparisons:

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - SA | +0.0040 | [+0.0035, +0.0044] | [+0.0016, +0.0055] | +0.0020 | +0.0075 |
| HRcov - SA_GEN | +0.0043 | [+0.0039, +0.0048] | [+0.0017, +0.0060] | +0.0024 | +0.0075 |
| HRcov - HIST | +0.0529 | [+0.0520, +0.0539] | [+0.0471, +0.0565] | +0.0772 | +0.0144 |
| HRcov - base | +0.0404 | [+0.0396, +0.0411] | [+0.0367, +0.0427] | +0.0557 | +0.0079 |
| SA - HIST | +0.0489 | [+0.0480, +0.0499] | [+0.0449, +0.0517] | +0.0752 | +0.0069 |
| SA - base | +0.0364 | [+0.0355, +0.0372] | [+0.0332, +0.0389] | +0.0537 | +0.0004 |
| HRcov - EXT_alone_edgeremind | +0.0454 | [+0.0445, +0.0462] | [+0.0427, +0.0488] | +0.0692 | +0.0128 |
| SA - EXT_alone_edgeremind | +0.0414 | [+0.0406, +0.0422] | [+0.0382, +0.0461] | +0.0673 | +0.0053 |
| HRcov - EXT_blend_edgeremind | +0.0230 | [+0.0224, +0.0237] | [+0.0213, +0.0243] | +0.0370 | +0.0032 |
| SA - EXT_blend_edgeremind | +0.0190 | [+0.0183, +0.0198] | [+0.0168, +0.0214] | +0.0350 | -0.0043 |
| HRcov - EXT_alone_counttrucola | +0.0219 | [+0.0213, +0.0225] | [+0.0198, +0.0234] | +0.0333 | +0.0216 |
| SA - EXT_alone_counttrucola | +0.0179 | [+0.0173, +0.0185] | [+0.0161, +0.0198] | +0.0313 | +0.0141 |
| HRcov - EXT_blend_counttrucola | +0.0037 | [+0.0033, +0.0042] | [+0.0029, +0.0045] | +0.0047 | +0.0030 |
| SA - EXT_blend_counttrucola | -0.0003 | [-0.0009, +0.0003] | [-0.0021, +0.0022] | +0.0027 | -0.0045 |

External comparators and SA_ERM:

| method | gate | variant | selection-part MRR | selected (a, tau) |
|---|---|---|---|---|
| edgeremind | PASS | alone | 0.6287 | – |
| edgeremind | PASS | blend | 0.6423 | (0.4, 1.0) |
| counttrucola | PASS | alone | 0.6374 | – |
| counttrucola | PASS | blend | 0.6570 | (0.6, 0.5) |

Selected external comparator: **EXT_blend_counttrucola** (variant with the higher selection-part MRR per method, then the better method). Standalone comparator: **SA** (selection-part MRR SA 0.6579, SA_ERM 0.6576).

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - EXT_blend_counttrucola | +0.0037 | [+0.0033, +0.0042] | [+0.0029, +0.0045] | +0.0047 | +0.0030 |
| SA - EXT_blend_counttrucola | -0.0003 | [-0.0009, +0.0003] | [-0.0021, +0.0022] | +0.0027 | -0.0045 |
| HRcov - SA_ERM | +0.0051 | [+0.0046, +0.0055] | [+0.0044, +0.0056] | +0.0032 | +0.0078 |
| SA_ERM - SA | -0.0011 | [-0.0014, -0.0008] | [-0.0032, +0.0003] | -0.0013 | -0.0003 |
| HRcov - ERM_ORDER | +0.0427 | [+0.0419, +0.0435] | [+0.0403, +0.0457] | +0.0654 | +0.0102 |
| HRcov - EXT_alone_edgeremind | +0.0454 | [+0.0445, +0.0462] | [+0.0427, +0.0488] | +0.0692 | +0.0128 |
| HRcov - EXT_blend_edgeremind | +0.0230 | [+0.0224, +0.0237] | [+0.0213, +0.0243] | +0.0370 | +0.0032 |
| HRcov - EXT_alone_counttrucola | +0.0219 | [+0.0213, +0.0225] | [+0.0198, +0.0234] | +0.0333 | +0.0216 |

Ladder (descriptive; each rung adds one base component):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| ladder: +C - SA | +0.0008 | [+0.0005, +0.0011] | [-0.0014, +0.0021] | -0.0013 | +0.0048 |
| ladder: +S - +C | +0.0029 | [+0.0026, +0.0033] | [+0.0023, +0.0034] | +0.0032 | +0.0032 |
| ladder: +T - +S | +0.0003 | [+0.0003, +0.0003] | [+0.0002, +0.0003] | +0.0000 | -0.0001 |
| ladder: HRcov - +T (+P) | -0.0000 | [-0.0003, +0.0002] | [-0.0003, +0.0002] | +0.0000 | -0.0003 |
| HRcov - SA | +0.0040 | [+0.0035, +0.0044] | [+0.0016, +0.0055] | +0.0020 | +0.0075 |

+T reproduces HRcov: |Δ| = 0.0000 (≤ 0.002: True).

Width sweep (descriptive):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| width: SA_W50 - SA | -0.0009 | [-0.0011, -0.0007] | [-0.0014, -0.0005] | -0.0012 | -0.0014 |
| width: SA_W150 - SA | -0.0002 | [-0.0004, -0.0001] | [-0.0006, +0.0001] | -0.0005 | +0.0002 |
| width: HR_W50 - HRcov | +0.0004 | [+0.0002, +0.0006] | [-0.0001, +0.0008] | +0.0007 | -0.0002 |
| width: HR_W150 - HRcov | -0.0011 | [-0.0013, -0.0009] | [-0.0014, -0.0008] | -0.0019 | +0.0001 |
| width: HR_W50 - SA_W50 | +0.0053 | [+0.0049, +0.0057] | [+0.0030, +0.0069] | +0.0038 | +0.0087 |
| HRcov - SA | +0.0040 | [+0.0035, +0.0044] | [+0.0016, +0.0055] | +0.0020 | +0.0075 |
| width: HR_W150 - SA_W150 | +0.0031 | [+0.0027, +0.0036] | [+0.0009, +0.0044] | +0.0006 | +0.0074 |

REL_LIN diagnostic (not decision-relevant):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| REL_LIN: HRcov - REL_LIN | +0.0068 | [+0.0063, +0.0073] | [+0.0057, +0.0077] | +0.0118 | +0.0001 |
| REL_LIN: REL_LIN - REL_LIN_global | +0.0084 | [+0.0079, +0.0089] | [+0.0076, +0.0090] | +0.0127 | +0.0024 |
| REL_LIN: HRcov - REL_LIN_global | +0.0152 | [+0.0146, +0.0158] | [+0.0135, +0.0163] | +0.0245 | +0.0026 |

Checks: base-rank identity with the stored HRcov file: mean |Δ RR| = 0.00e+00 (ok: True); harness − official MRR flagged (> 1e-4): {"EXT_alone_edgeremind": 0.02908, "EXT_alone_counttrucola": 0.13342}.

## S2: polecat, tirgn (n = 532,636 test queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR | valid-select MRR |
|---|---|---|---|---|---|---|
| frozen base | 0.2205 | 0.1429 | 0.2498 | 0.3750 |  | – |
| HIST (RecB_xf history order alone) | 0.1991 | 0.1378 | 0.2222 | 0.3193 |  | – |
| EdgeReMIND order alone (history tie-break; SA_ERM tail) | 0.1655 | 0.1001 | 0.1848 | 0.2961 |  | – |
| SA (W100, 16 base-free features) | 0.2752 | 0.1922 | 0.3137 | 0.4399 | 0.2750, 0.2753, 0.2753 | 0.2824 |
| SA_ERM (EdgeReMIND top-50 + base-free block, 16 features) | 0.2709 | 0.1882 | 0.3089 | 0.4349 | 0.2710, 0.2711, 0.2707 | 0.2767 |
| SA_GEN (sensitivity: + RecB generator slots) | 0.2753 | 0.1925 | 0.3136 | 0.4399 | 0.2752, 0.2757, 0.2752 | 0.2822 |
| +C (base top-50 candidates) | 0.2807 | 0.1954 | 0.3194 | 0.4497 | 0.2808, 0.2808, 0.2805 | 0.2899 |
| +S (+ base-score features, fusion) | 0.2825 | 0.1967 | 0.3214 | 0.4528 | 0.2824, 0.2823, 0.2828 | 0.2929 |
| +T (+ base tail order) | 0.2825 | 0.1967 | 0.3214 | 0.4528 | 0.2824, 0.2823, 0.2828 | 0.2921 |
| HRcov (stored v7.7, W100) | 0.2824 | 0.1968 | 0.3212 | 0.4529 | 0.2825, 0.2823, 0.2823 | – |
| SA W50 | 0.2738 | 0.1924 | 0.3134 | 0.4366 | 0.2741, 0.2734, 0.2739 | 0.2806 |
| SA W150 | 0.2756 | 0.1921 | 0.3139 | 0.4403 | 0.2757, 0.2758, 0.2754 | 0.2829 |
| HRcov-recipe W50 (refit) | 0.2809 | 0.1967 | 0.3205 | 0.4489 | 0.2810, 0.2806, 0.2810 | 0.2906 |
| HRcov-recipe W150 (refit) | 0.2827 | 0.1967 | 0.3212 | 0.4533 | 0.2823, 0.2826, 0.2831 | 0.2926 |
| REL_LIN (relation offsets) | 0.2706 | 0.1849 | 0.3090 | 0.4398 | 0.2706, 0.2704, 0.2707 | 0.2805 |
| global linear (refit) | 0.2702 | 0.1843 | 0.3088 | 0.4396 | 0.2704, 0.2697, 0.2704 | 0.2804 |
| EXT_alone EdgeReMIND (faithful, s1337) | 0.1654 | 0.1001 | 0.1848 | 0.2961 |  | – |
| EXT_blend EdgeReMIND + base | 0.2286 | 0.1498 | 0.2590 | 0.3857 |  | – |
| EXT_alone CountTRuCoLa (faithful) | 0.2557 | 0.1785 | 0.2911 | 0.4082 |  | – |
| EXT_blend CountTRuCoLa + base | 0.2655 | 0.1831 | 0.3017 | 0.4277 |  | – |

Decision and context comparisons:

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - SA | +0.0072 | [+0.0069, +0.0075] | [+0.0066, +0.0078] | +0.0046 | +0.0130 |
| HRcov - SA_GEN | +0.0070 | [+0.0067, +0.0074] | [+0.0064, +0.0077] | +0.0042 | +0.0130 |
| HRcov - HIST | +0.0833 | [+0.0827, +0.0839] | [+0.0816, +0.0850] | +0.0590 | +0.1336 |
| HRcov - base | +0.0619 | [+0.0614, +0.0624] | [+0.0607, +0.0631] | +0.0539 | +0.0779 |
| SA - HIST | +0.0761 | [+0.0756, +0.0767] | [+0.0746, +0.0777] | +0.0544 | +0.1206 |
| SA - base | +0.0547 | [+0.0542, +0.0553] | [+0.0534, +0.0560] | +0.0493 | +0.0649 |
| HRcov - EXT_alone_edgeremind | +0.1170 | [+0.1164, +0.1177] | [+0.1151, +0.1190] | +0.0966 | +0.1568 |
| SA - EXT_alone_edgeremind | +0.1098 | [+0.1092, +0.1105] | [+0.1080, +0.1117] | +0.0921 | +0.1438 |
| HRcov - EXT_blend_edgeremind | +0.0538 | [+0.0533, +0.0543] | [+0.0526, +0.0550] | +0.0470 | +0.0672 |
| SA - EXT_blend_edgeremind | +0.0466 | [+0.0461, +0.0472] | [+0.0454, +0.0479] | +0.0424 | +0.0542 |
| HRcov - EXT_alone_counttrucola | +0.0267 | [+0.0263, +0.0271] | [+0.0259, +0.0275] | +0.0183 | +0.0446 |
| SA - EXT_alone_counttrucola | +0.0195 | [+0.0191, +0.0199] | [+0.0188, +0.0203] | +0.0137 | +0.0317 |
| HRcov - EXT_blend_counttrucola | +0.0169 | [+0.0165, +0.0173] | [+0.0162, +0.0176] | +0.0137 | +0.0251 |
| SA - EXT_blend_counttrucola | +0.0097 | [+0.0093, +0.0101] | [+0.0090, +0.0105] | +0.0091 | +0.0122 |

External comparators and SA_ERM:

| method | gate | variant | selection-part MRR | selected (a, tau) |
|---|---|---|---|---|
| edgeremind | PASS | alone | 0.1755 | – |
| edgeremind | PASS | blend | 0.2438 | (0.4, 4.0) |
| counttrucola | PASS | alone | 0.2647 | – |
| counttrucola | PASS | blend | 0.2757 | (0.6, 2.0) |

Selected external comparator: **EXT_blend_counttrucola** (variant with the higher selection-part MRR per method, then the better method). Standalone comparator: **SA** (selection-part MRR SA 0.2824, SA_ERM 0.2767).

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - EXT_blend_counttrucola | +0.0169 | [+0.0165, +0.0173] | [+0.0162, +0.0176] | +0.0137 | +0.0251 |
| SA - EXT_blend_counttrucola | +0.0097 | [+0.0093, +0.0101] | [+0.0090, +0.0105] | +0.0091 | +0.0122 |
| HRcov - SA_ERM | +0.0114 | [+0.0111, +0.0118] | [+0.0108, +0.0121] | +0.0086 | +0.0179 |
| SA_ERM - SA | -0.0043 | [-0.0045, -0.0041] | [-0.0047, -0.0039] | -0.0040 | -0.0050 |
| HRcov - ERM_ORDER | +0.1169 | [+0.1163, +0.1176] | [+0.1150, +0.1189] | +0.0966 | +0.1568 |
| HRcov - EXT_alone_edgeremind | +0.1170 | [+0.1164, +0.1177] | [+0.1151, +0.1190] | +0.0966 | +0.1568 |
| HRcov - EXT_blend_edgeremind | +0.0538 | [+0.0533, +0.0543] | [+0.0526, +0.0550] | +0.0470 | +0.0672 |
| HRcov - EXT_alone_counttrucola | +0.0267 | [+0.0263, +0.0271] | [+0.0259, +0.0275] | +0.0183 | +0.0446 |

Ladder (descriptive; each rung adds one base component):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| ladder: +C - SA | +0.0055 | [+0.0053, +0.0057] | [+0.0051, +0.0059] | +0.0032 | +0.0098 |
| ladder: +S - +C | +0.0017 | [+0.0015, +0.0019] | [+0.0014, +0.0021] | +0.0013 | +0.0031 |
| ladder: +T - +S | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| ladder: HRcov - +T (+P) | -0.0001 | [-0.0002, +0.0000] | [-0.0003, +0.0000] | +0.0001 | +0.0001 |
| HRcov - SA | +0.0072 | [+0.0069, +0.0075] | [+0.0066, +0.0078] | +0.0046 | +0.0130 |

+T reproduces HRcov: |Δ| = 0.0001 (≤ 0.002: True).

Width sweep (descriptive):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| width: SA_W50 - SA | -0.0014 | [-0.0015, -0.0013] | [-0.0015, -0.0013] | +0.0002 | -0.0033 |
| width: SA_W150 - SA | +0.0004 | [+0.0003, +0.0005] | [+0.0003, +0.0005] | -0.0001 | +0.0004 |
| width: HR_W50 - HRcov | -0.0015 | [-0.0016, -0.0014] | [-0.0017, -0.0013] | -0.0001 | -0.0040 |
| width: HR_W150 - HRcov | +0.0003 | [+0.0002, +0.0004] | [+0.0002, +0.0004] | -0.0001 | +0.0005 |
| width: HR_W50 - SA_W50 | +0.0071 | [+0.0068, +0.0074] | [+0.0065, +0.0077] | +0.0043 | +0.0123 |
| HRcov - SA | +0.0072 | [+0.0069, +0.0075] | [+0.0066, +0.0078] | +0.0046 | +0.0130 |
| width: HR_W150 - SA_W150 | +0.0071 | [+0.0068, +0.0074] | [+0.0065, +0.0077] | +0.0045 | +0.0130 |

REL_LIN diagnostic (not decision-relevant):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| REL_LIN: HRcov - REL_LIN | +0.0118 | [+0.0115, +0.0121] | [+0.0113, +0.0123] | +0.0118 | +0.0131 |
| REL_LIN: REL_LIN - REL_LIN_global | +0.0004 | [+0.0003, +0.0005] | [+0.0003, +0.0005] | +0.0006 | +0.0002 |
| REL_LIN: HRcov - REL_LIN_global | +0.0122 | [+0.0119, +0.0125] | [+0.0117, +0.0127] | +0.0124 | +0.0133 |

Checks: base-rank identity with the stored HRcov file: mean |Δ RR| = 1.29e-08 (ok: True); harness − official MRR flagged (> 1e-4): {"EXT_alone_edgeremind": 0.00034, "EXT_alone_counttrucola": 0.00184}.

## S3: polecat, cogn (n = 532,636 test queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR | valid-select MRR |
|---|---|---|---|---|---|---|
| frozen base | 0.2914 | 0.2049 | 0.3316 | 0.4603 |  | – |
| HIST (RecB_xf history order alone) | 0.1991 | 0.1378 | 0.2222 | 0.3193 |  | – |
| EdgeReMIND order alone (history tie-break; SA_ERM tail) | 0.1655 | 0.1001 | 0.1848 | 0.2961 |  | – |
| SA (W100, 16 base-free features) | 0.2752 | 0.1922 | 0.3137 | 0.4399 | 0.2750, 0.2753, 0.2753 | 0.2824 |
| SA_ERM (EdgeReMIND top-50 + base-free block, 16 features) | 0.2709 | 0.1882 | 0.3089 | 0.4349 | 0.2710, 0.2711, 0.2707 | 0.2767 |
| SA_GEN (sensitivity: + RecB generator slots) | 0.2753 | 0.1925 | 0.3136 | 0.4399 | 0.2752, 0.2757, 0.2752 | 0.2822 |
| +C (base top-50 candidates) | 0.2987 | 0.2135 | 0.3373 | 0.4661 | 0.2987, 0.2988, 0.2987 | 0.3103 |
| +S (+ base-score features, fusion) | 0.3046 | 0.2185 | 0.3442 | 0.4745 | 0.3041, 0.3049, 0.3050 | 0.3158 |
| +T (+ base tail order) | 0.3046 | 0.2185 | 0.3442 | 0.4745 | 0.3040, 0.3048, 0.3049 | 0.3410 |
| HRcov (stored v7.7, W100) | 0.3040 | 0.2178 | 0.3438 | 0.4741 | 0.3042, 0.3042, 0.3036 | – |
| SA W50 | 0.2738 | 0.1924 | 0.3134 | 0.4366 | 0.2741, 0.2734, 0.2739 | 0.2806 |
| SA W150 | 0.2756 | 0.1921 | 0.3139 | 0.4403 | 0.2757, 0.2758, 0.2754 | 0.2829 |
| HRcov-recipe W50 (refit) | 0.3038 | 0.2183 | 0.3436 | 0.4735 | 0.3036, 0.3033, 0.3045 | 0.3406 |
| HRcov-recipe W150 (refit) | 0.3048 | 0.2182 | 0.3445 | 0.4755 | 0.3050, 0.3045, 0.3049 | 0.3406 |
| REL_LIN (relation offsets) | 0.2988 | 0.2129 | 0.3387 | 0.4681 | 0.2988, 0.2989, 0.2987 | 0.3357 |
| global linear (refit) | 0.2987 | 0.2128 | 0.3387 | 0.4681 | 0.2987, 0.2987, 0.2987 | 0.3356 |
| EXT_alone EdgeReMIND (faithful, s1337) | 0.1654 | 0.1001 | 0.1848 | 0.2961 |  | – |
| EXT_blend EdgeReMIND + base | 0.2929 | 0.2048 | 0.3331 | 0.4653 |  | – |
| EXT_alone CountTRuCoLa (faithful) | 0.2557 | 0.1785 | 0.2911 | 0.4082 |  | – |
| EXT_blend CountTRuCoLa + base | 0.2966 | 0.2113 | 0.3347 | 0.4653 |  | – |

Decision and context comparisons:

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - SA | +0.0288 | [+0.0283, +0.0292] | [+0.0279, +0.0297] | +0.0256 | +0.0342 |
| HRcov - SA_GEN | +0.0287 | [+0.0282, +0.0291] | [+0.0278, +0.0295] | +0.0253 | +0.0342 |
| HRcov - HIST | +0.1049 | [+0.1042, +0.1056] | [+0.1031, +0.1067] | +0.0801 | +0.1548 |
| HRcov - base | +0.0126 | [+0.0123, +0.0128] | [+0.0121, +0.0130] | +0.0130 | +0.0138 |
| SA - HIST | +0.0761 | [+0.0756, +0.0767] | [+0.0746, +0.0777] | +0.0544 | +0.1206 |
| SA - base | -0.0162 | [-0.0168, -0.0157] | [-0.0173, -0.0151] | -0.0127 | -0.0204 |
| HRcov - EXT_alone_edgeremind | +0.1386 | [+0.1379, +0.1393] | [+0.1366, +0.1407] | +0.1177 | +0.1780 |
| SA - EXT_alone_edgeremind | +0.1098 | [+0.1092, +0.1105] | [+0.1080, +0.1117] | +0.0921 | +0.1438 |
| HRcov - EXT_blend_edgeremind | +0.0111 | [+0.0108, +0.0114] | [+0.0106, +0.0117] | +0.0130 | +0.0088 |
| SA - EXT_blend_edgeremind | -0.0177 | [-0.0182, -0.0171] | [-0.0187, -0.0166] | -0.0126 | -0.0254 |
| HRcov - EXT_alone_counttrucola | +0.0483 | [+0.0478, +0.0488] | [+0.0473, +0.0492] | +0.0394 | +0.0659 |
| SA - EXT_alone_counttrucola | +0.0195 | [+0.0191, +0.0199] | [+0.0188, +0.0203] | +0.0137 | +0.0317 |
| HRcov - EXT_blend_counttrucola | +0.0074 | [+0.0071, +0.0076] | [+0.0070, +0.0078] | +0.0065 | +0.0088 |
| SA - EXT_blend_counttrucola | -0.0214 | [-0.0219, -0.0209] | [-0.0224, -0.0204] | -0.0191 | -0.0254 |

External comparators and SA_ERM:

| method | gate | variant | selection-part MRR | selected (a, tau) |
|---|---|---|---|---|
| edgeremind | PASS | alone | 0.1755 | – |
| edgeremind | PASS | blend | 0.3045 | (0.1, 4.0) |
| counttrucola | PASS | alone | 0.2647 | – |
| counttrucola | PASS | blend | 0.3073 | (0.3, 4.0) |

Selected external comparator: **EXT_blend_counttrucola** (variant with the higher selection-part MRR per method, then the better method). Standalone comparator: **SA** (selection-part MRR SA 0.2824, SA_ERM 0.2767).

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| HRcov - EXT_blend_counttrucola | +0.0074 | [+0.0071, +0.0076] | [+0.0070, +0.0078] | +0.0065 | +0.0088 |
| SA - EXT_blend_counttrucola | -0.0214 | [-0.0219, -0.0209] | [-0.0224, -0.0204] | -0.0191 | -0.0254 |
| HRcov - SA_ERM | +0.0331 | [+0.0326, +0.0335] | [+0.0321, +0.0340] | +0.0296 | +0.0392 |
| SA_ERM - SA | -0.0043 | [-0.0045, -0.0041] | [-0.0047, -0.0039] | -0.0040 | -0.0050 |
| HRcov - ERM_ORDER | +0.1385 | [+0.1378, +0.1392] | [+0.1365, +0.1406] | +0.1177 | +0.1780 |
| HRcov - EXT_alone_edgeremind | +0.1386 | [+0.1379, +0.1393] | [+0.1366, +0.1407] | +0.1177 | +0.1780 |
| HRcov - EXT_blend_edgeremind | +0.0111 | [+0.0108, +0.0114] | [+0.0106, +0.0117] | +0.0130 | +0.0088 |
| HRcov - EXT_alone_counttrucola | +0.0483 | [+0.0478, +0.0488] | [+0.0473, +0.0492] | +0.0394 | +0.0659 |

Ladder (descriptive; each rung adds one base component):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| ladder: +C - SA | +0.0235 | [+0.0231, +0.0239] | [+0.0227, +0.0243] | +0.0213 | +0.0262 |
| ladder: +S - +C | +0.0059 | [+0.0056, +0.0061] | [+0.0055, +0.0063] | +0.0050 | +0.0084 |
| ladder: +T - +S | -0.0000 | [-0.0000, -0.0000] | [-0.0001, -0.0000] | +0.0000 | +0.0000 |
| ladder: HRcov - +T (+P) | -0.0006 | [-0.0007, -0.0005] | [-0.0007, -0.0005] | -0.0007 | -0.0004 |
| HRcov - SA | +0.0288 | [+0.0283, +0.0292] | [+0.0279, +0.0297] | +0.0256 | +0.0342 |

+T reproduces HRcov: |Δ| = 0.0006 (≤ 0.002: True).

Width sweep (descriptive):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| width: SA_W50 - SA | -0.0014 | [-0.0015, -0.0013] | [-0.0015, -0.0013] | +0.0002 | -0.0033 |
| width: SA_W150 - SA | +0.0004 | [+0.0003, +0.0005] | [+0.0003, +0.0005] | -0.0001 | +0.0004 |
| width: HR_W50 - HRcov | -0.0002 | [-0.0003, -0.0001] | [-0.0003, -0.0001] | +0.0004 | -0.0006 |
| width: HR_W150 - HRcov | +0.0008 | [+0.0007, +0.0009] | [+0.0007, +0.0009] | +0.0004 | +0.0014 |
| width: HR_W50 - SA_W50 | +0.0300 | [+0.0296, +0.0305] | [+0.0291, +0.0309] | +0.0258 | +0.0370 |
| HRcov - SA | +0.0288 | [+0.0283, +0.0292] | [+0.0279, +0.0297] | +0.0256 | +0.0342 |
| width: HR_W150 - SA_W150 | +0.0292 | [+0.0287, +0.0296] | [+0.0283, +0.0301] | +0.0261 | +0.0352 |

REL_LIN diagnostic (not decision-relevant):

| comparison | Δ MRR | query one-sided 95% [L, U] | timestamp-block [L, U] | Δ H@1 | Δ H@10 |
|---|---|---|---|---|---|
| REL_LIN: HRcov - REL_LIN | +0.0052 | [+0.0050, +0.0054] | [+0.0049, +0.0056] | +0.0049 | +0.0060 |
| REL_LIN: REL_LIN - REL_LIN_global | +0.0001 | [+0.0000, +0.0001] | [+0.0000, +0.0001] | +0.0001 | -0.0000 |
| REL_LIN: HRcov - REL_LIN_global | +0.0053 | [+0.0051, +0.0055] | [+0.0049, +0.0056] | +0.0050 | +0.0060 |

Checks: base-rank identity with the stored HRcov file: mean |Δ RR| = 0.00e+00 (ok: True); harness − official MRR flagged (> 1e-4): {"HRcov": 0.00159, "base(stored)": 0.03162, "base": 0.03162, "+T": 0.00159, "HR_W50": 0.00283, "HR_W150": 0.00113, "REL_LIN": 0.00164, "REL_LIN_global": 0.00164, "EXT_alone_edgeremind": 0.00034, "EXT_blend_edgeremind": 0.00014, "EXT_alone_counttrucola": 0.00184, "EXT_blend_counttrucola": 0.00047}.

## Implementation checks

- Pattern lists: the re-derived HRcov utility block equals the stored HRcov pattern block in 100% of validation and test queries (S1, S2, S3), and the 5 pattern features are identical (max |diff| 0).
- Evidence features recomputed with the HRcov code equal the stored float16 features to rounding (max |diff| 0.0039).
- History order (RecB_xf) and EdgeReMIND order: the top-75 and r_tail are identical between the order pass and the pool-assembly pass (fraction 1.0 in every configuration).
- Tail-order identity: re-ranking with the tail scores reproduces the tail rank for 100% of test queries in every SA, SA_ERM and base-tail pool. Exception: CognTKE HU75 0.99976, because of ties in the CognTKE base order.
- Official vs harness: the per-batch Evaluator vs formula deviation is 0 in every pass, and history-tail systems (HIST, SA, SA_GEN, SA_ERM, +C, +S) have identical official and harness ranks. Seed-max |harness − official| MRR above 1e-4: S1: EXT_alone_counttrucola 0.13342, EXT_alone_edgeremind 0.02908; S2: EXT_alone_counttrucola 0.00184, EXT_alone_edgeremind 0.00034; S3: base(stored) 0.03162, base 0.03162, HR_W50 0.00283, EXT_alone_counttrucola 0.00184, REL_LIN 0.00164, REL_LIN_global 0.00164, HRcov 0.00159, +T 0.00159, HR_W150 0.00113, EXT_blend_counttrucola 0.00047, EXT_alone_edgeremind 0.00034, EXT_blend_edgeremind 0.00014. Cause: the harness rank is strict (only scores above the gold count), while the official rank averages ties. For S3 every base-tail system (base, HRcov, +T, HR_W50, HR_W150, REL_LIN, REL_LIN_global) inherits CognTKE's large tied tail: unreached entities share one log p, so golds ranked in the tail sit in tie blocks. The gap is largest for the base row (0.032) and 0.001–0.003 for the re-rankers. The external 'alone' scorers have tied floors (CountTRuCoLa scores 0 outside its top-200; EdgeReMIND outside its top-1000). This gives the large S1 CountTRuCoLa gap of 0.133 and the EdgeReMIND gap of 0.029. S1/S2 TiRGN base-tail systems and all blends with TiRGN have no gap above 1e-4. All decisions use the official tie-averaged ranks, as registered.
- Base-rank identity with the stored HRcov official files: S1 exact; S2 mean |ΔRR| 1.3e-8 (0.011% of queries differ, TiRGN replay); S3 exact. The external-blend passes reproduce the same base rows (S1/S3 exact, S2 3.1e-8).
- Ladder reproduction: +T vs stored HRcov |ΔMRR| = 4.9e-5 (S1), 1.0e-4 (S2), 6.0e-4 (S3); limit 0.002.
- EXT_alone with the amendment floor (minimum listed score) agrees with the sibling's full-vector official MRR within 7e-5 (EdgeReMIND smallpedia 0.613039 vs 0.613084, polecat 0.165385 vs 0.165455; CountTRuCoLa smallpedia 0.636524 vs 0.636580, polecat identical).
- External dumps: harness row order, gold identity and keys checked by the sibling (MANIFEST.json, all 8 entries PASS) and re-asserted in v78_ext.py / v78_hist.py.

## Deviations

- Reproduction gates were evaluated by this track from the sibling's protocol note (v78_external_protocol_note.json) and its meta.json numbers. The sibling had not yet written a formal gate verdict: MANIFEST.json covers dump integrity only, and EdgeReMIND seeds 1338-1341 were still running. EdgeReMIND: seed-1337 faithful test MRR 0.6148 vs published 0.6136 ± 0.0017. CountTRuCoLa: official 0.6366 vs published 0.644. Both are within the 0.01 tolerance (PASS).
- EXT_adapted (optional) was not run. It was not supplied by the sibling and could not be built cheaply in this track, so it is 'not assessed (optional, not run)'. The external comparator was therefore chosen among {alone, blend}.
- serverA SSH was unreachable for about 11 h (VPN). The detached polecat job kept running and completed normally, so no restart or checkpoint resume was needed. Outputs above 100 MB were downloaded separately.
- Smoke runs (2,000-3,000 queries) were inspected only for consistency fields (harness−official differences, rank equality, identity); no smoke metric was read.
- EdgeReMIND scores outside its top-1000 take the floor (amendment). Its tail is exactly constant in < 1% of queries, and the effect on MRR is ≤ 7e-5 (see checks). CountTRuCoLa's released code keeps only the top-200 entities per query on large datasets; all others score 0, which is the method's own convention.
- S1 'additionally 3 seeds': every learned arm uses ranker seeds 0-2. No additional base seeds were trained (the rule ties seeds to the HRcov runs, which use one base run).

## Files

- v78_rule_amendment_1.json (+ _registered.json)
- v78_results.json
- v78_verdict.json
- official per-query arrays: official_test_{smallpedia,polecat}_v78_hist.npz, official_test_{ds}_v78_{tirgn,cogn}.npz (saved as checkpoints); EXT / SA_ERM arrays on serverA under tracks/hardening/v78/<ds>/
- code: v78_common, v78_hist, v78_lists, v78_lookup, v78_pools, v78_train, v78_official, v78_ext, v78_chain.sh, v78_ext_chain.sh (serverA tracks/hardening/v78/code_*), v78_agg.py, v78_report.py