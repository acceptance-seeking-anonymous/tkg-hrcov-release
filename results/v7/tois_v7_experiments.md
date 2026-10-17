# TOIS v7 experiments: consolidated results (Sections 2-4)

All test numbers are **pre-specified later analyses**: rule `tois_v7_rule.json` (registered 2026-10-14T10:13:36Z, sha256 bfef91681490...) was registered before any v7 test number was read. Default bases are TiRGN on tkgl-polecat and on tkgl-smallpedia. B4 and C3 use the tkgl-icews Stage B setting (RE-GCN and LogCL bases, full test set of 4,651,378 queries). Evaluator: official py-tgb for the TGB datasets; harness ranks for ICEWS18/GDELT (A items only). Statistics: paired Poisson(1) query bootstrap and timestamp-block bootstrap, B = 2000, seed 0, shared draws within an item. A difference is called present only if the one-sided lower bound L > 0 under both bootstraps (Holm within items with several primaries). A difference ≥ 0.005 is called excluded only if the timestamp-block one-sided bound is inside ±0.005. MDE_80 = 2.486 × SE_ts (sensitivity only). Learned arms: seeds 0-2, per-query RR averaged over seeds.

## Summary

| item | result (polecat / smallpedia unless stated) | call |
|---|---|---|
| B1 MLP64x2 − LIN | +0.01217 [+0.01158, +0.01278] / +0.01522 [+0.01313, +0.01653] (ts CI) | present (+) / present (+) |
| B1 MLP64x2 − QBLEND | +0.05210 [+0.05079, +0.05347] / +0.03082 [+0.02677, +0.03327] (ts CI) | present (+) / present (+) |
| B2 PT_common − RB_common | +0.00341 [+0.00306, +0.00376] / +0.00007 [-0.00018, +0.00037] (ts CI) | present (+); gain ≥ 0.005 excluded / not established; bounded within ±0.005 |
| B3 FULL − NB_NF | +0.00183 [+0.00147, +0.00219] / +0.00217 [+0.00149, +0.00292] (ts CI) | present (+); gain ≥ 0.005 excluded / present (+); gain ≥ 0.005 excluded |
| C1 FULL − noDyadLR | +0.00015 [-0.00001, +0.00030] / -0.00006 [-0.00036, +0.00023] (ts CI) | present (+); gain ≥ 0.005 excluded / not established; bounded within ±0.005 |
| A5 UNMASKED - FULL (polecat) | +0.00002 [-0.00011, +0.00015] | not established; bounded within ±0.005 |
| A5 MULTIPOS - FULL (polecat) | -0.00034 [-0.00048, -0.00019] | present (−); loss ≥ 0.005 excluded |
| C2 UTIL12 - CONF12 | -0.00036 [-0.00052, -0.00021] / +0.00004 [-0.00025, +0.00036] | present (−); loss ≥ 0.005 excluded / not established; bounded within ±0.005 |
| C2 UTIL12 - UTIL1 | +0.00237 [+0.00201, +0.00271] / +0.00039 [-0.00001, +0.00082] | present (+); gain ≥ 0.005 excluded / not established; bounded within ±0.005 |
| B4 HRCOV200 (RE-GCN) - G1 (tkgl-icews) | +0.04001 [+0.03925, +0.04075] | present (+) |
| B4 HRCOV150 (LogCL) - G1 (tkgl-icews) | +0.03912 [+0.03844, +0.03982] | present (+) |
| C3 RE-GCN-RN base - RE-GCN base [RN effect on the base] (tkgl-icews, partial) | +0.03438 [+0.03300, +0.03585] | present (+) |
| C3 HRCOV200_RN - HRCOV200 [RN effect on the re-ranker features, harness] (tkgl-icews, partial) | +0.00938 [+0.00894, +0.00986] | present (+) |
| C3 HRCOV200 - RE-GCN base [gain without RN] (tkgl-icews, partial) | +0.10063 [+0.09952, +0.10179] | present (+) |
| C3 HRCOV200_RN - RE-GCN-RN base [HRCOV with RN features vs the RN base] (tkgl-icews, partial) | +0.07564 [+0.07513, +0.07612] | present (+) |
| C4 second TiRGN polecat seed | not run | declared cut |


## Figures

!B/C summary: (a) ΔMRR with timestamp-block 95% CI (thick) and query-bootstrap 95% CI (white inner line) for every B/C/A5 comparison on the two default TiRGN bases; (b) HRCOV_PAT − G1 decomposition (base + pattern features and fusion on a common pool; candidate source with common inputs; remainder = pool width 100 vs 150, generator features, tail order), ts 95% CI; (c) tkgl-icews Stage B test MRR, official evaluator.

A-item figures from the earlier v7 delivery: `a6_temporal_stability.png`, `a7_gain_by_base_confidence.png`. Per-comparison rows for all items: `tois_v7_experiments.csv`.

## A items (reanalysis of stored outputs)

### A1 Membership and loss decomposition (`a1_membership_decomposition.json`)

| dataset | base MRR | disagreements stored list vs rank ≤ 50 | in-pool share v1 (stored) / v2 (rank ≤ 50) | out-of-pool share of loss v1 / v2 |
|---|---|---|---|---|
| polecat | 0.22049 | 456 (0.086%; in list & rank > 50: 0) | 0.5317 / 0.5326 | 0.599 / 0.598 |
| smallpedia | 0.61806 | 1,327 (0.813%; in list & rank > 50: 0) | 0.7574 / 0.7655 | 0.631 / 0.612 |
| ICEWS18 | 0.33701 | 135 (0.136%; in list & rank > 50: 0) | 0.7413 / 0.7427 | 0.388 / 0.386 |
| GDELT | 0.21670 | 899 (0.147%; in list & rank > 50: 0) | 0.6215 / 0.6229 | 0.480 / 0.478 |

Interpretation: the stored unfiltered top-50 never contains a gold with filtered rank > 50; the two pool definitions differ by ≤ 0.81% of queries, and the out-of-pool loss share is 0.39-0.63 under either definition.

### A2 Exact oracle (`a2_exact_oracle.json`)

| dataset | T50 recall / oracle | T100 | U50 | POOL_sel | HRCOV_PAT pool (mean size) | RB pool (mean size) | realised HRCOV_PAT / PT oracle |
|---|---|---|---|---|---|---|---|
| polecat | 0.5317 / 0.5333 | 0.5906 / 0.5913 | 0.5803 / 0.5812 | 0.6343 / 0.6347 | 0.6107 / 0.6115 (94.3) | 0.5899 / 0.5907 (99.2) | 0.28238 (46%) |
| smallpedia | 0.7574 / 0.7589 | 0.7819 / 0.7825 | 0.7682 / 0.7691 | 0.7942 / 0.7946 | 0.7614 / 0.7621 (82.5) | 0.7714 / 0.7720 (95.6) | 0.65842 (86%) |
| ICEWS18 | 0.7413 / 0.7430 | 0.8098 / 0.8104 | 0.7705 / 0.7716 | 0.8340 / 0.8345 | 0.7831 / 0.7840 (98.1) | 0.7821 / 0.7830 (99.5) | 0.37088 (47%) |
| GDELT | 0.6215 / 0.6242 | 0.7320 / 0.7332 | 0.6975 / 0.6992 | 0.7871 / 0.7878 | 0.6778 / 0.6793 (99.9) | 0.7325 / 0.7336 (99.8) | 0.26933 (40%) |

Interpretation: the exact oracle adds only 0.0003-0.0027 over recall; HRCOV_PAT realises 40-86% of its own pool's oracle.

### A3 Hits@k (`a3_hits.json`)

| run | system | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| tkgl-polecat | TiRGN | base | 0.22049 | 0.1429 | 0.2498 | 0.3750 |
| tkgl-polecat | TiRGN | BLEND | 0.23481 | 0.1576 | 0.2649 | 0.3879 |
| tkgl-polecat | TiRGN | POOL | 0.23486 | 0.1576 | 0.2649 | 0.3879 |
| tkgl-polecat | TiRGN | HR | 0.26943 | 0.1895 | 0.3076 | 0.4269 |
| tkgl-polecat | TiRGN | HRCOV_PAT | 0.28238 | 0.1968 | 0.3212 | 0.4529 |
| tkgl-polecat | LogCL | base | 0.17846 | 0.1161 | 0.2016 | 0.3023 |
| tkgl-polecat | LogCL | BLEND | 0.21303 | 0.1440 | 0.2403 | 0.3499 |
| tkgl-polecat | LogCL | POOL | 0.21308 | 0.1440 | 0.2403 | 0.3499 |
| tkgl-polecat | LogCL | HR | 0.23610 | 0.1689 | 0.2709 | 0.3683 |
| tkgl-polecat | LogCL | HRCOV_PAT | 0.27080 | 0.1884 | 0.3092 | 0.4362 |
| tkgl-smallpedia | TiRGN | base | 0.61806 | 0.5523 | 0.6691 | 0.7239 |
| tkgl-smallpedia | TiRGN | BLEND | 0.62746 | 0.5613 | 0.6819 | 0.7276 |
| tkgl-smallpedia | TiRGN | POOL | 0.62740 | 0.5613 | 0.6819 | 0.7277 |
| tkgl-smallpedia | TiRGN | HR | 0.65725 | 0.6067 | 0.6989 | 0.7301 |
| tkgl-smallpedia | TiRGN | HRCOV_PAT | 0.65842 | 0.6080 | 0.6999 | 0.7318 |
| tkgl-smallpedia | RE-GCN (TGB config) | base | 0.59067 | 0.5391 | 0.6266 | 0.6763 |
| tkgl-smallpedia | RE-GCN (TGB config) | BLEND | 0.62494 | 0.5576 | 0.6827 | 0.7248 |
| tkgl-smallpedia | RE-GCN (TGB config) | POOL | 0.62491 | 0.5576 | 0.6827 | 0.7248 |
| tkgl-smallpedia | RE-GCN (TGB config) | HR | 0.64214 | 0.5986 | 0.6794 | 0.7035 |
| tkgl-smallpedia | RE-GCN (TGB config) | HRCOV_PAT | 0.65856 | 0.6111 | 0.6994 | 0.7270 |
| ICEWS18 | TiRGN | base | 0.33701 | 0.2316 | 0.3817 | 0.5425 |
| ICEWS18 | TiRGN | BLEND | 0.34991 | 0.2456 | 0.3952 | 0.5529 |
| ICEWS18 | TiRGN | POOL | 0.34986 | 0.2456 | 0.3952 | 0.5529 |
| ICEWS18 | TiRGN | HR | 0.36700 | 0.2626 | 0.4146 | 0.5702 |
| ICEWS18 | TiRGN | HRCOV_PAT | 0.37088 | 0.2654 | 0.4188 | 0.5763 |
| ICEWS18 | RE-GCN | base | 0.28691 | 0.1933 | 0.3219 | 0.4717 |
| ICEWS18 | RE-GCN | BLEND | 0.32376 | 0.2266 | 0.3662 | 0.5111 |
| ICEWS18 | RE-GCN | POOL | 0.32373 | 0.2266 | 0.3662 | 0.5111 |
| ICEWS18 | RE-GCN | HR | 0.34183 | 0.2455 | 0.3869 | 0.5288 |
| ICEWS18 | RE-GCN | HRCOV_PAT | 0.35922 | 0.2568 | 0.4070 | 0.5591 |
| GDELT | TiRGN | base | 0.21670 | 0.1363 | 0.2327 | 0.3760 |
| GDELT | TiRGN | BLEND | 0.25945 | 0.1726 | 0.2863 | 0.4287 |
| GDELT | TiRGN | POOL | 0.25945 | 0.1726 | 0.2863 | 0.4287 |
| GDELT | TiRGN | HR | 0.26602 | 0.1799 | 0.2943 | 0.4343 |
| GDELT | TiRGN | HRCOV_PAT | 0.26933 | 0.1816 | 0.2977 | 0.4418 |
| GDELT | LogCL | base | 0.21139 | 0.1336 | 0.2272 | 0.3639 |
| GDELT | LogCL | BLEND | 0.25822 | 0.1708 | 0.2854 | 0.4289 |
| GDELT | LogCL | POOL | 0.25821 | 0.1708 | 0.2854 | 0.4289 |
| GDELT | LogCL | HR | 0.25360 | 0.1712 | 0.2793 | 0.4144 |
| GDELT | LogCL | HRCOV_PAT | 0.26590 | 0.1794 | 0.2938 | 0.4359 |

| run | comparison | ΔMRR [ts CI] | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] |
|---|---|---|---|---|---|
| tkgl-polecat | TiRGN | HRCOV_PAT − BLEND | +0.0476 [+0.0465, +0.0488] | +0.0392 [+0.0379, +0.0405] | +0.0563 [+0.0547, +0.0579] | +0.0649 [+0.0634, +0.0666] |
| tkgl-polecat | TiRGN | HRCOV_PAT − POOL | +0.0475 [+0.0464, +0.0487] | +0.0392 [+0.0379, +0.0405] | +0.0563 [+0.0547, +0.0579] | +0.0649 [+0.0634, +0.0666] |
| tkgl-polecat | LogCL | HRCOV_PAT − BLEND | +0.0578 [+0.0564, +0.0592] | +0.0444 [+0.0431, +0.0459] | +0.0689 [+0.0672, +0.0708] | +0.0864 [+0.0844, +0.0884] |
| tkgl-polecat | LogCL | HRCOV_PAT − POOL | +0.0577 [+0.0563, +0.0591] | +0.0444 [+0.0431, +0.0459] | +0.0689 [+0.0672, +0.0708] | +0.0864 [+0.0845, +0.0884] |
| tkgl-smallpedia | TiRGN | HRCOV_PAT − BLEND | +0.0310 [+0.0270, +0.0333] | +0.0467 [+0.0410, +0.0502] | +0.0179 [+0.0151, +0.0199] | +0.0043 [+0.0036, +0.0047] |
| tkgl-smallpedia | TiRGN | HRCOV_PAT − POOL | +0.0310 [+0.0271, +0.0334] | +0.0467 [+0.0410, +0.0502] | +0.0179 [+0.0151, +0.0199] | +0.0042 [+0.0035, +0.0047] |
| tkgl-smallpedia | RE-GCN (TGB config) | HRCOV_PAT − BLEND | +0.0336 [+0.0290, +0.0364] | +0.0535 [+0.0466, +0.0576] | +0.0167 [+0.0136, +0.0190] | +0.0022 [+0.0014, +0.0029] |
| tkgl-smallpedia | RE-GCN (TGB config) | HRCOV_PAT − POOL | +0.0337 [+0.0291, +0.0364] | +0.0535 [+0.0466, +0.0576] | +0.0167 [+0.0136, +0.0190] | +0.0022 [+0.0014, +0.0030] |
| ICEWS18 | TiRGN | HRCOV_PAT − BLEND | +0.0210 [+0.0197, +0.0222] | +0.0198 [+0.0181, +0.0215] | +0.0236 [+0.0219, +0.0253] | +0.0234 [+0.0212, +0.0259] |
| ICEWS18 | TiRGN | HRCOV_PAT − POOL | +0.0210 [+0.0198, +0.0223] | +0.0198 [+0.0181, +0.0215] | +0.0236 [+0.0219, +0.0253] | +0.0234 [+0.0212, +0.0259] |
| ICEWS18 | RE-GCN | HRCOV_PAT − BLEND | +0.0355 [+0.0339, +0.0371] | +0.0303 [+0.0282, +0.0324] | +0.0408 [+0.0389, +0.0429] | +0.0480 [+0.0456, +0.0508] |
| ICEWS18 | RE-GCN | HRCOV_PAT − POOL | +0.0355 [+0.0340, +0.0371] | +0.0303 [+0.0282, +0.0324] | +0.0408 [+0.0389, +0.0429] | +0.0480 [+0.0456, +0.0508] |
| GDELT | TiRGN | HRCOV_PAT − BLEND | +0.0099 [+0.0092, +0.0106] | +0.0090 [+0.0082, +0.0098] | +0.0114 [+0.0105, +0.0123] | +0.0131 [+0.0119, +0.0143] |
| GDELT | TiRGN | HRCOV_PAT − POOL | +0.0099 [+0.0092, +0.0106] | +0.0090 [+0.0082, +0.0098] | +0.0114 [+0.0105, +0.0123] | +0.0131 [+0.0119, +0.0143] |
| GDELT | LogCL | HRCOV_PAT − BLEND | +0.0077 [+0.0070, +0.0084] | +0.0086 [+0.0077, +0.0094] | +0.0084 [+0.0074, +0.0094] | +0.0069 [+0.0057, +0.0082] |
| GDELT | LogCL | HRCOV_PAT − POOL | +0.0077 [+0.0070, +0.0084] | +0.0086 [+0.0077, +0.0094] | +0.0084 [+0.0074, +0.0094] | +0.0069 [+0.0057, +0.0082] |

Interpretation: HRCOV_PAT's gain over the controls is present at H@1, H@3 and H@10 in every run; on smallpedia it is concentrated at H@1.

### A4 Training selection (`a4_training_selection.json`)

| dataset | HR retained / fit population | HRCOV_PAT retained / fit population |
|---|---|---|
| polecat | 77,514 / 139,236 (0.557) | 86,236 / 139,236 (0.619) |
| smallpedia | 81,561 / 105,606 (0.772) | 81,874 / 105,606 (0.775) |
| ICEWS18 | 48,040 / 63,514 (0.756) | 50,235 / 63,514 (0.791) |
| GDELT | 103,905 / 163,232 (0.637) | 112,200 / 163,232 (0.687) |

Interpretation: gold-in-pool retention drops most for no-pair and first-seen queries (polecat HRCOV_PAT: recurrent 0.936 vs first-seen 0.044); breakdowns by stratum and frequency are in the JSON.

### A5 Multi-positive masking

Code reading: other true same-timestamp answers are masked in every trainer (fit loss, calibration and filtered evaluation). The pre-specified sensitivity variants are reported under B/C below.

### A6 Temporal sensitivity (`a6_temporal_sensitivity.json`)

| dataset | comparison | Δ | leave-one-period-out range | moving-block 95% CI | timestamps with Δ < 0 |
|---|---|---|---|---|---|
| polecat | HRCOV_PAT − BLEND | +0.04757 | [+0.04714, +0.04827] | [+0.04632, +0.04878] (l = 7) | 2 / 304 |
| polecat | HRCOV_PAT − POOL | +0.04752 | [+0.04708, +0.04822] | [+0.04627, +0.04873] (l = 7) | 2 / 304 |
| smallpedia | HRCOV_PAT − BLEND | +0.03096 | [+0.03024, +0.03151] | [+0.02491, +0.03364] (l = 3) | 0 / 17 |
| ICEWS18 | HRCOV_PAT − BLEND | +0.02097 | [+0.02071, +0.02133] | [+0.01979, +0.02214] (l = 4) | 0 / 34 |
| GDELT | HRCOV_PAT − BLEND | +0.00988 | [+0.00951, +0.01027] | [+0.00874, +0.01095] (l = 8) | 29 / 348 |

Interpretation: no single period drives the margin; leaving out any period moves the polecat margin by < 0.001.

### A7 Selective gain and loss by base confidence (`a7_selective_gain_loss.json`)

| dataset | base p1 bin | share | MRR base → HRCOV_PAT | Δ vs base [ts CI] | P(override of correct base top-1) |
|---|---|---|---|---|---|
| polecat | [0,0.1) | 0.403 | 0.1188 → 0.1858 | +0.0670 [+0.0652, +0.0687] | 0.3376 |
| polecat | [0.1,0.2) | 0.367 | 0.2345 → 0.2979 | +0.0634 [+0.0614, +0.0655] | 0.2354 |
| polecat | [0.2,0.3) | 0.143 | 0.3126 → 0.3688 | +0.0562 [+0.0539, +0.0587] | 0.1417 |
| polecat | [0.3,0.5) | 0.067 | 0.4393 → 0.4822 | +0.0429 [+0.0403, +0.0455] | 0.0643 |
| polecat | [0.5,0.7) | 0.014 | 0.6026 → 0.6401 | +0.0375 [+0.0317, +0.0436] | 0.0261 |
| polecat | [0.7,0.9) | 0.004 | 0.7196 → 0.7478 | +0.0282 [+0.0204, +0.0362] | 0.0057 |
| polecat | [0.9,1.0] | 0.001 | 0.7292 → 0.7654 | +0.0362 [+0.0104, +0.0598] | 0.0115 |
| smallpedia | [0,0.1) | 0.341 | 0.3132 → 0.3845 | +0.0713 [+0.0640, +0.0764] | 0.2207 |
| smallpedia | [0.1,0.2) | 0.187 | 0.5928 → 0.6589 | +0.0661 [+0.0627, +0.0687] | 0.0691 |
| smallpedia | [0.2,0.3) | 0.106 | 0.6739 → 0.6958 | +0.0219 [+0.0197, +0.0249] | 0.0249 |
| smallpedia | [0.3,0.5) | 0.124 | 0.7606 → 0.7688 | +0.0082 [+0.0072, +0.0098] | 0.0093 |
| smallpedia | [0.5,0.7) | 0.041 | 0.8951 → 0.9031 | +0.0080 [+0.0063, +0.0096] | 0.0063 |
| smallpedia | [0.7,0.9) | 0.066 | 0.9683 → 0.9691 | +0.0008 [+0.0001, +0.0013] | 0.0011 |
| smallpedia | [0.9,1.0] | 0.136 | 0.9904 → 0.9905 | +0.0001 [-0.0000, +0.0003] | 0.0001 |
| ICEWS18 | [0,0.1) | 0.236 | 0.1644 → 0.2054 | +0.0410 [+0.0375, +0.0447] | 0.3113 |
| ICEWS18 | [0.1,0.2) | 0.344 | 0.2849 → 0.3245 | +0.0396 [+0.0371, +0.0423] | 0.1904 |
| ICEWS18 | [0.2,0.3) | 0.213 | 0.3869 → 0.4207 | +0.0338 [+0.0311, +0.0361] | 0.1118 |
| ICEWS18 | [0.3,0.5) | 0.162 | 0.5216 → 0.5412 | +0.0195 [+0.0169, +0.0224] | 0.0566 |
| ICEWS18 | [0.5,0.7) | 0.035 | 0.7135 → 0.7192 | +0.0057 [+0.0017, +0.0090] | 0.0163 |
| ICEWS18 | [0.7,0.9) | 0.009 | 0.8095 → 0.8122 | +0.0027 [-0.0004, +0.0061] | 0.0015 |
| ICEWS18 | [0.9,1.0] | 0.001 | 0.9091 → 0.9103 | +0.0012 [-0.0012, +0.0041] | 0.0000 |
| GDELT | [0,0.1) | 0.552 | 0.1521 → 0.2030 | +0.0509 [+0.0478, +0.0542] | 0.3000 |
| GDELT | [0.1,0.2) | 0.332 | 0.2609 → 0.3173 | +0.0564 [+0.0531, +0.0599] | 0.1473 |
| GDELT | [0.2,0.3) | 0.085 | 0.3699 → 0.4243 | +0.0544 [+0.0512, +0.0581] | 0.0905 |
| GDELT | [0.3,0.5) | 0.029 | 0.4689 → 0.5081 | +0.0393 [+0.0353, +0.0435] | 0.0670 |
| GDELT | [0.5,0.7) | 0.002 | 0.5691 → 0.5922 | +0.0231 [+0.0125, +0.0338] | 0.0358 |
| GDELT | [0.7,0.9) | 0.000 | 0.5842 → 0.6443 | +0.0601 [-0.0017, +0.1384] | 0.0000 |
| GDELT | [0.9,1.0] | 0.000 | 1.0000 → 1.0000 | +0.0000 [+0.0000, +0.0000] | 0.0000 |

Interpretation: the gain is largest when the base is unsure (p1 < 0.1) and shrinks towards zero for confident queries; overrides of a correct base top-1 occur in 22-34% of such queries at p1 < 0.1 but in at most 1.1% at p1 ≥ 0.7.

## B and C items (new fits; official scoring)

### B1 Ranker form (`b1_ranker_form.json`)

**Interpretation.** Non-linearity adds value on both bases: MLP64x2 - LIN = +0.01217 (polecat) / +0.01522 (smallpedia) and MLP64x2 - QBLEND = +0.05210 / +0.03082, all present under both bootstraps (Holm-adjusted); MLP16 recovers 77% / 53% of the LIN-to-MLP64x2 gap.

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| LIN | 0.27015 | 0.1843 | 0.3088 | 0.4396 | 0.27040, 0.26966, 0.27039 |
| MLP16 | 0.27954 | 0.1936 | 0.3184 | 0.4504 | 0.27961, 0.27887, 0.28012 |
| QBLEND | 0.23023 | 0.1534 | 0.2593 | 0.3824 | 0.22993, 0.23034, 0.23040 |
| HRCOV_PAT (stored) | 0.28238 | 0.1968 | 0.3212 | 0.4529 | 0.28250, 0.28234, 0.28230 |
| BLEND | 0.23481 | 0.1576 | 0.2649 | 0.3879 | - |
| POOL | 0.23486 | 0.1576 | 0.2649 | 0.3879 | - |
| base | 0.22049 | 0.1429 | 0.2498 | 0.3750 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP64x2 - LIN | primary | +0.01217 | [+0.01180, +0.01254] | [+0.01158, +0.01278] | +0.01269 | +0.0123 [+0.0114, +0.0132] | +0.0126 [+0.0118, +0.0134] | +0.0131 [+0.0124, +0.0138] | 0.00077 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| MLP64x2 - QBLEND | primary | +0.05210 | [+0.05154, +0.05268] | [+0.05079, +0.05347] | +0.05320 | +0.0432 [+0.0417, +0.0447] | +0.0621 [+0.0604, +0.0638] | +0.0702 [+0.0686, +0.0720] | 0.00169 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| MLP16 - LIN | secondary | +0.00938 | [+0.00907, +0.00971] | [+0.00885, +0.00989] | +0.00982 | +0.0093 [+0.0085, +0.0100] | +0.0096 [+0.0089, +0.0103] | +0.0109 [+0.0102, +0.0115] | 0.00065 | present (+) |
| MLP64x2 - MLP16 | secondary | +0.00279 | [+0.00261, +0.00298] | [+0.00253, +0.00305] | +0.00301 | +0.0030 [+0.0026, +0.0034] | +0.0030 [+0.0026, +0.0034] | +0.0022 [+0.0019, +0.0025] | 0.00033 | present (+); gain ≥ 0.005 excluded |

**smallpedia** (TiRGN; 163,172 queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| LIN | 0.64320 | 0.5835 | 0.6939 | 0.7293 | 0.64300, 0.64327, 0.64332 |
| MLP16 | 0.65128 | 0.5967 | 0.6970 | 0.7304 | 0.65253, 0.65013, 0.65116 |
| QBLEND | 0.62760 | 0.5610 | 0.6830 | 0.7279 | 0.62754, 0.62763, 0.62762 |
| HRCOV_PAT (stored) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| BLEND | 0.62746 | 0.5613 | 0.6819 | 0.7276 | - |
| POOL | 0.62740 | 0.5613 | 0.6819 | 0.7277 | - |
| base | 0.61806 | 0.5523 | 0.6691 | 0.7239 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP64x2 - LIN | primary | +0.01522 | [+0.01450, +0.01592] | [+0.01313, +0.01653] | +0.01633 | +0.0245 [+0.0210, +0.0267] | +0.0060 [+0.0048, +0.0069] | +0.0026 [+0.0020, +0.0029] | 0.00212 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| MLP64x2 - QBLEND | primary | +0.03082 | [+0.02995, +0.03170] | [+0.02677, +0.03327] | +0.03298 | +0.0470 [+0.0409, +0.0507] | +0.0168 [+0.0142, +0.0186] | +0.0039 [+0.0034, +0.0044] | 0.00408 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| MLP16 - LIN | secondary | +0.00808 | [+0.00755, +0.00861] | [+0.00658, +0.00907] | +0.00893 | +0.0132 [+0.0107, +0.0149] | +0.0031 [+0.0020, +0.0040] | +0.0011 [+0.0006, +0.0015] | 0.00156 | present (+) |
| MLP64x2 - MLP16 | secondary | +0.00714 | [+0.00662, +0.00768] | [+0.00628, +0.00779] | +0.00770 | +0.0113 [+0.0099, +0.0125] | +0.0029 [+0.0024, +0.0033] | +0.0014 [+0.0011, +0.0017] | 0.00093 | present (+) |

Expansion rule: second base added = False (MLP64x2 - LIN: ts CI contains 0.005 = False, opposite signs = False; MLP64x2 - QBLEND: ts CI contains 0.005 = False, opposite signs = False).

Notes: (1) QBLEND (query-conditioned blend on the PT pool, no calibration): polecat best selection epoch was the first epoch for all three seeds; QBLEND is below the full-row valid-selected BLEND on polecat (0.23023 vs 0.23481) and equal on smallpedia (0.62760 vs 0.62746).

### B2 Matched-width pool (pattern prior vs RecB) (`b2_matched_pool.json`)

**Interpretation.** At matched width (W = 100) with common inputs, the pattern-prior pool beats the RecB pool by +0.00341 on polecat (present; gain >= 0.005 excluded) and ties on smallpedia (+0.00007; |delta| >= 0.005 excluded); with generator-specific features the polecat gap widens to +0.00978 (smallpedia +0.00015), i.e. the pattern features, not the candidate set alone, carry most of the polecat difference.

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| PT_common | 0.27601 | 0.1913 | 0.3139 | 0.4437 | 0.27471, 0.27613, 0.27719 |
| RB_common | 0.27260 | 0.1905 | 0.3103 | 0.4343 | 0.27202, 0.27280, 0.27299 |
| PT_gen (= FULL) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| RB_gen | 0.27254 | 0.1905 | 0.3102 | 0.4343 | 0.27188, 0.27226, 0.27349 |
| base | 0.22049 | 0.1429 | 0.2498 | 0.3750 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| PT_common - RB_common | primary | +0.00341 | [+0.00319, +0.00364] | [+0.00306, +0.00376] | +0.00370 | +0.0008 [+0.0005, +0.0011] | +0.0037 [+0.0032, +0.0041] | +0.0094 [+0.0087, +0.0101] | 0.00044 | present (+); gain ≥ 0.005 excluded |
| PT_gen - RB_gen | secondary | +0.00978 | [+0.00947, +0.01011] | [+0.00917, +0.01039] | +0.01032 | +0.0061 [+0.0054, +0.0069] | +0.0112 [+0.0104, +0.0120] | +0.0184 [+0.0175, +0.0194] | 0.00079 | present (+) |

**smallpedia** (TiRGN; 163,172 queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| PT_common | 0.65753 | 0.6068 | 0.6993 | 0.7311 | 0.65736, 0.65779, 0.65745 |
| RB_common | 0.65746 | 0.6067 | 0.6993 | 0.7312 | 0.65730, 0.65767, 0.65742 |
| PT_gen (= FULL) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| RB_gen | 0.65827 | 0.6076 | 0.7001 | 0.7319 | 0.65843, 0.65816, 0.65821 |
| base | 0.61806 | 0.5523 | 0.6691 | 0.7239 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| PT_common - RB_common | primary | +0.00007 | [-0.00020, +0.00036] | [-0.00018, +0.00037] | +0.00032 | +0.0001 [-0.0002, +0.0007] | -0.0000 [-0.0006, +0.0005] | -0.0001 [-0.0004, +0.0003] | 0.00035 | not established; bounded within ±0.005 |
| PT_gen - RB_gen | secondary | +0.00015 | [-0.00016, +0.00047] | [-0.00019, +0.00061] | +0.00052 | +0.0003 [-0.0002, +0.0011] | -0.0002 [-0.0013, +0.0006] | -0.0001 [-0.0008, +0.0006] | 0.00049 | not established; bounded within ±0.005 |

Expansion rule: second base added = False (PT_common - RB_common: ts CI contains 0.005 = False, opposite signs = False).

Notes: (1) Fit rows retained (gold in pool): polecat PT 86,236 vs RB 83,985 of 139,236; smallpedia PT 81,874 vs RB 83,088 of 105,606. (2) Pool recall / exact oracle MRR (A2): polecat PT 0.6107 / 0.6115, RB 0.5899 / 0.5907; smallpedia PT 0.7614 / 0.7621, RB 0.7714 / 0.7720. (3) PT_gen (= FULL re-fit, 25 features) - RB_gen (20 common + 5 RecB features) on polecat = (PT_gen - PT_common) + (PT_common - RB_common) + (RB_common - RB_gen) = +0.00631 +0.00341 +0.00006.

### B3 Common-pool base ablation and HRCOV_PAT − G1 decomposition (`b3_common_pool_base_ablation.json`)

**Interpretation.** On the common PT pool, base features + alpha log p fusion add +0.00183 (polecat) / +0.00217 (smallpedia): present on both, gain >= 0.005 excluded on both. The HRCOV_PAT - G1 gap (+0.03408 / +0.00732, re-fit) decomposes into base-and-pattern contribution on a common pool +0.00863 / +0.00525, candidate source with common inputs +0.00438 / -0.00011, and the remainder (pool width 100 vs 150, generator features, tail order) +0.02108 / +0.00218.

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| NB_F | 0.28039 | 0.1949 | 0.3192 | 0.4497 | 0.28035, 0.28025, 0.28056 |
| B_NF | 0.28213 | 0.1965 | 0.3212 | 0.4524 | 0.28161, 0.28248, 0.28231 |
| NB_NF | 0.28049 | 0.1951 | 0.3195 | 0.4495 | 0.28040, 0.28040, 0.28068 |
| EV_NF_PT | 0.27369 | 0.1900 | 0.3107 | 0.4382 | 0.27384, 0.27354, 0.27370 |
| EV_NF_RB | 0.26932 | 0.1882 | 0.3059 | 0.4279 | 0.26936, 0.26965, 0.26894 |
| G1 | 0.24824 | 0.1780 | 0.2830 | 0.3847 | 0.24871, 0.24773, 0.24829 |
| HRCOV_PAT (stored) | 0.28238 | 0.1968 | 0.3212 | 0.4529 | 0.28250, 0.28234, 0.28230 |
| base | 0.22049 | 0.1429 | 0.2498 | 0.3750 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL - NB_NF | primary | +0.00183 | [+0.00162, +0.00205] | [+0.00147, +0.00219] | +0.00214 | +0.0015 [+0.0010, +0.0020] | +0.0019 [+0.0014, +0.0025] | +0.0032 [+0.0027, +0.0037] | 0.00046 | present (+); gain ≥ 0.005 excluded |
| FULL - NB_F | secondary | +0.00194 | [+0.00173, +0.00215] | [+0.00160, +0.00227] | +0.00222 | +0.0017 [+0.0012, +0.0022] | +0.0022 [+0.0017, +0.0027] | +0.0029 [+0.0025, +0.0034] | 0.00043 | present (+); gain ≥ 0.005 excluded |
| FULL - B_NF | secondary | +0.00019 | [+0.00015, +0.00024] | [+0.00013, +0.00025] | +0.00024 | +0.0002 [+0.0001, +0.0003] | +0.0002 [+0.0001, +0.0003] | +0.0002 [+0.0001, +0.0003] | 0.00007 | present (+); gain ≥ 0.005 excluded |
| FULL - EV_NF_PT [base contribution, common pool] | secondary | +0.00863 | [+0.00829, +0.00896] | [+0.00798, +0.00933] | +0.00924 | +0.0066 [+0.0057, +0.0075] | +0.0106 [+0.0097, +0.0116] | +0.0145 [+0.0136, +0.0153] | 0.00086 | present (+) |
| EV_NF_PT - EV_NF_RB [candidate source, common inputs] | secondary | +0.00438 | [+0.00418, +0.00459] | [+0.00404, +0.00473] | +0.00466 | +0.0018 [+0.0015, +0.0022] | +0.0048 [+0.0044, +0.0052] | +0.0103 [+0.0096, +0.0111] | 0.00044 | present (+); gain ≥ 0.005 excluded |
| EV_NF_RB - G1 [pool width, generator features, tail order] | secondary | +0.02108 | [+0.02066, +0.02149] | [+0.02028, +0.02192] | +0.02178 | +0.0102 [+0.0096, +0.0109] | +0.0230 [+0.0219, +0.0240] | +0.0432 [+0.0417, +0.0447] | 0.00104 | present (+) |
| HRCOV_PAT(stored) - G1 [total] | secondary | +0.03414 | [+0.03363, +0.03468] | [+0.03292, +0.03536] | +0.03517 | +0.0188 [+0.0177, +0.0199] | +0.0382 [+0.0368, +0.0397] | +0.0681 [+0.0662, +0.0702] | 0.00154 | present (+) |
| FULL - G1 [total, re-fit] | secondary | +0.03408 | [+0.03355, +0.03462] | [+0.03288, +0.03534] | +0.03513 | +0.0187 [+0.0175, +0.0198] | +0.0384 [+0.0370, +0.0400] | +0.0680 [+0.0660, +0.0700] | 0.00157 | present (+) |

**smallpedia** (TiRGN; 163,172 queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| NB_F | 0.65788 | 0.6071 | 0.6999 | 0.7314 | 0.65802, 0.65787, 0.65776 |
| B_NF | 0.65845 | 0.6081 | 0.6998 | 0.7318 | 0.65850, 0.65808, 0.65876 |
| NB_NF | 0.65625 | 0.6058 | 0.6977 | 0.7287 | 0.65652, 0.65633, 0.65590 |
| EV_NF_PT | 0.65317 | 0.6015 | 0.6960 | 0.7266 | 0.65278, 0.65323, 0.65350 |
| EV_NF_RB | 0.65327 | 0.6015 | 0.6964 | 0.7266 | 0.65350, 0.65330, 0.65302 |
| G1 | 0.65110 | 0.6028 | 0.6937 | 0.7201 | 0.65123, 0.65119, 0.65087 |
| HRCOV_PAT (stored) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| base | 0.61806 | 0.5523 | 0.6691 | 0.7239 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL - NB_NF | primary | +0.00217 | [+0.00178, +0.00255] | [+0.00149, +0.00292] | +0.00279 | +0.0022 [+0.0008, +0.0037] | +0.0022 [+0.0015, +0.0028] | +0.0031 [+0.0023, +0.0038] | 0.00092 | present (+); gain ≥ 0.005 excluded |
| FULL - NB_F | secondary | +0.00053 | [+0.00022, +0.00085] | [+0.00011, +0.00096] | +0.00090 | +0.0009 [+0.0002, +0.0016] | -0.0001 [-0.0006, +0.0005] | +0.0004 [-0.0000, +0.0009] | 0.00054 | present (+); gain ≥ 0.005 excluded |
| FULL - B_NF | secondary | -0.00003 | [-0.00010, +0.00004] | [-0.00010, +0.00004] | +0.00003 | -0.0001 [-0.0002, +0.0000] | +0.0001 [-0.0000, +0.0001] | +0.0000 [-0.0001, +0.0001] | 0.00008 | not established; bounded within ±0.005 |
| FULL - EV_NF_PT [base contribution, common pool] | secondary | +0.00525 | [+0.00476, +0.00576] | [+0.00441, +0.00615] | +0.00600 | +0.0065 [+0.0050, +0.0081] | +0.0039 [+0.0031, +0.0047] | +0.0053 [+0.0047, +0.0058] | 0.00110 | present (+) |
| EV_NF_PT - EV_NF_RB [candidate source, common inputs] | secondary | -0.00011 | [-0.00039, +0.00016] | [-0.00040, +0.00024] | +0.00018 | +0.0000 [-0.0006, +0.0006] | -0.0005 [-0.0010, +0.0000] | -0.0001 [-0.0004, +0.0002] | 0.00041 | not established; bounded within ±0.005 |
| EV_NF_RB - G1 [pool width, generator features, tail order] | secondary | +0.00218 | [+0.00174, +0.00264] | [+0.00136, +0.00292] | +0.00280 | -0.0013 [-0.0023, -0.0002] | +0.0027 [+0.0016, +0.0037] | +0.0065 [+0.0051, +0.0077] | 0.00097 | present (+); gain ≥ 0.005 excluded |
| HRCOV_PAT(stored) - G1 [total] | secondary | +0.00732 | [+0.00680, +0.00786] | [+0.00617, +0.00850] | +0.00828 | +0.0052 [+0.0036, +0.0069] | +0.0061 [+0.0051, +0.0070] | +0.0117 [+0.0105, +0.0125] | 0.00147 | present (+) |
| FULL - G1 [total, re-fit] | secondary | +0.00732 | [+0.00680, +0.00786] | [+0.00617, +0.00850] | +0.00828 | +0.0052 [+0.0036, +0.0069] | +0.0061 [+0.0051, +0.0070] | +0.0117 [+0.0105, +0.0125] | 0.00147 | present (+) |

Expansion rule: second base added = False (FULL - NB_NF: ts CI contains 0.005 = False, opposite signs = False).

Notes: (1) FULL - EV_NF_PT removes base features 0-7, pattern features 19-23 and the fusion at once (the rule's 'base contribution, common pool' term); FULL - NB_NF removes base features and fusion but keeps pattern features. (2) G1 = stored G1_K150 (RecB-trained top-150 pool, 19 features, official files from serverB). Decomposition terms are exact differences of seed-mean MRR and sum to FULL(re-fit) - G1. (3) G1 official files (from serverB) share the 'rows' array with the TiRGN files exactly; their 'base' column is G1's own generator, the trained RecB (official MRR 0.19922 polecat, 0.60459 smallpedia), not TiRGN, so G1 rows are aligned by 'rows', not by base RR.

### B4 G1 on tkgl-icews Stage B (`b4_g1_tkgl_icews.json`)

**Interpretation.** G1 does not close the gap on tkgl-icews: HRCOV - G1 = +0.04001 (RE-GCN) and +0.03912 (LogCL), present under both bootstraps (Holm). G1 beats both neural bases and both BLEND controls on MRR (e.g. G1 - LogCL BLEND +0.01065), but at H@10 it trails the LogCL base (-0.0096), RE-GCN BLEND (-0.0050) and LogCL BLEND (-0.0198).

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| RE-GCN base | 0.27469 | 0.1767 | 0.3092 | 0.4720 | - |
| RE-GCN BLEND | 0.31060 | 0.2110 | 0.3503 | 0.5063 | - |
| RE-GCN POOL | 0.31053 | 0.2110 | 0.3503 | 0.5063 | - |
| HRCOV200 (RE-GCN) | 0.37531 | 0.2682 | 0.4253 | 0.5870 | 0.37522, 0.37491, 0.37582 |
| LogCL base | 0.31271 | 0.2126 | 0.3537 | 0.5110 | - |
| LogCL BLEND | 0.32465 | 0.2249 | 0.3661 | 0.5212 | - |
| LogCL POOL | 0.32461 | 0.2249 | 0.3661 | 0.5212 | - |
| HRCOV150 (LogCL) | 0.37443 | 0.2668 | 0.4243 | 0.5874 | 0.37533, 0.37400, 0.37395 |
| G1 | 0.33530 | 0.2490 | 0.3824 | 0.5014 | 0.33481, 0.33568, 0.33542 |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| HRCOV200 (RE-GCN) - G1 | primary | +0.04001 | [+0.03981, +0.04021] | [+0.03925, +0.04075] | +0.04062 | +0.0193 [+0.0187, +0.0198] | +0.0430 [+0.0421, +0.0438] | +0.0856 [+0.0842, +0.0872] | 0.00095 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| HRCOV150 (LogCL) - G1 | primary | +0.03912 | [+0.03893, +0.03932] | [+0.03844, +0.03982] | +0.03969 | +0.0179 [+0.0174, +0.0183] | +0.0420 [+0.0411, +0.0428] | +0.0860 [+0.0846, +0.0875] | 0.00088 | present (+) (Holm p query/ts <0.0005/<0.0005) |
| G1 - RE-GCN base | secondary | +0.06062 | [+0.06033, +0.06091] | [+0.05986, +0.06138] | +0.06125 | +0.0723 [+0.0714, +0.0733] | +0.0732 [+0.0723, +0.0741] | +0.0294 [+0.0282, +0.0305] | 0.00096 | present (+) |
| G1 - LogCL base | secondary | +0.02260 | [+0.02233, +0.02286] | [+0.02201, +0.02313] | +0.02305 | +0.0364 [+0.0358, +0.0369] | +0.0287 [+0.0280, +0.0293] | -0.0096 [-0.0107, -0.0085] | 0.00071 | present (+) |
| G1 - RE-GCN BLEND | secondary | +0.02470 | [+0.02446, +0.02496] | [+0.02394, +0.02547] | +0.02536 | +0.0380 [+0.0370, +0.0389] | +0.0321 [+0.0313, +0.0330] | -0.0050 [-0.0059, -0.0041] | 0.00095 | present (+) |
| G1 - LogCL BLEND | secondary | +0.01065 | [+0.01042, +0.01089] | [+0.01013, +0.01119] | +0.01110 | +0.0241 [+0.0234, +0.0247] | +0.0163 [+0.0156, +0.0169] | -0.0198 [-0.0206, -0.0191] | 0.00067 | present (+) |
| HRCOV200 (RE-GCN) - RE-GCN BLEND | context | +0.06471 | [+0.06449, +0.06494] | [+0.06353, +0.06594] | +0.06569 | +0.0572 [+0.0560, +0.0584] | +0.0751 [+0.0737, +0.0765] | +0.0807 [+0.0793, +0.0821] | 0.00151 | present (+) |
| HRCOV150 (LogCL) - LogCL BLEND | context | +0.04978 | [+0.04958, +0.04996] | [+0.04888, +0.05071] | +0.05054 | +0.0419 [+0.0411, +0.0428] | +0.0582 [+0.0572, +0.0594] | +0.0662 [+0.0651, +0.0674] | 0.00115 | present (+) |

Notes: (1) HRCOV200 (RE-GCN) and HRCOV150 (LogCL) are the Stage B claim systems already scored officially (official_test_icews_regcn.npz, official_test_icews_logcl_sys.npz); G1 official per-query arrays: tracks/hardening/v7/b4/official_test_icews_recb_g1.npz on <HOST> (not transferred, ~4.65M x 9 arrays). (2) The trainer's exact test MRR equals the official value per seed (0.33481 / 0.33568 / 0.33542); the optimistic-tie harness column inside the official-scoring log reads 0.347-0.348 because the RecB tail has many exact ties, which the official evaluator averages. Official values are the registered ones. (3) LogCL base RR in the two LogCL official files differs on 0.004% of queries (GPU replay near-ties; mean |dRR| 1.2e-7). Aggregation computed on <HOST> (CPU): v7_b4c3_agg.py. (4) ICEWS18 / GDELT extension (rule: only if resources allow): not run (declared cut).

### C1 Without ev_dyad_lr (`c1_no_dyad_lr.json`)

**Interpretation.** ev_dyad_lr is not load-bearing: FULL - noDyadLR = +0.00015 (polecat; one-sided L > 0 under both bootstraps, i.e. detectable but negligible) and -0.00006 (smallpedia); a contribution >= 0.005 is excluded on both bases (U_ts +0.00028 / +0.00019).

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| noDyadLR | 0.28218 | 0.1965 | 0.3212 | 0.4526 | 0.28195, 0.28240, 0.28218 |
| HRCOV_PAT (stored) | 0.28238 | 0.1968 | 0.3212 | 0.4529 | 0.28250, 0.28234, 0.28230 |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL - noDyadLR | primary | +0.00015 | [+0.00002, +0.00028] | [-0.00001, +0.00030] | +0.00028 | +0.0001 [-0.0001, +0.0004] | +0.0002 [-0.0001, +0.0005] | +0.0001 [-0.0001, +0.0003] | 0.00019 | present (+); gain ≥ 0.005 excluded |

**smallpedia** (TiRGN; 163,172 queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| noDyadLR | 0.65848 | 0.6082 | 0.6997 | 0.7317 | 0.65860, 0.65844, 0.65840 |
| HRCOV_PAT (stored) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL - noDyadLR | primary | -0.00006 | [-0.00026, +0.00013] | [-0.00036, +0.00023] | +0.00019 | -0.0002 [-0.0007, +0.0003] | +0.0002 [-0.0001, +0.0005] | +0.0001 [-0.0001, +0.0004] | 0.00038 | not established; bounded within ±0.005 |

Expansion rule: second base added = False (FULL - noDyadLR: ts CI contains 0.005 = False, opposite signs = False).

Notes: (1) Fused arms: official = exact harness rank to <= 1e-7 MRR (hrank stored alongside rank). (2) Score-alone arms (B_NF, NB_NF, EV_NF_*): official scoring orders the promoted list by the ranker score and breaks exact ties by list position (base top-50 order, then the generator block); the preliminary exact ranks used strict-greater (optimistic) ties. Difference official vs preliminary: polecat EV_NF_PT -0.00012, NB_NF -0.00006; smallpedia EV_NF_PT -0.00183, EV_NF_RB -0.00192, NB_NF -0.00011 (all other arms 0). The registered numbers are the official ones. (3) Base replay: per-query base RR identical to the stored control/HRCOV_PAT files except near-tie blocks (polecat 0.011-0.012% of queries, mean |dRR| < 1e-8; smallpedia exactly identical). (4) FULL re-fit vs stored HRCOV_PAT: smallpedia identical per seed; polecat -0.00005, ts CI [-0.00017, +0.00006] (GPU non-determinism).

### C2 Pattern design (utility vs confidence ordering, 1-hop vs 2-hop) (`c2_pattern_design.json`)

**Interpretation.** The base-mass penalty in the pattern utility is not what makes the pattern prior work: ordering by confidence alone (CONF12) is marginally better on polecat (UTIL12 - CONF12 = -0.00036, present (−), loss ≥ 0.005 excluded; test recall 0.6150 vs 0.6107) and equal on smallpedia (+0.00004, bounded within ±0.005). 2-hop patterns add a small, positive amount over 1-hop only (UTIL12 - UTIL1 = +0.00237 polecat, present, gain ≥ 0.005 excluded; +0.00039 smallpedia, bounded within ±0.005); every variant stays +0.0430 or more above the polecat POOL control.

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| UTIL12 | 0.28232 | 0.1967 | 0.3213 | 0.4525 | 0.28237, 0.28243, 0.28214 |
| CONF12 | 0.28268 | 0.1970 | 0.3215 | 0.4531 | 0.28260, 0.28249, 0.28294 |
| UTIL1 | 0.27995 | 0.1968 | 0.3182 | 0.4442 | 0.28004, 0.27977, 0.28004 |
| UTIL2 | 0.27782 | 0.1930 | 0.3167 | 0.4465 | 0.27816, 0.27727, 0.27804 |
| FULL (HRCOV_PAT re-fit, PT layout) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| HRCOV_PAT (stored) | 0.28238 | 0.1968 | 0.3212 | 0.4529 | 0.28250, 0.28234, 0.28230 |
| base | 0.22049 | 0.1429 | 0.2498 | 0.3750 | - |
| POOL | 0.23486 | 0.1576 | 0.2649 | 0.3879 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| UTIL12 - CONF12 | primary | -0.00036 | [-0.00050, -0.00023] | [-0.00052, -0.00021] | -0.00024 | -0.0003 [-0.0005, -0.0000] | -0.0001 [-0.0004, +0.0001] | -0.0005 [-0.0008, -0.0003] | 0.00019 | present (−); loss ≥ 0.005 excluded (Holm p query/ts 1/1) |
| UTIL12 - UTIL1 | primary | +0.00237 | [+0.00214, +0.00259] | [+0.00201, +0.00271] | +0.00266 | -0.0001 [-0.0006, +0.0003] | +0.0031 [+0.0026, +0.0036] | +0.0083 [+0.0076, +0.0090] | 0.00045 | present (+); gain ≥ 0.005 excluded (Holm p query/ts <0.0005/<0.0005) |
| UTIL12 - UTIL2 | secondary | +0.00449 | [+0.00423, +0.00475] | [+0.00405, +0.00493] | +0.00487 | +0.0038 [+0.0032, +0.0044] | +0.0046 [+0.0040, +0.0053] | +0.0060 [+0.0055, +0.0066] | 0.00056 | present (+); gain ≥ 0.005 excluded |
| UTIL12 - FULL (same candidates, recomputed features, RB-style column layout) | fidelity | -0.00001 | [-0.00013, +0.00010] | [-0.00014, +0.00012] | +0.00010 | +0.0001 [-0.0001, +0.0003] | -0.0001 [-0.0003, +0.0002] | -0.0001 [-0.0003, +0.0001] | 0.00016 | not established; bounded within ±0.005 |
| UTIL12 - POOL | context | +0.04745 | [+0.04690, +0.04802] | [+0.04630, +0.04869] | +0.04845 | +0.0391 [+0.0378, +0.0405] | +0.0564 [+0.0548, +0.0581] | +0.0646 [+0.0630, +0.0664] | 0.00148 | present (+) |
| CONF12 - POOL | context | +0.04782 | [+0.04726, +0.04838] | [+0.04667, +0.04904] | +0.04881 | +0.0394 [+0.0381, +0.0407] | +0.0566 [+0.0549, +0.0582] | +0.0651 [+0.0635, +0.0669] | 0.00150 | present (+) |
| UTIL1 - POOL | context | +0.04509 | [+0.04452, +0.04567] | [+0.04396, +0.04627] | +0.04606 | +0.0392 [+0.0380, +0.0405] | +0.0533 [+0.0517, +0.0549] | +0.0563 [+0.0548, +0.0579] | 0.00145 | present (+) |
| UTIL2 - POOL | context | +0.04296 | [+0.04244, +0.04352] | [+0.04197, +0.04398] | +0.04379 | +0.0354 [+0.0343, +0.0365] | +0.0518 [+0.0504, +0.0532] | +0.0586 [+0.0572, +0.0601] | 0.00126 | present (+) |

**smallpedia** (TiRGN; 163,172 queries, 17 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| UTIL12 | 0.65865 | 0.6083 | 0.6999 | 0.7319 | 0.65811, 0.65891, 0.65893 |
| CONF12 | 0.65861 | 0.6082 | 0.6999 | 0.7321 | 0.65807, 0.65880, 0.65895 |
| UTIL1 | 0.65826 | 0.6078 | 0.7000 | 0.7315 | 0.65818, 0.65825, 0.65834 |
| UTIL2 | 0.65757 | 0.6071 | 0.6993 | 0.7303 | 0.65719, 0.65748, 0.65803 |
| FULL (HRCOV_PAT re-fit, PT layout) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| HRCOV_PAT (stored) | 0.65842 | 0.6080 | 0.6999 | 0.7318 | 0.65848, 0.65807, 0.65871 |
| base | 0.61806 | 0.5523 | 0.6691 | 0.7239 | - |
| BLEND | 0.62746 | 0.5613 | 0.6819 | 0.7276 | - |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| UTIL12 - CONF12 | primary | +0.00004 | [-0.00017, +0.00026] | [-0.00025, +0.00036] | +0.00030 | +0.0001 [-0.0004, +0.0006] | +0.0000 [-0.0003, +0.0003] | -0.0002 [-0.0006, +0.0001] | 0.00038 | not established; bounded within ±0.005 (Holm p query/ts 0.346/0.374) |
| UTIL12 - UTIL1 | primary | +0.00039 | [+0.00016, +0.00063] | [-0.00001, +0.00082] | +0.00076 | +0.0005 [-0.0001, +0.0011] | -0.0000 [-0.0003, +0.0004] | +0.0003 [-0.0001, +0.0008] | 0.00054 | not established; bounded within ±0.005 (Holm p query/ts <0.0005/0.062) |
| UTIL12 - UTIL2 | secondary | +0.00108 | [+0.00074, +0.00140] | [+0.00079, +0.00147] | +0.00139 | +0.0012 [+0.0007, +0.0018] | +0.0006 [-0.0005, +0.0016] | +0.0016 [+0.0009, +0.0022] | 0.00044 | present (+); gain ≥ 0.005 excluded |
| UTIL12 - FULL (same candidates, recomputed features, RB-style column layout) | fidelity | +0.00023 | [+0.00002, +0.00045] | [-0.00005, +0.00046] | +0.00043 | +0.0003 [-0.0003, +0.0008] | +0.0001 [-0.0002, +0.0003] | +0.0001 [-0.0001, +0.0002] | 0.00034 | not established; bounded within ±0.005 |
| UTIL12 - BLEND | context | +0.03119 | [+0.03033, +0.03205] | [+0.02718, +0.03363] | +0.03335 | +0.0470 [+0.0411, +0.0507] | +0.0180 [+0.0151, +0.0201] | +0.0043 [+0.0037, +0.0048] | 0.00402 | present (+) |
| CONF12 - BLEND | context | +0.03115 | [+0.03029, +0.03200] | [+0.02718, +0.03355] | +0.03327 | +0.0469 [+0.0409, +0.0505] | +0.0180 [+0.0152, +0.0199] | +0.0045 [+0.0039, +0.0051] | 0.00400 | present (+) |
| UTIL1 - BLEND | context | +0.03080 | [+0.02995, +0.03167] | [+0.02682, +0.03317] | +0.03286 | +0.0465 [+0.0407, +0.0500] | +0.0180 [+0.0151, +0.0200] | +0.0040 [+0.0034, +0.0045] | 0.00393 | present (+) |
| UTIL2 - BLEND | context | +0.03011 | [+0.02921, +0.03098] | [+0.02582, +0.03272] | +0.03241 | +0.0458 [+0.0395, +0.0497] | +0.0174 [+0.0146, +0.0191] | +0.0027 [+0.0021, +0.0031] | 0.00428 | present (+) |

Expansion rule: second base added = False (UTIL12 - CONF12: ts CI contains 0.005 = False, opposite signs = False; UTIL12 - UTIL1: ts CI contains 0.005 = False, opposite signs = False).

Notes: (1) UTIL12 reproduces the stored HRCOV_PAT pattern block exactly (block identity 1.0); its features are recomputed from the exact base lookups in the RB-style 25-column layout (0-18 base + evidence, 19 is_extra, 20-24 pattern features), and UTIL12 - FULL re-fit = -0.00001 (polecat) / +0.00023 (smallpedia), i.e. the re-implementation is faithful. (2) Block budget <= 50 after T50 for all variants; UTIL1 blocks are smaller (mean 12.6 vs 44.3 candidates on polecat test) because 1-hop patterns alone instantiate fewer candidates; the budget was not refilled from other sources (fixed by the rule). (3) Process: the first C2 chain (job) completed the lists and exact lookups but its feature stage failed on a docker argument-order bug (env flags after the image name); features, fits and official scoring were re-run unchanged from the stored lists/lookups (job) after a 2,000-query smoke. No configuration was changed. (4) Expansion rule: polecat UTIL12 - CONF12 is negative with CI excluding 0 while smallpedia's CI contains 0, and no timestamp CI contains 0.005, so the primaries are not ambiguous and no second base is added.

### C3 Relation normalisation on tkgl-icews, partial (`c3_relnorm_tkgl_icews.json`)

**Interpretation.** Relation normalisation helps the base (+0.03438, RE-GCN-RN ≈ RE-GCN BLEND) and the re-ranker features (+0.00938); the HRCOV margin survives when the competitor also gets RN: HRCOV200_RN - RE-GCN-RN base = +0.07564 vs HRCOV200 - RE-GCN base = +0.10063 without RN (the gain shrinks by 0.0250 but remains large).

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| RE-GCN base | 0.27469 | 0.1767 | 0.3092 | 0.4720 | - |
| RE-GCN-RN base | 0.30906 | 0.2097 | 0.3483 | 0.5062 | - |
| RE-GCN BLEND | 0.31060 | 0.2110 | 0.3503 | 0.5063 | - |
| HRCOV200 (RE-GCN) | 0.37531 | 0.2682 | 0.4253 | 0.5870 | 0.37522, 0.37491, 0.37582 |
| HRCOV200 (RE-GCN) [harness] | 0.37531 | 0.2682 | 0.4253 | 0.5870 | 0.37522, 0.37491, 0.37582 |
| HRCOV200_RN (RE-GCN base, RN features) [harness] | 0.38470 | 0.2763 | 0.4356 | 0.5983 | 0.38509, 0.38466, 0.38434 |
| G1 | 0.33530 | 0.2490 | 0.3824 | 0.5014 | 0.33481, 0.33568, 0.33542 |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| RE-GCN-RN base - RE-GCN base [RN effect on the base] | primary-partial | +0.03438 | [+0.03419, +0.03456] | [+0.03300, +0.03585] | +0.03555 | +0.0330 [+0.0317, +0.0344] | +0.0391 [+0.0375, +0.0407] | +0.0342 [+0.0328, +0.0357] | 0.00179 | present (+) |
| HRCOV200_RN - HRCOV200 [RN effect on the re-ranker features, harness] | primary-partial | +0.00938 | [+0.00928, +0.00949] | [+0.00894, +0.00986] | +0.00978 | +0.0081 [+0.0076, +0.0085] | +0.0103 [+0.0097, +0.0108] | +0.0113 [+0.0108, +0.0119] | 0.00059 | present (+) |
| HRCOV200 - RE-GCN base [gain without RN] | primary-partial | +0.10063 | [+0.10038, +0.10090] | [+0.09952, +0.10179] | +0.10156 | +0.0916 [+0.0904, +0.0927] | +0.1162 [+0.1148, +0.1175] | +0.1150 [+0.1139, +0.1162] | 0.00142 | present (+) |
| HRCOV200_RN - RE-GCN-RN base [HRCOV with RN features vs the RN base] | primary-partial | +0.07564 | [+0.07539, +0.07589] | [+0.07513, +0.07612] | +0.07605 | +0.0666 [+0.0660, +0.0672] | +0.0874 [+0.0867, +0.0880] | +0.0922 [+0.0916, +0.0928] | 0.00064 | present (+) |
| HRCOV200_RN - RE-GCN base | secondary | +0.11001 | [+0.10976, +0.11027] | [+0.10861, +0.11152] | +0.11126 | +0.0996 [+0.0982, +0.1011] | +0.1264 [+0.1248, +0.1282] | +0.1264 [+0.1249, +0.1279] | 0.00182 | present (+) |
| HRCOV200_RN - G1 | secondary | +0.04939 | [+0.04919, +0.04959] | [+0.04832, +0.05049] | +0.05030 | +0.0273 [+0.0265, +0.0281] | +0.0532 [+0.0520, +0.0545] | +0.0970 [+0.0951, +0.0989] | 0.00138 | present (+) |
| HRCOV200_RN - RE-GCN BLEND | secondary | +0.07410 | [+0.07388, +0.07432] | [+0.07260, +0.07567] | +0.07539 | +0.0653 [+0.0639, +0.0668] | +0.0853 [+0.0836, +0.0871] | +0.0920 [+0.0903, +0.0938] | 0.00192 | present (+) |
| HRCOV200 - RE-GCN BLEND [without RN] | secondary | +0.06471 | [+0.06449, +0.06494] | [+0.06353, +0.06594] | +0.06569 | +0.0572 [+0.0560, +0.0584] | +0.0751 [+0.0737, +0.0765] | +0.0807 [+0.0793, +0.0821] | 0.00151 | present (+) |
| RE-GCN-RN base - RE-GCN BLEND | secondary | -0.00154 | [-0.00176, -0.00132] | [-0.00308, +0.00000] | -0.00024 | -0.0014 [-0.0028, +0.0001] | -0.0020 [-0.0037, -0.0002] | -0.0002 [-0.0020, +0.0016] | 0.00195 | present (−); loss ≥ 0.005 excluded |

Declared cuts: HRCOV on the RE-GCN-RN base (requires the full Stage B pipeline - features, pattern lists, exact lookups, fits, official scoring - on the RN base; > 24 GPU-h budget share); RecB on RN data at full scale (only a 4.65M-query sample profile exists: recbRN_profile.json) and therefore G1-RN; BLEND / POOL on the RE-GCN-RN base; official py-tgb scoring of the HRCOV200_RN promoted lists (harness ranks used; identity with official checked on HRCOV200).

Notes: (1) HRCOV200_RN = HRCOV200 recipe on the ORIGINAL RE-GCN base with features (evidence, patterns) computed on the relation-normalised history (feat_regcn_rn, scheme 'rn'); evaluation on original ids/conflict sets. (2) Because HRCOV200_RN keeps the original base, 'HRCOV200_RN - RE-GCN-RN base' compares the RN re-ranker with the RN base, not an RN-base re-ranker.

### C4 Second base seed (`c4_base_seed.json`)

**Declared cut.** A second TiRGN polecat seed under the same recipe needs ~51.5 GB GPU memory (peak allocated in the original run: L = 10, 8 epochs with patience, 9,126 s training + 1,186 s top-200 dump on a 96 GB card). GPU 1 of <HOST> is shared with two other tracks (45-51 GB in use by others during v7), so the v7 cap of ~30 GB on GPU 1 cannot hold it; GPU 0 is reserved for G8 (CognTKE training, ETA ~04:00-04:30Z 15 Oct). Reducing memory would change the recipe (batch / history length), which the rule forbids. What it would take: After G8 releases GPU 0: TiRGN polecat seed-2 training + dump (~3 h), the HRCOV_PAT feature pipeline on the new base (pattern lists reused; exact base lookups + features ~2-3 h CPU/GPU), HRCOV_PAT fits (3 seeds, < 1 h), valid-selected BLEND/POOL controls and official scoring (~2 h): roughly 8-9 h wall, ~5 GPU-h, on GPU 0 only.

### A5 sensitivity: UNMASKED and MULTIPOS (polecat TiRGN) (`a5_multi_positive_sensitivity.json`)

**Interpretation.** Other true answers are already masked in Eq. (4) (code reading, A5); the two pre-specified sensitivity variants on polecat TiRGN change MRR by +0.00002 (UNMASKED) and -0.00034 (MULTIPOS); |delta| >= 0.005 is excluded for both, so the masking choice does not drive the result.

**polecat** (TiRGN; 532,636 queries, 304 timestamps)

| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |
|---|---|---|---|---|---|
| MLP64x2 (FULL, re-fit) | 0.28232 | 0.1966 | 0.3214 | 0.4527 | 0.28203, 0.28253, 0.28242 |
| UNMASKED | 0.28235 | 0.1967 | 0.3214 | 0.4526 | 0.28275, 0.28217, 0.28212 |
| MULTIPOS | 0.28199 | 0.1962 | 0.3210 | 0.4524 | 0.28117, 0.28218, 0.28261 |
| HRCOV_PAT (stored) | 0.28238 | 0.1968 | 0.3212 | 0.4529 | 0.28250, 0.28234, 0.28230 |

| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |
|---|---|---|---|---|---|---|---|---|---|---|
| UNMASKED - FULL | secondary | +0.00002 | [-0.00009, +0.00013] | [-0.00011, +0.00015] | +0.00013 | +0.0001 [-0.0001, +0.0003] | -0.0000 [-0.0002, +0.0002] | -0.0001 [-0.0003, +0.0001] | 0.00016 | not established; bounded within ±0.005 |
| MULTIPOS - FULL | secondary | -0.00034 | [-0.00045, -0.00022] | [-0.00048, -0.00019] | -0.00022 | -0.0004 [-0.0006, -0.0001] | -0.0004 [-0.0006, -0.0002] | -0.0003 [-0.0005, -0.0001] | 0.00018 | present (−); loss ≥ 0.005 excluded |

Notes: (1) Run on polecat TiRGN only, as pre-specified. UNMASKED: other true same-timestamp answers kept as negatives in the listwise softmax; MULTIPOS: loss = -log sum over true answers in the pool of the softmax. Fit queries with >= 1 other true answer in top-50: polecat 20.0% (A5 code reading).

## Deviations, cuts and process notes

1. Fused arms: official = exact harness rank to <= 1e-7 MRR (hrank stored alongside rank).
2. Score-alone arms (B_NF, NB_NF, EV_NF_*): official scoring orders the promoted list by the ranker score and breaks exact ties by list position (base top-50 order, then the generator block); the preliminary exact ranks used strict-greater (optimistic) ties. Difference official vs preliminary: polecat EV_NF_PT -0.00012, NB_NF -0.00006; smallpedia EV_NF_PT -0.00183, EV_NF_RB -0.00192, NB_NF -0.00011 (all other arms 0). The registered numbers are the official ones.
3. Base replay: per-query base RR identical to the stored control/HRCOV_PAT files except near-tie blocks (polecat 0.011-0.012% of queries, mean |dRR| < 1e-8; smallpedia exactly identical).
4. FULL re-fit vs stored HRCOV_PAT: smallpedia identical per seed; polecat -0.00005, ts CI [-0.00017, +0.00006] (GPU non-determinism).
5. Polecat official-scoring group 02 (A5, B3 NB arms, C1) was stopped once to respect the GPU-1 cap and re-run unchanged in two halves (02a, 02b); C2's GPU stages were held until it finished.
6. C2 feature stage failed once (docker argument order) and was re-run unchanged from the stored lists and lookups after a smoke test.
7. GPU use (<HOST> GPU 1, container wall-hours on a shared card): A2 scan ≈ 0.7 h, B fits ≈ 0.8 h, B/A5/C1 official scoring ≈ 7.8 h, B4 G1 features + fits + official ≈ 3.4 h, C2 lookups + fits + official ≈ 1.9 h; total ≈ 15 h (< 24 GPU-h budget). GPU 0 (G8) untouched.
8. Declared cuts: C3 HRCOV/G1/BLEND on the RE-GCN-RN base and official scoring of HRCOV200_RN; C4 second TiRGN seed; B4 extension to ICEWS18/GDELT.
