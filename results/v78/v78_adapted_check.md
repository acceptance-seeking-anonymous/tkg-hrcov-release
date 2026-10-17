# v7.8 post-verdict check: external comparator with the label-adapted CountTRuCoLa run

**Label: registered-optional comparator run (amendment `v78_rule_amendment_1`, field `EXT.EXT_adapted`), executed and evaluated after the v7.8 framing verdict was read. The registered verdict (`F4_CONDITIONAL`, results version `b037d61d`) stays as recorded; this check is reported next to it.**

Protocol decisions for the adapted run were saved before any adapted number: `v78_external_protocol_note_2_ext_adapted.json` (version `03fb6eb1`, sha256 `ce9ef0ebc845dafd…`, 2026-10-16T08:20:08Z). Rule `v78_framing_rule` (sha256 `8e05c67b7ce8f3a4…`), amendment 1 (sha256 `8fffffcaad75b066…`).

## Result

Re-applying the rule's `external_comparator` selection over {EXT_alone, EXT_blend, EXT_adapted, EXT_adapted_blend} (CountTRuCoLa) and {EXT_alone, EXT_blend} (EdgeReMIND) selects **EXT_adapted_blend (CountTRuCoLa)** in all three configurations, by selection-part margins of +0.00043 (S1), +0.00024 (S2) and +0.00012 (S3) MRR over the registered EXT_blend. **No call changes and the outcome label does not change**: external calls stay S1 MATCHES, S2 BEATS, S3 BEATS; SA does not beat the external comparator in S1 or S3 and beats it in S2, as before; outcome `F4_CONDITIONAL` (sensitivity with SA_GEN: `F4_CONDITIONAL`), the same as registered.

| config | external comparator (registered → re-applied) | HRcov − X: Δ MRR | query bootstrap [L, U] | ts-block bootstrap [L, U] | call (registered → re-applied) | SA beats X |
|---|---|---|---|---|---|---|
| S1 (tkgl-smallpedia, TiRGN) | EXT_blend_counttrucola → EXT_adapted_blend_counttrucola | +0.0032 | [+0.0027, +0.0036] | [+0.0023, +0.0040] | MATCHES → MATCHES | False → False |
| S2 (tkgl-polecat, TiRGN) | EXT_blend_counttrucola → EXT_adapted_blend_counttrucola | +0.0168 | [+0.0164, +0.0172] | [+0.0161, +0.0175] | BEATS → BEATS | True → True |
| S3 (tkgl-polecat, CognTKE) | EXT_blend_counttrucola → EXT_adapted_blend_counttrucola | +0.0073 | [+0.0071, +0.0076] | [+0.0069, +0.0077] | BEATS → BEATS | False → False |

L, U = 5th / 95th bootstrap percentiles (one-sided 95% bounds), B = 2000. The registered HRcov − EXT_blend rows are reproduced exactly by the same aggregation (all 30 registered comparison pairs per configuration identical, shared bootstrap draws).

## Selection (validation selection part, harness ranks with ties averaged)

| config | EdgeReMIND alone | EdgeReMIND blend | CTR alone | CTR blend (registered X) | CTR adapted alone | CTR adapted blend | adapted-blend grid point (a, τ) | registered blend (a, τ) |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.62868 | 0.64228 | 0.63741 | 0.65702 | 0.63850 | **0.65745** | (0.6, 0.5) | (0.6, 0.5) |
| S2 | 0.17545 | 0.24381 | 0.26467 | 0.27571 | 0.26486 | **0.27595** | (0.6, 2.0) | (0.6, 2.0) |
| S3 | 0.17545 | 0.30452 | 0.26467 | 0.30728 | 0.26486 | **0.30739** | (0.05, 0.5) | (0.3, 4.0) |

CTR = CountTRuCoLa. Selection rows: S1 56,460; S2/S3 60,764 (same rows as the registered EXT selection; asserted).

## Test (official evaluator, all test queries)

| config | system | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| S1 | HRcov | 0.6584 | 0.6080 | 0.6999 | 0.7318 |
| S1 | SA | 0.6544 | 0.6060 | 0.6967 | 0.7243 |
| S1 | EXT_alone_counttrucola | 0.6365 | 0.5747 | 0.6812 | 0.7102 |
| S1 | EXT_blend_counttrucola | 0.6547 | 0.6033 | 0.6977 | 0.7288 |
| S1 | EXT_adapted_counttrucola | 0.6373 | 0.5762 | 0.6816 | 0.7109 |
| S1 | EXT_adapted_blend_counttrucola | 0.6552 | 0.6041 | 0.6979 | 0.7290 |
| S2 | HRcov | 0.2824 | 0.1968 | 0.3212 | 0.4529 |
| S2 | SA | 0.2752 | 0.1922 | 0.3137 | 0.4399 |
| S2 | EXT_alone_counttrucola | 0.2557 | 0.1785 | 0.2911 | 0.4082 |
| S2 | EXT_blend_counttrucola | 0.2655 | 0.1831 | 0.3017 | 0.4277 |
| S2 | EXT_adapted_counttrucola | 0.2559 | 0.1786 | 0.2911 | 0.4083 |
| S2 | EXT_adapted_blend_counttrucola | 0.2656 | 0.1833 | 0.3017 | 0.4277 |
| S3 | HRcov | 0.3040 | 0.2178 | 0.3438 | 0.4741 |
| S3 | SA | 0.2752 | 0.1922 | 0.3137 | 0.4399 |
| S3 | EXT_alone_counttrucola | 0.2557 | 0.1785 | 0.2911 | 0.4082 |
| S3 | EXT_blend_counttrucola | 0.2966 | 0.2113 | 0.3347 | 0.4653 |
| S3 | EXT_adapted_counttrucola | 0.2559 | 0.1786 | 0.2911 | 0.4083 |
| S3 | EXT_adapted_blend_counttrucola | 0.2967 | 0.2114 | 0.3347 | 0.4653 |

## Further comparisons (MRR; L, U = 5th / 95th percentiles, i.e. one-sided 95% bounds)

| config | pair | Δ | query [L, U] | ts-block [L, U] |
|---|---|---|---|---|
| S1 | HRcov - EXT_adapted_blend_counttrucola | +0.0032 | [+0.0027, +0.0036] | [+0.0023, +0.0040] |
| S1 | SA - EXT_adapted_blend_counttrucola | -0.0008 | [-0.0014, -0.0002] | [-0.0026, +0.0017] |
| S1 | HRcov - EXT_blend_counttrucola | +0.0037 | [+0.0033, +0.0042] | [+0.0029, +0.0045] |
| S1 | SA - EXT_blend_counttrucola | -0.0003 | [-0.0009, +0.0003] | [-0.0021, +0.0022] |
| S1 | HRcov - EXT_adapted_counttrucola | +0.0211 | [+0.0205, +0.0218] | [+0.0190, +0.0227] |
| S2 | HRcov - EXT_adapted_blend_counttrucola | +0.0168 | [+0.0164, +0.0172] | [+0.0161, +0.0175] |
| S2 | SA - EXT_adapted_blend_counttrucola | +0.0096 | [+0.0092, +0.0100] | [+0.0089, +0.0104] |
| S2 | HRcov - EXT_blend_counttrucola | +0.0169 | [+0.0165, +0.0173] | [+0.0162, +0.0176] |
| S2 | SA - EXT_blend_counttrucola | +0.0097 | [+0.0093, +0.0101] | [+0.0090, +0.0105] |
| S2 | HRcov - EXT_adapted_counttrucola | +0.0265 | [+0.0261, +0.0270] | [+0.0258, +0.0273] |
| S3 | HRcov - EXT_adapted_blend_counttrucola | +0.0073 | [+0.0071, +0.0076] | [+0.0069, +0.0077] |
| S3 | SA - EXT_adapted_blend_counttrucola | -0.0215 | [-0.0220, -0.0209] | [-0.0225, -0.0205] |
| S3 | HRcov - EXT_blend_counttrucola | +0.0074 | [+0.0071, +0.0076] | [+0.0070, +0.0078] |
| S3 | SA - EXT_blend_counttrucola | -0.0214 | [-0.0219, -0.0209] | [-0.0224, -0.0204] |
| S3 | HRcov - EXT_adapted_counttrucola | +0.0481 | [+0.0476, +0.0486] | [+0.0472, +0.0491] |

## Checks

- EXT_adapted alone on the selection rows equals the sibling's official per-query reciprocal ranks (max |Δrr| 2.0e-08); selection MRR 0.6385 (smallpedia) and 0.2649 (polecat) vs the sibling's 0.6385 and 0.2649.
- EXT_adapted alone test MRR 0.6373 (smallpedia) and 0.2559 (polecat), equal to the sibling's official 0.6373 and 0.2559.
- Frozen-base reciprocal ranks inside the adapted scoring equal the v7.8 base (mean |Δrr| 2.1e-08).
- Validation rows inside the adapted learning data (NaN in the dumps) were never requested (asserted per run).
- Harness vs official: EXT_adapted alone shows the same tie-block gap as EXT_alone on tkgl-smallpedia (harness − official +0.1318); decisions use official ranks.

## Caveats

1. Timing: the adapted run, its blend and this re-selection were made after the v7.8 verdict was read. The amendment lists EXT_adapted as optional, so the selection is the registered procedure, but the decision to run it was taken post-verdict; the registered verdict is unchanged and remains the reference.
2. The selection margins are small (at most 0.00043 MRR) and the test MRR differences EXT_adapted_blend − EXT_blend are +0.00053 (S1), +0.00010 (S2), +0.00006 (S3); the label adaptation changes the comparator only marginally, and HRcov − X moves by the same amounts.
3. tkgl-polecat (S2, S3): CountTRuCoLa learned on all 372,024 validation facts below the cut, more fit-window label access than the HRcov ranker (120,000 fit queries); declared in protocol note 2. The direction of this deviation favours the comparator.
4. EdgeReMIND was not adapted (out of scope of protocol note 2).

File: `v78_adapted_check.json` (sha256 of its canonical body `14a04ecc3883958a…`). Code: `v78_ext.py --tag _adapted`, `v78_adapted_chain.sh`, `v78_adapted_agg.py`, `v78_adapted_report.py`.
