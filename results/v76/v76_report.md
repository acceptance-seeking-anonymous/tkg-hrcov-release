# Manuscript A v7.6 - reviewer-requested experiments (M2b, M3a, M4, M6)

**Rule provenance.**

| File | Registered UTC | Registered sha256 (recorded in file; = sha256 of *_registered.json) | sha256 of the saved file bytes |
|---|---|---|---|
| v76_rule.json | 2026-10-15T12:07:43Z | edcfca14128e22f6aaec415eca54da083d2678354bd4cdc95f475638863e0a04 | 4daf8b0f99271ff8f3018762dd12f03695a406e89c3837af44d4aa7370b8471d |
| v76_rule_amendment_1.json | 2026-10-15T12:49:21Z | bc8cd8f87192ee980dc0fdde8a3a8e8bd3770bdf01339bb133be16df260697ff | f02c5f6b85dc726be51392a9ef54ba28722ceb73321532f7aab01527b3b458d1 |

The two hashes per file differ by construction: write_prereg: registered sha256 = sha256 of json.dumps(body, indent=1, sort_keys=True, default=str), where body = rule + registered_utc and WITHOUT the sha256 field; the saved file is json.dump(body + sha256, indent=1) in insertion order, so its file-bytes sha256 differs. The exact hashed bytes are stored as *_registered.json; v76_rule_hash_check.py re-derives both from the saved files. The amendment is a separate file registered 42 min after the rule; it cites the rule's registered hash and changes only the M2b LambdaMART selection statistic. Each file has one stored version.

Rule: v76_rule.json (registered 2026-10-15T12:07:43Z, registered sha256 edcfca14128e..., before any test number of these arms), amendment 1 (2026-10-15T12:49:21Z, registered sha256 bc8cd8f87192...: tie-exact LambdaMART selection statistic). Statistics as tois_v7_rule.json: official py-tgb 2.3.0 per-query RR; paired Poisson query bootstrap and timestamp-block bootstrap (B = 2000, seed 0, shared draws); present iff the one-sided lower bound > 0 under both (Holm across an item's primary contrasts on a base); 'excluded' iff the timestamp-block upper bound of |d| < 0.005. All test sets were inspected in earlier rounds: every number below is a pre-specified later analysis, not an untouched confirmation.

Finiteness: every new array was scanned before aggregation (dumps, full rows, features, RecB rows, fitted models, promoted lists, official outputs): 0 NaN and 0 +inf; the only -inf values sit at padded candidate slots of feature log-prob tables (0 at real candidates).

## M2b - LambdaMART on the same features, pool, rows and fusion

**Card.** LightGBM 4.6.0 lambdarank (CPU, 16 threads) on the 25 HRCOV_PAT features of the W = 100 pool; one group per fit query (valid, unfiltered candidates; label 1 = gold); fit rows = validation ts < 70% cut with the gold in the pool (cap 120,000); grid num_leaves {15, 31, 63} x learning_rate {0.03, 0.1}, <= 3,000 rounds, early stopping (patience 100) and grid choice on the selection-row MRR of the exact fused-rank statistic of the LambdaMART score (ties resolved as the promoted list resolves them, amendment 1); grid on seed 0, seeds 1-2 refit; calibrated alpha/beta fusion with the base log-prob as for the MLP. MLP64_PAT (= HRCOV_PAT ranker) and LIN_PAT re-fit in the same run on the same rows.

| Base | Better control | LIN | LambdaMART | MLP (HRCOV_PAT) | MLP stored | LMART grid point |
|---|---|---|---|---|---|---|
| tkgl-polecat TiRGN (s42) | 0.2349 (POOL) | 0.2702 | 0.2817 | 0.2823 | 0.2824 | 63 leaves, lr 0.03 |
| tkgl-polecat CognTKE | 0.2941 (BLEND) | 0.2987 | 0.3037 | 0.3040 | 0.3040 | 63 leaves, lr 0.03 |
| tkgl-smallpedia TiRGN | 0.6275 (BLEND) | 0.6432 | 0.6595 | 0.6584 | 0.6584 | 63 leaves, lr 0.03 |
| tkgl-smallpedia RE-GCN (TGB) | 0.6249 (BLEND) | 0.6379 | 0.6601 | 0.6586 | 0.6586 | 31 leaves, lr 0.1 |

MRR, official evaluator, test; learned arms = 3-seed means. H@10 in v76_results.json.

| Base | Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |
|---|---|---|---|---|---|
| tkgl-polecat TiRGN (s42) | MLP64_PAT_fused - LMART_PAT_fused | +0.0006 | [+0.0004, +0.0009] | [+0.0004, +0.0010] | present (+); within +-0.005 (both directions excluded) |
| tkgl-polecat TiRGN (s42) | LMART_PAT_fused - LIN_PAT_fused | +0.0115 | [+0.0111, +0.0119] | [+0.0109, +0.0122] | present (+) |
| tkgl-polecat TiRGN (s42) | LMART_PAT_fused - better control | +0.0468 | [+0.0463, +0.0474] | [+0.0457, +0.0480] | present (+) |
| tkgl-polecat CognTKE | MLP64_PAT_fused - LMART_PAT_fused | +0.0002 | [+0.0000, +0.0005] | [-0.0000, +0.0005] | present (+); within +-0.005 (both directions excluded) |
| tkgl-polecat CognTKE | LMART_PAT_fused - LIN_PAT_fused | +0.0050 | [+0.0048, +0.0053] | [+0.0046, +0.0054] | present (+) |
| tkgl-polecat CognTKE | LMART_PAT_fused - better control | +0.0097 | [+0.0093, +0.0100] | [+0.0092, +0.0102] | present (+) |
| tkgl-smallpedia TiRGN | MLP64_PAT_fused - LMART_PAT_fused | -0.0010 | [-0.0015, -0.0006] | [-0.0015, -0.0006] | present (-); within +-0.005 (both directions excluded) |
| tkgl-smallpedia TiRGN | LMART_PAT_fused - LIN_PAT_fused | +0.0163 | [+0.0155, +0.0170] | [+0.0141, +0.0177] | present (+) |
| tkgl-smallpedia TiRGN | LMART_PAT_fused - better control | +0.0320 | [+0.0311, +0.0329] | [+0.0281, +0.0345] | present (+) |
| tkgl-smallpedia RE-GCN (TGB) | MLP64_PAT_fused - LMART_PAT_fused | -0.0015 | [-0.0019, -0.0011] | [-0.0022, -0.0006] | present (-); within +-0.005 (both directions excluded) |
| tkgl-smallpedia RE-GCN (TGB) | LMART_PAT_fused - LIN_PAT_fused | +0.0222 | [+0.0214, +0.0230] | [+0.0188, +0.0242] | present (+) |
| tkgl-smallpedia RE-GCN (TGB) | LMART_PAT_fused - better control | +0.0351 | [+0.0342, +0.0361] | [+0.0304, +0.0380] | present (+) |

Secondary (alone scores, exact fused-rank statistic): tkgl-polecat TiRGN (s42): MLP - LMART +0.0008; tkgl-polecat CognTKE: MLP - LMART +0.0002; tkgl-smallpedia TiRGN: MLP - LMART -0.0007; tkgl-smallpedia RE-GCN (TGB): MLP - LMART -0.0015.

**Cost.**

| Base | LMART fit per seed (s, incl. per-round selection) | best rounds | LMART grid total (s) | MLP fit per seed (s, GPU) | Scoring latency, ms / 1,000 queries: LMART CPU / MLP CPU / MLP GPU |
|---|---|---|---|---|---|
| tkgl-polecat TiRGN (s42) | 126.0, 127.9, 111.5 | 555, 600, 516 | 650.0 | 12.4, 13.4 (seeds 1-2) | 155.1 / 16.3 / 2.6 |
| tkgl-polecat CognTKE | 97.6, 89.3, 79.5 | 458, 400, 332 | 533.5 | 16.5, 9.8 (seeds 1-2) | 116.5 / 18.1 / 2.7 |
| tkgl-smallpedia TiRGN | 54.7, 43.9, 42.6 | 326, 236, 228 | 267.6 | 10.4, 13.2 (seeds 1-2) | 38.9 / 17.9 / 3.5 |
| tkgl-smallpedia RE-GCN (TGB) | 19.6, 24.2, 18.2 | 76, 123, 67 | 205.4 | 10.9, 14.8 (seeds 1-2) | 7.5 / 16.8 / 2.4 |

**Reading.** LambdaMART matches the MLP within the 0.005 bar on all four bases (MLP - LambdaMART from -0.0015 to +0.0006); the small differences are present in both directions (MLP ahead on the two tkgl-polecat bases, LambdaMART ahead on the two tkgl-smallpedia bases). Like the MLP, LambdaMART exceeds the linear scorer (+0.0050 to +0.0222; on CognTKE the margin is at the bar) and the better recurrence control (+0.0097 to +0.0468) on every base. The added value is therefore a non-linear candidate-level ranker over these features, not the particular 2x64 network. Cost: the MLP (5,889 parameters) scores at 16-18 ms per 1,000 queries on CPU (2-4 ms on GPU); LambdaMART (67-600 trees) is 9.5x and 6.4x slower on the tkgl-polecat bases, 2.2x slower on tkgl-smallpedia TiRGN, and 2.2x faster on tkgl-smallpedia RE-GCN, whose selected model has 67-123 trees. Fitting cost is minutes for either learner.

**Caveats.** Amendment 1: the first LambdaMART run selected round 1 everywhere because the v7 statistic counts tied scores optimistically; it was discarded (its log printed round-1 test MRRs; nothing was chosen from them). LambdaMART fit times include the per-round selection evaluation on 56-61k queries; MLP times are GPU. LogCL (optional base) was not run.

## M3a - base trained with the validation-fit labels (tkgl-smallpedia TiRGN)

**Card.** sp_tirgn.py recipe unchanged except that the training snapshots are train plus the 7 validation snapshots with ts < 105 (52,803 facts); epoch selection on the 3 validation-select snapshots only (28,230 facts). Best epoch 25 of 30 (no early stop), validation-select MRR 0.6337; training 1313 s on <HOST> GPU 0 (no memory issue). BLEND / POOL grids as hardening, selected on the 56,460 validation-select queries (selected: BLEND tr LOGLIN (0.25, 1e-05), POOL K = 100).

| System | MRR |
|---|---|
| Original TiRGN base | 0.6181 |
| Validation-trained base | 0.6215 |
| Better control, original base (BLEND) | 0.6275 |
| Better control, validation-trained base (BLEND) | 0.6320 |
| HRCOV_PAT, original base (stored v7.4 run) | 0.6584 |
| HRCOV_PAT, validation-trained base (secondary) | 0.6548 |

| Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |
|---|---|---|---|---|
| **primary** HRCOV_PAT(original base) - better control(validation-trained base) | +0.0264 | [+0.0255, +0.0274] | [+0.0232, +0.0285] | present (+) |
| validation-trained base - original base | +0.0035 | [+0.0025, +0.0044] | [+0.0016, +0.0058] | present (+) |
| better control(validation-trained) - better control(original) | +0.0045 | [+0.0037, +0.0053] | [+0.0027, +0.0067] | present (+) |
| HRCOV_PAT(original base) - better control(original base) | +0.0310 | [+0.0301, +0.0318] | [+0.0270, +0.0333] | present (+) |
| [secondary] HRCOV_PAT(validation-trained base) - better control(validation-trained base) | +0.0228 | [+0.0220, +0.0236] | [+0.0191, +0.0249] | present (+) |

**Reading.** Giving the base the validation-fit labels raises it by +0.0035 and the better recurrence control by +0.0045. HRCOV_PAT on the original base still exceeds the better control on the validation-trained base by +0.0264 [+0.0232, +0.0285] (vs +0.0310 on the original base): validation supervision of the base closes about 15% of the margin, and the supervision asymmetry does not explain it. On the validation-trained base, HRCOV_PAT keeps +0.0228 over that base's own better control (secondary; its fit rows are in-sample for this base).

**Caveats.** One dataset and one base (as specified); single training seed; the validation-fit facts are a superset of the ranker's fit rows (all facts of those timestamps); RecB's trained parameters inside the controls are the v7.4 ones (selected on all validation queries).

## M4 - cross-fitted RecB (parameters selected without the ranker's fit labels)

**Card.** RecB_xf: TGB train() selection of (lambda_r, alpha_r) from the stored exact validation ranks of all 182 configurations, restricted to validation-select queries (ts >= cut). tkgl-polecat: 161,448 queries, parameters changed for 16 of 32 directed relations; tkgl-smallpedia: 56,460 queries, changed for 209 of 374 (70 without validation-select queries fall back to the TGB defaults). Optimism of the original parameters on the fit rows (validation MRR, original - xf): polecat +0.0004, smallpedia +0.0028. G1_xf = G1 recipe unchanged on RecB_xf candidates/features (K selected on validation-select: K = 150 on both); RecB-feature arms re-fit in one run per base with no RecB features, with trained-RecB features (tr) and with RecB_xf features (xf).

**polecat** - G1 0.2482, G1_xf 0.2480; RecB generator row tr 0.1992, xf 0.1989; RecB-feature arms (TiRGN): PAT 0.2823, PAT_RBtr 0.2824, PAT_RBxf 0.2821, HR19 0.2695, HR19_RBtr 0.2693, HR19_RBxf 0.2695.

| Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |
|---|---|---|---|---|
| **G1_xf - G1** | -0.0002 | [-0.0004, -0.0001] | [-0.0004, -0.0001] | present (-); within +-0.005 (both directions excluded) |
| RecB_xf - RecB_tr (generator rows) | -0.0003 | [-0.0004, -0.0001] | [-0.0005, -0.0001] | present (-); within +-0.005 (both directions excluded) |
| **HRCOV[TiRGN] - G1_xf** | +0.0344 | [+0.0339, +0.0349] | [+0.0332, +0.0356] | present (+) |
| HRCOV[TiRGN] - G1 | +0.0341 | [+0.0336, +0.0347] | [+0.0329, +0.0354] | present (+) |
| **HRCOV[LogCL] - G1_xf** | +0.0228 | [+0.0223, +0.0234] | [+0.0215, +0.0242] | present (+) |
| HRCOV[LogCL] - G1 | +0.0226 | [+0.0220, +0.0231] | [+0.0213, +0.0239] | present (+) |
| **HRCOV[CognTKE] - G1_xf** | +0.0560 | [+0.0554, +0.0567] | [+0.0546, +0.0574] | present (+) |
| HRCOV[CognTKE] - G1 | +0.0558 | [+0.0551, +0.0564] | [+0.0544, +0.0572] | present (+) |
| **PAT_RBxf - PAT** | -0.0002 | [-0.0003, -0.0000] | [-0.0003, -0.0000] | present (-); within +-0.005 (both directions excluded) |
| **HR19_RBxf - HR19** | +0.0001 | [-0.0001, +0.0002] | [-0.0001, +0.0002] | not present; within +-0.005 (both directions excluded) |
| PAT_RBtr - PAT | +0.0001 | [-0.0001, +0.0002] | [-0.0001, +0.0002] | not present; within +-0.005 (both directions excluded) |
| HR19_RBtr - HR19 | -0.0002 | [-0.0003, -0.0000] | [-0.0003, +0.0000] | present (-); within +-0.005 (both directions excluded) |
| PAT_RBxf - PAT_RBtr | -0.0003 | [-0.0003, -0.0002] | [-0.0004, -0.0001] | present (-); within +-0.005 (both directions excluded) |
| HR19_RBxf - HR19_RBtr | +0.0002 | [+0.0001, +0.0003] | [+0.0001, +0.0003] | present (+); within +-0.005 (both directions excluded) |

**smallpedia** - G1 0.6511, G1_xf 0.6521; RecB generator row tr 0.6046, xf 0.6043; RecB-feature arms (TiRGN): PAT 0.6584, PAT_RBtr 0.6504, PAT_RBxf 0.6519, HR19 0.6570, HR19_RBtr 0.6266, HR19_RBxf 0.6304.

| Contrast | Delta MRR | 95% CI (query) | 95% CI (ts-block) | Verdict |
|---|---|---|---|---|
| **G1_xf - G1** | +0.0010 | [+0.0006, +0.0013] | [-0.0004, +0.0037] | not present; within +-0.005 (both directions excluded) |
| RecB_xf - RecB_tr (generator rows) | -0.0003 | [-0.0005, -0.0001] | [-0.0014, +0.0015] | not present; within +-0.005 (both directions excluded) |
| **HRCOV[TiRGN] - G1_xf** | +0.0064 | [+0.0058, +0.0069] | [+0.0032, +0.0084] | present (+) |
| HRCOV[TiRGN] - G1 | +0.0073 | [+0.0068, +0.0079] | [+0.0062, +0.0085] | present (+) |
| **HRCOV[RE-GCN (TGB)] - G1_xf** | +0.0065 | [+0.0059, +0.0071] | [+0.0033, +0.0085] | present (+) |
| HRCOV[RE-GCN (TGB)] - G1 | +0.0075 | [+0.0069, +0.0080] | [+0.0063, +0.0085] | present (+) |
| **PAT_RBxf - PAT** | -0.0065 | [-0.0070, -0.0060] | [-0.0077, -0.0050] | present (-); gain >= 0.005 excluded |
| **HR19_RBxf - HR19** | -0.0266 | [-0.0272, -0.0260] | [-0.0291, -0.0227] | present (-); gain >= 0.005 excluded |
| PAT_RBtr - PAT | -0.0080 | [-0.0086, -0.0075] | [-0.0089, -0.0074] | present (-); gain >= 0.005 excluded |
| HR19_RBtr - HR19 | -0.0304 | [-0.0311, -0.0298] | [-0.0327, -0.0267] | present (-); gain >= 0.005 excluded |
| PAT_RBxf - PAT_RBtr | +0.0015 | [+0.0012, +0.0019] | [+0.0008, +0.0028] | present (+); within +-0.005 (both directions excluded) |
| HR19_RBxf - HR19_RBtr | +0.0038 | [+0.0033, +0.0042] | [+0.0029, +0.0047] | present (+); within +-0.005 (both directions excluded) |

**Reading.** Cross-fitting changes G1 by -0.0002 on tkgl-polecat (present, below the bar) and +0.0010 on tkgl-smallpedia (not present under the timestamp-block bootstrap; a gain >= 0.005 is excluded). G1 was therefore not materially understated: the base-contribution range HRCOV - G1 moves from +0.0073..+0.0558 to +0.0064..+0.0560. The RecB-score-as-feature loss on tkgl-smallpedia TiRGN persists with cross-fitted parameters (HR19 -0.0266, HRCOV_PAT -0.0065; with trained parameters in the same run -0.0304 / -0.0080; v7.4, harness, bridge runs: -0.0269 / -0.0119). Cross-fitting recovers only 0.0015-0.0038, so the symptom is not explained by selecting RecB's parameters on the ranker's fit labels; on tkgl-polecat all RecB-feature effects stay within +-0.0003.

**Caveats.** Validation-select has fewer queries per relation than full validation (smallpedia: 70 relations fall back to defaults). The v7.4 RecB-feature values come from different (bridge) runs scored with the harness evaluator and are not paired with these. The default-mode RecB sensitivity for G1 was not run.

## M6 - base-seed variation on tkgl-smallpedia (lowest priority, descriptive)

**RE-GCN TGB configuration, tkgl-smallpedia** (3 base seeds)

| Base seed | Base | BLEND | POOL | HRCOV_PAT | Margin (HRCOV_PAT - better control) | 95% CI (ts-block) |
|---|---|---|---|---|---|---|
| 1 | 0.5907 | 0.6249 | 0.6249 | 0.6586 | +0.0336 | [+0.0290, +0.0364] |
| 43 | 0.5889 | 0.6248 | 0.6247 | 0.6578 | +0.0330 | [+0.0282, +0.0362] |
| 44 | 0.5907 | 0.6251 | 0.6251 | 0.6586 | +0.0335 | [+0.0290, +0.0364] |

Margin mean 0.0334 +- 0.0003 (SD, n-1), range 0.0330..0.0336; base MRR 0.5901 +- 0.0010; HRCOV_PAT MRR 0.6578..0.6586.

Seeds 43/44 on <HOST> (RTX 4090), G3 recipe unchanged except --seed; finiteness: 0 NaN/+inf, -inf only at padded slots.

**TiRGN, tkgl-smallpedia** (3 base seeds)

| Base seed | Base | BLEND | POOL | HRCOV_PAT | Margin (HRCOV_PAT - better control) | 95% CI (ts-block) |
|---|---|---|---|---|---|---|
| 42 | 0.6181 | 0.6275 | 0.6274 | 0.6584 | +0.0310 | [+0.0270, +0.0333] |
| 43 | 0.6135 | 0.6234 | 0.6234 | 0.6576 | +0.0342 | [+0.0290, +0.0374] |
| 44 | 0.6150 | 0.6283 | 0.6283 | 0.6583 | +0.0299 | [+0.0260, +0.0323] |

Margin mean 0.0317 +- 0.0022 (SD, n-1), range 0.0299..0.0342; base MRR 0.6155 +- 0.0023; HRCOV_PAT MRR 0.6576..0.6584.

Seeds 43/44: sp_tirgn.py recipe unchanged except --seed (<HOST> GPU 0, job); best epoch 16 / 22 of 30, full-validation MRR 0.6515 / 0.6539, training 857 / 1039 s; controls and HRCOV_PAT pipeline as for seed 42 (selection on validation only). Finiteness: dumps, controls and HRCOV_PAT outputs 0 NaN/inf; the 53,355,889 -inf values of the two seeds' feature log-prob tables are all at padded candidate slots (0 at real candidates).

Existing runs counted as one seed each (TiRGN pilot-2 seed 42; RE-GCN G3 seed 1, the same TGB script and defaults). The intervals are conditional on each trained base; the SD over base seeds is the descriptive between-seed spread.

**Reading.** Every base seed keeps a present margin over its better recurrence control (lowest timestamp-block lower bound +0.0260). The between-seed SD of the margin is 0.0022 for TiRGN and 0.0003 for RE-GCN. For TiRGN the variation comes from the base and its control (base SD 0.0023; BLEND 0.6234..0.6283), while HRCOV_PAT stays within a range of 0.0008. The single-seed TiRGN margin reported earlier (+0.0310, seed 42) lies within the three-seed range. Completed 2026-10-15, well inside the window (stop rule: no new seed after 2026-10-19T10:00Z).

## Deviations and clarifications

- **D1 (M2b, amendment (registered at 12:49:21Z, after the discarded first LMART run had printed round-1 test MRRs; no test number of the amended statistic existed)).** LMART selection statistic and exact ranks resolve ties as the promoted list does (the evaluator's statistic for tied scores). The first LMART run (<HOST> job, round-1 selection at every grid point) is discarded, including the round-1 test MRRs its log printed and the LMART rows of that run's official files. Detection: every grid point of the first run stopped at round 1 on the validation-select statistic; the cause (the v7 statistic counts tied tree scores optimistically) was diagnosed on validation rows. The round-1 test MRRs printed by that run's log were seen before the amendment was written; they were not used to choose anything. Verification of the tie-exact rank (disclosure of the full check history): (i) identity with the v7 statistic on untied scores, max |difference| 0.0; (ii) a first tie check agreed with the list-position rank for 205 of 240 queries because its synthetic candidate rows drew entity ids with replacement (duplicate ids within a row make the id-based reference ambiguous; real candidate rows have unique ids); (iii) after redrawing the synthetic rows with unique ids (permutations), agreement was 360/360 (ties at scale 2) and 353/353 (scale 0.5). The registered amendment lists only (i) and (iii).
- **D2 (M2b (RE-GCN TGB base, official scoring only), clarification, no effect on any result).** v76_p2_serverA.sh line 23 and v76_p2b_serverA.sh line 21 passed the RE-GCN test full-row file in both slots of --fullrows. The validation slot is never opened in --stage test: hd_blend.py line 280 sets sp_need = ('test',), line 333 opens only the test path, and fullrow_source (lines 229-241) memory-maps only the needed split, yielding facts only for validation snapshots. Every RTROWS call was --stage test --no_controls (v76_p2_serverA.sh:79, v76_p2b_serverA.sh:26). No selection or early stopping used these files; the RE-GCN controls are the stored G3 controls, selected on <HOST> with regcn_valid_logp.f32. Identity: rr_base of both RE-GCN official files of the run is byte-identical to the stored G3 base (sha256 of the float32 array 79b9b324cd7f6140...; MRR 0.590669). Fix: the slot now reads /nonexistent/UNUSED_IN_TEST_STAGE (fails loudly if read); the three RE-GCN official passes (g1, g2, lm2) were re-run with it in <HOST> job (all rc = 0, so the slot was never opened). Re-check: all 48 per-query arrays of the three re-run official files are byte-identical to the original files (g1 19/19, g2 16/16, lm2 13/13; rr_base sha256 79b9b324cd7f6140... in all three, equal to the stored G3 base).
- **D3 (infrastructure, no protocol change).** phase-2 job was cancelled after (i) the LMART tie issue (D1), (ii) CUDA out-of-memory of G1_xf features and the LMART latency step while the M3a TiRGN training filled GPU 0, and (iii) a wrong TiRGN tree path for the polecat replay. Those steps were re-run unchanged in job; M3a training, M3a controls, the v76_btrain arm fits and the smallpedia TiRGN / CognTKE / RE-GCN official passes of the MLP/LIN/M4 arms from 69f852c5 completed normally and are used. Cancelling 69f852c5 did not stop four of its docker containers (CognTKE and RE-GCN official passes g1/g2). They kept running on GPU 0 while phase 2b started, and they finished; their outputs are the ones used (the RE-GCN g1/g2 files were reproduced byte for byte in the D2 re-check). Phase 2b reached ALLDONE, and all of its outputs are in the finiteness scan.
- **D4 (M4, note).** hd_recb.py --stage test for RecB_xf used P = 8 for polecat and 16 for smallpedia; P only affects the 'faithful' variant (not used by G1 or the RecB features, which use the clean variant). The stored runs report P_emulated = 8.
- **D5 (rule files, provenance clarification, no protocol change).** The sha256 recorded for v76_rule.json (edcfca14...) is the registration hash produced by write_prereg, not the sha256 of the saved file (4daf8b0f...). write_prereg adds registered_utc, hashes json.dumps(body, indent=1, sort_keys=True, default=str) (keys sorted, no sha256 field), then adds the sha256 field and writes json.dump(body, indent=1) in insertion order. The saved file therefore differs from the hashed bytes in key order and by the embedded sha256 field; its content is unchanged. The rule file has a single version (saved 12:07:47Z, file sha256 4daf8b0f...), and amendment 1 is a separate single-version file (registered 12:49:21Z; registered sha256 bc8cd8f8..., file sha256 f02c5f6b...) that cites edcfca14... It was not appended to the rule file. The exact hashed bytes are now stored as v76_rule_registered.json (aec1141a / 2efa22b5; file sha256 = edcfca14...) and v76_rule_amendment_1_registered.json (a0b4fbae / c9800bf3; = bc8cd8f8...); v76_rule_hash_check.py (5340c102 / 689306e0) re-derives both registered hashes from the saved files and checks the amendment's citation (ALL OK).
