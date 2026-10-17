# v78 external comparators: EdgeReMIND and CountTRuCoLa reproduction and per-query dumps

*Updated 2026-10-16: adds the registered-optional EXT_adapted CountTRuCoLa run (executed after the framing verdict); see the section 'EXT_adapted'.*

Rule: `v78_framing_rule` (sha256 `8e05c67b7ce8f3a4…`, registered 2026-10-15T17:34:21Z), components **EXT_alone** and **reproduction_gate**. Protocol decisions (gate definition, history boundary, CountTRuCoLa config caveat) were saved before any number existed: file `v78_external_protocol_note.json` (version `redacted-id`). This report contains no HRcov, SA, blend or other comparison number.

## Outcome

| Method | Dataset | Published test MRR | Reproduced test MRR (official py-tgb 2.3.0) | Δ | Tolerance | Gate |
|---|---|---|---|---|---|---|
| EdgeReMIND (seed 1337) | tkgl-smallpedia | 0.6136 ± 0.0017 | 0.6148 | +0.0012 | 0.01 | **PASS** |
| CountTRuCoLa | tkgl-smallpedia | 0.644 | 0.6366 | -0.0074 | 0.01 | **PASS** |

* **EdgeReMIND reproduced** on tkgl-smallpedia. The unmodified script with seed 1337 gives test MRR 0.6148 and val 0.6554 (published 0.6136 ± 0.0017 / 0.6554 ± 0.0009; the paper's seed-1337 value is 0.6130). Seeds 1337-1341 give 0.6138 ± 0.0017 test and 0.6554 ± 0.0003 val. A second, independent re-training inside `er_run.py` gave bit-identical val/test MRR, and the per-query `Evaluator.eval` re-score with the authors' cutoff matches the authors' aggregate to within 1e-9.
* **CountTRuCoLa reproduced** on tkgl-smallpedia. The official evaluator gives test MRR 0.6366 (Δ -0.0074 vs 64.4). The authors' own evaluator (random tie-breaking) gives 0.6441, with H@1/3/10 0.5949/0.6874/0.7174; this matches Table 15 (59.5/68.7/71.7) to the reported precision. The 0.0075 gap between the two evaluators comes from the tie convention: in 17% of test queries the gold shares its non-zero score with another candidate, and random tie-breaking gives E[1/rank] > 1/E[rank].
* **tkgl-polecat was assessed for both methods** (stop rule not triggered). EdgeReMIND took 3.9 h, CountTRuCoLa 5.8 h; peak memory was 43 / 103 GB summed PSS and host MemAvailable never fell below 348 GB. With the authors' own protocol, both again agree with the published numbers: EdgeReMIND test 0.1698 vs 0.1707 ± 0.0046 (paper's seed-1337 value 0.1698); CountTRuCoLa 0.2557 official / 0.2557 own vs 25.6.
* EXT_alone numbers for the later decision are the **official, HRcov-boundary** rows marked 'dumped' below. For EdgeReMIND they are lower than the authors'-cutoff numbers (test −0.0017 smallpedia, −0.0044 polecat), because the released evaluation lets a query see earlier 200-event batches of its own timestamp. CountTRuCoLa already respects t′ < t.
* All 8 dumps (2 methods × 2 datasets × {val, test}) pass the independent alignment and consistency check (`MANIFEST.json`): gold equals the harness gold on every row, with 0 mismatches.

## Setup

* Host: serverA (CPU only); both methods are CPU-only. Isolated venvs under `v78_external/env/` (`er`: torch 2.8.0+cpu, torch-geometric 2.8.1, py-tgb 2.3.0, tgm editable from the clone; `ct`: the repository's `requirements.txt`, ray 2.59.0, with the repository's vendored `tgb`). Freezes: `env/er_freeze.txt`, `env/ct_freeze.txt`.
* EdgeReMIND: https://github.com/BryantPollard/EdgeReMIND @ `41c0db2df31bf4ab60e6e4fec8176932314e3ec8` (a TGM fork). Configuration is the released one, identical for all datasets (γ = 2, n = 3, 30 epochs, Adam lr 1e-3, K = 20 relation-aware negatives, smoothing window 3, ≤ 2 M training queries, base decay 1/(7·86400), 200-event loader batches).
* CountTRuCoLa: https://github.com/JuliaGast/counttrucola_submission @ `5d9aedf4106926d13befa141e0855d684f3f5f9d`; repository cited in arXiv:2509.09474v4 (26 Aug 2026, latest); JuliaGast/counttrucola (HEAD 9523726, 2026-10-15) not used. Configuration: `rule_based/config-default.yaml` with its `DATASET_OVERRIDES` (run as `rule_based/main.py --params DATASET_NAME=<ds> NUM_CPUS=<n>`).
* Data: TGB 2.0 zips cached by the harness (`tracks/datasets/raw/`, md5 tkgl-smallpedia 3eb574d2…, tkgl-polecat 3ef419cf…, both equal to the md5 published for the TGB 2.0 datasets), staged into each venv's dataset directory; nothing was downloaded or deleted elsewhere.
* Scoring: per query `tgb.linkproppred.evaluate.Evaluator.eval` (py-tgb 2.3.0) on the gold score and the scores of the shipped negatives (`TKGNegativeEdgeSampler.query_batch`, 1-vs-all, time-aware filtered), ties averaged. H@k use the evaluator's averaged rank (rank ≤ k), as in the harness. Both query directions (inverse relations r + R), as in TGB 2.0 and the harness.

## Published vs reproduced numbers

### tkgl-smallpedia

| Run | split | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| EdgeReMIND published (5 seeds) | test | 0.6136 ± 0.0017 | — | — | — |
| EdgeReMIND published (5 seeds) | val | 0.6554 ± 0.0009 | — | — | — |
| EdgeReMIND published, seed 1337 (Fig. 5) | test | 0.6130 | — | — | — |
| EdgeReMIND reproduced, authors' cutoff (gate) | test | 0.6148 | 0.5413 | 0.6824 | 0.7199 |
| EdgeReMIND reproduced, authors' cutoff (gate) | val | 0.6554 | 0.5985 | 0.7033 | 0.7342 |
| EdgeReMIND, HRcov boundary t′ < t (**dumped, EXT_alone**) | test | 0.6131 | 0.5387 | 0.6811 | 0.7190 |
| EdgeReMIND, HRcov boundary t′ < t (**dumped, EXT_alone**) | val | 0.6540 | 0.5973 | 0.7017 | 0.7320 |
| CountTRuCoLa published | test | 0.644 | 0.595 | 0.687 | 0.717 |
| CountTRuCoLa reproduced, authors' evaluator (random ties) | test | 0.6441 | 0.5949 | 0.6874 | 0.7174 |
| CountTRuCoLa reproduced, authors' evaluator (random ties) | val | 0.6666 | 0.6271 | 0.7015 | 0.7241 |
| CountTRuCoLa reproduced, official evaluator (**gate, dumped, EXT_alone**) | test | 0.6366 | 0.5748 | 0.6813 | 0.7102 |
| CountTRuCoLa reproduced, official evaluator (**gate, dumped, EXT_alone**) | val | 0.6610 | 0.6135 | 0.6951 | 0.7215 |

### tkgl-polecat

| Run | split | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| EdgeReMIND published (5 seeds) | test | 0.1707 ± 0.0046 | — | — | — |
| EdgeReMIND published (5 seeds) | val | 0.1769 ± 0.0049 | — | — | — |
| EdgeReMIND published, seed 1337 (Fig. 5) | test | 0.1698 | — | — | — |
| EdgeReMIND reproduced, authors' cutoff | test | 0.1698 | — | — | — |
| EdgeReMIND reproduced, authors' cutoff | val | 0.1757 | — | — | — |
| EdgeReMIND, HRcov boundary t′ < t (**dumped, EXT_alone**) | test | 0.1655 | 0.1001 | 0.1848 | 0.2961 |
| EdgeReMIND, HRcov boundary t′ < t (**dumped, EXT_alone**) | val | 0.1705 | 0.1085 | 0.1874 | 0.2973 |
| CountTRuCoLa published | test | 0.256 | 0.178 | 0.291 | 0.408 |
| CountTRuCoLa reproduced, authors' evaluator (random ties) | test | 0.2557 | 0.1785 | 0.2911 | 0.4082 |
| CountTRuCoLa reproduced, authors' evaluator (random ties) | val | 0.2625 | 0.1867 | 0.2981 | 0.4101 |
| CountTRuCoLa reproduced, official evaluator (**dumped, EXT_alone**) | test | 0.2557 | 0.1785 | 0.2911 | 0.4082 |
| CountTRuCoLa reproduced, official evaluator (**dumped, EXT_alone**) | val | 0.2625 | 0.1866 | 0.2981 | 0.4101 |

The EdgeReMIND rows with the authors' cutoff on polecat have no H@k: the released `evaluate_parallel` returns MRR only, and per-query scoring with the authors' cutoff was run for smallpedia only. The papers do not report H@k for EdgeReMIND.

### EdgeReMIND tkgl-smallpedia seed spread (unmodified script, gate statistics only; no dumps for seeds ≠ 1337)

| seed | test MRR | val MRR |
|---|---|---|
| 1337 | 0.6148 | 0.6554 |
| 1338 | 0.6159 | 0.6555 |
| 1339 | 0.6117 | 0.6551 |
| 1340 | 0.6143 | 0.6558 |
| 1341 | 0.6125 | 0.6555 |
| mean ± SD (n = 5) | 0.6138 ± 0.0017 | 0.6554 ± 0.0003 |
| published (n = 5, same seeds) | 0.6136 ± 0.0017 | 0.6554 ± 0.0009 |

## Resources (serverA, 48 cores / 503 GB, shared)

| Run | workers | wall | peak memory (summed PSS of process tree) |
|---|---|---|---|
| EdgeReMIND smallpedia gate, seed 1337 (unmodified script) | 16 | 11.3 min | 9.5 GB (RSS-sum 32.3 GB) |
| EdgeReMIND smallpedia seeds 1338-1341 (unmodified script) | 16 | 10.7, 11.9, 12.3, 12.0 min | 11.1, 9.2, 10.1, 10.1 GB |
| EdgeReMIND smallpedia dump re-score (val+test, both cutoffs) | 12 | 18.8 min | 10.7 GB |
| EdgeReMIND polecat (train/select 1.23 h, authors' test eval 0.94 h, strict dumps val+test 1.77 h) | 16 | 3.94 h | 42.8 GB (RSS-sum 118.9 GB; paper 69.6 GB) |
| CountTRuCoLa smallpedia main.py (examples + learn + apply + own eval) | 16 | 19.7 min | 8.5 GB |
| CountTRuCoLa smallpedia official re-score (val+test) | 16 | 0.5 min | 2.1 GB |
| CountTRuCoLa polecat main.py | 12 | 5.82 h (paper 3.8 h with 20 threads) | 103.1 GB |
| CountTRuCoLa polecat official re-score (val+test) | 12 | 3.3 min | 11.1 GB |

Peak memory is the maximum over 10-s samples of the summed proportional set size of the run's process tree (fork-shared pages counted once); the summed RSS, which double-counts shared pages and is closer to what the EdgeReMIND paper reports, is given in `ext_repro_results.json`. A guard killed the run if host MemAvailable fell below 80 GB; it never triggered (minimum host MemAvailable during the runs is recorded in the same file).

## Dumps

Root: `$WORK_DIR/tracks/v78_external/dumps/<method>/<dataset>/{val,test}/` (symlinks into `v78_external/runs/...`). The format is described in `v78_external/README.md`; the checks are in `v78_external/MANIFEST.json`. Total size about 27 GB of dumps plus 6.7 GB of CountTRuCoLa rankings files in `repos/counttrucola_submission/files/rankings/`.

| dump | rows (val / test) | top-K | gold in top-K (test) | tail constant (test) |
|---|---|---|---|---|
| edgeremind_s1337/tkgl-smallpedia | 162,066 / 163,172 | 1000 | 0.833 | 0.0016 |
| edgeremind_s1337/tkgl-polecat | 533,472 / 532,636 | 1000 | 0.682 | 0.0000 |
| counttrucola/tkgl-smallpedia | 162,066 / 163,172 | 1000 (≤ 200 scored) | 0.721 | 1.0000 |
| counttrucola/tkgl-polecat | 533,472 / 532,636 | 1000 (≤ 200 scored) | 0.658 | 1.0000 |

Each split directory holds `gold.npy` [Q] int32, `keys_harness.npy` [Q,4] (s, r, o, ts_id), `top_ids.npy` [Q,1000] int32 (score descending, ties by id), `top_scores.npy` [Q,1000] (float32 for EdgeReMIND, float64 for CountTRuCoLa, the native dtypes), `floor.npy` (score used for every entity outside the list), `tail_min.npy`, `n_gt_floor.npy`, `gold_score.npy`, `rr_official.npy` and `rank_official.npy` (official evaluator on the full score vector; use these for EXT_alone). A `meta.json` per method/dataset records commit, config, metrics, timings and alignment. Full row: `s = full(N, floor); s[top_ids] = top_scores`. This is exact for CountTRuCoLa and an upper-bound approximation of the tail for EdgeReMIND.

## Alignment check

* Harness order: `split_queries` of `hardening/v76/code/hd_blend.py` (stable sort by ts id; per timestamp the forward block in file order, then the inverse block (o, r+R, s, t)). Entity, relation and TGB ids are identical (harness assertion, re-checked here: py-tgb 2.3.0 codes the inverse relation as r + 283 / r + 16 = r + R).
* EdgeReMIND: per-query outputs were produced in the TGM/TGB event order and mapped to harness rows by the (s, r, o, raw t) key. On both datasets the two multisets are identical, and the mapping is the identity permutation: the TGM stream is already in harness order.
* CountTRuCoLa: rows are generated directly in harness order. Each row's rankings line is looked up by (s, r, CountTRuCoLa ts id), using the timestamp-id rule copied from `RuleDataset.reformat_ts`. On all four splits there are 0 rankings lines without a harness query and 0 harness queries without a line.
* Independent check (`code/verify_dumps.py` → `MANIFEST.json`), all 8 dumps PASS. gold.npy equals the harness gold recomputed from `harness_data/<ds>/<split>.txt` and equals column 2 of the harness's stored query list `hardening/v76/recb_xf/recb_csr_<split>_<ds>.npz['sro']` (0 mismatches on 162,066 / 163,172 / 533,472 / 532,636 rows). keys_harness equals that stored list. The lists are sorted and duplicate-free, with floor ≤ last listed score. The official rank never exceeds the bound implied by the listed scores, and MRR/H@k recomputed from `rr_official`/`rank_official` equal meta.json.

## EXT_adapted: CountTRuCoLa with the validation fit window in its learning data

**Label: registered-optional comparator run (amendment `v78_rule_amendment_1`, field `EXT.EXT_adapted`), executed after the v78 framing verdict was read.** The protocol decisions were saved before the run: file `v78_external_protocol_note_2_ext_adapted.json` (version `redacted-id`). No HRcov comparison or blend is computed here.

* **Adaptation:** the released code and config (commit 5d9aedf, same `DATASET_OVERRIDES`) were run from a code-identical copy (`repos/counttrucola_adapted`). One wrapper change was made: every validation fact with harness ts id < cut moves from the TGB val mask to the train mask before `RuleDataset` indexes the data. Rule examples, rules and confidence parameters are therefore re-learned on train plus the validation fit window, while the application history (t′ < t, union of all splits) is unchanged.
* **Split (the harness's):** cut = `uts[int(0.7·len(uts))]` over the harness validation rows' timestamps.
  * tkgl-smallpedia: cut ts id 105 (year 2005); 105,606 fit rows; 56,460 selection rows. Split file `hardening/v78/smallpedia/hist/valid_hist.npz`, sha256 `f4b4910d…`.
  * tkgl-polecat: cut ts id 1423 (raw 1637730000). Harness rows are in `pc_tirgn/work/valid_sample_idx.npy` (sha256 `604f241d…`), identical to `hardening/v78/polecat/hist/valid_hist.npz['rows']` (sha256 `363f10c9…`). The sample has 60,764 selection rows. All 372,024 validation facts below the cut were moved (deviation 2).
* **No choice was made on the selection part:** the settings are the released ones, so there was nothing to select.

| Dataset | Run | Selection part (harness rows) MRR | H@1 | H@3 | H@10 | Test MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|---|---|---|
| tkgl-smallpedia | **EXT_adapted, official** (n_sel = 56,460) | **0.6385** | 0.5842 | 0.6785 | 0.7096 | **0.6373** | 0.5763 | 0.6817 | 0.7109 |
| tkgl-smallpedia | EXT_alone (faithful) on the same rows, official | 0.6374 | 0.5825 | 0.6774 | 0.7088 | 0.6366 | 0.5748 | 0.6813 | 0.7102 |
| tkgl-smallpedia | EXT_adapted, authors' evaluator (random ties; selection = all ts ≥ cut) | 0.6439 | 0.5989 | 0.6834 | 0.7095 | 0.6444 | 0.5955 | 0.6877 | 0.7176 |
| tkgl-polecat | **EXT_adapted, official** (n_sel = 60,764) | **0.2649** | 0.1872 | 0.3010 | 0.4160 | **0.2559** | 0.1786 | 0.2911 | 0.4083 |
| tkgl-polecat | EXT_alone (faithful) on the same rows, official | 0.2647 | 0.1872 | 0.3007 | 0.4157 | 0.2557 | 0.1785 | 0.2911 | 0.4082 |
| tkgl-polecat | EXT_adapted, authors' evaluator (random ties; selection = all ts ≥ cut) | 0.2655 | 0.1875 | 0.3012 | 0.4180 | 0.2559 | 0.1787 | 0.2911 | 0.4083 |

On tkgl-polecat, over all 161,448 validation rows with ts ≥ cut, EXT_adapted gives 0.2655 versus 0.2653 for EXT_alone (descriptive). The EXT_alone rows are given only so the adapted run can be read in context; choosing between the variants is the hardening track's `external_comparator` step.

Resources (serverA, 12 CPUs each, threads pinned):

| Dataset | main.py wall | peak memory (summed PSS) | official re-score | rules |
|---|---|---|---|---|
| tkgl-smallpedia | 3.6 min | 6.9 GB (host MemAvailable ≥ 467 GB) | 0.5 min, 3.2 GB | 1577 (faithful: 1223) |
| tkgl-polecat | 159.8 min | 105.0 GB (host MemAvailable ≥ 387 GB) | 2.3 min, 10.6 GB | 1024 (faithful: 1024) |

Dumps: `v78_external/dumps/counttrucola_adapted/<dataset>/{val,test}/`, in the same format and harness order as `counttrucola`, plus `in_learning_data.npy`. Validation rows inside the learning data are NaN and must be skipped. The README section `counttrucola_adapted` documents this. The independent check (`MANIFEST.json`, now 12 dumps) passes for all dumps: gold equals the harness gold on every row with 0 mismatches, learning rows are all NaN, and the metrics recompute exactly.

Deviations (EXT_adapted):

1. Timing/label: run after the v78 framing verdict was read; registered-optional comparator (amendment field EXT.EXT_adapted). Not part of the registered verdict computation unless the hardening track applies the rule's external_comparator selection to it.
2. tkgl-polecat fit part: all 372,024 validation facts with ts id < 1423 were moved into the learning data, not only the 139,236 fit rows of the 200,000-row HRcov sample. The cut is the harness's (identical over the sample and over all valid rows). Reason, recorded before the run: CountTRuCoLa's learning counts every head fact absent from its train index as a negative, so moving only sampled facts would make about 62% of the true fit-window facts negatives. Consequence: CountTRuCoLa has more fit-window label access than the HRcov ranker (<= 120,000 fit queries). tkgl-smallpedia: both readings coincide (105,606 rows).
3. Implementation: one in-memory change (TGB val/train masks) in a wrapper around the released main.py, on a code-identical repo copy. The first attempt for both datasets placed the wrapper at the repo root, so ray workers could not import rule_based modules (ModuleNotFoundError: apply_dataset). The polecat attempt was killed after about 2 min. The smallpedia attempt ended without usable rankings, and its run directory was overwritten by the re-run. No number from either attempt was read or used. The wrapper was moved to rule_based/ (as the released entry point) and both runs were restarted from scratch.
4. Validation dumps: rows with ts id < cut were learning data. No ranking exists for them; they are flagged in_learning_data=True with NaN scores/rr/rank (top_ids = 0..999, n_cand = n_gt_floor = -1). Only selection-window rows are scored.
5. No hyper-parameter or variant choice was made (released settings), so the selection part was used only for reporting. CPU count 12 for both datasets (faithful smallpedia run: 16); resource setting only. BLAS/OpenMP threads pinned to 1.
6. z-/f-rule statistics follow the released code on the modified splits. In val-mode application the train-split counts now include the fit window, which lies entirely before every selection query. In test-mode application train+val totals are unchanged, so test predictions change only through the re-learned rules and confidence parameters.
7. Out of scope / not done: EdgeReMIND adapted, adapted blend, any HRcov comparison or bootstrap.

## Notes for the consumer (hardening agent)

* EXT_alone per-query rr: dumps/<method>/<ds>/test/rr_official.npy (harness test order).
* Blending: rebuild a row as full(N, floor) with top_ids/top_scores; EdgeReMIND scores are raw linear scores (not probabilities), CountTRuCoLa scores are noisy-or confidences in [0,1] with floor 0.
* Polecat valid: the dumps cover all 533,472 rows; index with pilot_polecat/work/valid_sample_idx.npy if the HRcov fit/selection parts refer to the subsample.

## Deviations

1. Seeds: per-query dumps exist for EdgeReMIND seed 1337 only. The gate number is seed 1337; seeds 1338-1341 were added as unmodified-script runs only (no dumps), giving a 5-seed test MRR of 0.6138 ± 0.0017 (paper: 0.6136 ± 0.0017). EdgeReMIND polecat: seed 1337 only (paper: 5 seeds). CountTRuCoLa: one run per dataset; the method takes no seed argument and the paper also reports one run.
2. BLAS/OpenMP threads were pinned to 1 per process (OMP/OPENBLAS/MKL_NUM_THREADS=1) for every run. The first EdgeReMIND smallpedia attempt used the library defaults: 48 threads in each of 16 workers, host load about 500. I killed it after about 15 min and used none of its numbers. This is a resource setting only; the re-trained theta reproduced the gate run's val/test MRR bit for bit.
3. Worker counts are resource settings, not hyper-parameters, and they differ from the protocol note's '16 workers per method'. EdgeReMIND used --num-workers 16 for the gate runs and polecat, and 12 for the smallpedia dump re-score, which overlapped other runs. CountTRuCoLa used NUM_CPUS=16 on smallpedia and 12 on polecat, to keep cores free for co-tenants, instead of the config default of 1 (the paper used up to 20 threads). The 26 h run_timeout set on the polecat jobs was only a scheduler backstop; the pre-registered stop rule (<= 24 h wall per seed) was evaluated on measured wall time: 3.9 h (EdgeReMIND) and 5.8 h (CountTRuCoLa).
4. py-tgb 2.3.0 was used (required by the EdgeReMIND repo and used by our harness), not the py-tgb 2.2.0 named in the EdgeReMIND paper. CountTRuCoLa's own run uses its vendored tgb copy; our official re-score uses py-tgb 2.3.0.
5. EdgeReMIND history boundary in the DUMPS (EXT_alone): every query was re-scored with the trained, unchanged theta and the strict cutoff t' < t (the HRcov boundary). The released code instead cuts history at the start of each 200-event batch: earlier batches of the same timestamp are visible and the current batch is hidden. Effect on test MRR: smallpedia 0.6148 -> 0.6131, polecat 0.1698 -> 0.1655. The gate uses the authors' cutoff. Per-query arrays for the authors' cutoff were dumped for smallpedia only (polecat: aggregate MRR only).
6. EdgeReMIND tkgl-polecat: the unmodified example script was not run separately. er_run.py runs the verbatim training/selection code and the authors' evaluate_parallel; on smallpedia the two paths gave bit-identical val and test MRR.
7. CountTRuCoLa gate number = official py-tgb Evaluator over its rankings with ties averaged, as pre-declared. The authors' own evaluation breaks ties randomly (EVAL_TYPE=random): smallpedia test 0.6441 vs 0.6366 official; polecat 0.2557 under both. Both are reported; the dumps carry the official per-query rr. The gold shares its non-zero score with at least one other listed candidate in 17.0% of smallpedia test queries (2.0% on polecat), which accounts for the gap.
8. The CountTRuCoLa released config differs from paper Table 6 for smallpedia: the code sets C_X_COUNT=20 (large_data_flag: > 1M quads incl. inverses), while Table 6 lists Cx-count 3. I used the released config unchanged, as pre-declared. A Table-6 run was not needed because the gate passed.
9. CountTRuCoLa dumps: the released code stores at most 200 non-zero candidates per query (large_data_flag, both datasets). Each top-1000 list therefore holds <= 200 scored entities, followed by ties at the floor 0 ordered by entity id (n_gt_floor and n_cand give the split).
10. EdgeReMIND dump floor = highest score outside the top-1000, an upper bound of the tail. The tail is constant in only 0.16% of smallpedia test queries and 0% on polecat, because scores are linear in history features and can be negative. The exact per-query official rr/rank and the exact gold score are stored separately and do not depend on the truncation.
11. Validation dumps cover all harness valid rows. The harness's polecat HRcov fitting used a valid subsample (pilot_polecat/work/valid_sample_idx.npy), which the consumer must index.
12. Script bugs were fixed during the run, with no effect on numbers. (a) The first CountTRuCoLa smallpedia val re-score matched the test rankings file ('pvalue' contains 'val') and was re-run. (b) The EdgeReMIND smallpedia dump needed two restarts from the saved theta: one for a wrong assertion that the strict cutoff is never later than the authors' cutoff (it is later for 0.5-6% of queries, when a batch straddles a timestamp boundary), one for a dict-cleanup bug. The history boundary of the dumps was not changed by fix (b): every dumped score uses the pre-registered strict cutoff (index position of the first event at t, verified per row by asserting ALL_T[cut-1] < t and ALL_T[cut] == t). Only an incorrect consistency assertion comparing it with the authors' batch cutoff was dropped; that comparison is logged as cutoff_stats in meta.json.
13. The CountTRuCoLa smallpedia wall time (1179 s vs 280 s in the paper) is inflated: the run overlapped the oversubscribed first EdgeReMIND attempt.
14. Only serverA was used; serverB was not needed. A VPN outage (about 2026-10-15T21:35Z to 2026-10-16T06:15Z) cut SSH to both hosts. Both polecat jobs ran detached, finished during the outage (EdgeReMIND 22:40Z, CountTRuCoLa 00:07Z) and were harvested intact afterwards. No job died and nothing was restarted from a checkpoint.
15. Not in scope / not done: EXT_adapted, EXT_blend, SA_ERM, any HRcov comparison or bootstrap.

