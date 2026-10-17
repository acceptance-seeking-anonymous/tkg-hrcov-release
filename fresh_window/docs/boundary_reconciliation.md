# Window boundaries: GDELT25X vs GDELTFW (reviewer item 4)

**Short answer.** The 7-month shift was applied to the builder's calendar boundaries, as the registered rule says. The
GDELT25X train split ends on 14 June 2025 because the official GDELT server has no event exports for 15 June to 1 July 2025.
The builder's train/validation boundary was 1 July 2025, so this 17-day outage left the last 16 days of the training interval
and the first day of validation without facts. GDELTFW has no outage, so its training split runs to its boundary,
31 January 2026. The applied rule matches the registered rule text; there is no deviation and no rerun. The
manuscript should describe the effective GDELT25X extent and the shifted boundaries separately (§4).

All records were read without modification:
- serverB job `29022264` (the GDELT25X build): command, wrapper, builder, build summary, log and missing-export list;
- serverA: the GDELT25X snapshot used by the forward holdout and its `timestamp2id.txt`;
- the code-release data directory;
- the GDELTFW build job `faac377d`.

Copies are in the log bundle v2 in `fresh_window/logs/`, under `boundary_sources/` and `track/data/GDELTFW/`.

## 1. Builder arguments

The interval is half-open, `[start, end)`. Days are the DATEADDED UTC day. Train is `start <= day < valid`; validation is
`valid <= day < test`; test is `test <= day < end`.

| argument | GDELT25X (built 2026-10-13, serverB) | GDELTFW (built 2026-10-16, serverA) | shift |
|---|---|---|---|
| `--start` | 2025-01-01 | 2025-08-01 | +7 months |
| `--valid` (first validation day) | 2025-07-01 | 2026-02-01 | +7 months |
| `--test` (first test day) | 2025-08-01 | 2026-03-01 | +7 months |
| `--end` (exclusive) | 2025-10-01 | 2026-05-01 | +7 months |
| builder | `ps_build_gdelt.py`. The dates are set as `PS_G_START=20250101 PS_G_END=20251001 PS_G_VALID=20250701 PS_G_TEST=20250801` in `ps_job_ccA.sh` (`boundary_sources/serverB_GDELT25X_build_job_29022264/`). | `build_gdelt25x.py --start 20250801 --end 20260501 --valid 20260201 --test 20260301` (`jobs/faac377d…/cmd.sh:L10`) | |
| export files requested / retrieved | 26,208 / 24,527 (1,681 missing) | 26,208 / 26,208 (0 missing, no skip list) | |

The two builders are the same code path, from the export filters to the split files. The release builder adds a skip-list
option and a relation-name table, and drops the corpus/document outputs that the story pilot needed.

## 2. First and last day with events per split

Time ids are ranks of the days that have at least one retained event. A day without events gets no id.

| split | GDELT25X first day | last day | ts ids | days with events / calendar days | facts | GDELTFW first day | last day | ts ids | days with events / calendar days | facts |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2025-01-01 | **2025-06-14** | 0–164 | 165 / 181 | 514,299 | 2025-08-01 | 2026-01-31 | 0–183 | 184 / 184 | 509,717 |
| valid | **2025-07-02** | 2025-07-31 | 165–194 | 30 / 31 | 94,778 | 2026-02-01 | 2026-02-28 | 184–211 | 28 / 28 | 76,117 |
| test | 2025-08-01 | 2025-09-30 | 195–255 | 61 / 61 | 182,359 | 2026-03-01 | 2026-04-30 | 212–272 | 61 / 61 | 228,584 |
| all | | | 256 ids | 256 / 273 | 791,436 | | | 273 ids | 273 / 273 | 814,418 |

Sources:
- GDELT25X: `boundary_sources/serverA_GDELT25X_snapshot/timestamp2id.txt` and `boundary_sources/serverB_GDELT25X_build_job_29022264/build_gdelt25x_summary.json` (`t_range` per split).
- GDELTFW: `track/data/GDELTFW/timestamp2id.txt` and `track/data/GDELTFW/build_summary.json`.

## 3. Missing or skipped export files

GDELTFW has none: `missing_export_files.txt` is empty and `build.log` reports "downloaded 26208 / 26208 (skipped by list: 0)".

GDELT25X is missing 1,681 of 26,208 files. The list written by the build job is byte-identical to the release copy
`gdelt25x_missing_export_files.txt`.

| date (UTC) | missing 15-minute files | split |
|---|---|---|
| 2025-06-15 … 2025-06-30 (16 days) | 96 of 96 on every day (1,536) | train interval |
| 2025-07-01 | 96 of 96 | validation interval |
| 2025-06-14 | 24 | train |
| 2025-06-12 | 9 | train |
| 2025-07-02 | 9 | validation |
| 2025-01-05, 03-11, 03-27, 04-24, 04-29, 05-03, 05-06 | 1 each | train |
| 2025-08-01 … 2025-09-30 | 0 | test |

By split: train interval 1,576, validation 105, test 0. The 17 days with all files missing are exactly the 17 calendar
days that have no time id.

The builder tried each file three times (60 s timeout). On 2026-10-17 at 02:47 UTC, a HEAD request to the official server
returned 404 for two of the listed files (2025-06-20 12:00 and 2025-07-01 00:00) and 200 for two controls (2025-06-13 and
2025-06-14 12:00). Only these four files were checked. The gap therefore appears to be in GDELT's published exports rather
than in our download, but the check is not exhaustive.

## 4. Where 14 June 2025 comes from

14 June 2025 is the last day with events in the GDELT25X train split. It is not a builder argument. The train boundary was
1 July 2025, but no export exists from 15 June 00:00 to 1 July 23:45, so 15–30 June add no facts. Validation's first day with
events is 2 July, not 1 July.

The manuscript reports the effective extent correctly ("training 2025-01-01 to 2025-06-14", Table `tab:datasets`
footnote; §Forward holdout). Shifting that effective end by 7 months gives 14 January 2026. Shifting the recipe boundary,
which the rule prescribes, gives 31 January 2026 as the last training day. Because GDELTFW has every export, its training
split reaches that boundary.

## 5. Interval rule that was applied

GDELTFW = the GDELT25X builder with each of its four date arguments moved by +7 calendar months:
- `start` 2025-01-01 → 2025-08-01
- `valid` 2025-07-01 → 2026-02-01
- `test` 2025-08-01 → 2026-03-01
- `end` 2025-10-01 → 2026-05-01

Splits are half-open DATEADDED-day intervals. Everything else (filters, entity and relation coding, deduplication, time
indexing) is unchanged.

## 6. Does it match the registered rule text?

**Yes.** The rule's data clause (`v79_fresh_window_rule.json`, registered 2026-10-16 14:24:12 UTC) reads: "Every date of
the GDELT25X recipe shifted forward by exactly 7 calendar months." It derives the new test window 2026-03-01..2026-04-30
from the GDELT25X test window and states that "validation and training windows shift by the same offset". The recipe's
dates are the four builder arguments, and each moved by exactly 7 calendar months, so **there is no deviation from the
registered rule**.

The difference the reviewer found comes from the GDELT25X export outage, not from the shift:
- Read as effective data extents, GDELT25X trains to 14 June and validates from 2 July.
- GDELTFW trains to 31 January and validates from 1 February.
- So GDELTFW's training split covers the 17 days at the end of its interval that GDELT25X lacked.

**Manuscript text to correct (no rerun needed):**
1. §6.4, "with every date shifted forward by seven months", should name the shifted dates. Suggested wording: "the GDELT25X builder with its four calendar boundaries (2025-01-01, 2025-07-01, 2025-08-01, 2025-10-01) shifted forward by seven months; GDELT25X's training data end on 2025-06-14 only because GDELT published no event exports for 15 June to 1 July 2025, whereas the new window has every export".
2. §6.4, "its first two months are the validation and test months of GDELT25X", is inaccurate. GDELTFW's first two training months (August and September 2025) are the **test** months of GDELT25X. GDELT25X validated on July 2025, which lies outside GDELTFW.
3. The dataset table footnote and §Forward holdout say GDELT25X "validates on July 2025". Its first validation day with events is 2 July, because 1 July has no exports. Optional precision.

## 7. Which data selected the entity and relation vocabulary

**The training split only, in both datasets.** Both builders apply the same steps:
1. The filters: root events, both actors coded, at least one actor country in the Middle-East set, URL first-seen day, self-loops removed.
2. Keep entities with at least 10 occurrences as subject or object among train events, and CAMEO event codes with at least 5 train events.
3. Drop facts of every split that use any other entity or relation (`ps_build_gdelt.py` and `build_gdelt25x.py`, the lines with `ef[ef >= 10]` and `rf[rf >= 5]`).

| dataset | train events that selected the vocabulary | entities | relations |
|---|---|---|---|
| GDELT25X | DATEADDED 2025-01-01 … 2025-06-14 (effective; 15–30 June have no exports) | 1,173 | 218 |
| GDELTFW | DATEADDED 2025-08-01 … 2026-01-31 | 1,209 | 215 |

Validation and test facts never add entities or relations. In GDELTFW, every entity and relation id of every split occurs
in train: the sanity check of 14:46:37 UTC found 1,209 / 1,209 entities and 215 / 215 relations.
