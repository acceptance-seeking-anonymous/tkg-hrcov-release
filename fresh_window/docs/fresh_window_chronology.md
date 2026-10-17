# Fresh-window chronology (GDELTFW)

**Verified:** No test-row prediction, rank or metric was computed before 2026-10-16 16:34:01 UTC, when Stage 2 finished verifying the 201-file freeze manifest written at 16:32:23 UTC; every test step is launched only after that check, and the first one (TiRGN test rows) began at about 16:34:02 UTC.

**Supplement S12.** Its sentence that no prediction, rank or metric on any split was computed before the freeze is not correct: validation predictions, ranks and metrics were computed before the freeze for fitting and selection, as Section 6.4 reports. Suggested replacement:

> Before the freeze (manifest 16:32 UTC), predictions, ranks and metrics were computed on the training and validation splits only, for fitting and selection (Section 6.4); test.txt was read before the freeze only for split bookkeeping, hashing, time-aware filter indices and history indices restricted to earlier time stamps, and no test-row prediction, rank or metric existed before Stage 2 began scoring at 16:34 UTC.

Freeze: manifest of 201 files written at 2026-10-16T16:32:23Z, verified at 2026-10-16T16:34:01Z as the first Stage-2 step. Sources refer to the log bundle v2 in `fresh_window/logs/` (paths `track/`, `jobs/`, `scans/`, `inventory/`, `orchestration_commands.tsv`), to `fresh_window_code_v1.tar.gz` (prefix `fresh_window_code_v1/`) or to the artifact store. All times are UTC.

## 1. Ordered events

| # | UTC | cat. | event | split | source |
|---|---|---|---|---|---|
| 1 | 14:24:12 | b | fresh-window rule registered (lead) |  | artifact v79_fresh_window_rule.json (artifact 13197939-8bbd-4f32-88c2-9197a2badc8b, version e10ab60d-9569-49b6-bd1e-6cf415138a78, created 2026-10-16T14:24:13.099Z); field registered_utc |
| 2 | 14:24:46 | b | Amendment 1 registered (lead) |  | artifact v79_fresh_window_rule_amendment_1.json (artifact a4c9bafc-09fe-4dc9-b6b3-aedeaab12d59, version 6debf06f-fc19-49ac-be4e-44b87bbb47da, created 2026-10-16T14:24:47.417Z); field registered_utc |
| 3 | 14:30:28 | a | Uninspected scan of serverA (read-only, files dated >= 2025-10-01), job 24868d2a |  | scans/serverA_24868d2a-00f2-4201-a925-29f33feb31d4/scan.json:L20 |
| 4 | 14:30:29 | a | Uninspected scan of serverB (read-only, files dated >= 2025-10-01), job 7281237c |  | scans/serverB_7281237c-9381-4054-a66d-2109569cbc8e/scan.json:L16 |
| 5 | 14:30:30 | a | Uninspected scan of serverC (read-only, files dated >= 2025-10-01), job 6f3c1306 |  | scans/serverC_6f3c1306-74ba-414c-ab05-e93115855078/scan.json:L1 |
| 6 | 14:31:48 | a | GDELTFW build job starts (download of 26,208 official 15-minute exports, 2025-08-01..2026-04-30) |  | track/logs/build.log:L1 |
| 7 | 14:42:59 | a | All 26,208 export files downloaded (0 missing) (671.1 s after the build start; missing_export_files.txt written 14:43:00Z (inventory/track_files.tsv:L424 (mtime of data/GDELTFW/missing_export_files.txt))) |  | track/logs/build.log:L22 |
| 8 | 14:45:13 | a | Filtering: 1,552,574 Middle-East root events in the window (805.4 s after the build start) |  | track/logs/build.log:L23 |
| 9 | 14:45:38 | a | GDELTFW split files written (train/valid/test.txt, id maps, stat.txt) | construction | inventory/track_files.tsv:L427 (mtime of data/GDELTFW/test.txt) |
| 10 | 14:45:39 | a | Build exits 0 (1,209 entities, 215 relations, 273 days) |  | track/logs/build.log:L25 |
| 11 | 14:45:40 | a | Build job records line counts and SHA-256 of every split file (test.txt read: wc -l, sha256sum) | test.txt read: bookkeeping | jobs/faac377d-9748-42af-9403-7352ff11995d/out/wc.txt:L5; jobs/faac377d-9748-42af-9403-7352ff11995d/out/data_sha256.txt:L5 |
| 12 | 14:46:37 | a | Read-only sanity check of the built splits: entity/relation coverage of every split by train, timestamp range per split (test: ts 212..272, 61 days); no model involved | test.txt read: bookkeeping | orchestration_commands.tsv:L14 |
| 13 | 14:46:47 | a | Uninspected check recorded: PASS |  | artifact fresh_window_uninspected_check.json (artifact 950da451-f3b4-42a8-ac66-4d7c5998d0d6, version f255c8f4-129c-4055-a56d-808c26de0265, created 2026-10-16T14:47:01.087Z); field recorded_utc |
| 14 | 14:49:29 | b | Implementation note registered (configurations, CountTRuCoLa candidates C0/C1/C2, freeze plan) |  | artifact fresh_window_implementation_note.json (artifact eab28671-aec2-4a8c-a096-f897e6aa8f12, version 7ef9ae3a-6fbf-433b-b6a1-536c6512f598, created 2026-10-16T14:49:35.045Z); field registered_utc |
| 15 | 14:50:01 | - | Fresh-window code directory created from the forward-holdout code (constant substitution only) |  | orchestration_commands.tsv:L15 |
| 16 | 14:50:41 | c | Stage 1A starts (job c2ac489f) |  | track/logs/stage1_chain.log:L1 |
| 17 | 14:50:42 | c | TiRGN training starts on serverA GPU 0 (train split; validation filtered MRR per epoch for model selection; --no_dump) | train/valid | track/logs/stage1_chain.log:L27 |
| 18 | 14:50:42 | c | TiRGN loader builds time snapshots of all three splits (train 184, valid 28, test 61; num_times 273 = time-encoder input size); history vocabulary and filter index over all facts, both masked to earlier time stamps | test.txt read: split bookkeeping / filter | track/logs/tirgn_train.log:L17 |
| 19 | 14:53:01 | d | Base-independent preparation (validation strata; pattern tables on the early 70% of train timestamps; event index over all facts, queried strictly before t) | test.txt read: row counts / time-masked event index | track/logs/stage1_chain.log:L28 |
| 20 | 14:53:18 | d | RecB validation grid and selection (BLEND component; validation ranks and MRR) | valid | track/logs/stage1_chain.log:L29 |
| 21 | 14:53:26 | d | RecB validation CSR (--only_valid; no test rows) | valid | track/logs/stage1_chain.log:L30 |
| 22 | 14:53:44 | e | CountTRuCoLa Stage 1, attempt 1 (C0, C1, C2 one minute apart) |  | jobs/7a51145d-80e6-4474-9d1b-132197567505/h/C0/run.txt:L1 |
| 23 | 14:53:54 | e | Attempt 1 fails at dataset construction in the wrapper (relation2id format); no rule learned, no ranking |  | jobs/7a51145d-80e6-4474-9d1b-132197567505/h/C0/run.txt:L2; jobs/7a51145d-80e6-4474-9d1b-132197567505/h/C0/ct_tail.txt:L40 |
| 24 | 14:57:05 | e | Smoke-test script written |  | orchestration_commands.tsv:L21 |
| 25 | 14:57:29 | e | Adapter smoke test: CountTRuCoLa RuleDataset built from GDELTFW (all three splits loaded; split sizes printed); no learning, no scoring | test.txt read: split bookkeeping | orchestration_commands.tsv:L22 |
| 26 | 14:57:37 | e | CountTRuCoLa Stage 1 C0 starts: learn on train, apply to validation only (EVAL_TESTSET_FLAG=False) | train/valid | track/ct/C0/run.txt:L1; track/ct/C0/ct_stage1.log:L24 |
| 27 | 14:58:37 | e | CountTRuCoLa Stage 1 C1 starts: learn on train, apply to validation only (EVAL_TESTSET_FLAG=False) | train/valid | track/ct/C1/run.txt:L1; track/ct/C1/ct_stage1.log:L24 |
| 28 | 14:59:37 | e | CountTRuCoLa Stage 1 C2 starts: learn on train, apply to validation only (EVAL_TESTSET_FLAG=False) | train/valid | track/ct/C2/run.txt:L1; track/ct/C2/ct_stage1.log:L24 |
| 29 | 15:01:01 | c | TiRGN training ends (best epoch 11 by validation filtered MRR); no dump of any split | valid | track/logs/tirgn_train.log:L55; inventory/track_files.tsv:L595 (mtime of GDELTFW/work/base_tirgn/tirgn_GDELTFW_epoch_log.json) |
| 30 | 15:01:08 | c | TiRGN validation rows dumped (full log-prob rows for the 152,234 validation queries) | valid | track/logs/stage1_chain.log:L35; inventory/track_files.tsv:L597 (mtime of GDELTFW/work/fullrows/tirgn_GDELTFW_valid_scores_f32.bin) |
| 31 | 15:01:48 | d | Stage 1B starts (job dc39717c) |  | track/logs/stage1_chain.log:L38 |
| 32 | 15:01:59 | d | SA part of Stage 1B fails at the selection-cut parse (container banner in the captured value); every SA step rc != 0 in 2-11 s, nothing computed |  | track/logs/stage1_chain.log:L65; track/logs/stage1_chain.log:L60 |
| 33 | 15:02:20 | d | HRcov candidate lists, validation | valid | track/logs/stage1_chain.log:L66 |
| 34 | 15:02:35 | d | HRcov features, validation | valid | track/logs/stage1_chain.log:L67 |
| 35 | 15:02:57 | d | BLEND grid scored on validation (13 betas x lambda; validation MRR) | valid | track/logs/stage1_chain.log:L68 |
| 36 | 15:02:58 | d | BLEND / RecB-POOL selection on validation | valid | track/logs/stage1_chain.log:L69 |
| 37 | 15:04:46 | d | HRcov re-ranker seed 0 fitted on the validation fit part, epoch chosen on the selection part | valid | inventory/track_files.tsv:L626 (mtime of GDELTFW/work/train_tirgn/models/HRCOV_PAT_s0.pt) |
| 38 | 15:06:10 | d | HRcov re-ranker seed 1 fitted on the validation fit part, epoch chosen on the selection part | valid | inventory/track_files.tsv:L628 (mtime of GDELTFW/work/train_tirgn/models/HRCOV_PAT_s1.pt) |
| 39 | 15:06:52 | d | Stage 1B-SA rerun starts (job 9ccd0e7d; crash recovery of the failed SA part, identical inputs) |  | track/logs/stage1_chain.log:L70 |
| 40 | 15:06:53 | d | RecB_xf per-relation selection (cross-fitted, validation) | valid | track/logs/stage1_chain.log:L73 |
| 41 | 15:07:00 | d | RecB_xf validation CSR (--only_valid) | valid | track/logs/stage1_chain.log:L74 |
| 42 | 15:07:04 | d | SA strict history order, validation | valid | track/logs/stage1_chain.log:L75 |
| 43 | 15:07:06 | d | HRcov re-ranker seed 2 fitted on the validation fit part, epoch chosen on the selection part | valid | inventory/track_files.tsv:L653 (mtime of GDELTFW/work/train_tirgn/models/HRCOV_PAT_s2.pt) |
| 44 | 15:07:07 | d | HRcov fit and freeze complete (selection-part MRR per seed in train_fh.json); Stage 1B ends | valid | track/logs/stage1_chain.log:L76 |
| 45 | 15:07:40 | d | SA candidate lists, validation | valid | track/logs/stage1_chain.log:L78 |
| 46 | 15:08:05 | d | SA pools/features, validation | valid | track/logs/stage1_chain.log:L79 |
| 47 | 15:08:14 | d | SA_W100 seed 0 fitted (validation fit part; selection part for the epoch) | valid | inventory/track_files.tsv:L689 (mtime of GDELTFW/work/sa/train/models/SA_W100_s0.pt) |
| 48 | 15:08:23 | d | SA_W100 seed 1 fitted (validation fit part; selection part for the epoch) | valid | inventory/track_files.tsv:L690 (mtime of GDELTFW/work/sa/train/models/SA_W100_s1.pt) |
| 49 | 15:08:38 | d | SA_W100 seed 2 fitted (validation fit part; selection part for the epoch) | valid | inventory/track_files.tsv:L691 (mtime of GDELTFW/work/sa/train/models/SA_W100_s2.pt) |
| 50 | 15:08:38 | d | SA fit and freeze complete; Stage 1B-SA ends | valid | track/logs/stage1_chain.log:L80 |
| 51 | 15:42:19 | e | CountTRuCoLa memory watchdog starts (job eea61388) |  | track/logs/ct_memwatch.log:L1 |
| 52 | 16:07:06 | e | C1 / C2 resident memory 135 / 132 GB (the values quoted in the manuscript) |  | track/logs/ct_memwatch.log:L50 |
| 53 | 16:07:36 | e | Last sample before the stop: C1 / C2 at 135 / 134 GB, host MemAvailable 198 GB |  | track/logs/ct_memwatch.log:L51 |
| 54 | 16:07:48 | e | Manual stop of C1 and C2 (window 50): projected memory/time infeasible; no rule set, ranking or score existed |  | track/logs/ct_memwatch.log:L52; track/logs/ct_memwatch.log:L53; command at 16:07:53Z: orchestration_commands.tsv:L33 |
| 55 | 16:07:54 | e | CountTRuCoLa C0 rule set learned on train (rule files written) | train | inventory/track_files.tsv:L699 (mtime of ct/C0/repo/files/rules/tkgl-gdeltfw-learn-cyc1nonrec-cyc1rec-crules-multi-window10-0-0-ruleset-ids.txt) |
| 56 | 16:08:05 | b | Addendum 1 registered: C1/C2 dropped; comparator = C0 alone or its 52 blends, by selection-part MRR |  | artifact fresh_window_implementation_note_addendum_1.json (artifact 9eaf6fe4-71c3-4bd1-acf7-c465a0af4ac8, version dede10e9-7120-436c-a8d0-d9c77306dc4a, created 2026-10-16T16:08:06.779Z); field registered_utc |
| 57 | 16:08:05 | e | C1 main process ends (SIGTERM, rc 241); no validation or test ranking file |  | track/ct/C1/run.txt:L2; track/ct/C1/run.txt:L3 |
| 58 | 16:08:09 | e | C2 main process ends (SIGTERM, rc 241); no validation or test ranking file |  | track/ct/C2/run.txt:L2; track/ct/C2/run.txt:L3 |
| 59 | 16:18:41 | e | CountTRuCoLa results CSV written by the released main.py: validation columns only, all -1 (built-in evaluation disabled); config_rankings_path holds the release's default name of the test-rankings file, which did not exist before 16:58:39Z | valid | inventory/track_files.tsv:L717 (mtime of ct/C0/repo/files/results/tkgl-gdeltfw-results.csv); content: track/ct/C0/repo/files/results/tkgl-gdeltfw-results.csv:L2 |
| 60 | 16:19:03 | e | C0 rules applied to validation; validation rankings written (no test rankings: EVAL_TESTSET_FLAG=False) | valid | track/ct/C0/run.txt:L2; inventory/track_files.tsv:L715 (mtime of ct/C0/repo/files/rankings/tkgl-gdeltfw/tkgl-gdeltfw10-rankings_val_conf_0_corr_conf_0_noisyor_crules_frules_zrules_pvalue_30_num_top_rules_10_multi.txt) |
| 61 | 16:19:13 | e | C0 validation rankings scored by fw_ct_score.py --split val | valid | track/ct/C0/run.txt:L9 |
| 62 | 16:19:43 | f | Stage 1C: CountTRuCoLa C0 alone and 52 TiRGN blends scored on the validation selection part (50,192 queries) (job 953e4300 is recorded as exit 1: the final harvest `cp` of the non-existent C2 dump_val/meta.json (C2 was stopped) returns 1 after ext_valid rc=0 (jobs/job_status.tsv)) | valid | track/logs/stage1_chain.log:L83; track/logs/ext_valid.log:L18 |
| 63 | 16:20:19 | f | External comparator frozen: C0 blend a=0.1, tau=0.5 (selection-part MRR 0.4077 vs alone 0.3684, base 0.4041) (the file was staged into the freeze at 16:32:23Z and saved as an artifact at 16:33:17Z) |  | registration/ext_comparator.json field selected_utc (code release); artifact ext_comparator.json (artifact fa82718d-bddf-4d50-8316-76b908439c07, version b4d773ad-9d7e-4af8-924c-9907844f149c, created 2026-10-16T16:33:17.831Z) |
| 64 | 16:20:47 | - | Latency protocol on a fixed 2,000-query validation slice (job c46772b5) | valid | orchestration_commands.tsv:L35; inventory/track_files.tsv:L756 (mtime of GDELTFW/work/latency/latency_systems.json); inventory/track_files.tsv:L760 (mtime of GDELTFW/work/latency/latency_ct.json) |
| 65 | 16:32:23 | g | SHA-256 freeze manifest written: 201 files, 0 missing (rule documents, code, data, base model, validation rows, fitted rankers, selections, CT Stage-1 outputs, latency) |  | jobs/117823f0-d9fc-4870-95e1-83bbedc15ba9/h/freeze_out.txt:L1; track/freeze/freeze_manifest_remote.json |
| 66 | 16:33:17 | g | Manifest with chronology, frozen code tarball, Stage-1 validation summary and comparator saved as artifacts |  | artifact fresh_window_freeze_manifest.json (artifact f1c68442-9ec6-4698-8cc4-97d9a34368ea, version 8911bb05-7d65-43ca-82a0-f123ab6476f9, created 2026-10-16T16:33:17.773Z); artifact fw_code_frozen.tgz (artifact 35a008b8-3c30-4519-b62b-8ee751439cfd, version 14bba154-e925-43c3-9746-3d92aa958dfb, created 2026-10-16T16:33:17.794Z) |
| 67 | 16:33:56 | i | Stage 2 starts (job 7349980c) |  | track/logs/stage2_test.log:L1 |
| 68 | 16:34:01 | g | Freeze verified at the start of Stage 2: 201 files, 0 problems |  | track/logs/stage2_test.log:L27; track/logs/stage2_verify.json (verified_utc, n_checked 201, problems []) |
| 69 | 16:34:02 | h | FIRST TEST-ROW PREDICTION, TiRGN base: test log-prob rows (457,168 queries) computed about 16:34:02-16:34:13Z (approximate start = log end 16:34:13Z minus the logged 11.1 s; launched only after the freeze check ended at 16:34:01Z; no test output of any system has an earlier mtime) | test | track/logs/stage2_test.log:L45; track/logs/tirgn_rows_test.log:L19 (11.1 s, file closed 16:34:13Z); inventory/track_files.tsv:L779 (mtime of GDELTFW/work/fullrows/tirgn_GDELTFW_test_scores_f32.bin) |
| 70 | 16:35:05 | h | First test-row prediction, CountTRuCoLa comparator: frozen C0 rules applied to the test set (Stage-2 process started 16:34:02Z) | test | inventory/track_files.tsv:L776 (mtime of ct/C0/.stage2_marker); track/ct/C0/ct_stage2.log:L45 (serverA local time UTC+8); track/ct/C0/ct_stage2.log:L101 |
| 71 | 16:35:47 | h | First test-row prediction, RecB (component of BLEND): recurrency scores of the test rows 16:35:47-16:36:14Z | test | track/logs/stage2_test.log:L48; inventory/track_files.tsv:L792 (mtime of GDELTFW/work/recb_stage2/recb_test_ranks_GDELTFW.npz) |
| 72 | 16:35:47 | i | Test strata (evaluation only) written | test | track/logs/stage2_test.log:L47 |
| 73 | 16:36:44 | h | First test-row computation, SA: strict history order of the test rows 16:36:44-16:36:50Z | test | track/logs/stage2_test.log:L52 (block starts after recb_xf_test, track/logs/stage2_test.log:L49) |
| 74 | 16:36:44 | h | First test-row computation, HRcov: candidate lists from the TiRGN test rows 16:36:44-16:38:23Z | test | track/logs/stage2_test.log:L53 |
| 75 | 16:36:44 | i | RecB_xf test CSR (component of SA) complete; recomputed validation CSR of RecB and RecB_xf identical to the frozen arrays | test | track/logs/stage2_test.log:L49; track/logs/stage2_test.log:L50 |
| 76 | 16:39:05 | h | First test-row score of the HRcov re-ranker (frozen seeds 0-2) 16:39:05-16:39:51Z | test | track/logs/stage2_test.log:L57; inventory/track_files.tsv:L858 (mtime of GDELTFW/work/train_tirgn/ranks_test.npz) |
| 77 | 16:39:05 | i | HRcov test features complete (dump/rank agreement 1.0) | test | track/logs/stage2_test.log:L55; track/logs/features_test.log:L27 |
| 78 | 16:39:45 | h | First test-row score of the SA ranker (frozen seeds 0-2) 16:39:45-16:40:01Z | test | track/logs/stage2_test.log:L58; inventory/track_files.tsv:L865 (mtime of GDELTFW/work/sa/train/ranks_test_SA_W100.npz) |
| 79 | 16:40:09 | h | First test-row score of BLEND (frozen MIX 0.3) and official scoring of base, BLEND and HRcov 16:40:09-16:40:26Z | test | track/logs/stage2_test.log:L60; inventory/track_files.tsv:L871 (mtime of GDELTFW/work/test_out/official_test_gdeltfw_tirgn.npz) |
| 80 | 16:40:09 | i | SA official scoring on test | test | track/logs/stage2_test.log:L59; inventory/track_files.tsv:L868 (mtime of GDELTFW/work/test_out/official_test_gdeltfw_sa.npz) |
| 81 | 16:58:39 | i | CountTRuCoLa test rankings file complete | test | inventory/track_files.tsv:L874 (mtime of ct/C0/repo_stage2/files/rankings/tkgl-gdeltfw/tkgl-gdeltfw10-rankings_test_conf_0_corr_conf_0_noisyor_crules_frules_zrules_pvalue_30_num_top_rules_10_multi.txt) |
| 82 | 17:09:43 | i | CountTRuCoLa Stage-2 main.py returns rc 1 in its results-CSV writer (pandas dtype error) after rule application finished; rankings complete; not rerun | test | track/logs/stage2_test.log:L61; track/ct/C0/ct_stage2.log:L305 |
| 83 | 17:09:52 | i | Validation re-application by the frozen CountTRuCoLa rules reproduces the Stage-1 validation ranks exactly (max |dRR| 0.0, n 152,234) | valid | track/logs/stage2_test.log:L63; inventory/track_files.tsv:L891 (mtime of ct/C0/dump_val_stage2/meta.json) |
| 84 | 17:10:14 | h | First test-row score of the CountTRuCoLa comparator blend (frozen a=0.1, tau=0.5) and its official scoring 17:10:14-17:10:22Z | test | track/logs/stage2_test.log:L65; inventory/track_files.tsv:L909 (mtime of GDELTFW/work/test_out/official_test_gdeltfw_ext.npz) |
| 85 | 17:10:14 | i | CountTRuCoLa test rankings scored (fw_ct_score.py --split test) | test | track/logs/stage2_test.log:L64; inventory/track_files.tsv:L906 (mtime of ct/C0/dump_test/meta.json) |
| 86 | 17:10:22 | i | Stage 2 ends; one stage-time line per test step (no step repeated) |  | track/logs/stage2_test.log:L66 |
| 87 | 17:12:18 | i | Aggregation, paired query bootstrap and timestamp-block bootstrap (fw_agg.py, run once, locally): outcome R4_OTHER (SA: SMALL, comparator: BEATS) | test | fresh_window_verdict.json field decided_utc; artifact fresh_window_verdict.json (artifact aaf891b3-b045-4dad-8a8f-0146dd247513, version 5b185ac5-16e3-485f-8810-9b4ee702ebe9, created 2026-10-16T17:13:48.316Z) |
| 88 | 17:13:48 | i | Rank cross-checks (pipeline ranks = scorer strict ranks) added; results, verdict and latency table saved |  | artifact fresh_window_results.json (artifact 6a1de31b-6e61-4e6e-89cc-163042aead0f, version 637824ba-6bdf-40ca-a734-7394c7234795, created 2026-10-16T17:13:48.336Z); artifact fresh_window_latency_table.csv (artifact c3991052-aacb-4207-9b5b-695462337417, version 6d4620fa-3789-4365-81db-aee046625d24, created 2026-10-16T17:13:48.348Z) |
| 89 | 17:15:19 | i | Report saved |  | artifact fresh_window_report.md (artifact 3a67d023-9dba-4cdc-b60b-f41537ab0ca8, version e1fd450a-735f-488a-950e-fafc89e2cdb5, created 2026-10-16T17:15:19.398Z) |

Categories: **a** data download, construction, uninspected check; **b** registration of rule, amendment, implementation note, addendum; **c** base training and validation row dump; **d** HRcov, SA and BLEND fitting and validation scoring; **e** CountTRuCoLa Stage-1 runs incl. the stopped window-50 configurations; **f** comparator and blend selection; **g** freeze-manifest hashing; **h** first test-row prediction of every system; **i** test scoring, aggregation, crash-recovery reruns; **-** other

## 2. Predictions, ranks and metrics computed before the freeze (validation and training only)

- TiRGN: validation filtered MRR after each of 17 epochs (14:50-15:01Z, track/logs/tirgn_train.log; best epoch 11) and full validation log-prob rows (15:01:08Z)
- RecB / RecB_xf: validation ranks and selection (14:53Z, 15:06:53Z)
- BLEND / RecB-POOL: validation grid MRR and selection (15:02:57-15:02:58Z)
- HRcov: fit-part training and selection-part MRR per seed (15:04-15:07Z, train_fh.json)
- SA_W100: fit-part training and selection-part MRR per seed (15:08Z, train_fw_sa.json)
- CountTRuCoLa C0: validation rankings and validation MRR (16:19Z); C1/C2 produced none
- Comparator: selection-part MRR of C0 alone and 52 blends (16:19:43Z) and the selection (16:20:19Z)
- Latency on a 2,000-query validation slice (16:20-16:31Z)

## 3. Every place test.txt was read before the freeze

| # | UTC | process | code location | what is read | why | test row scored |
|---|---|---|---|---|---|---|
| 1 | 14:45:38Z | GDELTFW builder (job faac377d) | fresh_window_code_v1/01_window_build/build_gdelt25x.py:L188 | writes test.txt | construction | no |
| 2 | 14:45:40Z | build job (wc -l, sha256sum of every split file) | fresh_window_code_v1/01_window_build/run_build_gdeltfw.sh:L15 | line count and SHA-256 of test.txt | split bookkeeping | no |
| 3 | 14:46:37Z | read-only sanity check (orchestration command) | fresh_window/logs/orchestration_commands.tsv:L14 (verbatim command in fresh_window_chronology.md, section 4) | entity/relation ids and time stamps of every split | split bookkeeping (vocabulary closure, ts range per split: test 212..272, 61 days) | no |
| 4 | 14:50:41Z..14:53:01Z | fh_prep.py --stage 1 (Stage 1A) | fresh_window_code_v1/code/oih_common.py:L97; fresh_window_code_v1/code/fh_prep.py:L102; fresh_window_code_v1/code/fh_prep.py:L114; fresh_window_code_v1/code/fh_prep.py:L137 | all three splits loaded by the harness Split class | split bookkeeping (row counts: test 457,168 in track/logs/prep_stage1.log:L20) and the event index, which is queried strictly before the query time (fresh_window_code_v1/code/patterns.py:L43); strata built for validation only | no |
| 5 | 14:50:42Z..15:01:01Z | TiRGN training, hd_tirgn.py via fh_tirgn_train.py --no_dump (Stage 1A) | fresh_window_code_v1/code/hd_tirgn.py:L91; fresh_window_code_v1/code/hd_tirgn.py:L93; fresh_window_code_v1/code/hd_tirgn.py:L61; fresh_window_code_v1/code/hd_tirgn.py:L101; fresh_window_code_v1/code/hd_tirgn.py:L125; fresh_window_code_v1/code/hd_tirgn.py:L81; fresh_window_code_v1/code/fh_tirgn_train.py:L11; fresh_window_code_v1/code/hd_tirgn.py:L238 | all three splits | split bookkeeping (time snapshots; num_times = 273 distinct time stamps is the input size of TiRGN's time encoder, track_extra/tirgn_rrgcn_num_times_grep.txt), history vocabulary masked to first occurrences before the query time, time-aware filter index keyed by (s, r, t); only validation snapshots are evaluated (fresh_window_code_v1/code/hd_tirgn.py:L168); the dump that would score test is disabled | no |
| 6 | 14:51..14:53:26Z | fh_recb.py --stage valid; fh_recb.py --stage test --only_valid (Stage 1A) | fresh_window_code_v1/code/fh_recb.py:L312; fresh_window_code_v1/code/fh_recb.py:L323; fresh_window_code_v1/code/fh_recb.py:L345 | all three splits | time-aware filter and online recurrency counts (facts before the query time); with --only_valid only the validation CSR is built | no |
| 7 | 14:53:44Z..14:53:54Z and 14:57:37Z..16:19:03Z | CountTRuCoLa Stage 1 (attempt 1; C0, C1, C2) through fw_ct_main.py | fresh_window_code_v1/code_ct/fw_ct_main.py:L30 | all three splits written into CountTRuCoLa's dataset files with split codes | dataset construction required by the released RuleDataset (split bookkeeping); rules learned on train and applied to validation only (EVAL_TESTSET_FLAG=False; track/ct/*/ct_stage1.log:L24) | no |
| 8 | 14:57:29Z | adapter smoke test (orchestration command) | fresh_window/logs/orchestration_commands.tsv:L22; fresh_window_code_v1/code_ct/fw_ct_dataset_smoke.sh | RuleDataset of all three splits; sizes printed | split bookkeeping | no |
| 9 | 15:01:01Z..15:01:08Z | fh_tirgn_rows.py --split valid (TiRGN validation rows) | fresh_window_code_v1/code/hd_tirgn_replay.py:L19; fresh_window_code_v1/code/hd_tirgn_replay.py:L36 | all three splits | history vocabulary and snapshots; the replay stops after the last needed split (validation), so no test snapshot is scored | no |
| 10 | 15:01:48Z..15:02:58Z | fh_lists.py / fh_features.py --splits valid; fh_blend.py --stage valid (Stage 1B) | fresh_window_code_v1/code/fh_lists.py:L114; fresh_window_code_v1/code/fh_lists.py:L118; fresh_window_code_v1/code/fh_blend.py:L283; fresh_window_code_v1/code/fh_blend.py:L285; fresh_window_code_v1/code/fh_blend.py:L207 | all three splits | time-aware filter, event index queried before t, test query list defined but not processed (--splits valid); fh_blend scored 0 test rows (track/logs/blend_valid.log:L20-21, counter 'test': 0) | no |
| 11 | 15:01:49Z..15:01:59Z and 15:06:52Z..15:08:05Z | v78_hist.py / v78_lists.py / v78_pools.py --splits valid (SA, Stage 1B and 1B-SA) | fresh_window_code_v1/code_sa/v78_hist.py:L25; fresh_window_code_v1/code_sa/v78_hist.py:L22; fresh_window_code_v1/code_sa/v78_lists.py:L142; fresh_window_code_v1/code_sa/v78_pools.py:L36 | all three splits | time-aware filter and online history counts; only the splits named in --splits (valid) are processed | no |
| 12 | 16:19:37Z..16:19:43Z | fw_ext.py --stage valid (Stage 1C) | fresh_window_code_v1/code_sa/fw_ext.py:L51; fresh_window_code_v1/code_sa/fw_ext.py:L66 | all three splits | time-aware filter; base rows are read for the validation split only (the test-rows path in the argument did not exist yet: first written 16:34:12Z) | no |
| 13 | 16:20:47Z..16:31:22Z | fw_latency.py / fw_ct_latency.py (latency protocol) | fresh_window_code_v1/code/fw_latency.py:L70; fresh_window_code_v1/code/fw_latency.py:L123; fresh_window_code_v1/code/fw_latency.py:L43 | all three splits (as the production pipelines load them) | filter and event index; the timed slice is 2,000 validation queries of the first selection-part time stamp | no |
| 14 | 16:32:23Z | fw_freeze.py (freeze job 117823f0) | fresh_window_code_v1/code/fw_freeze.py:L36 | SHA-256 of test.txt | freeze-manifest hashing | no |

Modules run before the freeze that do not read test.txt: code/fh_train.py (HRcov fit) refuses to run if test features exist (fresh_window_code_v1/code/fh_train.py:L41); code_sa/v76_recb_xf_select.py reads valid.txt only (fresh_window_code_v1/code_sa/v76_recb_xf_select.py:L21); code_ct/fw_ct_score.py --split val reads valid.txt only (fresh_window_code_v1/code_ct/fw_ct_score.py:L25); fw_uninspected_scan.py ran before GDELTFW existed (14:30Z) and reads time-stamp tables of other datasets.

## 4. Search for test-stage outputs before the freeze

- Scope: serverA: every file of tracks/phase2_fresh_window except raw/ (the 26,208 downloaded GDELT export zips): 911 files; the 308 files of the 12 serverA job directories; the scan job directories on serverB and serverC (inventory/*.tsv, mtime UTC).
- Files whose name contains 'test' with an mtime before the freeze: data/GDELTFW/test.txt (the split, 14:45:38Z); code/fulltest_eval.py (a harness library whose name contains 'test'; copied 14:50:02Z).
- Test-stage outputs: 91 test-named output files; the earliest mtime is 2026-10-16T16:34:12Z (GDELTFW/work/fullrows/tirgn_GDELTFW_test_scores_f32.bin); the CountTRuCoLa test-rankings file was first created at 16:58:39Z (ct/C0/repo_stage2), and no rankings_test file exists in ct/C0/repo, ct/C0/repo_latency, ct/C1 or ct/C2.
- Text search: 135 lines of the pre-freeze logs/JSONs contain the substring 'test'; none is a prediction, rank or metric on test rows (classes above). Classes: library file name fulltest_eval.py in provenance fields (18); split row/fact counts or test-split day range (bookkeeping) (30); CountTRuCoLa option echo (EVAL_TESTSET_FLAG=False) or options dump (57); Ray library warning (token authentication banner) (6); validation rankings path (CountTRuCoLa) (3); no test ranking file (rankings ... test=none) (9); 0 test rows scored (blend_valid progress counter) (4); other track's process in a host process listing (2); scanner command line (docker inspect / test -e) (2); build command (builder --test argument) (1); line count / SHA-256 of test.txt (build job) (2); docker image tag 'latest' (substring match) (1).

Sanity-check command of 14:46:37Z (verbatim, user name redacted):

```
cd /home/[REDACTED:~ANON]/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window/data/GDELTFW && cat missing_export_files.txt; echo; python3 - <<'EOF'
import collections
N,R=1209,215
ents=collections.Counter(); rels=set(); ts={}
for sp in ('train','valid','test'):
    tt=set()
    for ln in open(sp+'.txt'):
        s,r,o,t=map(int,ln.split()[:4]); ents[s]+=1; ents[o]+=1; rels.add(r); tt.add(t)
        if s==o: print('selfloop',sp)
    ts[sp]=(min(tt),max(tt),len(tt))
tr_e=set(); tr_r=set()
for ln in open('train.txt'):
    s,r,o,t=map(int,ln.split()[:4]); tr_e.add(s); tr_e.add(o); tr_r.add(r)
print('ents_any',len(ents),'ents_train',len(tr_e),'rels_any',len(rels),'rels_train',len(tr_r),'ts',ts)
EOF
# output: ents_any 1209 ents_train 1209 rels_any 215 rels_train 215 ts {'train': (0, 183, 184), 'valid': (184, 211, 28), 'test': (212, 272, 61)}
```

## 5. Notes

- Stage 1C job 953e4300 is recorded as exit 1: its last command copies dump_val/meta.json of C2, which does not exist because C2 was stopped; ext_valid itself returned 0 (track/logs/stage1_chain.log:L83).
- Both crash-recovery reruns happened in Stage 1 on train/validation only: attempt 1 failed at dataset construction (14:53:54Z) and was rerun with the fixed wrapper (14:57:37Z); the SA part of Stage 1B failed at the cut parse (15:01:49-15:01:59Z, nothing computed) and was rerun as Stage 1B-SA (15:06:52Z). No test step was rerun: stage2_test.log has exactly one stage-time line per step, and there is one Stage-2 job.
- CountTRuCoLa Stage-2 main.py returned rc 1 in its results-CSV writer after rule application had finished (17:09:43Z); the test rankings were complete (16:58:39Z) and were scored by fw_ct_score.py; the step was not rerun.
- An earlier manuscript draft gave 135 and 132 GB for the stopped window-50 configurations; these are the watchdog values at 16:07:06Z. The last sample before the stop (16:07:36Z) was 135 and 134 GB, and the process-tree peaks recorded by memmon were 135.3 and 134.6 GB PSS (track/ct/C1|C2/mem_ct_stage1.json).
- stage2_test.log / stage1_chain.log stage-time lines are written when a step ENDS: `[stage-time] <step> rc=<rc> <seconds since the stage started>s <UTC end time>` (fw_stage2.sh:L22). The freeze check runs first and aborts Stage 2 on any mismatch (fw_stage2.sh:L24-26); all test blocks are launched after it. Start times of test steps are therefore bounded below by 16:34:01Z; the TiRGN start of about 16:34:02Z is its log end (16:34:13Z, file mtime) minus the 11.1 s the step logged.
- All times are UTC. serverA writes local time (UTC+8) in some third-party log lines (CountTRuCoLa / Ray); those were converted. Times of orchestration commands come from the command log of the analysis workstation (orchestration_commands.tsv).
