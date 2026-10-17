# fresh_window_code_v1: code for the v79 fresh-window comparison (GDELTFW)

This package holds every script and configuration that ran for the v79 fresh-window comparison:

- the window build from GDELT raw exports and the uninspected check;
- TiRGN base training and dumps;
- HRcov, SA and BLEND fitting and scoring;
- the CountTRuCoLa runs: the default configuration that was used, and the window-50 configurations that were stopped;
- the latency harness, and aggregation with both bootstraps.

It also holds the registered rule documents and outcome documents.

**Not included:** no GDELT data or GDELTFW files, no dumps, rankings, scores or model weights. No third-party code is included
either: the TiRGN model tree, CountTRuCoLa and py-tgb are referenced by commit and checksum.

## Provenance

- `code/`, `code_sa/` and `code_ct/` are byte-identical to the Stage-1 frozen code tarball (`fw_code_frozen.tgz`, artifact
  14bba154-e925-43c3-9746-3d92aa958dfb). Stage 2 verified the corresponding host files against the SHA-256 manifest:
  201 files, 0 problems.
- Three scripts ran on the host but were not part of the frozen set: `code_ct/fw_ct_memwatch.sh`,
  `code_ct/fw_ct_stop_window50.sh` and `code_ct/fw_ct_dataset_smoke.sh`.
- Some inherited files in `code/` were copied by the setup step and never invoked: `fh_freeze.py`, `fh_gpu_wait.sh`,
  `g7_cluster_bootstrap.py`, `hd_agg.py`.
- `steps/` holds the orchestration and local-analysis cells **verbatim** from the session archive, in the order they ran,
  including the failed attempts. They are not reconstructions. The only edit is that the archive's display redaction of the
  user name was restored to `/home/anon`.
- `diffs/` shows every change against the files each script was derived from (forward-holdout harness, hardening v76/v78,
  v78_external). Changes of the fh_* files: the setup sed substitutions (GDELT25X -> GDELTFW, N/R 1173/218 -> 1209/215,
  track name) and the all-entity patch of `fh_blend.py`.
- Local workspace paths in `steps/` map to this package as follows:
  - `fw/code_fw/` and `fw/code/` -> `code/` (the scanner -> `02_uninspected_check/`);
  - `fw/code_sa/` -> `code_sa/`;
  - `fw/code_ct/` -> `code_ct/`;
  - `fw/out/` and `fw/freeze_in/` -> `registration/`.

## Prerequisites

- **Host paths.** The scripts hard-code `H=/home/anon/cs_scratch/open_ideas_harness`, `FW=$H/tracks/phase2_fresh_window` and
  `B=$H/tracks/v78_external`. Edit these to run elsewhere.
- **Runtime.** Docker image `tkg-gpu:latest` (`00_env/`), on serverA GPU 0 (NVIDIA RTX PRO 6000 Blackwell Max-Q). The GPU
  gate is `code/fw_gpu_wait.sh`.
- **TiRGN.** Upstream https://github.com/Liyyy2122/TiRGN at commit 4b74c98b24b200a138e22b7e69f948f0f8d0232a, staged at
  `$FW/stage/TiRGN`. The upstream repository has no licence file, so the tree is not vendored. Prepare it with
  manuscriptA_code_release `bases/fetch_upstream.sh` (upstream + `bases/patches/tirgn_stage.patch`), then verify with
  `md5sum -c 03_setup/tirgn_tree.md5` inside the tree (13 files).
- **CountTRuCoLa.**
  `git clone https://github.com/JuliaGast/counttrucola_submission $B/repos/counttrucola_submission && git -C $B/repos/counttrucola_submission checkout 5d9aedf4106926d13befa141e0855d684f3f5f9d`
  (the HEAD recorded at run time). Its venv is at `$B/env/ct`; see `00_env/ct_venv_pip_freeze.txt`. The released code is
  used unmodified.

## Order of commands to reproduce

Each remote step ran on serverA (serverA) in a fresh job directory. The files listed under "staged" were placed
flat in that directory, and outputs were written to `./h/`. Steps marked *local* ran on the analysis workstation on the job
outputs that were copied back (`hpc/<job_id>/`). Times are UTC on 2026-10-16. Each step's verbatim provenance is in `steps/`.

1. **Uninspected scan** (serverA, serverB, serverC; 14:30). Staged: `fw_uninspected_scan.py`. Run `bash 02_uninspected_check/run_scan.sh`; the scanner's first argument is the host label `ssh:server.<n>`. Outputs: `scan.json`, `scan.err`, `scan_mode.txt`. Provenance: `steps/01`.
2. **Window build** (serverA CPU; 14:31-14:45). Run `bash 01_window_build/run_build_gdeltfw.sh`. Equivalent direct call: `python3 -u build_gdelt25x.py --out $FW/data/GDELTFW --raw $FW/raw/gdelt --start 20250801 --end 20260501 --valid 20260201 --test 20260301 --workers 16`. Verify with `cd $FW/data/GDELTFW && sha256sum -c GDELTFW.sha256`. Provenance: `steps/02`.
3. *local* (14:46) **Record the uninspected check**, giving `registration/fresh_window_uninspected_check.json`. Provenance: `steps/03`.
4. *local* (14:49) **Register the implementation note.** Provenance: `steps/04`.
5. **Set up the code directory** (serverA). Run `bash 03_setup/fw_setup_code.sh`. Alternatively, copy `code/` to `$FW/code` and a verified TiRGN tree to `$FW/stage/TiRGN`. Provenance: `steps/05`.
6. **Stage 1A** (GPU 0; 14:50). Staged: `fw_gpu_wait.sh`, `fw_stage1a.sh`. Run `cp fw_gpu_wait.sh fw_stage1a.sh $FW/code/ && bash $FW/code/fw_stage1a.sh`. TiRGN's best epoch was 11. Provenance: `steps/06`.
7. **CountTRuCoLa Stage 1, first attempt** (CPU; 14:53). Staged: `fw_ct_main.py` (version before the fix), `fw_ct_score.py`, `fw_ct_stage1.sh`. Run `bash fw_ct_stage1.sh 10`. The attempt failed at dataset construction (relation2id format in the adapter) before any rule was learned. Provenance: `steps/07`.
8. **Adapter smoke test** (serverA). Run `PKG=<package root> bash code_ct/fw_ct_dataset_smoke.sh`. Provenance: `steps/08`.
9. **CountTRuCoLa Stage 1** (CPU; 14:57-16:19). Staged: `fw_ct_main.py`, `fw_ct_score.py`, `fw_ct_stage1.sh`. Run `bash fw_ct_stage1.sh 10`, which runs C0, C1 and C2 concurrently with 10 CPUs each. The exact commands are given below. Provenance: `steps/09`.
10. **Stage 1B** (GPU 0; finished 15:07). Staged: `fh_blend.py`, `fw_stage1b.sh`, all `code_sa/*.py`. Run `bash fw_stage1b.sh`. HRcov and BLEND completed; the SA part failed at the cut parse, with nothing computed. Provenance: `steps/10`.
11. **Stage 1B-SA** (GPU 0; finished 15:08). Staged: `fw_stage1b_sa.sh`, all `code_sa/*.py`. Run `bash fw_stage1b_sa.sh`. Provenance: `steps/11`.
12. **CountTRuCoLa memory watchdog** (serverA, concurrent with step 9). Staged: `fw_ct_memwatch.sh`. Run `bash fw_ct_memwatch.sh`. Provenance: `steps/12`.
13. **Manual stop of CountTRuCoLa C1/C2** (16:08). Run `bash code_ct/fw_ct_stop_window50.sh`. Provenance: `steps/13`.
14. *local* (16:08) **Register addendum 1.** Provenance: `steps/14`.
15. **Stage 1C** (GPU 0). Staged: `fw_ext.py`, `fw_stage1c_ext.sh`. Run `bash fw_stage1c_ext.sh`. Provenance: `steps/15`.
16. *local* (16:20) **Write `registration/ext_comparator.json`.** Provenance: `steps/16`.
17. **Latency** (GPU 0; until 16:31). Staged: `fw_latency.py`, `fw_latency.sh`, `fw_ct_latency.py`, `ext_comparator.json`. Run `bash fw_latency.sh`. Provenance: `steps/17`.
18. *local* **Stage the freeze inputs.** Provenance: `steps/18`.
19. **Freeze** (16:32). Staged: the six registration JSONs (v79 rule, amendment 1, implementation note, addendum 1, uninspected check, ext_comparator) plus `fw_stage2.sh`, `fw_freeze.py`, `fw_agg.py`, `v7_common.py`, `fw_freeze_job.sh`. Run `bash fw_freeze_job.sh`. Outputs: `freeze_manifest_remote.json` (201 files, 0 missing) and `fw_code_frozen.tgz`. Provenance: `steps/19`.
20. *local* **Stage-1 summary and manifest chronology**, giving `registration/fresh_window_freeze_manifest.json`. Provenance: `steps/20`, `steps/21`.
21. **Stage 2** (GPU 0 / CPU; from 16:34). No staged inputs. Run `bash $FW/code/fw_stage2.sh`. The script first verifies the freeze manifest (`stage2_verify: 201 files, 0 problems`), then makes one test pass per system. Provenance: `steps/22`.
22. *local* **Latency table.** Provenance: `steps/23`.
23. *local* **Aggregation and bootstraps.** Run `python3 code/fw_agg.py --tirgn h/official_test_gdeltfw_tirgn.npz --sa h/official_test_gdeltfw_sa.npz --ext h/official_test_gdeltfw_ext.npz --ts2id timestamp2id.txt --comparator registration/ext_comparator.json --out fresh_window_verdict.json --results fresh_window_results.json`, where `h/` holds the Stage-2 outputs and `timestamp2id.txt` comes from GDELTFW. Provenance: `steps/24`.
24. *local* **Verdict checks, results post-processing, report.** Provenance: `steps/25`, `steps/26`.

### CountTRuCoLa commands (exact)

Stage 1, per configuration `c` (run inside `$FW/ct/$c/repo`, an rsync copy of the release with `fw_ct_main.py` added to
`rule_based/`; `code_diff_vs_release.txt` recorded `Only in rule_based: fw_ct_main.py` for all three):

```
PYTHONPATH=$RD/repo/rule_based:$RD/repo $PY $CC/memmon.py --out $RD/mem_ct_stage1.json -- \
  $PY rule_based/fw_ct_main.py --fw_dd $DD --fw_root $RD/root --params DATASET_NAME=tkgl-gdeltfw NUM_CPUS=10 \
  EVAL_TESTSET_FLAG=False EVAL_VALSET_FLAG=True <CFG[c]>
COMMON="AGGREGATION_DECAY=0.8 NUM_TOP_RULES=10 RULE_TYPE_C=True F_UNSEEN_NEGATIVES=10 Z_RULES_FACTOR=0.1"
C0 (released default; USED)      LEARN_WINDOW_SIZE=10 DATAPOINT_THRESHOLD_MULTI=0  RULE_UNSEEN_NEGATIVES=30  $COMMON   main rc=0, 16:19:03Z
C1 (released icews14 entry)      LEARN_WINDOW_SIZE=50 DATAPOINT_THRESHOLD_MULTI=0  RULE_UNSEEN_NEGATIVES=30  $COMMON   stopped 16:08:05Z (rc=241)
C2 (released icews18 entry)      LEARN_WINDOW_SIZE=50 DATAPOINT_THRESHOLD_MULTI=50 RULE_UNSEEN_NEGATIVES=100 $COMMON  stopped 16:08:09Z (rc=241)
```

Stage 2, C0 only. Run in `$FW/ct/C0/repo_stage2` (from `code/fw_stage2.sh`). The frozen rules were re-applied to validation
and test, and the validation re-application had to reproduce the Stage-1 ranks (max abs deviation 0.0):

```
PYTHONPATH=$RD/repo_stage2/rule_based:$RD/repo_stage2 $PY $CC/memmon.py --out $RD/mem_ct_stage2.json -- \
  $PY rule_based/fw_ct_main.py --fw_dd $DD --fw_root $RD/root --params DATASET_NAME=tkgl-gdeltfw NUM_CPUS=10 \
  LOAD_PARAMS_FLAG=True CREATE_LEARN_DATA_FLAG=False EVAL_TESTSET_FLAG=True EVAL_VALSET_FLAG=True \
  LEARN_WINDOW_SIZE=10 DATAPOINT_THRESHOLD_MULTI=0 RULE_UNSEEN_NEGATIVES=30 AGGREGATION_DECAY=0.8 NUM_TOP_RULES=10 RULE_TYPE_C=True F_UNSEEN_NEGATIVES=10 Z_RULES_FACTOR=0.1
```

In Stage 2, the released `main.py` returned rc=1 from its results-CSV writer after rule application had finished. The
rankings were complete and were scored by `fw_ct_score.py`; the step was not rerun.

## Files (sha256, bytes, role)

`FILES.md` itself is not listed, because it would have to contain its own hash. The tarball's SHA-256 is reported with the artifact.

| file | sha256 | bytes | role |
|---|---|---|---|
| `00_env/ct_venv_pip_freeze.txt` | `07dbe64c9ed56271120bd27353a3ae953efac2a848d04895c160a9d1fdaf308d` | 728 | pip freeze of the CountTRuCoLa venv ($B/env/ct, Python 3.12.3). |
| `00_env/host_gpu.txt` | `3fc5e4b73335964934549d3b2501909bfd2afb83bbb488974a41bd56743b9950` | 173 | serverA GPUs (nvidia-smi) and CPU count. |
| `00_env/tkg-gpu_image.txt` | `9b9e98146755857f592684a725be679ef033342c20690782d3bea0dc85845ad0` | 108 | Docker image id and creation time of tkg-gpu:latest (runtime of every non-CT step). |
| `00_env/tkg-gpu_pip_freeze.txt` | `b44d27ee3985e5f7fd20419b32896b7b4d6c264b4cae9fb02c792b161e3aaca3` | 1627 | pip freeze inside tkg-gpu:latest (Python 3.12.3, torch 2.8.0+cu128). |
| `01_window_build/GDELT25X.sha256` | `7344f3c7fb05bfbbd171febf15c3ca581922bc13d3e2ad4d0f4ea668d03cdbe6` | 547 | Checksums of the earlier GDELT25X snapshot (read by the builder's --check; not GDELTFW). |
| `01_window_build/GDELTFW.sha256` | `b9db11ea8c01136d5be560ab0cb5d61ae106713b96d4042ff24ef818174acbaf` | 638 | SHA-256 of every GDELTFW output file as built (sha256sum -c format). |
| `01_window_build/build_gdelt25x.py` | `3bf22cc72e28ad7bda413d69c69e55761cf469d463f899dd71ee7bf1e9865fbb` | 12528 | Dataset builder (release copy, unmodified; run with the GDELTFW dates). |
| `01_window_build/build_summary.json` | `d385e1ae9f6cfbc8c76079b0a999f1cf3fa084c0c1349b78c9ad20a5c3fa98c4` | 695 | Builder summary written by the GDELTFW build (counts, splits, filters). |
| `01_window_build/builder_sha256_as_run.txt` | `ccbde9d7e30f20903338d431945eb042e75194d231e4e697b4da4c260a53af7d` | 260 | SHA-256 of the three builder files on serverA, recorded by the build job before it ran. |
| `01_window_build/gdelt25x_relation_names.tsv` | `650390e4d8c38bf92bd204cda3e061acc648463afefa3952d956fd8b995d47ff` | 8113 | Relation-name table (names column of relation2id.txt). |
| `01_window_build/run_build_gdeltfw.sh` | `f0c2bf220408961e932c75c7e972804746c87b413113ef59c2cce1e35ab55e98` | 1331 | Build job command as executed (job faac377d). |
| `02_uninspected_check/fw_uninspected_scan.py` | `5ae1edbd7e7b369c205e780ddd27eb030a0c446effa24407a03c0d8b03aaa750` | 4663 | Read-only scan for GDELT-derived files / time-stamp tables dated >= 2025-10-01 (stdlib). |
| `02_uninspected_check/run_scan.sh` | `dcdaeedf4ae1866b491781b6cce335360df6ad6d1010324b7f65a0dfde9d749a` | 494 | Scan job command on serverA (job 24868d2a); the serverB / serverC jobs used the same command with their host label. |
| `03_setup/fw_setup_code.sh` | `ee61c7e18f57669f32ea1d9bcdde384998301cfe6fbd7db293a74163f83dd46e` | 1610 | Creates $FW/code as a constant-substituted copy of the forward-holdout code; copies the TiRGN tree. |
| `03_setup/tirgn_tree.md5` | `5c0fbe0e14a0bd78c04057d2c711c17ac249c953590395c4a104a6d012915283` | 655 | MD5 of the 13 TiRGN tree files used (tree not vendored; see Prerequisites). |
| `03_setup/tirgn_tree.sha256` | `fa73fb5157c987694f90a4be48e30dc5943cb8f6c9e89afae0dc308dadeeeb3a` | 1071 | SHA-256 of the same 13 TiRGN tree files. |
| `code/fh_apply.py` | `0ab8c6fcdca50e1f88142fe8fb3f4d1e0de72a65b6f2d9f89e095d2343b2a671` | 4158 | Stage 2: apply the frozen HRcov re-rankers to test features. |
| `code/fh_blend.py` | `d8e9e82a337fbc7dcef1b2f912fe8789e0e571608a69b67fccdcb8998961ec1d` | 27104 | Scoring of base / BLEND / HRcov (all-entity ties-averaged + strict); BLEND grid and selection. GDELTFW entry + all-entity patch (diffs/). |
| `code/fh_common.py` | `b6a0d3cb28691bf5efe30cbcf4f817b2d926233c197b2af0086205d40b49b9cd` | 5189 | GDELTFW constants (N=1209, R=215), paths, shared helpers. |
| `code/fh_features.py` | `d1151f8c7a284f965de8af1a489b78ff0885f3f874f6d8b087336e3a7c40300b` | 11510 | HRcov candidate features. |
| `code/fh_freeze.py` | `6f7192ef64122d66c23d4552744fe8a038113ecba8580b66851a35f6a9c29fad` | 4360 | Inherited forward-holdout freeze script; superseded by fw_freeze.py, not invoked. |
| `code/fh_gpu_wait.sh` | `043b1ebbbc8d0c229f21fe8d44ae8ba320abb63e45cf2ea3bb4a98c028123246` | 1444 | Inherited GPU gate; superseded by fw_gpu_wait.sh, not invoked. |
| `code/fh_lists.py` | `f111d7ecab59385964393a7d608d1b8099caac38367ec00848e51b6b4acd5c2c` | 8204 | HRcov candidate lists. |
| `code/fh_prep.py` | `220156a6c7554bc47577992527f544429e2d1d68540c415fcee0987d0f48de10` | 9223 | Base-independent preparation (CPU). |
| `code/fh_recb.py` | `ef33c99162056af806985373454a71a279f57f4c920a0960360ed177bd499ef5` | 25508 | RecB (TGB recurrency baseline, vectorised): validation grid, selection, CSR rows. |
| `code/fh_tirgn_rows.py` | `9e12b19a14828b600798978324327315043f80e16be216cde24b9b7d5fb92d5f` | 5670 | TiRGN full log-prob rows + harness dump for one split. |
| `code/fh_tirgn_train.py` | `65a856ec9594a40249d56a2b53297f9ee4c3cd964c12a4f362525c7e8062a5bb` | 1219 | TiRGN training wrapper (runs hd_tirgn.py). |
| `code/fh_train.py` | `dce0c092b22498911675739e5085d2f3558a8530e5e4adef5f7fb1f95948eda0` | 8752 | Stage 1: HRcov (HRCOV_PAT) re-ranker fit + freeze, validation only, seeds 0-2. |
| `code/fulltest_eval.py` | `74849444912e3d8a52e422e16566f1a062bd1ea216feb762fc89311d66252031` | 15134 | Harness library (imported). |
| `code/fw_agg.py` | `d99eb9363e458c0d3a6391ddbb48a0dd201ef04c076dc62a6b9ac117eb25458e` | 6151 | v79 decision: metrics, paired query and timestamp-block bootstraps, outcome R1-R4, weekly deltas. |
| `code/fw_freeze.py` | `e04c3cd162d91ff6bf449806eed74259ba305f1c0c2b58effd42cb67df701b95` | 5680 | SHA-256 freeze manifest of every Stage-1 frozen file. |
| `code/fw_freeze_job.sh` | `06c31414eeae5c9accaa620af4289993ebb25eb7544a7991c721e360d5cc0711` | 1123 | End-of-Stage-1 job: install rule documents + Stage-2 code, run fw_freeze.py, tar the frozen code. |
| `code/fw_gpu_wait.sh` | `591c3d0626864eaa28d16c1e197acc8651e31e58911017708f31d408d4c13287` | 1181 | GPU gate sourced by the stage scripts: serverA GPU 0 only, > 35 GB free; logs every check. |
| `code/fw_latency.py` | `dd11c774efcb63d496371c0ce59fcb42c39b7185c763854ec86dcbc9684f5333` | 20423 | Latency protocol for base, BLEND, HRcov, SA (validation slice; batch 1 and batched). |
| `code/fw_latency.sh` | `e89d161a7be151ff2b3e0062e1f96bc29709f8e0d82b514b17c6518eb06ec266` | 2534 | Latency job script (runs fw_latency.py and code_ct/fw_ct_latency.py). |
| `code/fw_stage1a.sh` | `482009f9489334896c93227519b8511a1b5ba7bf950774b20dd17f87df1279da` | 3420 | Stage 1A: prep, RecB grid, TiRGN training (GPU 0), validation rows. |
| `code/fw_stage1b.sh` | `cc642dad92c8684af96c49cae23b2859866529bf87cf01d174faf98dba251ce8` | 5286 | Stage 1B: HRcov lists/features/fit, BLEND selection, RecB_xf, SA (its SA part failed at the cut parse). |
| `code/fw_stage1b_sa.sh` | `6c5308a0fe0c2aec7b35ca467e26c8b2d41ae730afeeb9523900288081cec9cf` | 4398 | Stage 1B-SA rerun: RecB_xf selection + valid CSR, SA history/lists/pools, SA_W100 fit. |
| `code/fw_stage1c_ext.sh` | `b595f70d51b7747279ad1860894393434ca754678b25283f0c69aad505d428ad` | 1622 | Stage 1C: CountTRuCoLa C0 alone and 52 TiRGN blends on the validation selection part. |
| `code/fw_stage2.sh` | `3f05ceffd27cf8878a1f6cf4ae453242e2cc48972a53037a1e0c6cde470a6831` | 9925 | Stage 2: verify the freeze manifest, then the single test pass of every system. |
| `code/g2_train.py` | `3eeb7bed20a31c442c35f037c0570fba49b2e49d9343dc0b17ef17cbdef02ebe` | 12634 | HR re-ranker recipe (imported by fh_train.py / fh_apply.py). |
| `code/g7_cluster_bootstrap.py` | `f68f68dd0ec69ae3063d3ec08cf60a8087757ccddb13a561aaf3a8da64f45747` | 8601 | Inherited, not invoked. |
| `code/hd_agg.py` | `d3d3924c3fbe29e499da8678c57d43231c0237f067522dec1f45d5965d9c69b2` | 14722 | Inherited, not invoked (imported only by g7_cluster_bootstrap.py). |
| `code/hd_select.py` | `4628caed6fbe340b006e930da248dc652a0c8b45a15007d61d0522f6cd36dc81` | 1351 | Validation-only blend selection (used by Stage 1B). |
| `code/hd_tirgn.py` | `6a885fc9d0911e7eb2a5aa1afbc374ce0dfee684f5a223adef40ec6f3e9a053d` | 15376 | TiRGN trainer (authors' model code from the TiRGN tree via the pure-torch DGL shim). |
| `code/hd_tirgn_replay.py` | `fc93c472b966767710aafae675e105968d8d66660061e6c52dfb7d7b6cab837e` | 4235 | TiRGN row source (checkpoint replay with online history). |
| `code/oih_common.py` | `9de692d458a95dca400565e59c2a9ccbe69ab2ee1d2961fca9002074ac7bbcc3` | 13312 | Harness library (imported). |
| `code/patterns.py` | `e1d91114e1f017bef21fb74081b32e83fbdf72fa2183a9bcb087272b5c9c3ba7` | 6525 | Harness library (imported). |
| `code/rs_features.py` | `abd1c3bba38ae0fb674975137aeced128a0f993b0a275f0353dff6787d1248ed` | 24681 | Harness library (imported). |
| `code/rt_common.py` | `94e82b2f44b9ff7462b4281898a9c316145d74d9b1a31caadd05894cd0d960c8` | 11341 | Harness library (imported). |
| `code/v7_common.py` | `d726bf31063ad8d0ada80930f31f2570e7bbb01a564cb38e8d6aa4301ee66374` | 5711 | Loaders and bootstrap statistics used by fw_agg.py. |
| `code_ct/fw_ct_dataset_smoke.sh` | `76fb514c87c27080ed0569884068142ebe12f9a35628a7fe5694dc4c14ec8c1c` | 2335 | Adapter smoke test (RuleDataset from GDELTFW; no learning, no scoring). |
| `code_ct/fw_ct_latency.py` | `0f0b9109acdf7ff192855c448ed2784277fd0c2bad3e7fd198be5adf9c83ecb8` | 5079 | Latency of CountTRuCoLa rule application with the frozen Stage-1 rules. |
| `code_ct/fw_ct_main.py` | `5f86ed858bbf9ddb1f7c349ba7add34cc525a903baded4af2d3a87b45add1b67` | 5604 | Adapter: runs the released CountTRuCoLa rule_based/main.py on GDELTFW (id files, dataset entry). |
| `code_ct/fw_ct_memwatch.sh` | `48834367ab65ef843536f609fc9f79537ef9cb697b5946b9f909320e140c3cb6` | 1524 | Host-memory watchdog for the three concurrent CT Stage-1 runs (ran on the host; not in the frozen set). |
| `code_ct/fw_ct_score.py` | `05a802f672d4c6217f9d5dd44f1599558ff396ce0ec1ef8c627b4f151e6fa901` | 8921 | Scores CountTRuCoLa ranking files (validation / test) into harness dumps. |
| `code_ct/fw_ct_stage1.sh` | `844c70b6f2773419d733dc02c33814c4e1c5e83d69205bbf35c0bec67bedbfda` | 3663 | CT Stage 1 for C0 / C1 / C2 (learn on train, apply to validation only). |
| `code_ct/fw_ct_stop_window50.sh` | `77f3ffd0946e55a06562ddb14237ddc3fe94c97d968d13f067b390fd30cdd853` | 1140 | Manual stop of CT C1 / C2 at 16:08Z (ran on the host; not in the frozen set). |
| `code_ct/memmon.py` | `ce0bb0ee7410180e419a85a0cb3c37f461ffcf640573f95391f06e49f67ba85e` | 2369 | Process-tree memory / wall-time monitor wrapper. |
| `code_sa/fw_ext.py` | `85780673e543b1365764f7e60267105669ad4a0e7812e10482a4050418355e7e` | 8977 | EXT_alone / EXT_blend of CountTRuCoLa with TiRGN: selection (Stage 1C) and test scoring (Stage 2). |
| `code_sa/fw_pc_shim.py` | `2d723fc2f0caf65bcab5970f9583f660584da7feb85bd3691a9247434e3e62bb` | 478 | 'common' module that v78_lists / v78_pools import, for GDELTFW. |
| `code_sa/fw_sa_official.py` | `dd86b227e9a40a98f0c0dab63942f2e61dad268bcd9330c8c5e47a58a128c391` | 5019 | Official all-entity scoring of SA (no shipped negatives). |
| `code_sa/fw_sa_train.py` | `c24ad73f0f5a08b1be85d72fcf8c6cbd748fe2f60f7d7e4fe2512cdd87bbf436` | 8583 | SA_W100 fit (validation) and apply (test). |
| `code_sa/hd_blend.py` | `41220518c01d563b8ae6f35a5f372c613eca3abbd9559019872e7e9fd251fcf8` | 26049 | v78 hd_blend.py + GDELTFW entry. |
| `code_sa/hd_recb.py` | `617064ae3dac5c9f539a0968856eb0b6d3d1ebe211b02133ce72a871e4760c09` | 464 | Shim: hd_recb API for v76_recb_xf_select.py, backed by fh_recb.py (rewritten; the diff against v76 is not meaningful). |
| `code_sa/v76_recb_xf_select.py` | `9b39b411e775964abc99f4f0beedb9487387652ac6f4dae27d8eb203a7875f27` | 4860 | Cross-fitted RecB_xf selection (unchanged v76 file). |
| `code_sa/v78_common.py` | `bef0fa6ee373364ad9e5116f23c7b8483d311e9069ac05d05890d1ed68832f88` | 4240 | v7.8 SA shared paths / config / strict history order (+ GDELTFW entry). |
| `code_sa/v78_hist.py` | `f823083e3df5b8833d466d79fb10c3907dd93af23c6e8e720c89d498591f6dfe` | 6748 | v7.8 strict history order of RecB_xf. |
| `code_sa/v78_lists.py` | `5cd9806fe84bd1bd78c4ed7401b4ce94392f9ef6b4d3a96a3fcc9aaeb8c4e29b` | 13969 | v7.8 pattern blocks (unchanged). |
| `code_sa/v78_official.py` | `3c4bc5200333b414a5db73265c2c12c1e833f66ebee2ded7b7d74ff66b90e6a9` | 6337 | v7.8 official scoring (unchanged; GDELTFW scoring is fw_sa_official.py). |
| `code_sa/v78_pools.py` | `bde1461c05f7b2d23a4ebd7fa5a53aefa3695b29537a72a2afa9b4326b8bcc58` | 14625 | v7.8 pool assembly. |
| `code_sa/v78_train.py` | `a4389aa153a490bf4c7b4a5470002ab965b52b6fe018db7c18a0d49e1965ebf4` | 17197 | v7.8 re-ranker arms (unchanged; driven by fw_sa_train.py). |
| `diffs/INDEX.tsv` | `c4f1f09ff095f5bcf5cd64425b0478fb3dbffb6313a43a79965d08454be9a925` | 2755 | Each derived file, the original it came from, and the line counts of the change. |
| `diffs/code/fh_apply.py.diff` | `7b17d99b291cda421d3b3112fe23d4aaed09177dc7ea66d0d51c80110480f6d8` | 596 | Unified diff against the original: code/fh_apply.py |
| `diffs/code/fh_blend.py.diff` | `bff6e623d71e31869520e2774616544e1dc07f413f2fbe5209e3948eb1a6450b` | 5139 | Unified diff against the original: code/fh_blend.py |
| `diffs/code/fh_common.py.diff` | `ee84b9abbc5097df93efa57c6ef6482dc9c47d0280510c10594d3e815bb35c4a` | 3754 | Unified diff against the original: code/fh_common.py |
| `diffs/code/fh_features.py.diff` | `f7b9ec3737c1fbca02a95065ca75e27c6a076b7f49f856a42bada09f63bb9978` | 1391 | Unified diff against the original: code/fh_features.py |
| `diffs/code/fh_freeze.py.diff` | `b3311bdac4032733910e9982c892c7e239ab9a7c98003dfed7605b063fa2d349` | 3328 | Unified diff against the original: code/fh_freeze.py |
| `diffs/code/fh_gpu_wait.sh.diff` | `6d834ca949cee8090d629f5b6c7aa5ee61db6fca57def5f2032b18b40a10f8bc` | 613 | Unified diff against the original: code/fh_gpu_wait.sh |
| `diffs/code/fh_lists.py.diff` | `2f747ee72c9ce114e5e692edbb68e634bcd096f96a62806be0295c2a4394aa2a` | 597 | Unified diff against the original: code/fh_lists.py |
| `diffs/code/fh_prep.py.diff` | `1751cddbc48b83ca3bb8b8661735ab7dcd36500815b62584238b49f1bf4da75c` | 1508 | Unified diff against the original: code/fh_prep.py |
| `diffs/code/fh_recb.py.diff` | `66a6fc85485223d4fff295d9244bd97f8b88b2d1e92b58af7e07908815e06768` | 798 | Unified diff against the original: code/fh_recb.py |
| `diffs/code/fh_tirgn_rows.py.diff` | `d8f6a01bdf5b03b858287c77333fed3c11762780257604e7600073d777cdf066` | 1642 | Unified diff against the original: code/fh_tirgn_rows.py |
| `diffs/code/fh_tirgn_train.py.diff` | `3892c0627c9c7997fd0954dbecd8f6bcc867c9d3feb4f4dab7ec141588a2776f` | 952 | Unified diff against the original: code/fh_tirgn_train.py |
| `diffs/code/fh_train.py.diff` | `c54cfb7c2c4aa87fa0c1295b26480a73d784ebe64e5992530a5a9b091bf0eb0f` | 594 | Unified diff against the original: code/fh_train.py |
| `diffs/code/fw_agg.py.diff` | `1449c4a05c28fb4c035dda2ae2e9b92bb0e60d7e19400ffbe30e7d7dadea68c7` | 16963 | Unified diff against the original: code/fw_agg.py |
| `diffs/code/fw_freeze.py.diff` | `1cf45f716954e9bade0622e26c5f0097d65cae05abec8f018a62c9e8efd31200` | 5141 | Unified diff against the original: code/fw_freeze.py |
| `diffs/code/fw_gpu_wait.sh.diff` | `1b38b43788d7adbb311d9b7fe68c32a6516e16827d10c800d5fe6e2c1b5a9594` | 2479 | Unified diff against the original: code/fw_gpu_wait.sh |
| `diffs/code/fw_stage1a.sh.diff` | `04c08cdf00dc27202fb6588e71bfa3f7cf38b142d7b19b75091b526ba7a7d053` | 7883 | Unified diff against the original: code/fw_stage1a.sh |
| `diffs/code/fw_stage2.sh.diff` | `491f81b3fba6994d0f8b52e65842d9480fe3ebf2d08c0eddf8f8d457556b49ac` | 14690 | Unified diff against the original: code/fw_stage2.sh |
| `diffs/code_ct/fw_ct_main.py.diff` | `d5483a0cc7c1af06540cd57a442e8fc7b66e4efda2f44cbe2421c8beb95e75be` | 8671 | Unified diff against the original: code_ct/fw_ct_main.py |
| `diffs/code_ct/fw_ct_score.py.diff` | `2675dafaa88612c8b1db1261190850ab491e2dcc92d3b1c5c0d942fc87ed10ff` | 21159 | Unified diff against the original: code_ct/fw_ct_score.py |
| `diffs/code_sa/hd_blend.py.diff` | `66e882133463bde52dbf49e76329b7a7bbf06b109706bec010515221d5f5dcec` | 923 | Unified diff against the original: code_sa/hd_blend.py |
| `diffs/code_sa/hd_recb.py.diff` | `e4ccaefe71a7d5b29592075f32c46977e0d52dc2154f52d43e37dd461c799f24` | 25418 | Unified diff against the original: code_sa/hd_recb.py |
| `diffs/code_sa/v78_common.py.diff` | `1826e647d1625c9a45ab080a1bd948d90de51075b40c678af56d712040067daa` | 1503 | Unified diff against the original: code_sa/v78_common.py |
| `diffs/code_sa/v78_hist.py.diff` | `58636f316826c824e52245f8eade11554aa1dd2f6f90c8bdcefc942b5b07c209` | 3431 | Unified diff against the original: code_sa/v78_hist.py |
| `diffs/code_sa/v78_pools.py.diff` | `a89ad3378cbd4a03114dbced26673c07937b41f01eb76202c47e155d13bef070` | 5233 | Unified diff against the original: code_sa/v78_pools.py |
| `fresh_window_data_card.md` | `cc3e427f5cfb755b68c9032f0212b210ea6addf27f08895b0330d8d83195d360` | 8811 | Data card of GDELTFW: source, date ranges, filtering recipe, counts, GDELT licence terms. |
| `registration/ext_comparator.json` | `0be739fb69d7962c1c50b747142f261f7a8cd8906a331ecc35c267010047f914` | 912 | Frozen external comparator: CountTRuCoLa C0, blend a=0.1, tau=0.5 (artifact b4d773ad). |
| `registration/freeze_manifest_remote.json` | `4ce797b158efb7038ef25b5fd8d269131a5ac8f6a4ad8d1f18635bb8e44b61e0` | 54818 | SHA-256 freeze manifest written on serverA at the end of Stage 1 (201 files). |
| `registration/fresh_window_freeze_manifest.json` | `ee433c5b87993e3cd7f2d5f4509e0c864b9aa797d3df413f91a360a5a9dad98c` | 56946 | Same manifest + UTC chronology (artifact 8911bb05). |
| `registration/fresh_window_implementation_note.json` | `f677798b23240c81bc2312d83c51f96920848c48c2313086f756699517e5bf7d` | 10376 | Implementation note (artifact 7ef9ae3a). |
| `registration/fresh_window_implementation_note_addendum_1.json` | `5856d3bba6b7ba8d3f796964ba406158bbc7e54fa6857f137c3e66cf2f629d25` | 1714 | Addendum 1: C1/C2 dropped (artifact dede10e9). |
| `registration/fresh_window_uninspected_check.json` | `bfd93379c66b3914085fd11a53ce43648bc8325652d96e7ef76669ba3f8b754e` | 21591 | Uninspected check, PASS (artifact f255c8f4). |
| `registration/fresh_window_verdict.json` | `a96619d3e6cd71a3d32938201e3220969f2770f7e3316dc6c8994266cd62f627` | 5475 | Verdict: R4_OTHER, HRcov BEATS comparator, SMALL vs SA (artifact 5b185ac5). |
| `registration/v79_fresh_window_rule.json` | `14917cbaaef0306f4abb21516cf3941ae3b281a4b336dbdd2f9d43731d171929` | 4064 | v79 decision rule (lead; artifact e10ab60d). |
| `registration/v79_fresh_window_rule_amendment_1.json` | `2c69d03b316d9ff8ddcb5b2442f67ee036699869bf5c95424412ee832beaaf23` | 3426 | v79 amendment 1 (lead; artifact 6debf06f). |
| `steps/01_uninspected_scans.job.py` | `22b5267623e558a8a581264532088ee61c9ee93beb0a88b13a968a49b832c432` | 1682 | Verbatim job-submission cell. Read-only scan of serverA/serverB/serverC for any GDELT-derived file dated >= 2025-10-01 (jobs 24868d2a, 7281237c, 6f3c1306). |
| `steps/02_build_gdeltfw.job.py` | `6e03cc5fae3f6a8698abf75974e3652758fca2be8367cfc3d9a7cd89e5b5797e` | 2201 | Verbatim job-submission cell. Build GDELTFW from the official GDELT 2.0 15-minute exports (job faac377d). Same command as 01_window_build/run_build_gdeltfw.sh. |
| `steps/03_uninspected_record.local.py` | `8a030f56b60bd4f362b85627f155004b7942369e8622a6c081c0f2ec1cf78b95` | 10878 | Verbatim local cell(s). Assemble fresh_window_uninspected_check.json from the three scans, an artifact-store date search and the build summary. |
| `steps/04_implementation_note.local.py` | `0553fba4ce181bdd87df10b996a94087ae7da670d0fe74bea4407d29ca31009e` | 11646 | Verbatim local cell(s). Register fresh_window_implementation_note.json (write_prereg adds registered_utc + sha256). |
| `steps/05_setup_code_dir.ssh.py` | `9ebebc1a04f4e3bb8fa8b3b9e6b33a27fdb537a31c925f503d151feec57a2427` | 1863 | Verbatim command cell (serverA). Create the fresh-window code directory (same script as 03_setup/fw_setup_code.sh). |
| `steps/06_stage1a.job.py` | `168c50d84a11ee673ac7d77d1f492750e2b840baefcfd03f1f2c1312b762300e` | 1114 | Verbatim job-submission cell. Stage 1A (job c2ac489f): GDELTFW prep, RecB validation grid, TiRGN training on GPU 0, validation rows. |
| `steps/07_ct_stage1_attempt1.job.py` | `6bd8638e4096a5b920623ea86087b28279eb14d21bcac3e42f91eb92f483f2fc` | 1205 | Verbatim job-submission cell. CountTRuCoLa Stage 1, first attempt (job 7a51145d): failed at dataset construction in the adapter |
| `steps/08_ct_dataset_smoke.ssh.py` | `4aa497a1a215dcaeee51c2676924a238c6e72c410699c53626cf0c2215a3ad41` | 2282 | Verbatim command cell (serverA). Adapter smoke test (same commands as code_ct/fw_ct_dataset_smoke.sh). |
| `steps/09_ct_stage1.job.py` | `57862b31d54ddfd0a305afdd1ebd261074b9687e80b56e29f79a121d18861404` | 1154 | Verbatim job-submission cell. CountTRuCoLa Stage 1 rerun (job e7efbdf3): C0/C1/C2 concurrently, learn on train, apply to validation only. |
| `steps/10_stage1b.job.py` | `b46f90605aaacfecfc43a3b302caba4683654a13448a36d5744ae954b6e6fb1e` | 1837 | Verbatim job-submission cell. Stage 1B (job dc39717c): HRcov lists/features/fit, BLEND grid + selection, RecB_xf, SA (SA part failed at the |
| `steps/11_stage1b_sa.job.py` | `b33d091da119c23c51fba679dd11a9431901154d3d71101fefdc75b670c913ba` | 1014 | Verbatim job-submission cell. Stage 1B-SA rerun (job 9ccd0e7d): RecB_xf selection + valid CSR, SA history/lists/pools, SA_W100 fit. |
| `steps/12_ct_memwatch.job.py` | `3db9ddbfad4e2b04a69762a9b9d4ba92b6a1e3f75983ed46aa41da889cb2a971` | 1181 | Verbatim job-submission cell. Host-memory watchdog for the three concurrent CT runs (job eea61388). |
| `steps/13_ct_stop_window50.ssh.py` | `435ed54de1d73e4a7d53c3fc3e92ccd0dce263912288e92945023cf8578801ea` | 1392 | Verbatim command cell (serverA). Manual stop of CT C1/C2 at 16:08Z (same commands as code_ct/fw_ct_stop_window50.sh). |
| `steps/14_addendum_1.local.py` | `c6eb4683a48f8b7e0e02140c1a0b687d488425e8a7294ec841638e02aa7ab807` | 2193 | Verbatim local cell(s). Register fresh_window_implementation_note_addendum_1.json (C1/C2 dropped; comparator = C0 alone vs 52 C0 blends). |
| `steps/15_stage1c_ext.job.py` | `c9b4cd4de2ec67ec157cb2576f80a307acf490ddf26ae30d18e99c9b18bd4026` | 921 | Verbatim job-submission cell. Stage 1C (job 953e4300): CT C0 alone and 52 TiRGN blends on the validation selection part. |
| `steps/16_ext_comparator.local.py` | `9e539cf51f58310bd174c6d3fe5d36d8fd51e2457d7f5ed0396909f2918c4f74` | 1842 | Verbatim local cell(s). Select the external comparator from sel_ext_gdeltfw.json -> registration/ext_comparator.json. |
| `steps/17_latency.job.py` | `a9da05e1c76656554109a4c60b02a40933a76ac086b2a7b9468b2f31bd38797b` | 1137 | Verbatim job-submission cell. Latency protocol (job c46772b5): fw_latency.py (base/BLEND/HRcov/SA) + fw_ct_latency.py (CT comparator). |
| `steps/18_freeze_inputs.local.sh` | `488fdf2a47873acda380dfbc2292a958d0cace52fa937ca5373daf6cce986664` | 2719 | Verbatim local cell(s). Stage the registration JSONs into fw/freeze_in/ and write code/fw_freeze_job.sh (cell 622: tar option order fix). |
| `steps/19_freeze.job.py` | `b514e054db4f43ebd1c6c0145fba29a4ef387731d8c721ffa80338cc8377e89d` | 1286 | Verbatim job-submission cell. End of Stage 1 (job 117823f0): install rule documents + Stage-2 code, write freeze_manifest_remote.json, tar the frozen code. |
| `steps/20_stage1_summary.local.py` | `9a4a1654adfafdfedefe947375e8def14b6c5df9455de10c1c86a92c80074bbf` | 3334 | Verbatim local cell(s). Stage-1 validation summary (fresh_window_stage1_valid.json) from the job outputs. |
| `steps/21_freeze_manifest_chronology.local.py` | `5b986357e89cc715a0585dfa0d76f0644858feb748330738fa654de43c87e01f` | 2867 | Verbatim local cell(s). Add the UTC chronology to the host manifest -> registration/fresh_window_freeze_manifest.json. |
| `steps/22_stage2.job.py` | `88b87cd0b58f6199229cfb8abc4e80a4aa4d04d93107e25b70a38c1093ee296b` | 898 | Verbatim job-submission cell. Stage 2 (job 7349980c): verify the freeze manifest, then the single test pass per system. |
| `steps/23_latency_table.local.py` | `c4cddc979b96fc296d9454829604d3cf48c206ff8725b1edbb7e25189acafaa5` | 2225 | Verbatim local cell(s). Latency table (fresh_window_latency_table.csv) from latency_systems.json + latency_ct.json. |
| `steps/24_aggregate.local.sh` | `253a0f8163f880fc5ab0d8c514dde5dacb7027da7c94b6a5bc2e75e6c7490a76` | 934 | Verbatim local cell(s). Aggregation, paired query bootstrap and timestamp-block bootstrap, outcome R1-R4: code/fw_agg.py |
| `steps/25_verdict_checks.local.py` | `a9a8fcd85f37846a8df44604cccebea27d7b0c9abf842d0dbc64b5672168aca1` | 9768 | Verbatim local cell(s). Read-back of verdict/results; rank cross-checks (pipeline ranks vs scorer strict ranks); |
| `steps/26_report.local.py` | `81a552556a21aaf5e378bb65f2494c5e0c6d1a14ba81a8aa9bbda403d224faa8` | 14912 | Verbatim local cell(s). fresh_window_report.md: generator (660) and the two later text edits for report v2 (676 bash: corrections |
| `tests/fw_agg_patch_568.sh` | `d12b44419bdae2b33202259f31f08761fe9d94b2fd053ab494e14e875b367755` | 734 | Patch to fw_agg.py after the first synthetic check (already applied). |
| `tests/fw_agg_synthetic_check.sh` | `2680eadc7e2428d3bacd959d0c73c1cc55ce39be9eaf99ac9b04d601d025787d` | 2022 | fw_agg.py on synthetic score files, run before any test number existed. |
