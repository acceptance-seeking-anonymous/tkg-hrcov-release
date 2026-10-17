# fresh_window/ -- registered comparison on the uninspected 2026 GDELT window (`tab:fresh`)

This directory is the fresh-window code package (136 files listed in `FILES.md` with SHA-256, bytes and role), unpacked, plus
files added for the public release (this README, `FILES_RELEASE.tsv`, `ADDITIONS.tsv`, `fw_env.sh`, `fw_localize.sh`,
`verify_fresh_window.py`, `provenance/README.md`) and the review material in `reproduce/`, `docs/` and `logs/`. The package
holds everything that ran for the comparison: window build and uninspected check, TiRGN base training and dumps, HRcov / SA /
BASE⊕RecB fitting and scoring, the CountTRuCoLa runs (the default configuration that was used and the two window-50
configurations that were stopped), latency harness, aggregation with both bootstraps, and the registered rule and outcome
documents. It holds no GDELT data, no GDELTFW files, no dumps, rankings, scores or weights, and no third-party code.

Result: outcome **R4_OTHER (mixed)** (registered rule `registration/v79_fresh_window_rule.json` with amendment 1).
HRcov − SA = +0.0039 MRR, query bootstrap [+0.0036, +0.0042], timestamp-block [+0.0027, +0.0053]: SMALL (not BEATS).
HRcov − CountTRuCoLa comparator = +0.0359: BEATS. The comparator is CountTRuCoLa in its released default configuration
(learning window 10) blended with the base; stronger configurations were not assessed (the two window-50 configurations were
stopped before any rule or score existed, `registration/fresh_window_implementation_note_addendum_1.json`). The numbers are in
`../results/fresh_window/fresh_window_results.json` and `../results/fresh_window/fresh_window_latency_table.csv`.

## Layout

| path | content |
|---|---|
| `FILES.md` | package description: provenance, prerequisites, **order of commands to reproduce**, per-file SHA-256 |
| `fresh_window_data_card.md` | data card of GDELTFW (source, build command, splits, counts, checksums of every output file) |
| `00_env/` | runtime records: Docker image, pip freezes, GPU |
| `01_window_build/` | GDELTFW builder (the GDELT25X builder, unmodified), build command, SHA-256 of every output file |
| `02_uninspected_check/` | read-only scan recorded before any model was fitted |
| `03_setup/` | creation of the run's code directory; TiRGN tree checksums |
| `code/`, `code_sa/`, `code_ct/` | the code that ran (the frozen set of the single test pass, except three `code_ct/` helpers, see `FILES.md`) |
| `diffs/` | every change against the files each script was derived from |
| `registration/` | rule, amendment 1, implementation note, addendum 1, uninspected check, comparator record, freeze manifests, verdict |
| `tests/` | synthetic-data check of `code/fw_agg.py` run before any test number existed |
| `provenance/steps/` | **provenance only, not runnable**: the orchestration and analysis cells verbatim, in the order they ran |
| `reproduce/` | `make_fresh_window_table.py` (rebuild of `tab:fresh` from the released files, checked cell by cell against a manuscript TeX; `make_table9.py` is an alias), its output `fresh_window_table_check.csv` and the rebuilt LaTeX `fresh_window_table_rebuilt.tex` |
| `docs/` | chronology of the comparison (`fresh_window_chronology.md` / `.json`), reconciliation of the GDELT25X / GDELTFW boundaries, construction specification |
| `logs/` | run-log bundle v2 (355 files with `SHA256SUMS`) |
| `FILES_RELEASE.tsv`, `ADDITIONS.tsv` | per released file: SHA-256 of the source and of the released copy, and the redaction applied |

## Redaction and verification

Identifying paths, account and host names are replaced by neutral labels (`anon`, `serverA` / `serverB` / `serverC`,
`user1`, ...), analysis-platform identifiers by neutral text, and process-table lines of other users are removed from the
logs; code was not edited. The rules that would name hosts, accounts or the platform are not published; they come with the
unredacted originals, which are available to the editor and reviewers on request.

```bash
python3 fresh_window/verify_fresh_window.py                  # bundle-only
python3 fresh_window/verify_fresh_window.py --originals DIR [--log-bundle LOGS --docs DOCS]   # with the unredacted originals
```

Bundle-only mode checks every released file against `FILES_RELEASE.tsv` / `ADDITIONS.tsv`, every unredacted file against
`FILES.md`, every frozen code file against the freeze manifest (`registration/fresh_window_freeze_manifest.json`), the log
bundle against its `SHA256SUMS`, and the registration copies against `../registration/`. With the originals (the unredacted
package, log bundle and review documents) it also checks that the redaction of every original equals the released copy byte
for byte. The rule-document copies inside `logs/` are run records; the canonical documents are in `registration/` and are
checked by `../registration/verify_registration_hashes.py`.

## Prerequisites

- **Runtime.** The runs used the Docker image `tkg-gpu:latest` (`00_env/tkg-gpu_pip_freeze.txt`: Python 3.12.3, torch 2.8.0+cu128)
  on one GPU, and a separate venv for CountTRuCoLa (`00_env/ct_venv_pip_freeze.txt`). The repository's `../requirements.txt`
  lists the Python packages of the main pipeline; the exact versions of the run are in the two pip freezes.
- **GDELT raw exports.** The builder fetches the GDELT 2.0 15-minute export files of 2025-08-01 .. 2026-04-30
  (`fresh_window_data_card.md`, section 1). Verify a rebuild with `sha256sum -c 01_window_build/GDELTFW.sha256`.
- **TiRGN** (not vendored; the upstream repository has no licence file). Upstream https://github.com/Liyyy2122/TiRGN at commit
  `4b74c98b24b200a138e22b7e69f948f0f8d0232a`, prepared with `../bases/fetch_upstream.sh` (upstream plus
  `../bases/patches/tirgn_stage.patch`) and staged at `$FW/stage/TiRGN`; check with `md5sum -c 03_setup/tirgn_tree.md5` inside
  the tree (13 files).
- **CountTRuCoLa** (not vendored), used unmodified:
  ```bash
  git clone https://github.com/JuliaGast/counttrucola_submission $B/repos/counttrucola_submission
  git -C $B/repos/counttrucola_submission checkout 5d9aedf4106926d13befa141e0855d684f3f5f9d
  ```
  with its venv at `$B/env/ct` (`00_env/ct_venv_pip_freeze.txt`). The exact Stage-1 and Stage-2 commands are in `FILES.md`.
- **py-tgb 2.3.0** for the official evaluator (as in the rest of the repository).

## Running it

The scripts hard-code three locations: the harness root `H` (`/home/anon/cs_scratch/open_ideas_harness` in this release),
`FW=$H/tracks/phase2_fresh_window` and `B=$H/tracks/v78_external`. Do not edit the released files (their hashes are recorded);
make a localized copy instead:

```bash
export H=/data/tkg_work            # harness root; defaults to $WORK_DIR when configs/env.sh has been sourced
source fresh_window/fw_env.sh      # sets H, FW (default $H/tracks/phase2_fresh_window), B (default $H/tracks/v78_external)
bash fresh_window/fw_localize.sh /data/fw_run    # copies the runnable parts and rewrites the hard-coded paths in the copy
cd /data/fw_run
```

Then run the steps of `FILES.md`, section "Order of commands to reproduce", in that order, from the localized copy. This
plain-shell sequence is the runnable path: uninspected scan, window build, code setup, Stage 1A (TiRGN), CountTRuCoLa Stage 1
(default configuration C0), Stage 1B / 1B-SA (HRcov, BASE⊕RecB, SA), Stage 1C (comparator selection), latency, freeze,
Stage 2 (verifies the freeze manifest, then one test pass per system) and aggregation (`code/fw_agg.py`). Steps marked
*local* in `FILES.md` were analysis-side bookkeeping (registration, staging, summaries); their record is in `provenance/steps/`.
The window-50 CountTRuCoLa configurations (C1, C2) and the memory watchdog are not needed to reproduce the reported numbers.
The steps that touch the GPU expect one free card (`code/fw_gpu_wait.sh` is the gate).

## Outputs used in the paper

| output | where |
|---|---|
| systems, contrasts, calls, outcome, weekly contrasts | `../results/fresh_window/fresh_window_results.json` (`tab:fresh`, `tab:freshweek`) |
| latency stages | `../results/fresh_window/fresh_window_latency_table.csv` (`tab:fresh`, `tab:freshlat`) |
| verdict | `registration/fresh_window_verdict.json` (= `../registration/fresh_window_verdict.json`) |

`../tests/run_checks.sh` rebuilds every number of `tab:fresh`, `tab:freshweek` and `tab:freshlat` from these files and
compares them with the manuscript; `reproduce/make_fresh_window_table.py --tex <manuscript TeX>` checks `tab:fresh` cell by
cell against the TeX.
