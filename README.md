# Learning to Rank Historical Evidence for Temporal Knowledge Graph Forecasting

Code and result files for the manuscript. A small listwise re-ranker (HRcov) is plugged in after a frozen temporal knowledge
graph forecaster and re-ranks the base's top candidates together with history-reached candidates. The repository contains the
experiment code, the result files from which the tables and figures are built, the registered decision rules with a hash
checker, and scripts that rebuild the tables and check them against the printed numbers.

## Repository layout

| folder | content |
|---|---|
| `src/` | experiment code, one directory per pipeline (bases, candidate pools, features, re-rankers, controls, statistics) |
| `scripts/` | one entry point per experiment (`run_*.sh`), setup, table rebuild |
| `analysis/` | rebuild of the tables and figure data from `results/`; check against the manuscript |
| `results/` | result files the tables and figures are built from; `LARGE_FILES.tsv` lists the per-query arrays with SHA-256 |
| `registration/` | registered decision rules, manifest, hash checker |
| `fresh_window/` | registered comparison on an uninspected 2026 GDELT window: code, data card, logs, verifier |
| `bases/` | fetch scripts and patches for the third-party base models |
| `data/` | dataset download and conversion; GDELT25X builder with SHA-256 list |
| `configs/` | path configuration and experiment registry |
| `tests/` | CPU checks, GPU smoke test, expected values |

## Requirements

Python 3.12 (tested with 3.12.3) on Linux. Install the pinned packages with

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # torch 2.8.0 (CUDA 12.8 build), py-tgb 2.3.0, lightgbm 4.6.0, ...
```

The checks below run on CPU. Re-running the experiments needs an NVIDIA GPU; set the paths in `configs/paths.env`
(copy of `configs/paths.env.example`) and `source configs/env.sh`.

## Quick start

```bash
bash tests/run_checks.sh      # CPU, about one minute, no data needed
bash scripts/make_tables.sh   # rebuild the main tables and figure data from results/ (into results/tables/, results/figures/)
```

`tests/run_checks.sh` rebuilds the tables from `results/` and compares every number with the manuscript, checks the
registered rule files and the fresh-window package, compiles or parses every script and checks `SHA256SUMS`.

## Paper parts and code

| paper part | code |
|---|---|
| main comparison (`tab:main`, `fig:basestrength`, `tab:mainfull`) | `src/harness/`, `src/pilot_*/`, `src/hardening/` (HRCOV_PAT, recurrence controls, bootstrap decision), `src/presub/`, `src/g8_cogntke/`; `scripts/run_pilot_*.sh`, `run_hardening.sh`, `run_presub.sh`, `run_g8_cogntke.sh`; rebuilt by `analysis/tab_main.py`, `analysis/fig_basestrength_data.py` |
| decomposition and ladder (`fig:decomp`, `tab:mechanism`, `tab:ladder`, `tab:v78ladder`) | `src/v7/` and `src/v76/` (ranker form, matched pool, base ablation; `scripts/run_v7.sh`, `run_v76.sh`), `src/v78/` (ladder pools SA, +C, +S, +T, HRcov; `scripts/run_v78.sh ladder`); the shared-query rows of `tab:ladder` come from the fit-population audit |
| standalone and external comparisons (`tab:standalone`, `tab:v78sys`, `tab:v78ext`) | `src/v78/` (selection, contrasts, calls), `src/v78_external/` (EdgeReMIND, CountTRuCoLa); `scripts/run_v78.sh` |
| fit-population audit (`tab:fitpop`, `tab:fitrefit`, `tab:fitsize`, `tab:fitstrata`) | `src/fitpop/`; `scripts/run_fitpop.sh` |
| fresh window (`tab:fresh`, `tab:freshweek`, `tab:freshlat`) | `fresh_window/` (see `fresh_window/README.md`) |
| registered decision rules (`tab:rules`) | `registration/`: rule files, manifest, `verify_registration_hashes.py` (hash checker) |

Each entry point documents its stages in its header; `configs/experiments.yaml` lists, per experiment, the rule file, the
scripts and the result files. Identifying paths and host names are replaced by neutral labels.

## Licences

- Code written for this study: MIT (`LICENSE`). Everything else (documentation, result files, registration documents,
  logs): CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).
- Third-party code is not included; `bases/fetch_upstream.sh` and `bases/fetch_external_methods.sh` fetch it at the commits
  used in the paper, under its authors' terms (`THIRD_PARTY.md`).
- GDELT-derived data and results come from the GDELT Project (https://www.gdeltproject.org/); its terms of use require a
  citation of the GDELT Project and a link to its website.

## Data availability

The per-query rank and score arrays (7.7 GB) will be made available upon acceptance.
