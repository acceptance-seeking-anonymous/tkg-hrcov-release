# Large files (not in this repository)

`LARGE_FILES.tsv` lists the per-query arrays, score dumps, lookups, candidate lists and base checkpoints behind the paper's
numbers: tier, category, path (in the layout of this repository: `$WORK_DIR/...`, `$EXT_DIR/...`, `$RESULTS_DIR/...`, and
`$FW/...` for the fresh-window track, see `../fresh_window/README.md`), size in bytes, SHA-256, and the server it was found on
(`serverA`, `serverB`; a file found on both servers appears twice). Smoke-test outputs are excluded.

| tier | category (grouped) | rows | distinct SHA-256 | GB |
|---|---|---|---|---|
| 1-per-query | per-query evaluator output (reciprocal rank, H@k, rank) | 180 | 104 | 4.4 |
| 1-per-query | per-query strata and flags | 37 | 16 | 0.5 |
| 1-per-query | re-ranker per-query ranks | 231 | 155 | 2.4 |
| 1-per-query | test-query metadata | 11 | 11 | 0.4 |
| 2-intermediate | base checkpoint | 41 | 38 | 2.0 |
| 2-intermediate | base score dump (top-K + gold) | 45 | 44 | 111.2 |
| 2-intermediate | exact base lookup | 26 | 22 | 70.7 |
| 2-intermediate | promoted candidate lists | 575 | 527 | 137.5 |

Tier 1 holds the arrays needed to recompute every MRR, Hits@k and bootstrap contrast without retraining: 286 distinct
files, 7.7 GB. Tier 2 (base checkpoints, score dumps, exact lookups, promoted candidate lists) holds
631 further distinct files, about 321 GB. Formats: NumPy `.npz` / `.npy`, PyTorch `.pt`.

The per-query rank and score arrays (7.7 GB) will be made available upon acceptance, with an index that maps every tier-1 row of
`LARGE_FILES.tsv` to its file; any copy can be checked against `LARGE_FILES.tsv` with `sha256sum`. Tier 2 is kept by the
authors.

Sixteen paths under `$RESULTS_DIR/runs/` are listed for two or three different files (outputs of different runs that the
original job directories kept under the same file name; 36 files in all); the SHA-256 column identifies each file.

The rows under `$RESULTS_DIR/v78/` are the per-query arrays read by the ladder, external-comparator and fit-population
aggregators (`src/v78/`, `src/fitpop/`), in the flat layout those scripts read; three of them are the same files as rows listed
under other paths. `analysis/stage_v78_inputs.sh` places the small inputs shipped in `results/v78/runs/` and `results/fitpop/`
next to them and checks the SHA-256 of every large file against this table. The three rows under `$FW/` are the per-query
outputs of the single test pass on the 2026 GDELT window, from which `fresh_window/code/fw_agg.py` computed
`results/fresh_window/fresh_window_results.json`.
