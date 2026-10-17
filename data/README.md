# data/

| file | content |
|---|---|
| `download_tgb.py` | downloads tkgl-polecat / tkgl-smallpedia / tkgl-icews with the py-tgb 2.3.0 URL table into `$WORK_DIR/tracks/datasets/raw/` and converts them with `src/datasets/tgb_to_harness.py` into `$WORK_DIR/tracks/datasets/harness_data/<name>/` (split = TGB `generate_splits`; counts checked) |
| `build_gdelt25x.py` | builds GDELT25X from the GDELT 2.0 15-minute event exports (2025-01-01 to 2025-09-30; validation July 2025, test August-September 2025) |
| `gdelt25x_missing_export_files.txt` | the 1,681 of 26,208 export files that could not be retrieved when the dataset was built; pass it with `--skip-list` to rebuild the same snapshot |
| `gdelt25x_relation_names.tsv` | the snapshot's `relation2id.txt` (CAMEO code, id, name); the builder takes the names from it, because some names in the snapshot came from an earlier relation table rather than `CAMEO.eventcodes.txt` |
| `GDELT25X.sha256` | SHA-256 of the seven files of the snapshot (`build_gdelt25x.py --check --out DIR`) |

GDELT25X statistics (from the build record): 1,173 entities, 218 relations, 256 days; train 514,299 / valid 94,778 / test 182,359 facts. The builder was not re-run for this release.

**ICEWS18 and GDELT.** G2 and the development-history rows use the RE-GCN release splits. Place them at `$EXT_DIR/logcl_redun/data/ICEWS18` and `$EXT_DIR/gdelt_armAD/data/GDELT` (the directory names are those of the original runs; `src/harness/oih_common.py` reads them from there), each with `train.txt`, `valid.txt`, `test.txt`, `stat.txt`. Their TiRGN score dumps (`$EXT_DIR/logcl_redun/tirgn_scores_ICEWS18_traintest.npz`, `$EXT_DIR/gdelt_tirgn_dump/tirgn_scores_GDELT_traintest.npz`) were made before this study from TiRGN checkpoints (ICEWS18: the TiRGN authors' released checkpoint; GDELT: TiRGN trained with history 7 and angle 10) with the tree of `bases/patches/tirgn_gdelt_dump.patch`; their SHA-256 are in `results/LARGE_FILES.tsv`.
