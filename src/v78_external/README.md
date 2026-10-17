# v78_external — EdgeReMIND and CountTRuCoLa runs and per-query score dumps

Runs of 2026-10-15/16. Rule: `v78_framing_rule` (sha256 8e05c67b…, registered
2026-10-15T17:34:21Z), components EXT_alone and reproduction_gate. Protocol decisions written before any number:
file `v78_external_protocol_note.json`. This directory contains NO HRcov / SA / blend numbers.

## Layout

```
v78_external/
  README.md                this file
  MANIFEST.json            output of code/verify_dumps.py (alignment + consistency checks for every dump)
  repos/EdgeReMIND         git clone, HEAD 41c0db2df31bf4ab60e6e4fec8176932314e3ec8 (unmodified)
  repos/counttrucola_submission  git clone, HEAD 5d9aedf4106926d13befa141e0855d684f3f5f9d (unmodified code; run outputs in files/)
  env/er                   venv: torch 2.8.0+cpu, py-tgb 2.3.0, tgm (editable, repo above)      -> env/er_freeze.txt
  env/ct                   venv: repo requirements.txt (ray 2.59.0 ...); uses the repo's vendored tgb  -> env/ct_freeze.txt
  code/                    memmon.py, er_run.py, ct_score.py, verify_dumps.py, job scripts
  runs/edgeremind/<ds>/seed1337/{gate/, dump/, dump.log, mem_dump.json}
  runs/counttrucola/<ds>/{ct.log, <ds>-results.csv, dump/, mem_ct.json, mem_score.json}
  dumps/<method>/<ds>  ->  symlinks to the run dump directories (the consumer interface)
```

`<method>` is `edgeremind_s1337` or `counttrucola`; `<ds>` is `tkgl-smallpedia` or `tkgl-polecat`.

## Query order (identical to the harness)

Rows of every `val/` (= harness `valid`) and `test/` dump follow the harness order used for the HRcov base dumps:
`split_queries(Q[split], R, split)` of `hardening/v76/code/hd_blend.py` on `datasets/harness_data/<ds>/<split>.txt`
(stable sort by timestamp id; per timestamp first the forward block in file order, then the inverse block
`(o, r+R, s, t)`). The harness's own stored query list `hardening/v76/recb_xf/recb_csr_<split>_<ds>.npz['sro']` is
identical to it, and `keys_harness.npy` of every dump equals that array (checked in MANIFEST.json). Entity ids,
relation ids (inverse = r + R) and the TGB ids are the same (asserted by the harness and re-checked here); timestamps are
harness ts ids (`timestamp2id.txt`) in `keys_harness.npy`.

## Files per `<method>/<ds>/<split>/`

| file | shape / dtype | meaning |
|---|---|---|
| `gold.npy` | [Q] int32 | gold object id (== harness gold) |
| `keys_harness.npy` | [Q,4] int64 | (s, r, o, ts_id) of the harness row |
| `top_ids.npy` | [Q,1000] int32 | the method's 1000 highest-scoring entities, score descending, ties by entity id ascending |
| `top_scores.npy` | [Q,1000] float32 (EdgeReMIND) / float64 (CountTRuCoLa) | their scores (native dtype of the method) |
| `floor.npy` | [Q] | score assigned to every entity not in `top_ids` = the highest score outside the list (upper bound of the tail) |
| `tail_min.npy` | [Q] | lowest score over all entities; `tail_min == floor` means the tail is exactly constant |
| `n_gt_floor.npy` | [Q] int32 | number of entities scoring strictly above `floor` (entries after that in the list are ties at `floor`) |
| `gold_score.npy` | [Q] | exact score of the gold entity |
| `rr_official.npy`, `rank_official.npy` | [Q] float64 | per-query reciprocal rank / rank from the official `tgb.linkproppred.evaluate.Evaluator.eval` (py-tgb 2.3.0) on the FULL score vector against the shipped negatives (`TKGNegativeEdgeSampler.query_batch`, ties averaged = optimistic/pessimistic mean). Use these directly for EXT_alone; they do not depend on the top-1000 truncation |
| EdgeReMIND only: `rr_official_faithful_cutoff.npy`, `rank_official_faithful_cutoff.npy` | [Q] | same, but with the authors' history cutoff (smallpedia only) |
| EdgeReMIND only: `tgb_stream_pos.npy`, `cutoffs_strict_faithful.npy` | [Q], [Q,2] | position of the row in the TGM/TGB event stream; index cutoffs used |
| CountTRuCoLa only: `n_cand.npy` | [Q] int32 | number of entities with a non-zero score in the rankings file (<= 200: the released code stores the top 200 per query when the dataset has > 1M quads incl. inverses) |

Full score row reconstruction: `s = np.full(N, floor[i]); s[top_ids[i]] = top_scores[i]`. This is exact for CountTRuCoLa
(tail = 0) and for EdgeReMIND an upper-bound approximation of the tail (EdgeReMIND scores are linear in 15 history
features and can be negative; the tail is constant in < 1 % of queries — see `tail_exact_frac` in meta.json).
`<method>/<ds>/meta.json` holds config, commits, per-split MRR/H@1/3/10, alignment results and timings.

## History boundary

* CountTRuCoLa: rules fire on observations with t' < t (`filter_timesteps_within_apply`, `tt < t`) — the HRcov boundary.
* EdgeReMIND: the released evaluation uses position cutoffs over 200-event batches (all events of earlier batches,
  including earlier batches of the same timestamp; events of the current batch hidden). The dumps re-score every
  query with the trained theta (unchanged training) and the strict cutoff t' < t = HRcov boundary. The authors'
  cutoff numbers are recorded in meta.json (`test_mrr_authors`, `faithful_cutoff`) for the reproduction gate only.

## Reproduce

```
E=env/er/bin/python; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$E repos/EdgeReMIND/examples/linkproppred/tkgl/edgeremind.py --dataset tkgl-smallpedia --seed 1337 --num-workers 16   # gate
$E code/er_run.py --dataset <ds> --seed 1337 --num-workers 16 --out runs/edgeremind/<ds>/seed1337/dump [--faithful-perquery]
cd repos/counttrucola_submission && ../../env/ct/bin/python rule_based/main.py --params DATASET_NAME=<ds> NUM_CPUS=16
$E code/ct_score.py --dataset <ds> --out runs/counttrucola/<ds>/dump --rankings-val <..._val_...txt> --rankings-test <..._test_...txt>
$E code/verify_dumps.py
```

## counttrucola_adapted — EXT_adapted (added 2026-10-16; registered-optional comparator run AFTER the framing-rule verdict was read)

Definition: amendment `v78_rule_amendment_1` (sha256 8fffffca…), field `EXT.EXT_adapted`. Protocol decisions written
before the run: file `v78_external_protocol_note_2_ext_adapted.json`.

* Code and config: the same CountTRuCoLa commit and released config, run from a code-identical copy
  `repos/counttrucola_adapted` (diff -r against `repos/counttrucola_submission`, excluding run outputs: identical).
  Entry point `rule_based/ct_adapted_main.py` (= `code/ct_adapted_main.py`) imports the released `main.py` and changes one
  thing in memory: every validation fact (both directions) with harness ts id < cut moves from the TGB val mask to the
  train mask before `RuleDataset` indexes the data. Rules, examples and confidence parameters are thus learned on
  train + validation fit window; the val split holds only the selection window; the application history (union of
  splits, t' < t) is unchanged because no fact is duplicated. The moved counts are in `runs/counttrucola_adapted/<ds>/split_moved.json`.
* Split (the harness's): cut = `uts[int(0.7*len(uts))]` over the ts of the harness validation rows (hcpat_train.py / v78_train.py);
  tkgl-smallpedia cut ts id 105 (raw 2005; 105,606 fit rows, 56,460 selection rows, split file
  `hardening/v78/smallpedia/hist/valid_hist.npz`, sha256 f4b4910d…); tkgl-polecat cut ts id 1423 (raw 1637730000;
  harness rows = `pc_tirgn/work/valid_sample_idx.npy`, sha256 604f241d… = `hardening/v78/polecat/hist/valid_hist.npz['rows']`,
  sha256 363f10c9…; 60,764 selection rows in the sample, 161,448 valid rows with ts >= cut). On polecat ALL 372,024
  validation facts with ts < cut were moved (CountTRuCoLa treats head facts absent from its train index as negatives, so
  moving only the sampled rows would create false negatives).
* Dumps: `dumps/counttrucola_adapted/<ds>/{val,test}/`, same files, dtypes and harness order as `counttrucola`, plus
  `in_learning_data.npy` (val only meaningful): rows with ts id < cut were learning data, no ranking exists for them, and
  their `top_scores`, `floor`, `tail_min`, `gold_score`, `rr_official`, `rank_official` are NaN (`top_ids` = 0..999,
  `n_gt_floor` = `n_cand` = -1). Use only rows with `in_learning_data == False` (= the selection window). `meta.json`
  gives selection metrics over all selection-window rows and over the harness selection rows.
* No hyper-parameter or variant choice was made (released settings), so nothing was selected on the selection part.
