# Pilot 1 of 4 — coverage stack on TGB 2.0 `tkgl-polecat`: pipeline README

Scope: frozen extrapolation bases (RE-GCN, LogCL) → history re-ranker (6.2 arm B, `HR`) → coverage stack
(`HRCOV` = arm B over base top-50 ∪ 50 pattern-prior candidates ∪ 50 stage-2 bridge candidates) →
coverage-only references (`PAT_BUMP`, `BR_CS`) → 6.5 augmentation oracle and gated arms → aggregation into
`polecat_pilot_verdict.json`. Every number in the verdict was produced by the commands below on
`serverA` (2× RTX PRO 6000 Blackwell Max-Q, 48 CPU, docker image `tkg-gpu:latest`, torch 2.8.0+cu128),
except aggregation and figures (local laptop, CPU).

## 1. Code

| file | role |
|---|---|
| `pc_common.py` | constants, paths, dataset registration, split/query order helpers |
| `pc_tgb_verify.py` | TGB ↔ harness identity checks (id map, row order, conflict sets, evaluator formula) |
| `pc_prep.py` | valid sample, RecB-lite, TE tables, strata (base-independent) |
| `pc_lists.py` | pattern-prior list, stage-2 bridge list, C_cov, coverage features |
| `pc_lookup.py` | exact base log p / rank lookup for list members; official TGB eval (`--mode systems` for promoted lists) |
| `pc_features.py` | 30-feature tensors for valid/test |
| `pc_train.py` | HR, HR_R, HRCOV, HRCOV_R, HRCOVFEAT (3 seeds), calibration, promotion |
| `pc_refs.py` | PAT_BUMP and BR_CS (valid-selected), exact ranks |
| `pc_aug.py` | 6.5 gold-injection oracle + stop rule |
| `pc_augarms.py` | 6.5 utility-gated arms + matched comparators |
| `pc_density.py` | G5 density controls (random / degree-matched / popularity at matched |C|) |
| `pc_train_diag.py` | post-hoc block-attribution arms (HRCOV_PAT, _BR, _DYAD, _NODYAD) |
| `pc_agg.py` | aggregation, bootstrap, decision → verdict JSON |
| `pc_figs.py` | three report figures from the verdict JSON |
| `pc_report_fill.py` | renders report tables from the verdict JSON |
| `pc_chain.sh` | per-base stage driver |
| `pc_recb_official.sh` | official TGB RecurrencyBaseline script runner |

All scripts import `pc_common.py`, which registers dataset `POLECAT` in the shared harness's
`oih_common.DATASETS` **at import time only** (no file in the shared harness is modified).
`rescore_lib.HARNESS` is monkey-patched to the pilot directory so that rescoring caches land under the pilot.

## 2. Paths (host serverA)

| name | path |
|---|---|
| `H` (shared harness, read-only) | `$WORK_DIR` |
| `P` (this pilot) | `$H/tracks/pilot_polecat` — `code/`, `work/`, `dumps/`, `stage/` |
| `DD` (harness-format data) | `$H/tracks/datasets/harness_data/tkgl-polecat` |
| raw TGB zip | `$H/tracks/datasets/raw/tkgl-polecat.zip` |
| TGB repo clone (RecB official) | `$P/stage/tgb_repo` |
| base dumps / checkpoints | `$P/dumps/{regcn,logcl}_scores_POLECAT_traintest.npz`, `$P/dumps/{regcn,logcl}_POLECAT_seed42.pt` |
| per-stage outputs | `$P/work/{prep,lists_<b>,lookup_<b>,feat_<b>,train_<b>,refs_<b>,aug_<b>,augarms_<b>,density_<b>,diag_<b>}` |

Environment overrides: `PC_HARNESS`, `PC_TRACKS`, `PC_P`, `PC_DD` (used for the local smoke copy).

## 3. Verified TGB ↔ harness identities (`pc_tgb_verify.py`, job)

* Entity id map is the **identity** (`h2t == arange(150931)`); relation ids identical; TGB edge rows are in harness CSV order.
* TGB conflict sets (`tkgl-polecat_val_ns.pkl` / `_test_ns.pkl`) equal the harness time-aware filter for all
  454,516 valid and 447,006 test (s, r, t) keys ⇒ negatives = all entities minus conflict set.
* TGB evaluator rank = 1 + #greater + 0.5·#ties ⇒ harness exact filtered rank (`oih_common`) is the same statistic;
  official py-tgb `Evaluator` (py-tgb 2.3.0) reproduces harness MRR to 1e-7 for the bases and for `HR`/`HRCOV` seed 0.

## 4. Stage commands and measured run times

Containers: CPU stages `docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -v $EXT_DIR:$EXT_DIR ...`;
GPU stages add `--gpus device=<g> -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=8`. Pin one base per card.

| stage | command (inside container, cwd = job dir) | RE-GCN | LogCL |
|---|---|---|---|
| base training + full-row dump | harness `base_dump.py --model {regcn,logcl} --dataset POLECAT --epochs 12 --history 10 --seed 42` (GPU) | 2532 s + 124 s | 3721 s + 273 s |
| TGB verification | `python3 pc_tgb_verify.py` (CPU) | 1 run | — |
| prep (base-independent) | `python3 pc_prep.py --workers 24` → valid sample 200k (rng 12345), RecB-lite, TE tables (400k early-train queries), strata | 76 s | — |
| lists | `python3 pc_lists.py --base <b> --workers 12` (CPU): pattern prior L_pat(300), stage-2 bridge (8,000 coverage-hard train queries, 3 seeds) L_br(200), C_cov(150), 11 coverage features | 3330 s @12 | 3850 s @10 |
| lookup | `python3 pc_lookup.py --base <b>` (GPU; needs `pip install py-tgb`): exact base log p / rank for every list member + official TGB eval of base | 1863 s | 1274 s |
| features | `python3 pc_features.py --base <b>` (CPU): 30 features (8 base, 11 evidence, 11 coverage) | 189 s | 174 s |
| train | `python3 pc_train.py --base <b>` (GPU): HR, HR_R, HRCOV, HRCOV_R, HRCOVFEAT × 3 seeds | 409 s | 363 s |
| refs | `python3 pc_refs.py --base <b>` (CPU): PAT_BUMP, BR_CS (valid-selected) | 45 s | 41 s |
| aug oracle | `python3 pc_aug.py --base <b>` (GPU): gold-injection oracle, 20k test sample | 107 s | 105 s |
| aug gated arms | `python3 pc_augarms.py --base <b> --n_v1 5000 --n_v2 4000` (GPU) | 6387 s | 6978 s |
| density controls (G5) | `python3 pc_density.py --base <b> --workers 8` (CPU) | 149 s | 139 s |
| post-hoc diagnostic arms | `python3 pc_train_diag.py --base <b>` (GPU) | 548 s | 566 s |
| official eval of systems | `python3 pc_lookup.py --base <b> --mode systems --systems HR_s0,HRCOV_s0` (GPU) | 2286 s | 2694 s |
| official RecB (TGB script) | `docker run --shm-size=16g -e PYTHONPATH=$P/stage/tgb_repo ... bash -c "pip install py-tgb ray; bash pc_recb_official.sh <np> False"` (CPU) | test 2424 s, train+val 6140 s | — |
| aggregation | `python code/pc_agg.py --root aggroot --rule polecat_decision_rule.json --out polecat_pilot_verdict.json` (local) | ~10 min | — |
| figures | `exec(open('code/pc_figs.py').read())` with `VERDICT_PATH`, `OUT_PREFIX` set (local) | <1 min | — |

Chained per-base driver: `bash pc_chain.sh <base> <gpu> <workers> [smoke_n] [stages]`
(stages default `lists,lookup,features,train,refs,aug,collect`; `smoke_n>0` runs a smoke on n queries;
`collect` writes `bundle_<base>.tgz` with everything `pc_agg.py` needs). Measured chain wall time:
RE-GCN 5,204 s (lists + lookup; the chain then stopped at the 0.999 agreement assert, see pitfall 1) + 716 s
(resumed `features,train,refs,aug,collect`); LogCL 5,786 s in one pass.

Aggregation inputs (`aggroot/`): `bundle_<b>/{test_q.npz, ranks_test.npz, ranks_select.npz, refs_ranks_test.npz,
train_<b>.json, refs_<b>.json, feat_summary_<b>.json, lists_summary_<b>.json, lookup_<b>.json, aug_oracle_<b>.json,
dump_log_<b>.json, density_<b>.json, systems_<b>.json, train_diag_<b>.json, ranks_test_diag_<b>.npz,
augarms_ranks_test_<b>.npz, augarms_<b>.json}`, plus `prep_summary.json`, `recb_test_rank.npy`, `recb_official_default.json`.
Bootstrap: harness `paired_bootstrap` (Poisson(1), B = 2000; strata B = 1000).

## 5. Pitfalls met on this dataset (and fixes)

1. **Near-tie rank disagreement GPU dump vs exact lookup.** The re-scored lookup and the stored dump agree on
   0.9971–0.9977 of queries (max |Δrank| = 2, max |Δ log p(gold)| ≈ 1e-5): fp32 near-ties among 150,931 scores.
   The original 0.999 assert stopped the RE-GCN chain at `features`; relaxed to 0.99 and the measured agreement is
   recorded in the verdict. Systems scored with dump ranks are compared to the dump base; refs scored with lookup
   ranks are compared to `base_lookup` (the agg does this automatically).
2. **Quadratic per-query grouping.** `[pc[adm][pq[adm] == j] for j in qa]` is O(#queries × #pairs); fine on V2
   (4k queries), hours on the test pool (~180k queries). Use `group_pairs()` (sort + split) — fixed in `pc_augarms.py`.
3. **Root-owned files** from containers in the job dir: run containers with the job dir mounted and copy outputs
   out; never `rm` other tracks' dumps. `PYTHONDONTWRITEBYTECODE=1` avoids root-owned `__pycache__` in shared code dirs.
4. **Official RecB** (`examples/linkproppred/tkgl-polecat/recurrencybaseline.py`): ray workers do not inherit the
   repo path (`ModuleNotFoundError: modules`) — pass `-e PYTHONPATH=$P/stage/tgb_repo` to the container (and
   `--shm-size=16g` for ray's object store); train mode (λ, α grid on valid)
   is effectively serial per relation and projected > 8 h, so only default mode (λ 0.1, α 0.99) was run.
5. **Harvest caps.** Job outputs are harvested by glob; keep full-row dumps under `$P/dumps` and only copy
   compact `npz` / `json` bundles back.
6. **Static bridge index on train queries.** `BridgeIndex(S.q['train'])` is built once over all train facts and does not
   exclude a query's own fact, so the 8,000-query stage-2 training cohort and the 30,000-query density cohort see their
   own gold edge and later train facts (transductive features). Valid/test queries are unaffected (all train facts precede
   them). For a leak-free ranker, build the cohort features from facts with ts < t (e.g. per-snapshot incremental index).
7. **Augmentation sensitivity gate.** `pc_augarms.py` evaluates the Δ(1/rank) gate with seed 0 only on test (copy budget);
   the primary Δlog p gate runs all 3 seeds; matched comparators are built against the seed-0 primary gate.
8. **CPU budget.** ≤ ~24 workers in total on the shared host; `pc_lists` at 12 workers is the longest CPU stage.

## 6. Reuse for pilots 2–4

* `tkgl-smallpedia`, `tkgl-icews`: copy `pc_common.py`, change `DS`, `N_ENT`, `N_REL`, `DD`, `DUMPS`, `CKPT`, the
  published reference rows and the self-loop/inverse convention; rerun `pc_tgb_verify.py` first (the identity
  id-map and conflict-set equality must be re-checked, not assumed). Wikidata-derived smallpedia has yearly
  timestamps — `pat` TE tables' recency windows (30 snapshots) must be rescaled.
* Story-context pilot: `pc_lists.py`'s candidate-list interface (`L_pat`, `L_br`, `C_cov`) and `pc_train.py`'s
  promotion rule accept any extra candidate block of width ≤ 50 plus its features; add the block as a fourth
  column group in C_cov and append its features to `COV_FEATS`.
* Gate on the oracle first (`pc_aug.py`, ~2 min) before running the gated arms (`pc_augarms.py`, ~2–2.5 h).

---

# 7. Pilot 2 of 4 — static-KG fusion on TGB 2.0 `tkgl-smallpedia`: pipeline changes (appended 2026-10-12)

Pre-registered rule: `smallpedia_decision_rule.json`; verdict: `smallpedia_pilot_verdict.json`; epoch-grid selection: `base_selection.json`.

Scope: frozen bases (RE-GCN, LogCL, **TiRGN**) → history re-ranker (`HR`) → coverage stack (`HRCOV`, pilot 1, the
reference) → `HRCOV_ST` (HRCOV + 50 static-KG candidates + 12 static features) and controls (`HRCOV_DM` degree-matched
random block, `HRCOV_STFEAT`, `HRCOV_PAT`, `HR_ST`, `STONLY`) → official py-tgb evaluation → `smallpedia_pilot_verdict.json`.
Every number was produced on `serverA` (2× RTX PRO 6000 Blackwell Max-Q, docker `tkg-gpu:latest`,
torch 2.8.0+cu128, py-tgb 2.3.0) except aggregation and figures (local laptop, CPU).

## 7.1 Code (pilot 2)

| file | role | replaces / extends |
|---|---|---|
| `sp_common.py` | constants, paths, dataset `SMALLPEDIA` registration (import-time only), full-row memmap accessor, static-edge loader | `pc_common.py` |
| `sp_verify.py` | py-tgb identity checks (id map, relation map, row order, conflict sets, evaluator formula) **and** emission of the static Wikidata edges in the shared id space through py-tgb's own `static_data` processing | `pc_tgb_verify.py` |
| `sp_static.py` | static-KG index (typed CSR, both directions), typed path enumerator (S, SS, ST, TS over static ∪ temporal-before-t), path-type precision target encoding | new |
| `sp_prep.py` | valid sample, RecB-lite, pattern TE tables, **static path-type TE tables**, strata incl. static reach flags, uncapped 2-hop set sizes on a 20k test sample | `pc_prep.py` |
| `sp_base_grid.sh` | unmodified harness `base_dump.py` over an epoch-budget grid (valid-selected) | new |
| `sp_tirgn.py` | TiRGN train (valid-selected epoch) + harness-schema dump + full log-prob rows; authors' model code via the project's pure-torch DGL shim | new |
| `sp_fullrows.py` | GPU replay of RE-GCN / LogCL writing the FULL 47,433-entity log-prob row of every valid/test query (float32 memmap) | replaces `pc_lookup.py --mode lookup` |
| `sp_lists.py` | pattern prior + stage-2 bridge (pilot 1) + **static list L_st** + candidate layouts A [250] / B [150] + coverage and static features | `pc_lists.py` |
| `sp_features.py` | 42-feature tensors (8 base, 11 evidence, 11 coverage, 12 static) for both layouts; all base quantities exact from the full rows | `pc_features.py` |
| `sp_train.py` | 8 arms × 3 seeds over arbitrary column subsets of a layout (pilot-1 "B standalone, calibrated" recipe) | `pc_train.py` + `pc_train_diag.py` |
| `sp_listctl.py` | list-level density controls (uniform / degree-matched / popularity at matched list size) | part of `pc_density.py` |
| `sp_official.py` | official py-tgb `Evaluator` for the base and for promoted lists, CPU, read from the full rows | replaces `pc_lookup.py --mode systems` |
| `sp_chain.sh` | per-base driver: `fullrows,lists,features,train,listctl,official,collect` | `pc_chain.sh` |
| (v2 of the six files above adds the `SP_STATIC_MODE` switch of §7.8; mode `full` = pilot 2 unchanged) | | |
| `sp_agg.py`, `sp_figs.py` | aggregation, bootstrap, decision; three figures | `pc_agg.py`, `pc_figs.py` |

## 7.2 Paths (host serverA)

| name | path |
|---|---|
| `P` (this pilot) | `$H/tracks/pilot_smallpedia` — `code/`, `work/`, `dumps/`, `stage/` |
| `DD` | `$H/tracks/datasets/harness_data/tkgl-smallpedia` (read-only) |
| static edges (shared ids) | `$P/stage/static_edges.npz` (`h, t, p, n_nodes=309,794, n_prel=753`), names `$P/stage/static_names.json` |
| epoch-grid dumps | `$P/dumps/e{3,6,12,30,60}/{regcn,logcl}_scores_SMALLPEDIA_traintest.npz` (+ ckpt, dump log); selected budget symlinked to `$P/dumps/<base>_scores_SMALLPEDIA_traintest.npz` |
| TiRGN | tree copy `$P/stage/TiRGN/{src,rgcn,dgl_shim.py}`; dump `$P/dumps/tirgn_scores_SMALLPEDIA_traintest.npz` |
| full log-prob rows | `$P/dumps/<base>_{valid,test}_logp.f32` — 162,066 / 163,172 × 47,433 float32 = 30.8 + 31.0 GB per base (185 GB for 3 bases; disk had 1.5 TB free) |

## 7.3 Verified TGB ↔ harness identities (`sp_verify.py`, job)

* Entity map identity (47,433), relation map identity (283, inverse = +283), TGB rows = harness split files concatenated.
* Conflict sets equal for all 101,243 valid and 103,430 test (ts, s, r') keys; TGB negatives = exactly the 47,433 temporal entities.
* Static edges via py-tgb `ds.static_data`: 978,022 edges, 753 relation types; static processing EXTENDS the shared node-id dict
  with 262,361 static-only nodes (ids 47,433–309,793) — never candidates, only intermediates. 0 static-only↔static-only edges.
  4.14% of test facts have their (s, o) pair as a static edge (any property); 0.06% under the query's own property (reversed pair).
* Harness exact filtered rank vs official py-tgb Evaluator: MRR identical to 1e-6 for all three bases and all four seed-0 systems;
  per-query rank identical for 99.86% (RE-GCN, LogCL) / 98.7% (TiRGN) — the rest are ties only (TiRGN's log-mixture scores
  underflow to exact ties deep in the ranking; max |Δrank| 1885 at ranks > 1000, MRR effect < 1e-6).

## 7.4 Stage commands and measured run times

| stage | command | RE-GCN | LogCL | TiRGN |
|---|---|---|---|---|
| TGB verification + static emission | `python3 sp_verify.py` (CPU, needs `pip install py-tgb`) | 4 s (all) | | |
| prep (base-independent) | `python3 sp_prep.py --workers 20` | 45 s (all) | | |
| base grid | `bash sp_base_grid.sh <m> <gpu> 3,6,12,30,60` (harness `base_dump.py`, history 10) | e3 77 s … e60 976 s (dump-log `total_s`) | e3 176 s … e60 2,376 s (dump-log `total_s`) | — |
| TiRGN train + dump | `python3 sp_tirgn.py --epochs 30` (GPU) | — | — | 1,659 s train (25 epochs, best 20) + 128 s dump |
| full rows | `sp_fullrows.py --base <b>` (GPU) | 99 s | 102 s | (written by sp_tirgn.py) |
| lists | `sp_lists.py --base <b> --workers 10` | 131 s | 139 s | 129 s |
| features | `sp_features.py --base <b>` | 418 s | 405 s | 410 s |
| train (8 arms × 3 seeds) | `sp_train.py --base <b>` (GPU) | 2,341 s | 2,374 s | 2,613 s |
| list controls | `sp_listctl.py --base <b>` | 4 s | 4 s | 3 s |
| official eval (base + 4 systems) | `sp_official.py --base <b> [--systems ...]` (CPU, 10 workers) | 24 s | 23 s | 23 s |
| chain total | `bash sp_chain.sh <b> <gpu> 10 0` | 3,022 s | 3,052 s | 3,180 s |

(`sp_chain.sh` prints cumulative `[stage-time]` stamps; per-stage times above are differences of consecutive stamps.)
| aggregation | `python3 sp_agg.py --root aggroot --rule smallpedia_decision_rule.json --out smallpedia_pilot_verdict.json` (local) | ~3 min | | |

Aggregation inputs (`aggroot/`): `bundle_<b>/{test_q.npz, ranks_test.npz, ranks_select.npz, train_<b>.json, feat_summary_<b>.json,
lists_summary_<b>.json, listctl_<b>.json, official_{base,systems}_<b>.json + _ranks.npz, fullrows_<b>.json, dump_log_<b>.json}`, plus
`prep_summary.json`, `recb_test_rank.npy`, `tgb_verify.json`, `base_selection.json`, `static_names.json`, `static_edges.npz`, `relation2id.txt`.

## 7.5 Pipeline changes vs pilot 1 (and why)

1. **Full log-prob rows replace the per-list GPU lookup.** With N = 47k the full rows of valid + test fit on disk (62 GB per base);
   every downstream stage reads exact base log-probs, exact raw ranks and the exact filtered gold rank from them. Valid/test top-K,
   the HR pool (top-50) and the bridge anchors come from the full rows, not from the dump; the dump is used only for train rows
   (bridge cohort) and as a cross-check (rank agreement 99.89–99.91% for RE-GCN/LogCL, 100% for TiRGN). Pattern-prior base mass is
   therefore exact (pilot 1: top-200 + uniform tail).
2. **Candidate layouts with fixed block offsets.** Layout A = `C_cov` (pilot-1 definition, 150, variable pattern/bridge boundary) |
   static block (50) | degree-matched block (50); layout B = top-50 | 100 static. Arms are column subsets (`sp_train.ARMS`), so
   every arm shares one feature tensor and one exact-rank routine.
3. **Static path enumerator** (`sp_static.py`): intermediates with static degree > 1,000 do not expand (122 hub nodes);
   ST hop expands ≤ 128 static neighbours (lowest degree first), temporal hops use the 64 most recent distinct partners before t.
   Uncapped reach is still reported (membership via neighbour-set intersection).
4. **Epoch-budget selection on valid** for the harness bases: on smallpedia RE-GCN / LogCL overfit fast (full-valid MRR falls
   monotonically from 3 to 60 epochs: RE-GCN 0.490 → 0.396, LogCL 0.509 → 0.389), so the 12-epoch pilot-1 default would have
   handicapped them; both selected 3 epochs (grid edge — see pitfalls).
5. **TiRGN on smallpedia** with an on-the-fly global-history vocabulary (TiRGN's `rel_history_<t>.npz` would need 47,433² rows ≈ 18 GB
   index per timestamp × 125).
6. **No 6.5 augmentation stages** in this pilot (not run; see the report).

## 7.6 Pitfalls met on this dataset (and fixes)

1. **Concurrent LogCL processes on one card OOM.** The caching allocator of a running LogCL `base_dump.py` peaked at ~81 GB on
   one 95 GB card; a second LogCL job on the same card died with `CUDA out of memory` (and RE-GCN beside TiRGN at 63 GB). Run one
   LogCL per card, or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **Selected budget at the grid edge.** Both harness bases selected the smallest budget (3 epochs); 1–2 epochs were not tried
   (grid fixed in the pre-registered rule). Extend the grid downward when reusing on similar data.
3. **TiRGN ties.** TiRGN's `log(hr·p_hist + (1−hr)·p_raw)` underflows to identical values for many deep entities: harness strict-`>`
   ranks and TGB's `+0.5·ties` ranks differ for 1.3% of queries (all deep), MRR equal to 1e-6. Report the official evaluator row.
4. **grep buffering hides live logs.** `python ... | grep -v ... | tee log` buffers until exit; use `grep --line-buffered` or
   `python -u ... | tee`.
5. **Train-row bridge cohort is small and transductive.** Train rows of a 3-epoch base are near-memorised (train MRR 0.88 at 12
   epochs), so coverage-hard train queries are 45,884–53,774; the cohort keeps pilot-1's static `BridgeIndex` leak (pitfall 6 above).
6. **Static nodes outside the candidate space.** py-tgb extends the node-id dict with static-only nodes; negatives are still
   `0..47,432`. Never rank static-only ids; they are only path intermediates.
7. **`rs_features` recency window not rescaled.** `ev_log_ntrip_recent30` counts the last 30 snapshots = 30 years on yearly data
   (polecat: 30 days). Left unchanged (harness file is read-only); one of 42 features.

## 7.7 Reuse for pilots 3–4

* `tkgl-icews`: N = 87,856 → full test rows ≈ 4.65 M queries (2,325,689 facts × 2 directions) × 87,856 × 4 B ≈ 1.6 TB — **too large**; go back to pilot-1's
  per-list lookup (`pc_lookup.py`) or store only the top-2,000 + the requested lists. Keep the layout/column-subset training.
* Story-context channel: add its candidates as a fixed-offset block of layout A and its features as a new group; the arm
  machinery (`sp_train.ARMS`) needs only a new entry.
* Always report a degree-matched random block at matched size (`HRCOV_DM`) beside any new candidate source.

## 7.8 Direct-static-edge leakage control (addendum, appended 2026-10-12)

Pre-registered in `smallpedia_decision_rule_addendum_nodirect.json` before any addendum
list, model or number existed. One environment switch selects the static evidence; every output directory / file gets the mode suffix.

| `SP_STATIC_MODE` | suffix | static evidence | arm name in the verdict |
|---|---|---|---|
| `full` (default) | — | S, SS, ST, TS paths (pilot 2) | `HRCOV_ST` |
| `nodirect` | `_nd` | S paths (direct s–c static edge, any property / direction) removed; `st_1hop`, `st_log_n_S` ≡ 0 | `HRCOV_ST_NODIRECT` |
| `nodirect_strict` | `_nds` | as `nodirect`, and candidates with a direct edge to s lose ALL static paths and features (static degree without the s–c edges) | `HRCOV_ST_NODIRECT_STRICT` |

| file | role |
|---|---|
| `sp_static.py` | `enumerate_paths(..., nodirect=True)` skips shape S |
| `sp_lists.py` | in `_nd`/`_nds` modes: loads the pilot-2 bridge rankers from `work/lists_<b>/bridge_ranker_s*.pt` (no cohort, no retraining) and **asserts** that `C_cov` candidate ids and the 11 coverage features (layout A cols 0–149) equal the pilot-2 lists on every valid/test row, so the pilot-2 HRCOV ranks are a valid comparator |
| `sp_chain_nd.sh` | per-base driver: for each mode lists → features → `sp_train.py --arms HRCOV_ST` (3 seeds) → official seed 0 → `bundle_<b>_<sfx>.tgz` |
| `sp_leakage_audit.py` | direct-edge leakage audit (relation-pair classes, top-20, gain by class) from cached Wikidata labels and LLM judgements |
| `sp_agg_nd.py` | addendum statistics + pre-registered decision → `verdict["addendum_nodirect"]` |

Measured (host serverA, three bases in parallel, 14 workers each; RE-GCN + TiRGN training on GPU 0, LogCL on GPU 1):

| stage | RE-GCN | LogCL | TiRGN |
|---|---|---|---|
| lists `_nd` / `_nds` | 104 / 83 s | 104 / 79 s | 104 / 86 s |
| features `_nd` / `_nds` | 407 / 419 s | 408 / 412 s | 409 / 414 s |
| train HRCOV_ST × 3 seeds `_nd` / `_nds` | 494 / 445 s | 573 / 524 s | 525 / 455 s |
| official seed 0 `_nd` / `_nds` | 9 / 9 s | 10 / 8 s | 9 / 10 s |
| chain total (both modes) | 1,973 s | 2,120 s | 2,014 s |

Jobs c0010ec1 (RE-GCN), c676ce84 (LogCL), 6de5f663 (TiRGN). C_cov identity held on all rows (0 differing rows, both modes, three bases).
Local: `python3 sp_leakage_audit.py --root aggroot` (needs `wikidata_property_labels.json`, `leakage_pair_judgements.json`), then
`python3 sp_agg_nd.py --root aggroot --verdict <v2 verdict> --addendum <addendum json> --audit leakage_audit.json --out smallpedia_pilot_verdict.json`.

Pitfalls: (1) `torch.load` in torch 2.8 defaults to `weights_only=True` and refuses the numpy `mu`/`sd` stored with the bridge rankers —
pass `weights_only=False` (trusted, self-written files). (2) The `_nds` strict mode must not zero candidate-level static features
(a zero static degree on a candidate with static neighbours would itself flag "direct edge"); it sets the pair features to the
"no static path" values and recomputes the degree without the s–c edges. (3) wikidata.org `wbgetentities` returns HTTP 429 on rapid
batches; 50 ids per call with 2 s pauses and Retry-After back-off worked. (4) Semantic relation classes undercount leakage: the
largest lifts come from present-day outcome facts (e.g. `P131` located-in vs temporal `P1366` replaced-by) that a semantic judge
labels UNRELATED; use the retrained `_nd` arm, not the class split, as the control.


---

# 8. Pilot 3 of 4 — history re-ranker on TGB 2.0 `tkgl-icews`, STAGE A: pipeline changes (appended 2026-10-12)

Stage A is the "pilot of the pilot": bases trained on a contiguous recent window of train, every feature on the full
history, a uniform 10% test-query sample. Pre-registered rule: `stageA_decision_rule.json` (registered 2026-10-12T14:54:56Z, before any
base, list or model existed); verdict `icews_stageA_verdict.json`; report `icews_stageA_report.md`. Every number was produced on
`serverA` (2× RTX PRO 6000 Blackwell Max-Q, docker `tkg-gpu:latest`, torch 2.8.0+cu128, py-tgb 2.3.0) except aggregation
and figures (local laptop, CPU). Stage B (full 10.9M-fact base training, full test) was not started.

## 8.1 Code (pilot 3)

| file | role | replaces / extends |
|---|---|---|
| `pi_common.py` | constants (Stage A window, samples, grids), dataset `TKGL_ICEWS` registration (import-time only), **icews-safe `te_keys`** (monkey-patches `rt_common.te_keys`), vectorised `split_queries`, `filter_pairs`, chunked exact fused ranks, `SplitsView` (relation-mapped quads for history features) | `pc_common.py` |
| `pi_relnorm.py` | relation-normalisation map + per-merge evidence → `stage/relmap.npy`, `relnorm_mapping.csv` | new |
| `pi_relnorm_data.py` | relation-normalised harness data dir `stage/tkgl-icews-relnorm/` + post-hoc RecB-lite on it | new |
| `pi_verify.py` | py-tgb identity checks (entity / relation map, row order, inverse = r+391, conflict sets for every val/test key, evaluator formula) | `pc_tgb_verify.py` |
| `pi_prep.py` | samples, RecB-lite (patched profiler; full valid + full test ranks), TE tables (icews-safe keys), strata from the residual-track `Hist` (vectorised, full history) | `pc_prep.py` |
| `pi_base.py` | **window-trained** RE-GCN / LogCL (harness models), epoch grid with resumable checkpoints and in-run FULL-valid evaluation, sampled dump (valid + test samples, K = 200, gold lp, exact filtered rank, ties) | `base_dump.py` + `sp_base_grid.sh` |
| `pi_lists.py` | pattern-prior pool widening (pilot-1 `query_lists` without the bridge part), both relation schemes in one pass | `pc_lists.py` |
| `pi_replay.py` | GPU replay of the selected checkpoint: `lookup` (exact lp / raw rank of 300 requested candidates, gold lp, rank, ties, official base eval) and `systems` (official eval of promoted lists) | `pc_lookup.py` |
| `pi_features.py` | 25 features (8 base, 11 evidence, 6 pattern coverage), both schemes | `pc_features.py` |
| `pi_train.py` | HR, HRCOV_B (B ∈ {50,100,150}) × 3 seeds, pre-registered valid selection (B*, A*), then A*_RN | `sp_train.py` |
| `pi_recblite_official.py` | RecB-lite scored by py-tgb's Evaluator (lexicographic score vectors) | new |
| `pi_recb_official.py` | TGB RecurrencyBaseline (unmodified predictor) default mode, window 100, on the test sample | `pc_recb_official.sh` |
| `pi_chain.sh` | per-base driver `lists,lookup,features,train,official,collect` with per-stage completion markers (resume) | `sp_chain.sh` |
| `pi_agg.py`, `pi_figs.py` | aggregation, bootstrap, GO / NO-GO, strata, post-hoc block; three figures | `sp_agg.py`, `sp_figs.py` |

## 8.2 Paths (host serverA)

| name | path |
|---|---|
| `P` | `$H/tracks/pilot_icews` — `code/`, `work/`, `dumps/`, `stage/` (31 GB after Stage A) |
| `DD` (read-only) | `$H/tracks/datasets/harness_data/tkgl-icews` |
| relation-normalised data | `$P/stage/tkgl-icews-relnorm/` (+ `relmap.npy`, `relnorm_mapping.csv`, `relnorm_summary.json`) |
| TGB modules (copy of repo commit 3b83e5cf `modules/`) | `$P/stage/tgb_modules/modules` |
| base checkpoints | `$P/dumps/ckpt_<b>_stageA/ep<E>.pt` (model + optimiser + RNG), `grid_log.json`; selected `$P/dumps/<b>_stageA_best.pt` |
| sampled dumps | `$P/dumps/<b>_stageA_scores.npz` (`valid_*`, `test_*` keys) |
| per-stage outputs | `$P/work/{lists_<b>, lookup_<b>, feat_<b>_{orig,rn}, train_<b>, official_<b>}`; samples `work/{valid,test}_sample_idx.npy` |

## 8.3 Verified TGB ↔ harness identities (`pi_verify.py`, job)

* Entity map identity (87,856), relation ids = TGB edge types (391), TGB `num_rels` = 782 (inverse = r + 391), TGB rows = CSV order.
* Conflict sets equal for all 3,587,665 valid and 3,599,019 test (ts, s, r') keys; negatives = all 87,856 entities minus the conflict set.
* Base official vs harness on the test sample: MRR 0.263035 / 0.263035 (RE-GCN), 0.296899 / 0.296899 (LogCL); per-query rank identical
  for 99.93% / 99.91% (rest ties, max |Δrank| 1). Seed-0 systems: |official − harness| < 1e-9. RecB-lite: 0.301652 / 0.301652, per-query
  rank identical for 99.9998% (one float tie, |Δrank| 1).

## 8.4 Relation normalisation (`pi_relnorm.py`)

117 relation types first appear on/after 2020-01-01 (first use 2020-01-01 … 2021-03-01); every one is a zero-stripped CAMEO code with a
unique zero-padded original. All 117 merges accepted: 0 shared usage days for every pair, no original code used on/after 2020-01-01 (last
uses 2018-10-04 … 2019-12-31); the original is the most similar usage profile among the 266 codes active in 2018–2019 for 71 / 117 codes
and for 98.1% of the merged test facts (≤ 3rd most similar: 99.1%); median volume ratio new/old 0.88. Relmap is the identity on every
train relation, so pattern TE tables are shared by both schemes. 43.82% of test facts use a merged code (0% after normalisation); valid
has none (share 5e-6), so RELNORM changes only test-time inputs. The scouting note's "119 new types" (ids 272–390) includes ids 272 and 273, which first appear
before 2020 (272 occurs once, in 2014); the unseen-in-train test share (43.82%) is fully covered by the 117 merges.

## 8.5 Stage commands and measured run times

| stage | command | RE-GCN | LogCL |
|---|---|---|---|
| relnorm + prep + verify | `pi_relnorm.py`; `pi_prep.py --workers 20`; `pip install py-tgb; pi_verify.py` (CPU) | 21 s + 314 s (profiler 169 s, TE 115 s) + 101 s | — |
| base: window training + grid + dump | `pi_base.py --model <b>` (GPU) | 14 epochs, 154 s/epoch, full-valid eval 58 s/grid point, dump 75 s; total 2,775 s | 12 epochs, 459–516 s/epoch, full-valid eval 382–418 s/grid point, dump 126 s; total 8,995 s |
| lists (both schemes) | `pi_lists.py --base <b> --workers 20` (CPU) | 193 s | 192 s |
| lookup + official base | `pi_replay.py --base <b> --mode lookup` (GPU + CPU evaluator) | 257 s (official 147 s) | 336 s (official 151 s) |
| features (both schemes) | `pi_features.py --base <b>` (CPU) | 310 s | 320 s |
| train (4 arms × 3 seeds + A*_RN × 3) | `pi_train.py --base <b>` (GPU) | 768 s | 812 s |
| official systems (3 seed-0 systems) | `pi_replay.py --base <b> --mode systems` | 407 s | 469 s |
| RecB-lite official | `pi_recblite_official.py --workers 12` (CPU) | 52 s (shared) | |
| TGB RecB default, window 100 | `pi_recb_official.py --workers 12 --window 100` (CPU, needs `pip install py-tgb ray`, `--shm-size=16g`) | 127 s (shared) | |
| relnorm data + post-hoc RecB-lite RN | `pi_relnorm_data.py --nproc 16` (CPU) | ~10 min (shared) | |
| chain | `bash pi_chain.sh <b> <gpu> 20` | 1,993 s | 2,186 s |
| aggregation / figures | `pi_agg.py --root aggroot --rule stageA_decision_rule.json --out icews_stageA_verdict.json`; `exec(pi_figs.py)` (local) | ~25 min (≈ 110 paired bootstraps over 465k queries) | |

Aggregation inputs (`aggroot/`): `bundle_<b>/{test_q_orig.npz, test_q_rn.npz, ranks_test.npz, ranks_select.npz, train_<b>.json, lookup_<b>.json,
systems_<b>.json, base_<b>_stageA.json, ...}` plus `recb_test_sample_rank.npy`, `recblite_official*.{json,npy}`, `recb_official_w100*.{json,npy}`,
`recbRN_*`, `prep_summary.json`, `relnorm_summary.json`, `tgb_verify.json`. Download the npz files individually — `bundle_<b>.tgz` is ≈ 590 MB
because it carries `lookup_test.npz` (465k × 300 requested log-probs).

## 8.6 Pipeline changes vs pilots 1–2 (and why)

1. **Windowed base training with intact history.** Gradient steps only on the last 20% of train days (2,266,194 facts); RE-GCN's
   10-snapshot history reaches back before the window, LogCL's global copy-history is pre-rolled over all earlier train facts at the start of
   every epoch (otherwise its log-count input at evaluation would be on a different scale from training).
2. **One training run per base for the whole epoch grid.** Checkpoints (model + optimiser + RNG) at grid epochs, full-valid filtered MRR
   evaluated in the run, stopping rule fixed in the rule file; resumable from the last checkpoint. Equivalent to per-budget retraining
   because the recipe has no schedule that depends on the total budget. Selected: RE-GCN epoch 12 (14 tried), LogCL epoch 8 (12 tried) —
   both interior, unlike pilot 2.
3. **Sampled dumps, no train rows.** The dump holds only the 200,000 valid and 465,138 test sample queries (K = 200); the re-ranker never
   trains on train rows.
4. **Official evaluator, batched.** py-tgb `Evaluator.eval` on per-snapshot batches whose negatives are the full score row with the TGB
   conflict set (from the shipped `*_test_ns.pkl`) masked to −inf; per-query ranks by the evaluator's formula; the size-weighted batch means of
   `Evaluator.eval` are recorded beside them (not asserted; measured |difference| ≤ 1.1e-8); 20,000 queries additionally scored through `query_batch` (agree to float32 precision of 1/MRR). The pilot-1/2 per-query
   path would cost ~4 ms × 4.65M queries × systems at full test.
5. **`te_keys` collision fix.** The harness key `r·2e7 + pid` collides for R = 391 (max pattern id 29.97e6); `pi_common` uses `r·4e7 + pid`
   and replaces `rt_common.te_keys` at import time.
6. **No stage-2 bridge block.** HRCOV = base top-50 ++ B pattern-prior candidates (pilots 1–2: the bridge block adds ≈ 0; graph-unreachable
   facts are 0.4% of icews test). B ∈ {50,100,150} selected on valid; 6 coverage features instead of 11.
7. **Both relation schemes in one pass** (lists, features); the lookup requests the union of both schemes' widened candidates (300 ids).
8. **RecB-lite under the official evaluator** via score vectors that realise its lexicographic order (seen > unseen; recency-weighted count;
   train answer frequency) — closes the scouting open issue.
9. **Official TGB RecB on a sample** using the unmodified TGB predictor per relation on the sampled queries (fork pool instead of ray),
   mapped back to harness order.

## 8.7 Pitfalls met (and fixes)

0. `set -e; a | tail` inside `bash -c` (no pipefail) masks a failing step; `pi_relnorm.py` succeeded here, but use `set -eo pipefail`.
1. `docker run ... tkg-gpu:latest --shm-size=16g` passes the flag to the entrypoint (`exec: --: invalid option`); options go before the image.
2. Copying result files by base name into one local folder overwrites same-named files of another base — copy into per-base folders.
3. Bundles above the ~100 MB harvest threshold stay on the host; fetch the small npz files individually.
4. The valid period contains none of the new codes, so the RELNORM arm trains identically to the original arm (same epochs, α, β); its
   effect is purely the test-time features. Do not read identical valid numbers as a bug.
5. RecB-lite profiler ranks over the entities present in the data; identical to the TGB 1-vs-all ranking only when all entities occur
   (true for tkgl-icews; a toy smoke set with absent entities showed rank offsets).
6. LogCL evaluation cost is the per-query Python `GlobalHist.freq_vec` loop (≈ 385 s per full-valid evaluation of 4.65M queries); the full
   grid made LogCL training 2.5 h for a 2.27M-fact window.
7. Re-ranker block size B* = 150 is the top of the pre-registered grid on both bases (valid-select MRR 0.3743 / 0.3744 / 0.3748 for RE-GCN,
   0.3738 / 0.3744 / 0.3749 for LogCL) — extend the grid upward in Stage B.

## 8.8 Reuse for Stage B (full train, full test) and pilot 4

* Base training on all 10.86M train facts: expect ≈ 740 s/epoch RE-GCN and ≈ 2,200 s/epoch LogCL (× 4.8 facts) plus full-valid evaluation
  58 s / 385 s per grid point; the window bases' valid MRR (0.305 / 0.322) is the bar to beat.
* Full test (4,651,378 queries): dumps ≈ 7.4 GB per base (K = 200), lists ≈ 10 × 120 s per scheme, features 46.5 GB per scheme per base
  (W = 200, fp16), official systems ≈ 10 × 400 s per base. Keep `B_GRID` open upward (e.g. {150, 250, 400}).
* Retrain RE-GCN on `stage/tkgl-icews-relnorm/` as a separate base row (evaluation with the ORIGINAL ids / conflict sets) to measure the
  relation-code artefact on the base itself.
* Pilot 4 (story-context channel): add its candidates as a fixed-offset block after column 200 and its features as a new group; `pi_train.ARMS`
  takes column / feature lists directly.

# 9. Pilot 3 of 4 — `tkgl-icews` STAGE B (full train, full test): pipeline changes (appended 2026-10-14)

Result: `icews_stageB_report.md`; pre-registration `stageB_rule.json`.

## 9.1 Code (Stage B versions; all env-switchable, Stage A defaults unchanged)

* `pi_common.py`: `PI_STAGE` (stageA | stageB; Stage B work dir `work_stageB/`), `PI_TEST_FRAC` (1.0 = full test), `PI_BGRID` (150,200,300 -> W_MAX 350), `PI_LPLEN` (400), base names `regcn | logcl | regcnrn`, optional per-process CUDA cap `PI_GPU_MEM_GB`, `fast_global_hist()` (batched LogCL `GlobalHist.freq_vec`, identical output).
* `pi_base.py`: `--relnorm` (relation-normalised model inputs; filters/gold original), pre-roll = every snapshot before the first trained one, fast GlobalHist.
* `pi_lists.py / pi_replay.py / pi_features.py`: dump path from `DUMPS[base]` (stage-aware); `pi_features` loads npz members once per split (npz members are re-read on every access — fatal at 4.65M rows).
* `pi_train.py`: Stage B rule (B* over {150, 200, 300}, RN arm = HRCOV_B*), per-arm resume, every seed's promoted list saved; never deletes.
* `pi_prepB.py`: Stage B work dir from Stage A base-independent products + full-test strata (agree with the Stage A sample strata on every sampled row).
* `pi_blend.py`: copy of `tracks/hardening/code/hd_blend.py` (algorithm unchanged) + tkgl-icews entry, RN replay, `--base_only`, `--out_tag`; `hd_recb.py` and `hd_select.py` run unchanged.
* `pi_chainB.sh` (RE-GCN) / `pi_chainB2.sh` (LogCL, official scoring split into `official_blend` + `official_sys` passes); `pi_aggB.py`; `pi_figsB.py`; `pi_reportB.py`; `pi_ghcheck.py`; `pi_recbcost.py`.

## 9.2 Measured run times (host serverA; wall clock, with co-tenant load)

* Bases (all 10,861,600 train facts), epoch seconds — RE-GCN: 1: 1317, 2: 1336, 3: 1141, 4: 916, 5: 884, 6: 1624, 7: 1628, 8: 1635, 9: 1631, 10: 1624; full-valid eval s: 1: 91, 2: 112, 3: 95, 4: 64, 6: 141, 8: 128, 10: 139; dump 293 s.
* RE-GCN-RN: 1: 1638, 2: 1624, 3: 1637, 4: 1629, 5: 1623, 6: 919, 7: 772, 8: 772, 9: 773, 10: 866, 11: 1062, 12: 862; eval s: 1: 140, 2: 139, 3: 128, 4: 139, 6: 61, 8: 60, 10: 62, 12: 68.
* LogCL (epochs 1-2 harness GlobalHist, 3-12 batched): 1: 3648, 2: 4026, 3: 2840, 4: 1889, 5: 2386, 6: 2021, 7: 1553, 8: 1657, 9: 1663, 10: 1549, 11: 1541, 12: 1456; eval s: 1: 886, 2: 1175, 3: 418, 4: 417, 6: 229, 8: 354, 10: 312, 12: 259; dump 288 s.
* Hardening RecB (`hd_recb.py`, 20 workers): valid 14x13 grid 6,268 s (14.0e9 strict-recurrency pairs on valid); test stage + blend CSR 5,155 s.
* Official TGB RecB w100 full test 934 s (20 workers, needs `pip install ray`); RecB-lite official full test 436 s (16 workers).
* RE-GCN chain: lists 3,889 s (12 workers), lookup 530 s, features 3,306 s, re-rankers (5 arms x 3 seeds) 4,693 s, official one pass with 6 arms 9,780 s; blend valid grid 383 s.
* LogCL chain: lists 2,560 s, lookup 436 s, features 3,256 s, re-rankers 4,417 s, official pass 1 (base/blend/pool) 6,464 s, pass 2 (base + 3 seeds) 6,034 s; blend valid 455 s.
* RE-GCN-RN base-only official pass 3,350 s. Aggregation (both bases, ~180 bootstraps on 4.65M queries, 20 workers) 387 s.

## 9.3 Pitfalls met (and fixes)

1. LogCL on the full train peaks at ~83-96 GB on a 97.9 GB card; nothing can be co-scheduled with it. A co-scheduled RE-GCN OOM'd at start (LogCL survived). Use `PI_GPU_MEM_GB` caps when sharing a card so the newcomer fails instead of a co-tenant.
2. `GlobalHist.freq_vec` issues two H2D copies and one kernel per query; on late-train snapshots it cost ~8 ms/query. The batched version is 35.8x faster per call and torch.equal on real data; LogCL epochs fell from ~3,650-4,030 s to ~1,450-2,840 s.
3. `cmd | grep -v ... | tee log` block-buffers (grep without `--line-buffered`): no progress visible for hours. `docker logs <cid>` still shows the container's stdout.
4. bash reads a script incrementally: overwriting a driver script while a job runs it corrupts that job. Run every job from a job-private copy (`cp $P/code/pi_chainB2.sh ./chain_private.sh`).
5. Files written inside containers are root-owned; host-side `rm`/`mv` fails (`set -e` killed the RE-GCN chain at a prune step). Any file operation on them must run in a container. Per the investigators' instruction, no deletions on serverA without approval: the chain prune step must not be re-enabled.
6. Official scoring cost on the full test is dominated by per-batch negative construction + masking (~0.36 s per 512-query batch for one arm, ~1 s with 6 arms). Splitting into a blend pass and a systems pass lets the first start before the re-rankers finish. Two LogCL replays differ on 0.045% of base rows (GPU tie flips; MRR diff 8e-8), so score each arm entirely within one pass.
7. TGB RecB (`create_basis_dict(train+valid)`) gives no psi candidates to relations absent from train+valid, so the 117 new codes get xi only (unseen-code MRR 0.083). This is a property of the official method, not a bug in the re-implementation.
8. The valid period contains no merged code, so RN re-ranker arms train identically to the original arms (as in Stage A).

## 9.4 Reuse

* Relation normalisation is a 1-line input map for any base (`pi_base.py --relnorm`, `pi_blend.py` base names ending in `rn`). On tkgl-icews it is worth +0.034 MRR for RE-GCN alone, so any new tkgl-icews base should report both variants.
* The hardening comparators (`hd_recb.py` + `pi_blend.py` + `hd_select.py`) run on any harness-format dataset with a `DSCFG` entry.
