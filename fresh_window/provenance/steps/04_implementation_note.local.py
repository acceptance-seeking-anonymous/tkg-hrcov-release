"""Verbatim local notebook cell(s), session-archive index 450 (local python kernel, workspace root as cwd).
Register fresh_window_implementation_note.json (write_prereg adds registered_utc + sha256).
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
"""write_prereg helper (verbatim from the tkg-claims-protocol skill kernel used to register the v79 implementation note and addendum)."""
import json
import os
import hashlib
import shutil
import datetime


def write_prereg(path, rule):
    """Write a decision-rule JSON with a UTC registration time and content sha256. Save it as an artifact before any test number exists."""
    body = dict(rule)
    body["registered_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = json.dumps(body, indent=1, sort_keys=True, default=str)
    body["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with open(path, "w") as fh:
        json.dump(body, fh, indent=1, default=str)
    return body["sha256"]


# ===== cell 450
note = {
 "note_id": "v79_fresh_window_implementation_note",
 "implements": {"rule": "v79_fresh_window_rule.json (version e10ab60d-9569-49b6-bd1e-6cf415138a78, sha256 209bb1b5...)",
                "amendment": "v79_fresh_window_rule_amendment_1.json (version 6debf06f-fc19-49ac-be4e-44b87bbb47da, sha256 207689b8...)",
                "uninspected_check": "fresh_window_uninspected_check.json (version f255c8f4-129c-4055-a56d-808c26de0265, PASS)"},
 "status_at_registration": "GDELTFW built and checked; no model fitted, no rule learned, no score of any kind computed on any split of GDELTFW.",
 "data": {"name": "GDELTFW", "host_path": "serverA (serverA) tracks/phase2_fresh_window/data/GDELTFW",
          "N_entities": 1209, "R_relations": 215, "days": 273, "train": "2025-08-01..2026-01-31, ts 0..183, 509,717 facts",
          "valid": "2026-02-01..02-28, ts 184..211, 76,117 facts (152,234 queries incl. inverses)", "test": "2026-03-01..04-30, ts 212..272, 228,584 facts (457,168 queries)",
          "queries": "both directions (inverse relation r+R), harness split_queries order"},
 "validation_parts": "fit part = validation queries with ts < cut, selection part = ts >= cut, cut = uts[int(0.7*len(uts))] over the validation rows' ts ids (= ts 203, i.e. 2026-02-20; computed by the code, asserted). Used by HRcov, SA, RecB_xf and the CountTRuCoLa selection, exactly as in v7.7/v78.",
 "base": {"system": "TiRGN, GDELT25X forward-holdout recipe: hd_tirgn.py via fh_tirgn_train.py; history length 7, hr 0.3, lr 1e-3, angle 10, 2 layers, h 200, dropout 0.2, entity-only loss, <=30 epochs, patience 5, seed 42, dec_chunk 8192; epoch chosen on validation as in the recipe; full log-prob rows by fh_tirgn_rows.py (valid in Stage 1, test in Stage 2).",
          "hardware": "serverA GPU 0 only (RTX PRO 6000 Blackwell), process capped at 30 GB; load/memory recorded before each GPU stage; at registration GPU 0 carries one foreign process using 1.7 GB at 0% utilisation and GPU 1 is not used."},
 "code": {"pipeline": "tracks/phase2_fresh_window/code = copy of tracks/phase2_forward_holdout/code with ONLY these substitutions: dataset name GDELT25X->GDELTFW, dataset key gdelt25x->gdeltfw, track dir phase2_forward_holdout->phase2_fresh_window, N 1173->1209, R 218->215, GPU selection fixed to physical GPU 0; plus the evaluator extension below (fh_blend test stage). VALID_N stays 200000 (> 152,234 -> full validation, as for GDELT25X). A diff is stored with the freeze manifest.",
          "SA": "tracks/phase2_fresh_window/code_sa = copy of tracks/hardening/v78/code (v78_common, hd_blend, v78_hist, v78_lists, v78_pools, v78_train, v78_official) + v76_recb_xf_select.py, with ONLY: a gdeltfw dataset entry (no base pools; pools SA25/SA50/SA75 built, SA = SA_W100 on SA50), RECB_XF directory from the environment, a --splits option for v78_hist/v78_pools (Stage 1 valid, Stage 2 test; the snapshot stream and online counts are unchanged), v78_train split into --stage fit (validation only; models frozen) and --stage apply (test), all-entity negatives in v78_official (below). The HRcov feature-dir alignment in v78_hist is not used (SA rows = the HRcov validation rows = all validation queries).",
          "CountTRuCoLa": "JuliaGast/counttrucola_submission @ 5d9aedf (code unchanged), one copy of the tree per configuration, venv tracks/v78_external/env/ct, OMP/OPENBLAS/MKL/NUMEXPR threads = 1, NUM_CPUS 12. Wrapper fw_ct_main.py replaces only (i) the TGB dataset object given to RuleDataset by a harness-backed object with the same attributes (name 'tkgl-gdeltfw'; forward facts of train/valid/test in time order, then their inverses built as in the vendored add_inverse_quadruples (r + R); timestamps = unix seconds of the UTC day; masks = our splits; node_mapping.csv / rel_mapping.csv; num_rels = 2R, num_nodes = N asserted) and (ii) CountTRuCoLa's own evaluate() by a no-op (it needs TGB negative files that do not exist for GDELTFW; the rankings files are scored by our evaluator). The code's timestamp ids (reformat_ts, granularity 86400 s) equal the harness ts ids (no gaps; asserted)."},
 "systems": {
   "HRcov": "HRCOV_PAT arm of fh_train.py / g2_train.py (= v7.7 HRcov: hcpat_train.py states 'Same definition as the pre-registered G2 HRCOV_PAT; re-ranker code identical to g2_train.py'): base top-50 + <=50 pattern-prior block, W=100, 25 features, 2x64 listwise MLP on the fit part (<=120,000 queries, rng 0), epoch on selection-part MRR (patience 5), calibrated fusion alpha/beta on selection rows, seeds 0-2. HR is not fitted.",
   "SA": "v78 amendment 1 SA_W100: RecB_xf strict history order (RecB_xf score desc, train frequency desc, id asc), first block = history top-50, block = first <=50 conf_hist pattern candidates not in it, 16 base-free features (cols 8-18 evidence, 19 is_extra, 21-24 pattern), 2x64 MLP alone (no fusion), history tail, same fit/selection recipe, seeds 0-2.",
   "RecB_xf": "per-relation (lambda_r, alpha_r) selected by v76_recb_xf_select.py on the selection part from the stored 14x13 validation ranks of fh_recb.py --stage valid; CSR files by fh_recb.py (--only_valid in Stage 1).",
   "BLEND": "reference row: TiRGN (+) RecB, fh_blend.py grid on the full validation split + hd_select.py, exactly as in the forward holdout (v7.7 definition). RecB-POOL, RecB alone and RecB-lite are computed by the same scripts only as intermediates and are not reported (amendment: four systems + BLEND).",
   "CountTRuCoLa_configuration": "GDELTFW has no DATASET_OVERRIDES entry. The released tkgl-gdelt (LEARN_WINDOW_SIZE 200) and tkgl-polecat (150) entries cannot run on this window: RuleDataset.sanity_check_window_size calls exit() when the window exceeds half of the 273 timesteps. Candidates = released configurations that run on a daily 273-step window and come from daily political-event data: C0 = config-default.yaml top level (window 10), C1 = tkgl-icews14 entry (window 50), C2 = tkgl-icews18 entry (window 50, DATAPOINT_THRESHOLD_MULTI 50, RULE_UNSEEN_NEGATIVES 100; the tkgl-icews entry is identical). Values passed with --params; everything else from config-default.yaml; large-data behaviour automatic (1,628,836 quads incl. inverses > 1M: top-200 stored per query, C_X_COUNT 20). Stage 1 per config: learn on train, apply to validation only (EVAL_TESTSET_FLAG False).",
   "EXT": "v78 amendment EXT definitions: EXT_alone(c) = CountTRuCoLa scores (entities absent from the rankings line score 0, i.e. the floor = minimum of the 1,000 dumped scores); EXT_blend(c, a, tau) = (1-a) log p_TiRGN + a log_softmax(z/tau) over all N, z standardised over the top-1,000 dump (sd floor 1e-6), a in the 13 BLEND betas, tau in {0.5,1,2,4} (52 per config).",
   "external_comparator": "argmax of selection-part MRR (ties averaged) over the 3 EXT_alone and 156 EXT_blend candidates (configs C0-C2). Choosing among released configurations on validation is an implementation decision forced by the missing released entry; it can only favour the comparator. Only the selected comparator is applied to test (rules frozen from Stage 1, LOAD_PARAMS_FLAG; validation re-applied in the same run and required to reproduce the Stage-1 validation rankings)."},
 "evaluator": "Primary (all systems, Stage 2): time-aware filtered rank over all 1,209 entities with ties averaged, rank = 1 + #{e not filtered: s_e > s_gold} + 0.5 #{e != gold not filtered: s_e = s_gold} (the py-tgb Evaluator formula; negatives = all entities minus the other true objects of (s, r, t), i.e. TKGNegativeEdgeSampler 1-vs-all semantics, since no shipped negative file exists for GDELTFW). Score vectors: base = TiRGN log p; BLEND = frozen blend; HRcov = promoted list lifted over the base row (fh_blend lift); SA = promoted list lifted over N - position in the strict history order (v78_official); EXT = as defined. Harness strict rank 1 + #{>} reported alongside; pipeline ranks of fh_apply / v78_train are checked against the scorer (max abs deviation reported). HRcov and SA per-query RR = mean over seeds 0-2.",
 "statistics": "Per-query paired RR differences HRcov - X on all test queries; query bootstrap = Poisson(1) weights; timestamp-block bootstrap = multinomial resampling of the 61 test days; B = 2000, seed 0, draws shared across comparisons (v7_common.summ_multi); one-sided 95% bounds = 5th / 95th percentiles. BEATS(X): L > 0 under both AND point >= 0.005; MATCHES(X): U < 0.005 under both; SMALL otherwise. X in {SA, external_comparator}; BLEND and base descriptive. Outcome: R1_BEATS_BOTH if BEATS(SA) and BEATS(ext); else R2_SA_MATCHES if MATCHES(SA); else R3_EXTERNAL_MATCHES if MATCHES(ext); else R4_OTHER.",
 "latency": "Fixed slice = the first 2,000 validation queries of the selection part (harness order from ts 203). Same host (serverA), GPU 0, same container and thread settings for every system. Batch size 1 (each query processed alone through the stage's code path; median over 5 repeats) and batched (the production batching of each script; median over 5 repeats). Stages: base inference (TiRGN replay rows), candidate/feature construction (HRcov: pattern lists + 25 features; SA: history order + conf_hist lists + 16 features; BLEND: RecB scores; CountTRuCoLa: rule application of the frozen Stage-1 rules), ranker or blend scoring, end to end; reported as ms per query and s per 1,000 queries; descriptive only.",
 "discipline": "Stage 1 (train/validation only) ends with a SHA-256 freeze manifest saved as an artifact. Stage 2 then runs once per system on test. test.txt is read by the code only where the frozen pipelines read it for filters/split bookkeeping (as in the forward holdout); no test-row prediction, rank or metric is computed before the freeze. Crash-recovery reruns use identical inputs and are recorded.",
 "deviations_known_now": ["CountTRuCoLa configuration chosen on validation among three released configurations (no released entry for this dataset; the GDELT/POLECAT entries fail the code's own window check).",
                          "CountTRuCoLa's built-in evaluation is disabled (no TGB negative files); ranks come from our evaluator, as in v78 (which re-scored the rankings files with the official evaluator).",
                          "Only the selected external comparator is applied to test; the non-selected CountTRuCoLa variants are reported on validation only."]}
os.makedirs("fw/out", exist_ok=True)
r_ = write_prereg("fw/out/fresh_window_implementation_note.json", note)
print(r_)
