"""Verbatim local notebook cell(s), session-archive index 660, 676, 680 (local python kernel, workspace root as cwd).
fresh_window_report.md: generator (660) and the two later text edits for report v2 (676 bash: corrections
section; 680: artifact-link rewrite).
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 660
M_ = R_["metrics"]; D_ = R_["deltas_official"]
def f4(x): return f"{x:.4f}"
def row_sys(nm, lab):
    o = M_[nm]["official_ties_averaged"]; ps = o.get("per_seed_MRR")
    return f"| {lab} | {f4(o['MRR'])} | {f4(o['H1'])} | {f4(o['H3'])} | {f4(o['H10'])} | {f4(M_[nm]['harness_strict']['MRR'])} | {', '.join(f4(x) for x in ps) if ps else '-'} |"
def row_d(k, lab, callv=None):
    s = D_[k]; q, t = s["query"], s["ts"]
    return f"| {lab} | {s['delta']:+.4f} | {q['L']:+.4f} / {q['U']:+.4f} | [{q['CI95'][0]:+.4f}, {q['CI95'][1]:+.4f}] | {t['L']:+.4f} / {t['U']:+.4f} | [{t['CI95'][0]:+.4f}, {t['CI95'][1]:+.4f}] | {callv or 'descriptive'} |"
lat_rows = "\n".join(f"| {r['system']} | {r['setting']} | {r['base_inference_s_per_1k']:.3f} | {r['candidates_features_s_per_1k']:.3f} | {r['ranker_or_blend_s_per_1k']:.4f} | {r['end_to_end_s_per_1k']:.3f} |" for r in LT.to_dict(orient="records"))
wk = "\n".join(f"| {v['days'][:8]}..{v['days'][-8:]} | {v['n']:,} | {v['HRcov-SA']:+.4f} | {v['HRcov-external_comparator']:+.4f} | {v['HRcov-BLEND']:+.4f} |" for w, v in R_["weekly_deltas"].items())
st1 = json.load(open("fw/out/fresh_window_stage1_valid.json"))
hr_sel = st1["hrcov"]["seeds"]; sa_sel = st1["sa"]["seeds"]
rep = f"""# v79 fresh window: HRcov vs SA vs CountTRuCoLa + TiRGN on an uninspected GDELT window

**Outcome: `R4_OTHER`.** HRcov **BEATS** the CountTRuCoLa comparator (+0.0359 MRR). Against the matched standalone ranker SA the call is **SMALL**: HRcov is ahead by +0.0039 MRR, and the gain is positive under both bootstraps, but it is below the 0.005 bar. A MATCH is not established either, because the timestamp-block upper bound is 0.0053, which is above 0.005. Under amendment 1 this result is reported as mixed in the main text, with every call and interval shown below.

Rule: `v79_fresh_window_rule.json` ([artifact]({{{{artifact:e10ab60d-9569-49b6-bd1e-6cf415138a78}}}})) + `amendment_1` ([artifact]({{{{artifact:6debf06f-fc19-49ac-be4e-44b87bbb47da}}}})). Implementation note ([artifact]({{{{artifact:7ef9ae3a-6fbf-433b-b6a1-536c6512f598}}}})) and addendum 1 ([artifact]({{{{artifact:dede10e9-7120-436c-a8d0-d9c77306dc4a}}}})) were both registered before any CountTRuCoLa rule or any score existed. The decision is in [fresh_window_verdict.json]({{{{artifact:5b185ac5-16e3-485f-8810-9b4ee702ebe9}}}}) and every number is in [fresh_window_results.json]({{{{artifact:637824ba-6bdf-40ca-a734-7394c7234795}}}}).

## 1. Window and uninspected check
- **GDELTFW** uses the GDELT25X builder recipe with every date shifted by +7 months (release builder `build_gdelt25x.py`, no skip list). It has 1,209 entities, 215 relations and 273 days with no gaps.
  - Train: 2025-08-01..2026-01-31, 509,717 facts.
  - Valid: 2026-02-01..02-28, 76,117 facts, 152,234 queries.
  - Test: 2026-03-01..04-30, 228,584 facts, **457,168 queries** over 61 days.
- **Uninspected check: PASS**, recorded before any fitting ([fresh_window_uninspected_check.json]({{{{artifact:f255c8f4-129c-4055-a56d-808c26de0265}}}})). It covered three things:
  - Read-only scans of all three project hosts. There is no GDELT export file, built dataset, build summary or CC-NEWS reference dated on or after 2025-10-01. The latest is 2025-09-30.
  - The artifact store. Every date-like hit was classified, and none is a GDELT event date.
  - A one-time check of the builder output.

## 2. Frozen systems (Stage 1, validation only)
| System | Frozen choice | Validation (selection part, ts ≥ 203 = 2026-02-20, 50,192 queries) |
|---|---|---|
| TiRGN base | forward-holdout recipe; best epoch 11 of 17 (patience 5); GPU 0, 626 s | full-valid filtered MRR 0.3978; selection part 0.4041 |
| BLEND (reference) | TiRGN (+) RecB, MIX β 0.3, default λ (42-config grid, full validation) | full-valid 0.4116 |
| HRcov | HRCOV_PAT (v7.7: W = 100, 25 features, 2×64 MLP, calibrated fusion), seeds 0–2 | {', '.join(f4(v['select_MRR']) for v in hr_sel.values())} (best epochs {', '.join(str(v['best_epoch']) for v in hr_sel.values())}) |
| SA | SA_W100 (v78 am. 1: RecB_xf history top-50 + conf_hist block, 16 base-free features, history tail), seeds 0–2 | {', '.join(f4(v['select_MRR']) for v in sa_sel.values())} (best epochs {', '.join(str(v['best_epoch']) for v in sa_sel.values())}) |
| CountTRuCoLa comparator | released code @5d9aedf, default config C0 (window 10, c-rules); **blend** with TiRGN, a = 0.1, τ = 0.5 (best of alone + 52 blends) | blend 0.4077 vs alone 0.3684 (full-valid alone 0.3574) |

RecB_xf: λ_r and α_r were chosen per relation on the selection part, for 366 relations. The other 34 relations, which have no selection-part queries, keep the TGB defaults. Freeze: 201 files with SHA-256 hashes ([fresh_window_freeze_manifest.json]({{{{artifact:8911bb05-7d65-43ca-82a0-f123ab6476f9}}}}), 16:32:23Z). The frozen code is in [fw_code_frozen.tgz]({{{{artifact:14bba154-e925-43c3-9746-3d92aa958dfb}}}}) and the Stage-1 record is in [fresh_window_stage1_valid.json]({{{{artifact:9caa50d2-c4b1-4c6a-b737-f773b36e1806}}}}).

## 3. Test results (single pass, 457,168 queries)
Evaluator: time-aware filtered rank over all 1,209 entities, with ties averaged (the py-tgb Evaluator formula). The harness strict rank is shown alongside; it agrees to within 1e-5 on average. HRcov and SA use the per-query RR averaged over seeds 0–2.

| System | MRR | H@1 | H@3 | H@10 | MRR (strict) | per-seed MRR |
|---|---|---|---|---|---|---|
{row_sys('base', 'TiRGN base')}
{row_sys('BLEND', 'BLEND (reference)')}
{row_sys('HRcov', '**HRcov**')}
{row_sys('SA', 'SA')}
{row_sys('external_comparator', 'CountTRuCoLa C0 + TiRGN blend')}

## 4. Decision (B = 2000, seed 0, shared draws; L / U = one-sided 95% bounds)
| HRcov − X | point | query L / U | query 95% CI | ts-block L / U | ts-block 95% CI | call |
|---|---|---|---|---|---|---|
{row_d('HRcov-SA', 'SA', 'SMALL')}
{row_d('HRcov-external_comparator', 'CountTRuCoLa comparator', '**BEATS**')}
{row_d('HRcov-BLEND', 'BLEND (reference)')}
{row_d('HRcov-base', 'TiRGN base')}

BEATS requires L > 0 under both bootstraps and a point estimate ≥ 0.005. MATCHES requires U < 0.005 under both. For SA, both L values are positive (0.0036 query, 0.0027 block), but the point estimate (0.0039) is below 0.005. The query U (0.0042) is below 0.005 but the block U (0.0053) is not. The call is therefore **SMALL**, and the outcome is **R4_OTHER**. For context, SA − comparator = {D_['SA-external_comparator']['delta']:+.4f} and BLEND − base = {D_['BLEND-base']['delta']:+.4f}.

**What this means for the conditions analysis.** On polecat (v78), HRcov beat both comparators. On GDELTFW, the base-free learned history ranker comes within 0.004 MRR of HRcov. The frozen base still adds a gain over SA that is detectable but practically small. That gain is concentrated in the first two test weeks and is about 0.000–0.004 MRR afterwards (weekly table below). CountTRuCoLa with its released default configuration is a weak learner on this dense window: alone it scores below the TiRGN base on validation. Its validation-tuned blend is close to the base, and HRcov clears it by a wide margin. Both bootstrap intervals for HRcov − BLEND exclude zero by a wide margin.

| ISO week (days) | queries | HRcov − SA | HRcov − comparator | HRcov − BLEND |
|---|---|---|---|---|
{wk}

## 5. Latency (descriptive; validation slice of 2,000 queries at ts 203; serverA GPU 0 + 8 CPUs; median of 5 repeats)
Seconds per 1,000 queries ([fresh_window_latency_table.csv]({{{{artifact:6d4620fa-3789-4365-81db-aee046625d24}}}})):

| System | setting | base inference | candidates / features | ranker or blend | end to end |
|---|---|---|---|---|---|
{lat_rows}

Notes on the latency figures:
- **Batch 1** means each query goes alone through the stage's production code: one process on CPU, batch 1 on GPU. **Batched** means the production batching: lists use an 8-process pool, including its start-up; features run as one chunk; GPU stages make one call.
- The TiRGN encoder (0.032 s per timestamp) is amortised over the slice.
- CountTRuCoLa rule application uses the frozen rules, with NUM_CPUS = 1 for batch 1 and 10 Ray CPUs for batched. Its split statistics (2.6 s) are amortised over the 152,234 validation queries. The 2,000 harness queries correspond to 963 CountTRuCoLa (s, r, t) queries.
- The timed code reproduces the frozen validation lists, features and history order exactly.

## 6. Integrity checks
- Freeze verified before the test pass: 201 files, 0 problems. The recomputed validation CSRs for RecB and RecB_xf were byte-identical to the frozen ones.
- The frozen HRcov models reproduce their Stage-1 selection ranks (agreement 1.0). The SA history tail order on test is identical to the reference (1.0).
- Pipeline ranks equal the scorer's strict ranks for every HRcov and SA seed and for the base (maximum absolute difference 0). The formula agrees with the vendored py-tgb Evaluator to within 1e-8 on 2,000 queries.
- CountTRuCoLa: re-applying the frozen rules to validation in the test run reproduced the Stage-1 validation ranks exactly (maximum absolute RR difference 0.0). Every test query has a rankings line.

## 7. Deviations and incidents
1. **CountTRuCoLa configuration.** GDELTFW has no released configuration entry. The released gdelt and polecat entries cannot run, because the code's own window check calls exit() on a 273-step window. Three released configurations were declared as candidates. Two of them (C1/C2, window 50) were stopped at 16:08Z for memory and time: 135 GB and growing about 1.1 GB/min at 8% of c-rule example collection, with no rule or score yet in existence (addendum 1). The comparator therefore uses the released default configuration only. The window-50 configurations remain untested on this window.
2. CountTRuCoLa's built-in evaluation was disabled because no TGB negative files exist for GDELTFW. The only code change relative to the release is the dataset-adapter wrapper `fw_ct_main.py`. Ranks come from our scorer, as in v78.
3. In Stage 2, CountTRuCoLa's `main.py` exited with rc = 1 inside its own results-CSV writer (a pandas dtype error) after rule application had finished. The rankings were complete and validation reproduced exactly, so it was not rerun.
4. Only the selected comparator (C0 blend) was applied to test. CountTRuCoLa alone is reported on validation only. The scripts computed RecB alone, RecB-POOL and the history order alone as intermediates; per the amendment's scope, none of these is reported as a system.
5. Two launches failed before computing anything and were relaunched after the fix: the CountTRuCoLa adapter (id-file format) and the SA chain (container banner in the cut parse).
6. GPU 0 was shared with another track's process (1.7–10 GB) during Stage 1 and the latency run. This track's processes were capped at 30 GB, and GPU 1 was not used.
7. The fresh window's training months include 2025-08..09, which were GDELT25X's validation and test months; this follows from the +7-month shift. Every validation and test fact is dated after 2025-10-01.

## 8. Chronology (UTC, 2026-10-16)
Rule 14:24:12 → amendment 14:24:46 → scans 14:30 → build 14:45:39 → uninspected check 14:46:47 → implementation note → TiRGN 14:50–15:01 → HRcov / BLEND frozen 15:07 → SA frozen 15:08 → C1/C2 stopped 16:08 + addendum → CountTRuCoLa C0 validation 16:19 → comparator frozen 16:20:19 → latency 16:20–16:31 → freeze 16:32:23 → **Stage 2 (single pass) 16:33:56–17:10:22** → decision {V_['decided_utc'][11:19]}.
"""
open("fw/out/fresh_window_report.md", "w").write(rep)
print(len(rep)); print(rep[:1500])
# ===== cell 676 (bash cell; verbatim, commented out so this file stays valid Python)
#| cd fw/out && python3 - <<'EOF'
#| s = open("fresh_window_report.md").read()
#| add = """
#| ## 9. Corrections and clarifications to the saved records
#| - **Addendum 1, memory figures.** C1 and C2 grew from about 106 GB at 15:41Z to 135 / 132 GB at 16:07Z. That is about 1.1 GB/min on average. The last 30-second samples before the stop show 2–5 GB/min. The intent text at the stop said "projected > 250 GB" and the addendum says "> 300 GB". Under either rate, a run would exceed the host's free memory long before its progress-bar ETA of 3 h 48 min. The "C0 runs at 20 GB" in the addendum is C0's resident memory at the time of the stop; C0's peak over the whole run was 33.1 GB (PSS). The registered addendum is left unchanged, and these figures supersede it.
#| - **Uninspected check, scope.** The check covered GDELT export files, every timestamp2id.txt, every build summary and every CC-NEWS day reference on the three hosts. It also classified every date-like hit in this project's artifact store. Post-cutoff dates in files such as `evidence.json` and `provenance_audit_edges.csv` are Wikipedia revision timestamps and Wikidata retrieval dates from the smallpedia and extraction work. They are not GDELT facts. The classification is in the record's `artifact_store_check`.
#| - **Dataset-shape statistics before scoring.** Right after the build, a read-only sanity script printed the ts range and day count of each split (test: ts 212–272, 61 days), the number of entities and relations present, and whether self-loops occurred. These are shape statistics computed before any model existed, not outcome statistics. No prediction, rank or metric on any split was computed before the freeze.
#| - **Freeze-manifest chronology.** The `chronology_utc` list inside [fresh_window_freeze_manifest.json]({{artifact:8911bb05-7d65-43ca-82a0-f123ab6476f9}}) is in code order, not time order; for example, the 14:53:54Z CountTRuCoLa launch failure is listed after the 15:01:08Z TiRGN entry. The manifest is left unchanged because it is the frozen record. The time-ordered sequence is in Section 8 and in `stage2_chronology_utc` of the results file.
#| """
#| s = s.rstrip("\n") + "\n" + add
#| open("fresh_window_report.md", "w").write(s)
#| EOF
#| tail -8 fw/out/fresh_window_report.md | cut -c1-200
# ===== cell 680
txt = open("fw/out/fresh_window_report.md").read()
txt2, k = re.subn(r"\(/[^()\s]*?/v8911bb05_fresh_window_freeze_manifest\.json\)", "(" + host.artifact_marker("8911bb05-7d65-43ca-82a0-f123ab6476f9") + ")", txt)
left = re.findall(r"/Users/[^\s)]*", txt2)
open("fw/out/fresh_window_report.md", "w").write(txt2)
print(k, len(left), txt2.count("8911bb05-7d65-43ca-82a0-f123ab6476f9"))
