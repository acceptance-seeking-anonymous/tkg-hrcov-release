# Pilot 4 of 4 — story-context channel on MidEast-TE (Task 3), preliminary

**Plan steps covered:** (A) "Build time-masked text corpus for story context" and (B) "Pilot 4: story-context channel".
**Pre-registered rule:** `pilot4_decision_rule.json` (saved 2026-10-12 15:18 UTC, while the bases were still training; no test number had been read). **Verdict:** `story_pilot_verdict.json`. **Corpus card:** `story_corpus_card.md`.
**Hosts:** every model, feature and LLM number was produced on `serverB` (1× RTX 4090, docker `tkg-gpu:latest`, torch 2.8.0+cu128). Aggregation, bootstrap, figures and this report were produced on the local laptop from the harvested bundles. Host serverA was not used (both of its GPUs were busy with pilot 3).

## 0. Verdict

| question | answer |
|---|---|
| Pre-registered primary rule (HR+CTX vs HR ≥ +0.002 overall and on exact-recurrence-in-pool, p < 0.05, both bases, and beats HR+SHUF) | **PASSED on both bases** — RE-GCN +0.0042 overall / +0.0045 on R_in / +0.0046 vs SHUF; LogCL +0.0030 / +0.0036 / +0.0041 (each p < 0.0005: 0 of 2,000 bootstrap replicates ≤ 0). Label: **promising**. |
| Is the gain from *reading* the text? | **Mostly not.** A diagnostic arm that uses the same context articles but no text — whether the candidate takes part in an event extracted from one of the subject's recent articles (COOC) — gives **more**: +0.0050 / +0.0037. On top of COOC, the text reader adds +0.0007 (p = 0.003) / +0.0004 (p = 0.057) overall. Its only clear contribution is on dyadic-only golds (+0.0029 / +0.0021, p < 0.001). |
| LLM reader (Llama-3.1-8B-Instruct, 15k-query test sample) | **Does not count** under the sub-rule. On RE-GCN it lifts the re-ranker by +0.0047, but the same LLM scoring from the query alone (no text) gives +0.0042. The difference (+0.0005, p = 0.16) is not significant, so the gain is attributed to event memory or a generic prior. On LogCL the gain is +0.0010 (< +0.002). Provisional (sample). |
| Contamination control | Post-cutoff GDELT 2025-H1 slice (built from official exports, headline text). There the no-text LLM prior gives nothing (−0.0002 / −0.0009) and the LLM reader gives nothing that qualifies (+0.0007, p = 0.08 / −0.0001). The bge-m3 reader still lifts HR (+0.0016 / +0.0027), and COOC again lifts it more (+0.0040 / +0.0049). Under the same rule the slice passes on LogCL and fails on RE-GCN (descriptive only). |

What this means for the error analysis's top redesign option (story context for "which historical partner recurs"): time-masked news does carry usable signal for the exact-recurrence-in-pool category, and the gain is real, small (+0.003–0.005 MRR) and survives a stale-text control. However, the measurable part on this benchmark is "who appears in the subject's current articles". The extraction-level document links already expose that signal without reading. Reading adds value only where the pair's history is in another relation.

!Per-stratum deltas

*Figure 1. Δ MRR of each feature group added to the history re-ranker (HR), MidEast-TE full test (n = 89,896 queries, both directions), mean of 3 seeds, 95% paired-bootstrap CI. Strata use the base's own top-50 pool. Golds outside the top-50 (R_out, 4.0–4.4% of queries) are omitted because a top-50 re-ranker cannot change their rank (Δ = 0 for every arm). Dotted line = pre-registered +0.002.*

## 1. Corpus (plan step A)

Full details are in `story_corpus_card.md`.

* **Benchmark chosen:** MidEast-TE (SCTc-TE). It has 455,877 Vicuna-extracted atomic events (2015-02-19 – 2022-03-17), 2,794 entities and 234 CAMEO relations. Every event links to source-article md5s and URLs, and each article belongs to exactly one day.
* **Split:** we re-split chronologically with the authors' own TKG-baseline boundaries (`prepare_data_tkg_EGIRIS.ipynb`). This gives 347,449 / 47,780 / 44,948 unique facts; the event-row counts 360,190 / 49,125 / 46,562 match the notebook. The shipped CE split (by complex-event centre date) overlaps in time, so it cannot be used for extrapolation.
* **Text actually available:** full text only for the Egypt sub-corpus (14.3% of the 274,795 referenced articles). The Iran/Israel dump is "on S3" with no link and was not obtainable. For 65.4% of articles the richest shipped text is the headline recovered from the URL slug; 20.3% have no usable text. No URL was crawled. **The reader therefore mostly sees headlines.**
* **Licence:** the repository has no top-level licence. The articles are publisher content redistributed for research. No text is redistributed in any file.
* **Masking:** a query at day t sees only articles dated ≤ t−1, the same exposure as the structural bases. Window [t−7, t−1], ≤ 16 articles attached to an event involving the subject. Shuffled control: the same number of the subject's articles dated ≤ t−8.
* **Leakage audit:** windows containing an article dated ≥ t: 0 (valid and test, real and shuffled; asserted). Windows containing an article of the gold event: 0 (asserted). Legitimate signal: the gold object is named in 47.1% of test windows that have context (shuffled 38.8%) and co-participates in 58.7% (shuffled 50.8%). 86.9% of test queries have context.
* **Contamination:** MidEast-TE is entirely pre-cutoff for Llama-3.1 (Dec 2023). A post-cutoff slice was built from official GDELT 2.0 exports (2025-01-01 – 06-30, root events with a Middle-East actor). It has 1,024 country×role entities, 213 relations and 512,489 facts. Text = URL-slug headlines (80.0% of 248,622 URLs); no full text. 15,800 of 17,376 15-minute files were retrieved; the test month has only 14 event days.

!Corpus and audit

*Figure 3. (a) Text type of the 274,795 MidEast-TE source articles. (b) Leakage audit and gold-signal rates on the MidEast-TE test queries (n = 89,896). Rows 2–3 are conditioned on queries with ≥ 1 context article.*

## 2. Setup (plan step B)

* **Bases** (harness RE-GCN / LogCL code, vendored byte copies; h 200, 2 layers, history 10, seed 42). The epoch budget was selected on full-valid filtered MRR from {1, 2, 3, 4, 6, 8, 12, 16, 20}: RE-GCN chose 16 (valid 0.4419) and LogCL chose 12 (valid 0.4502); neither is at the grid edge. LogCL used a numerically identical `GlobalHist.freq_vec` (host-side fill, asserted equal to the harness version), which is 20× faster. Full valid/test log-prob rows give exact ranks (full-row vs dump rank agreement 1.000).
* **Arms** (3 seeds; all selection on valid): HR = pilot-1/2 recipe over the base top-50 (8 base + 11 history features). HR+CTX = + 10 bge-m3 reader features (hypothesis-sentence and name cosine to the context articles, within-pool z-scores, lexical mention count, query-level context size). HR+SHUF = the same 10 features from the shuffled context. HR+COOC and HR+CTX+COOC are diagnostics outside the rule. The LLM arms are trained on a 10k valid sample and evaluated on a 15k test sample. They use 20 options (the union of both bases' top lists), and the score is the log-softmax over option letters at the answer position.
* **Protocol:** time-aware filtered, both directions, 1-vs-all over 2,794 entities, full test (89,896 queries ≥ 50k). Bootstrap: harness Poisson(1) paired bootstrap, B = 2,000; p one-sided.

## 3. Main results — MidEast-TE full test (host serverB)

| arm | RE-GCN MRR | LogCL MRR |
|---|---|---|
| RecB-lite (strict recurrency, λ = 0.99 on valid) | 0.3990 | 0.3990 |
| frozen base | 0.4482 | 0.4569 |
| HR | 0.4817 | 0.4835 |
| **HR + CTX (bge-m3 story reader)** | **0.4859** | **0.4865** |
| HR + SHUF (shuffled stale text) | 0.4813 | 0.4825 |
| HR + COOC (same-article co-participation, no text) | 0.4867 | 0.4872 |
| HR + CTX + COOC | 0.4874 | 0.4875 |

Per-stratum results (shares: R_in 71.6% / 72.0%, R_out 4.4% / 4.0%, dyadic-only 17.1%, no history 6.9%):
* **R_in (the target category):** CTX +0.0045 / +0.0036, COOC +0.0058 / +0.0046, CTX on top of COOC +0.0004 (p = 0.14) / −0.0000.
* **Dyadic-only:** CTX +0.0057 / +0.0029, COOC +0.0049 / +0.0025, CTX on top of COOC +0.0029 / +0.0021 (p < 0.001). This is the only stratum where reading adds to co-participation.
* **No pair history:** CTX +0.0005 (p = 0.02) / −0.0003. HR itself loses here against the base (−0.0020 / −0.0050), as on ICEWS18.
* **No-context queries** (13.1%, where all CTX features are constant zero): HR+CTX − HR = +0.0013 (p = 0.01) on RE-GCN and HR+SHUF − HR = +0.0013. Re-training with extra feature columns therefore moves MRR by about ±0.001 even when those columns carry no information. Effects below ~0.001 should not be read as signal.
* **Ceilings:** gold in the base top-50 for 84.9% / 85.1% of queries. Perfect in-pool re-ranking would give MRR 0.851 / 0.852, so the headroom inside R_in is large (0.31 / 0.30 MRR-weighted) and the story channel recovers 1.2–1.5% of it.

## 4. LLM reader and contamination (sub-rule; provisional, 15k test sample)

!LLM arms

*Figure 2. Δ MRR of LLM-feature arms over the history re-ranker trained on the same 10k-query valid sample (n = 15,000 test queries per dataset, 3 seeds, 95% CI). The GDELT slice is a different dataset (GDELT-coded actors), so compare its pattern, not absolute values.*

| comparison (overall) | MidEast RE-GCN | MidEast LogCL | GDELT-25 RE-GCN | GDELT-25 LogCL |
|---|---|---|---|---|
| HR+LLM_CTX − HR_sub | +0.0047 (p < 0.001) | +0.0010 (p = 0.029) | +0.0007 (p = 0.079) | −0.0001 (p = 0.59) |
| HR+LLM_CTX − HR+LLM_SHUF | +0.0019 (p < 0.001) | +0.0018 (p < 0.001) | +0.0005 (p = 0.16) | +0.0001 (p = 0.40) |
| HR+LLM_CTX − HR+LLM_NOTEXT | +0.0005 (p = 0.16) | +0.0009 (p = 0.022) | +0.0009 (p = 0.025) | +0.0007 (p = 0.077) |
| HR+LLM_NOTEXT − HR_sub (memory / prior) | **+0.0042 (p < 0.001)** | +0.0001 (p = 0.39) | −0.0002 (p = 0.67) | −0.0009 (p = 0.99) |
| HR+CTX_sub − HR_sub (bge-m3, same sample) | +0.0034 (p < 0.001) | −0.0004 (p = 0.73) | +0.0004 (p = 0.31) | +0.0001 (p = 0.45) |

* The LLM's no-text prior lifts the RE-GCN re-ranker by +0.0042 on pre-cutoff MidEast-TE and by nothing on the post-cutoff slice. This is the pattern expected of event memory. However, the LogCL re-ranker gains nothing from the same prior (+0.0001), so the "memory" is information the LogCL features already carry, and the dataset changes between the two columns. Treat it as consistent with, not proof of, contamination.
* Reading versus not reading (LLM_CTX − LLM_NOTEXT) is +0.0005 to +0.0009 in all four cells. This is the same size as the noise floor in §3.
* Option coverage limits the LLM: the gold is among the 20 options for 77.2% (MidEast) and 73.5% (GDELT) of sampled test queries.
* On the 10k training sample the bge-m3 features do not help LogCL (−0.0004), whereas on the full valid split they do (+0.0030). The reader features need the full valid split to be learned.

## 5. Post-cutoff slice: main arms (GDELT 2025-H1, full test n = 85,956, host serverB)

| arm | RE-GCN MRR | LogCL MRR |
|---|---|---|
| RecB-lite | 0.3701 | 0.3701 |
| base (valid-selected 12 / 12 epochs) | 0.3883 | 0.3974 |
| HR | 0.4236 | 0.4234 |
| HR + CTX | 0.4252 | 0.4261 |
| HR + SHUF | 0.4231 | 0.4230 |
| HR + COOC | 0.4276 | 0.4283 |
| HR + CTX + COOC | 0.4275 | 0.4278 |

CTX − HR = +0.0016 / +0.0027 (R_in +0.0011 / +0.0026; vs SHUF +0.0021 / +0.0031). CTX on top of COOC = −0.0001 / −0.0005 overall, +0.0044 / +0.0032 on dyadic-only. The qualitative pattern of MidEast-TE replicates after the cutoff: the bi-encoder gain is not memory-driven, co-participation dominates, and reading helps on dyadic-only golds.

## 6. Negatives, caveats and open issues

1. **No base verification against a published row under this protocol.** SCTc-TE Table 6 reports RE-GCN MRR 0.2989 on MidEast-TE, under its CE protocol (CE events only, CE split by centre date, outliers as input graph, official code). Our RE-GCN gives 0.4482 on all test queries and 0.4713 on the CE-event subset (0.4506 forward only). The gap (+0.15) is larger than protocol details usually explain. It is unresolved; the likely cause is that their test CEs do not see ground-truth rolling history. Internal checks pass: LogCL > RE-GCN > RecB-lite, valid ≈ test, and full-row ranks equal the dump ranks.
2. **Headline-level text.** 86% of the test context articles are headlines (13.8% have full text), either slug-derived or titles. A full-text corpus (the S3 dump, or ACLED notes) might change the CTX-versus-COOC balance. This pilot cannot tell.
3. **COOC is a property of this dataset's construction.** Events are extracted from the same articles that form the context. A TKG without document identifiers cannot compute COOC, and there the text reader would be the only route to this signal. A follow-up would test HR+CTX on a TKG with the document links removed from the features.
4. **GPU nondeterminism.** Re-training LogCL changed epoch-1 valid MRR from 0.4303 to 0.4317, so the ps_base claim that epoch-e weights equal those of an e-epoch run holds statistically, not bitwise.
5. **LLM scope reduced** to 10k valid / 15k test queries (pre-registered; cost). The LLM sub-rule results are provisional.
6. **GDELT slice:** 9.1% of 15-minute files were not retrieved and the test month has only 14 event days. The actors are country×role codes, so it is not a like-for-like copy of MidEast-TE.
7. The shuffled control keeps the subject fixed (stale same-subject news). A random-subject shuffle would be weaker and was not run.
8. Not run: a cross-encoder reader; a full-test LLM pass (~4 h at 9k tokens/s); lag-2 masking sensitivity.

## 7. Mapping to Task 3 (and Task 2)

* **Task 3 (contextual TKG reasoning with LLMs):** the project frames the LLM as a reader of time-masked context, not as an event memory. The pilot shows the framing matters: on pre-cutoff data the LLM's no-text prior alone reproduces its reading gain (RE-GCN), and after the cutoff neither clears the bar. Any Task-3 claim about LLM in-context reading therefore needs (i) a post-cutoff evaluation period, (ii) a no-text arm, and (iii) a non-LLM reader on the same text. All three now exist in this pipeline.
* **What the story channel contributes:** a small, reliable re-ranking gain (+0.003–0.005 MRR) for "which historical partner recurs". It comes mostly from knowing who is in the subject's current news, which the document–event links already encode. For Task 3 this suggests treating the article as a retrieval unit (a subject's recent articles → their participants) and reserving the reader for relation-shift cases (dyadic-only golds), where it adds +0.002–0.004.
* **Task 2 (fusion):** COOC is a document-level co-participation edge type, an "eventuality" link between entities that never interact directly. Fusing it into the TKG as an extra relation type is a cheap, well-supported Task-2 experiment.

## 8. Reproduction (host serverB, `$WORK_DIR/tracks/pilot_story/`)

| stage | command | time |
|---|---|---|
| corpus + harness data | `python3 code/ps_build.py` (CPU) | 10 s |
| embeddings | `python3 code/ps_embed.py` (GPU) | 149 s |
| bases | `python3 code/ps_base.py --model {regcn,logcl}` (GPU) | 493 s / 501 s |
| contexts, audit, reader features, LLM inputs | `python3 code/ps_ctx.py` (GPU) | 334 s |
| HR features / RecB-lite | `python3 code/ps_features.py --base <b>`; `python3 code/ps_recb.py` (CPU) | ~15 s each / 27 s |
| arms (main) | `python3 code/ps_train.py --base <b> --mode main` (GPU) | 172 s / 158 s |
| LLM reader | `python3 code/ps_llm.py --max_tokens 7000` (GPU, bf16) | 2,638 s (75,000 prompts, 22.1M tokens) |
| arms (LLM) | `python3 code/ps_train.py --base <b> --mode llm` | 10 s |
| GDELT slice | `PS_DS=GDELT25 PS_P=$P/gdelt25` + `ps_build_gdelt.py`, then the same stages | build 1,282 s; LLM 2,324 s |
| aggregation / figures / tables / card (local) | `ps_agg.py --root aggroot --rule pilot4_decision_rule.json --out story_pilot_verdict.json --gdelt aggroot_gdelt`; `ps_figs.py`; `ps_report.py`; `ps_card.py` | < 1 min each |

Job drivers: `ps_job_build_base.sh`, `ps_job_logcl.sh`, `ps_job_embed.sh`, `ps_job_stage2.sh`, `ps_job_llm.sh`, `ps_job_gdelt_all.sh`, `ps_job_gdelt_resume.sh`. Jobs on serverB: 42eacec8 (build + RE-GCN), 5dc6907e (LogCL), e3baf788 (embed), 6c7400e6 (stage 2), 9cbdfb7f (LLM), 4fd33619 + 400e9be8 (GDELT slice).

---

# Appendix — full tables (rendered by `ps_report.py` from the verdict JSON)

### MidEast-TE: full-test rows (time-aware filtered, both directions, 3-seed mean of per-query RR)

| arm | RE-GCN MRR | sd(seeds) | H@1 | H@10 | LogCL MRR | sd(seeds) | H@1 | H@10 |
|---|---|---|---|---|---|---|---|---|
| frozen base | 0.4482 | 0.0000 | 0.3260 | 0.6849 | 0.4569 | 0.0000 | 0.3345 | 0.6927 |
| HR (history re-ranker) | 0.4817 | 0.0002 | 0.3579 | 0.7159 | 0.4835 | 0.0002 | 0.3604 | 0.7167 |
| HR + CTX (bge-m3 story reader) | 0.4859 | 0.0001 | 0.3632 | 0.7178 | 0.4865 | 0.0002 | 0.3640 | 0.7184 |
| HR + SHUF (shuffled stale text) | 0.4813 | 0.0008 | 0.3578 | 0.7154 | 0.4825 | 0.0010 | 0.3591 | 0.7160 |
| HR + COOC (same-article co-participation, no text) | 0.4867 | 0.0002 | 0.3647 | 0.7175 | 0.4872 | 0.0002 | 0.3651 | 0.7179 |
| HR + CTX + COOC | 0.4874 | 0.0005 | 0.3650 | 0.7182 | 0.4875 | 0.0002 | 0.3650 | 0.7185 |

n = 89,896 test queries.

### MidEast-TE: paired deltas (Δ MRR [95% CI], one-sided p; Poisson bootstrap B = 2000)

| comparison | stratum | RE-GCN | LogCL |
|---|---|---|---|
| HR_CTX-HR | overall | +0.0042 [+0.0036, +0.0049] p=0.000 | +0.0030 [+0.0024, +0.0037] p=0.000 |
| HR_CTX-HR | exact recurrence, gold in top-50 | +0.0045 [+0.0037, +0.0054] p=0.000 | +0.0036 [+0.0027, +0.0044] p=0.000 |
| HR_CTX-HR | pair history only | +0.0057 [+0.0045, +0.0070] p=0.000 | +0.0029 [+0.0016, +0.0041] p=0.000 |
| HR_CTX-HR | no pair history | +0.0005 [+0.0000, +0.0009] p=0.019 | -0.0003 [-0.0007, +0.0002] p=0.886 |
| HR_CTX-HR | has story context | +0.0047 [+0.0039, +0.0055] p=0.000 | +0.0036 [+0.0029, +0.0043] p=0.000 |
| HR_CTX-HR | no story context | +0.0013 [+0.0002, +0.0024] p=0.010 | -0.0004 [-0.0014, +0.0005] p=0.802 |
| HR_SHUF-HR | overall | -0.0003 [-0.0007, +0.0001] p=0.941 | -0.0010 [-0.0014, -0.0006] p=1.000 |
| HR_SHUF-HR | exact recurrence, gold in top-50 | -0.0003 [-0.0008, +0.0002] p=0.858 | -0.0003 [-0.0008, +0.0003] p=0.847 |
| HR_SHUF-HR | pair history only | -0.0002 [-0.0011, +0.0007] p=0.682 | -0.0044 [-0.0052, -0.0035] p=1.000 |
| HR_SHUF-HR | no pair history | -0.0011 [-0.0013, -0.0008] p=1.000 | -0.0014 [-0.0017, -0.0010] p=1.000 |
| HR_SHUF-HR | has story context | -0.0006 [-0.0010, -0.0001] p=0.993 | -0.0011 [-0.0015, -0.0006] p=1.000 |
| HR_SHUF-HR | no story context | +0.0013 [+0.0002, +0.0024] p=0.006 | -0.0009 [-0.0018, +0.0001] p=0.960 |
| HR_CTX-HR_SHUF | overall | +0.0046 [+0.0039, +0.0052] p=0.000 | +0.0041 [+0.0035, +0.0047] p=0.000 |
| HR_CTX-HR_SHUF | exact recurrence, gold in top-50 | +0.0048 [+0.0039, +0.0057] p=0.000 | +0.0038 [+0.0031, +0.0047] p=0.000 |
| HR_CTX-HR_SHUF | pair history only | +0.0059 [+0.0047, +0.0072] p=0.000 | +0.0073 [+0.0060, +0.0086] p=0.000 |
| HR_CTX-HR_SHUF | no pair history | +0.0015 [+0.0011, +0.0020] p=0.000 | +0.0011 [+0.0007, +0.0014] p=0.000 |
| HR_CTX-HR_SHUF | has story context | +0.0053 [+0.0045, +0.0060] p=0.000 | +0.0046 [+0.0039, +0.0053] p=0.000 |
| HR_CTX-HR_SHUF | no story context | -0.0000 [-0.0009, +0.0008] p=0.535 | +0.0005 [-0.0003, +0.0013] p=0.128 |
| HR_COOC-HR | overall | +0.0050 [+0.0043, +0.0057] p=0.000 | +0.0037 [+0.0030, +0.0043] p=0.000 |
| HR_COOC-HR | exact recurrence, gold in top-50 | +0.0058 [+0.0049, +0.0067] p=0.000 | +0.0046 [+0.0037, +0.0054] p=0.000 |
| HR_COOC-HR | pair history only | +0.0049 [+0.0037, +0.0061] p=0.000 | +0.0025 [+0.0014, +0.0037] p=0.000 |
| HR_COOC-HR | no pair history | +0.0006 [+0.0003, +0.0009] p=0.000 | -0.0007 [-0.0009, -0.0005] p=1.000 |
| HR_COOC-HR | has story context | +0.0057 [+0.0049, +0.0064] p=0.000 | +0.0042 [+0.0035, +0.0049] p=0.000 |
| HR_COOC-HR | no story context | +0.0006 [-0.0005, +0.0017] p=0.147 | +0.0002 [-0.0007, +0.0011] p=0.373 |
| HR_CTX_COOC-HR_COOC | overall | +0.0007 [+0.0002, +0.0012] p=0.003 | +0.0004 [-0.0001, +0.0008] p=0.057 |
| HR_CTX_COOC-HR_COOC | exact recurrence, gold in top-50 | +0.0004 [-0.0003, +0.0010] p=0.139 | -0.0000 [-0.0006, +0.0006] p=0.558 |
| HR_CTX_COOC-HR_COOC | pair history only | +0.0029 [+0.0018, +0.0040] p=0.000 | +0.0021 [+0.0011, +0.0032] p=0.000 |
| HR_CTX_COOC-HR_COOC | no pair history | -0.0005 [-0.0010, -0.0000] p=0.976 | +0.0005 [+0.0002, +0.0009] p=0.004 |
| HR_CTX_COOC-HR_COOC | has story context | +0.0007 [+0.0002, +0.0013] p=0.005 | +0.0006 [+0.0001, +0.0011] p=0.013 |
| HR_CTX_COOC-HR_COOC | no story context | +0.0006 [-0.0006, +0.0018] p=0.153 | -0.0011 [-0.0022, -0.0002] p=0.986 |
| HR_CTX-HR_COOC | overall | -0.0008 [-0.0014, -0.0002] p=0.992 | -0.0006 [-0.0012, -0.0000] p=0.980 |
| HR_CTX-HR_COOC | exact recurrence, gold in top-50 | -0.0012 [-0.0020, -0.0004] p=1.000 | -0.0010 [-0.0018, -0.0002] p=0.992 |
| HR_CTX-HR_COOC | pair history only | +0.0008 [-0.0004, +0.0021] p=0.091 | +0.0004 [-0.0008, +0.0016] p=0.246 |
| HR_CTX-HR_COOC | no pair history | -0.0002 [-0.0007, +0.0004] p=0.732 | +0.0004 [-0.0000, +0.0008] p=0.033 |
| HR_CTX-HR_COOC | has story context | -0.0010 [-0.0017, -0.0003] p=0.998 | -0.0006 [-0.0013, +0.0001] p=0.961 |
| HR_CTX-HR_COOC | no story context | +0.0008 [-0.0004, +0.0019] p=0.089 | -0.0006 [-0.0016, +0.0003] p=0.906 |
| HR-base | overall | +0.0335 [+0.0322, +0.0348] p=0.000 | +0.0266 [+0.0255, +0.0278] p=0.000 |
| HR-base | exact recurrence, gold in top-50 | +0.0416 [+0.0400, +0.0432] p=0.000 | +0.0299 [+0.0284, +0.0313] p=0.000 |
| HR-base | pair history only | +0.0227 [+0.0198, +0.0255] p=0.000 | +0.0321 [+0.0296, +0.0346] p=0.000 |
| HR-base | no pair history | -0.0020 [-0.0036, -0.0006] p=0.998 | -0.0050 [-0.0065, -0.0036] p=1.000 |
| HR-base | has story context | +0.0302 [+0.0288, +0.0315] p=0.000 | +0.0253 [+0.0241, +0.0265] p=0.000 |
| HR-base | no story context | +0.0555 [+0.0515, +0.0594] p=0.000 | +0.0358 [+0.0328, +0.0389] p=0.000 |

### MidEast-TE: per-stratum MRR

| stratum | share | RE-GCN base | RE-GCN HR | RE-GCN HR_CTX | RE-GCN HR_SHUF | RE-GCN HR_COOC | RE-GCN HR_CTX_COOC | LogCL base | LogCL HR | LogCL HR_CTX | LogCL HR_SHUF | LogCL HR_COOC | LogCL HR_CTX_COOC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact recurrence, gold in top-50 | 0.716 | 0.5687 | 0.6103 | 0.6149 | 0.6100 | 0.6161 | 0.6165 | 0.5797 | 0.6096 | 0.6131 | 0.6093 | 0.6141 | 0.6141 |
| exact recurrence, gold outside top-50 | 0.044 | 0.0094 | 0.0094 | 0.0094 | 0.0094 | 0.0094 | 0.0094 | 0.0096 | 0.0096 | 0.0096 | 0.0096 | 0.0096 | 0.0096 |
| pair history only | 0.171 | 0.2252 | 0.2479 | 0.2536 | 0.2477 | 0.2528 | 0.2557 | 0.2156 | 0.2477 | 0.2506 | 0.2433 | 0.2502 | 0.2523 |
| no pair history | 0.069 | 0.0311 | 0.0290 | 0.0295 | 0.0280 | 0.0296 | 0.0292 | 0.0352 | 0.0302 | 0.0299 | 0.0288 | 0.0295 | 0.0300 |
| has story context | 0.869 | 0.4599 | 0.4901 | 0.4948 | 0.4895 | 0.4958 | 0.4965 | 0.4668 | 0.4921 | 0.4956 | 0.4910 | 0.4963 | 0.4969 |
| no story context | 0.131 | 0.3703 | 0.4259 | 0.4272 | 0.4272 | 0.4264 | 0.4270 | 0.3911 | 0.4269 | 0.4265 | 0.4260 | 0.4271 | 0.4259 |
| complex-event facts | 0.484 | 0.4713 | 0.5099 | 0.5178 | 0.5100 | 0.5192 | 0.5207 | 0.4824 | 0.5130 | 0.5184 | 0.5116 | 0.5200 | 0.5211 |
| outlier facts | 0.516 | 0.4265 | 0.4551 | 0.4560 | 0.4545 | 0.4561 | 0.4562 | 0.4330 | 0.4559 | 0.4567 | 0.4551 | 0.4564 | 0.4561 |
| gold named in context | 0.409 | 0.6699 | 0.7111 | 0.7352 | 0.7110 | 0.7275 | 0.7363 | 0.6786 | 0.7148 | 0.7368 | 0.7131 | 0.7274 | 0.7352 |

### MidEast-TE: LLM reader sub-study (test sample n = 15,000; arms trained on the 10k valid sample)

| arm | RE-GCN MRR | LogCL MRR |
|---|---|---|
| frozen base | 0.4462 | 0.4555 |
| HR (LLM sample) | 0.4722 | 0.4794 |
| HR + CTX (LLM sample) | 0.4756 | 0.4790 |
| HR + LLM reading story text | 0.4769 | 0.4804 |
| HR + LLM reading shuffled text | 0.4750 | 0.4786 |
| HR + LLM, query only (no text) | 0.4764 | 0.4795 |

| comparison | stratum | RE-GCN | LogCL |
|---|---|---|---|
| HR_LLMCTX-HR_sub | overall | +0.0047 [+0.0034, +0.0060] p=0.000 | +0.0010 [-0.0000, +0.0021] p=0.029 |
| HR_LLMCTX-HR_sub | exact recurrence, gold in top-50 | +0.0046 [+0.0028, +0.0062] p=0.000 | +0.0008 [-0.0005, +0.0022] p=0.103 |
| HR_LLMCTX-HR_sub | pair history only | +0.0091 [+0.0061, +0.0120] p=0.000 | +0.0020 [-0.0006, +0.0046] p=0.068 |
| HR_LLMCTX-HR_sub | no pair history | -0.0018 [-0.0039, +0.0001] p=0.969 | +0.0010 [-0.0006, +0.0029] p=0.138 |
| HR_LLMCTX-HR_LLMSHUF | overall | +0.0019 [+0.0008, +0.0029] p=0.000 | +0.0018 [+0.0009, +0.0027] p=0.000 |
| HR_LLMCTX-HR_LLMSHUF | exact recurrence, gold in top-50 | +0.0016 [+0.0002, +0.0031] p=0.012 | +0.0016 [+0.0004, +0.0028] p=0.006 |
| HR_LLMCTX-HR_LLMSHUF | pair history only | +0.0043 [+0.0024, +0.0064] p=0.000 | +0.0039 [+0.0022, +0.0056] p=0.000 |
| HR_LLMCTX-HR_LLMSHUF | no pair history | -0.0002 [-0.0018, +0.0010] p=0.630 | +0.0004 [-0.0002, +0.0010] p=0.104 |
| HR_LLMCTX-HR_LLMNOTEXT | overall | +0.0005 [-0.0005, +0.0014] p=0.165 | +0.0009 [+0.0000, +0.0018] p=0.022 |
| HR_LLMCTX-HR_LLMNOTEXT | exact recurrence, gold in top-50 | +0.0006 [-0.0008, +0.0019] p=0.198 | +0.0009 [-0.0002, +0.0021] p=0.056 |
| HR_LLMCTX-HR_LLMNOTEXT | pair history only | +0.0004 [-0.0013, +0.0021] p=0.307 | +0.0009 [-0.0010, +0.0028] p=0.196 |
| HR_LLMCTX-HR_LLMNOTEXT | no pair history | -0.0001 [-0.0015, +0.0008] p=0.577 | +0.0011 [+0.0003, +0.0022] p=0.003 |
| HR_LLMNOTEXT-HR_sub | overall | +0.0042 [+0.0032, +0.0053] p=0.000 | +0.0001 [-0.0007, +0.0009] p=0.389 |
| HR_LLMNOTEXT-HR_sub | exact recurrence, gold in top-50 | +0.0040 [+0.0026, +0.0054] p=0.000 | -0.0001 [-0.0010, +0.0009] p=0.565 |
| HR_LLMNOTEXT-HR_sub | pair history only | +0.0086 [+0.0061, +0.0113] p=0.000 | +0.0011 [-0.0009, +0.0030] p=0.147 |
| HR_LLMNOTEXT-HR_sub | no pair history | -0.0017 [-0.0033, -0.0002] p=0.989 | -0.0001 [-0.0013, +0.0010] p=0.596 |
| HR_LLMSHUF-HR_sub | overall | +0.0028 [+0.0019, +0.0037] p=0.000 | -0.0008 [-0.0015, -0.0001] p=0.981 |
| HR_LLMSHUF-HR_sub | exact recurrence, gold in top-50 | +0.0029 [+0.0018, +0.0040] p=0.000 | -0.0007 [-0.0016, +0.0002] p=0.938 |
| HR_LLMSHUF-HR_sub | pair history only | +0.0047 [+0.0027, +0.0067] p=0.000 | -0.0019 [-0.0038, +0.0001] p=0.969 |
| HR_LLMSHUF-HR_sub | no pair history | -0.0016 [-0.0029, -0.0004] p=0.996 | +0.0006 [-0.0006, +0.0021] p=0.192 |
| HR_CTX_sub-HR_sub | overall | +0.0034 [+0.0020, +0.0048] p=0.000 | -0.0004 [-0.0017, +0.0009] p=0.732 |
| HR_CTX_sub-HR_sub | exact recurrence, gold in top-50 | +0.0025 [+0.0006, +0.0043] p=0.005 | +0.0001 [-0.0017, +0.0018] p=0.467 |
| HR_CTX_sub-HR_sub | pair history only | +0.0096 [+0.0067, +0.0126] p=0.000 | -0.0032 [-0.0054, -0.0009] p=0.997 |
| HR_CTX_sub-HR_sub | no pair history | -0.0002 [-0.0009, +0.0005] p=0.727 | +0.0013 [+0.0005, +0.0022] p=0.001 |
| HR_LLMCTX-HR_CTX_sub | overall | +0.0013 [-0.0003, +0.0029] p=0.059 | +0.0014 [+0.0001, +0.0029] p=0.018 |
| HR_LLMCTX-HR_CTX_sub | exact recurrence, gold in top-50 | +0.0021 [-0.0001, +0.0043] p=0.029 | +0.0008 [-0.0010, +0.0026] p=0.202 |
| HR_LLMCTX-HR_CTX_sub | pair history only | -0.0006 [-0.0039, +0.0027] p=0.644 | +0.0052 [+0.0022, +0.0083] p=0.000 |
| HR_LLMCTX-HR_CTX_sub | no pair history | -0.0016 [-0.0037, +0.0002] p=0.953 | -0.0004 [-0.0020, +0.0017] p=0.658 |

### GDELT 2025-H1 (post-cutoff): full-test rows (time-aware filtered, both directions, 3-seed mean of per-query RR)

| arm | RE-GCN MRR | sd(seeds) | H@1 | H@10 | LogCL MRR | sd(seeds) | H@1 | H@10 |
|---|---|---|---|---|---|---|---|---|
| frozen base | 0.3883 | 0.0000 | 0.2702 | 0.6347 | 0.3974 | 0.0000 | 0.2787 | 0.6469 |
| HR (history re-ranker) | 0.4236 | 0.0006 | 0.3017 | 0.6756 | 0.4234 | 0.0001 | 0.3006 | 0.6765 |
| HR + CTX (bge-m3 story reader) | 0.4252 | 0.0007 | 0.3029 | 0.6775 | 0.4261 | 0.0002 | 0.3034 | 0.6783 |
| HR + SHUF (shuffled stale text) | 0.4231 | 0.0008 | 0.3009 | 0.6752 | 0.4230 | 0.0003 | 0.3002 | 0.6763 |
| HR + COOC (same-article co-participation, no text) | 0.4276 | 0.0003 | 0.3062 | 0.6784 | 0.4283 | 0.0001 | 0.3062 | 0.6799 |
| HR + CTX + COOC | 0.4275 | 0.0003 | 0.3059 | 0.6778 | 0.4278 | 0.0004 | 0.3058 | 0.6791 |

n = 85,956 test queries.

### GDELT 2025-H1 (post-cutoff): paired deltas (Δ MRR [95% CI], one-sided p; Poisson bootstrap B = 2000)

| comparison | stratum | RE-GCN | LogCL |
|---|---|---|---|
| HR_CTX-HR | overall | +0.0016 [+0.0010, +0.0021] p=0.000 | +0.0027 [+0.0021, +0.0033] p=0.000 |
| HR_CTX-HR | exact recurrence, gold in top-50 | +0.0011 [+0.0004, +0.0019] p=0.002 | +0.0026 [+0.0019, +0.0033] p=0.000 |
| HR_CTX-HR | pair history only | +0.0040 [+0.0029, +0.0051] p=0.000 | +0.0042 [+0.0031, +0.0053] p=0.000 |
| HR_CTX-HR | no pair history | +0.0016 [+0.0008, +0.0025] p=0.001 | +0.0004 [-0.0003, +0.0012] p=0.152 |
| HR_CTX-HR | has story context | +0.0016 [+0.0011, +0.0021] p=0.000 | +0.0027 [+0.0021, +0.0033] p=0.000 |
| HR_CTX-HR | no story context | -0.0039 [-0.0098, +0.0019] p=0.909 | -0.0016 [-0.0084, +0.0049] p=0.687 |
| HR_SHUF-HR | overall | -0.0006 [-0.0010, -0.0002] p=0.997 | -0.0004 [-0.0008, +0.0000] p=0.959 |
| HR_SHUF-HR | exact recurrence, gold in top-50 | -0.0011 [-0.0017, -0.0006] p=1.000 | -0.0007 [-0.0012, -0.0001] p=0.997 |
| HR_SHUF-HR | pair history only | +0.0013 [+0.0005, +0.0020] p=0.000 | +0.0007 [-0.0000, +0.0015] p=0.030 |
| HR_SHUF-HR | no pair history | +0.0016 [+0.0008, +0.0024] p=0.000 | -0.0005 [-0.0011, +0.0003] p=0.906 |
| HR_SHUF-HR | has story context | -0.0005 [-0.0009, -0.0001] p=0.995 | -0.0004 [-0.0008, +0.0001] p=0.944 |
| HR_SHUF-HR | no story context | -0.0056 [-0.0130, +0.0014] p=0.935 | -0.0038 [-0.0103, +0.0025] p=0.890 |
| HR_CTX-HR_SHUF | overall | +0.0021 [+0.0016, +0.0027] p=0.000 | +0.0031 [+0.0025, +0.0036] p=0.000 |
| HR_CTX-HR_SHUF | exact recurrence, gold in top-50 | +0.0023 [+0.0016, +0.0030] p=0.000 | +0.0033 [+0.0025, +0.0040] p=0.000 |
| HR_CTX-HR_SHUF | pair history only | +0.0027 [+0.0016, +0.0037] p=0.000 | +0.0035 [+0.0025, +0.0046] p=0.000 |
| HR_CTX-HR_SHUF | no pair history | +0.0000 [-0.0004, +0.0005] p=0.447 | +0.0009 [+0.0002, +0.0016] p=0.010 |
| HR_CTX-HR_SHUF | has story context | +0.0021 [+0.0016, +0.0027] p=0.000 | +0.0031 [+0.0026, +0.0037] p=0.000 |
| HR_CTX-HR_SHUF | no story context | +0.0017 [-0.0019, +0.0053] p=0.203 | +0.0021 [-0.0015, +0.0058] p=0.119 |
| HR_COOC-HR | overall | +0.0040 [+0.0033, +0.0045] p=0.000 | +0.0049 [+0.0043, +0.0055] p=0.000 |
| HR_COOC-HR | exact recurrence, gold in top-50 | +0.0052 [+0.0043, +0.0059] p=0.000 | +0.0059 [+0.0051, +0.0067] p=0.000 |
| HR_COOC-HR | pair history only | +0.0007 [-0.0003, +0.0017] p=0.090 | +0.0028 [+0.0017, +0.0039] p=0.000 |
| HR_COOC-HR | no pair history | +0.0016 [+0.0008, +0.0025] p=0.000 | -0.0008 [-0.0014, -0.0003] p=1.000 |
| HR_COOC-HR | has story context | +0.0040 [+0.0034, +0.0047] p=0.000 | +0.0050 [+0.0044, +0.0056] p=0.000 |
| HR_COOC-HR | no story context | -0.0050 [-0.0094, -0.0010] p=0.994 | -0.0032 [-0.0075, +0.0008] p=0.938 |
| HR_CTX_COOC-HR_COOC | overall | -0.0001 [-0.0006, +0.0004] p=0.679 | -0.0005 [-0.0010, -0.0000] p=0.982 |
| HR_CTX_COOC-HR_COOC | exact recurrence, gold in top-50 | -0.0012 [-0.0018, -0.0006] p=1.000 | -0.0016 [-0.0022, -0.0010] p=1.000 |
| HR_CTX_COOC-HR_COOC | pair history only | +0.0044 [+0.0035, +0.0054] p=0.000 | +0.0032 [+0.0023, +0.0041] p=0.000 |
| HR_CTX_COOC-HR_COOC | no pair history | +0.0002 [-0.0007, +0.0011] p=0.331 | +0.0033 [+0.0020, +0.0047] p=0.000 |
| HR_CTX_COOC-HR_COOC | has story context | -0.0001 [-0.0006, +0.0004] p=0.669 | -0.0005 [-0.0010, -0.0001] p=0.987 |
| HR_CTX_COOC-HR_COOC | no story context | +0.0018 [-0.0036, +0.0070] p=0.256 | +0.0006 [-0.0054, +0.0062] p=0.397 |
| HR_CTX-HR_COOC | overall | -0.0024 [-0.0030, -0.0017] p=1.000 | -0.0022 [-0.0028, -0.0016] p=1.000 |
| HR_CTX-HR_COOC | exact recurrence, gold in top-50 | -0.0040 [-0.0048, -0.0032] p=1.000 | -0.0033 [-0.0041, -0.0025] p=1.000 |
| HR_CTX-HR_COOC | pair history only | +0.0033 [+0.0022, +0.0044] p=0.000 | +0.0015 [+0.0003, +0.0026] p=0.008 |
| HR_CTX-HR_COOC | no pair history | +0.0000 [-0.0009, +0.0009] p=0.512 | +0.0012 [+0.0004, +0.0021] p=0.001 |
| HR_CTX-HR_COOC | has story context | -0.0024 [-0.0031, -0.0018] p=1.000 | -0.0023 [-0.0029, -0.0017] p=1.000 |
| HR_CTX-HR_COOC | no story context | +0.0011 [-0.0037, +0.0059] p=0.322 | +0.0016 [-0.0044, +0.0073] p=0.296 |
| HR-base | overall | +0.0354 [+0.0340, +0.0368] p=0.000 | +0.0260 [+0.0247, +0.0271] p=0.000 |
| HR-base | exact recurrence, gold in top-50 | +0.0491 [+0.0473, +0.0508] p=0.000 | +0.0324 [+0.0309, +0.0338] p=0.000 |
| HR-base | pair history only | -0.0039 [-0.0056, -0.0022] p=1.000 | +0.0105 [+0.0088, +0.0122] p=0.000 |
| HR-base | no pair history | -0.0068 [-0.0087, -0.0050] p=1.000 | -0.0090 [-0.0121, -0.0061] p=1.000 |
| HR-base | has story context | +0.0352 [+0.0338, +0.0365] p=0.000 | +0.0258 [+0.0246, +0.0270] p=0.000 |
| HR-base | no story context | +0.0559 [+0.0427, +0.0697] p=0.000 | +0.0438 [+0.0297, +0.0578] p=0.000 |

### GDELT 2025-H1 (post-cutoff): per-stratum MRR

| stratum | share | RE-GCN base | RE-GCN HR | RE-GCN HR_CTX | RE-GCN HR_SHUF | RE-GCN HR_COOC | RE-GCN HR_CTX_COOC | LogCL base | LogCL HR | LogCL HR_CTX | LogCL HR_SHUF | LogCL HR_COOC | LogCL HR_CTX_COOC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact recurrence, gold in top-50 | 0.738 | 0.4764 | 0.5255 | 0.5267 | 0.5244 | 0.5307 | 0.5295 | 0.4845 | 0.5169 | 0.5195 | 0.5162 | 0.5228 | 0.5213 |
| exact recurrence, gold outside top-50 | 0.064 | 0.0115 | 0.0115 | 0.0115 | 0.0115 | 0.0115 | 0.0115 | 0.0120 | 0.0120 | 0.0120 | 0.0120 | 0.0120 | 0.0120 |
| pair history only | 0.177 | 0.1994 | 0.1955 | 0.1995 | 0.1968 | 0.1962 | 0.2006 | 0.1828 | 0.1933 | 0.1975 | 0.1940 | 0.1960 | 0.1992 |
| no pair history | 0.022 | 0.0417 | 0.0349 | 0.0365 | 0.0365 | 0.0365 | 0.0367 | 0.0447 | 0.0357 | 0.0361 | 0.0352 | 0.0349 | 0.0381 |
| has story context | 0.989 | 0.3881 | 0.4232 | 0.4248 | 0.4227 | 0.4273 | 0.4271 | 0.3972 | 0.4230 | 0.4257 | 0.4226 | 0.4280 | 0.4274 |
| no story context | 0.011 | 0.4078 | 0.4637 | 0.4598 | 0.4580 | 0.4586 | 0.4604 | 0.4159 | 0.4597 | 0.4581 | 0.4559 | 0.4565 | 0.4571 |
| gold named in context | 0.262 | 0.7053 | 0.7402 | 0.7574 | 0.7393 | 0.7557 | 0.7668 | 0.7171 | 0.7396 | 0.7639 | 0.7395 | 0.7583 | 0.7643 |

### GDELT 2025-H1 (post-cutoff): LLM reader sub-study (test sample n = 15,000; arms trained on the 10k valid sample)

| arm | RE-GCN MRR | LogCL MRR |
|---|---|---|
| frozen base | 0.3888 | 0.3969 |
| HR (LLM sample) | 0.4164 | 0.4173 |
| HR + CTX (LLM sample) | 0.4169 | 0.4174 |
| HR + LLM reading story text | 0.4172 | 0.4172 |
| HR + LLM reading shuffled text | 0.4167 | 0.4170 |
| HR + LLM, query only (no text) | 0.4163 | 0.4164 |

| comparison | stratum | RE-GCN | LogCL |
|---|---|---|---|
| HR_LLMCTX-HR_sub | overall | +0.0007 [-0.0003, +0.0017] p=0.079 | -0.0001 [-0.0012, +0.0010] p=0.594 |
| HR_LLMCTX-HR_sub | exact recurrence, gold in top-50 | +0.0010 [-0.0002, +0.0023] p=0.061 | +0.0009 [-0.0005, +0.0022] p=0.107 |
| HR_LLMCTX-HR_sub | pair history only | +0.0001 [-0.0020, +0.0024] p=0.469 | -0.0042 [-0.0068, -0.0018] p=1.000 |
| HR_LLMCTX-HR_sub | no pair history | -0.0007 [-0.0019, +0.0003] p=0.897 | -0.0018 [-0.0051, +0.0001] p=0.964 |
| HR_LLMCTX-HR_LLMSHUF | overall | +0.0005 [-0.0004, +0.0014] p=0.155 | +0.0001 [-0.0008, +0.0012] p=0.398 |
| HR_LLMCTX-HR_LLMSHUF | exact recurrence, gold in top-50 | +0.0007 [-0.0005, +0.0019] p=0.150 | +0.0006 [-0.0006, +0.0019] p=0.174 |
| HR_LLMCTX-HR_LLMSHUF | pair history only | +0.0000 [-0.0015, +0.0017] p=0.451 | -0.0018 [-0.0037, +0.0002] p=0.962 |
| HR_LLMCTX-HR_LLMSHUF | no pair history | -0.0000 [-0.0003, +0.0003] p=0.566 | +0.0008 [+0.0004, +0.0012] p=0.000 |
| HR_LLMCTX-HR_LLMNOTEXT | overall | +0.0009 [+0.0000, +0.0019] p=0.025 | +0.0007 [-0.0002, +0.0017] p=0.076 |
| HR_LLMCTX-HR_LLMNOTEXT | exact recurrence, gold in top-50 | +0.0008 [-0.0004, +0.0020] p=0.083 | +0.0011 [-0.0001, +0.0023] p=0.036 |
| HR_LLMCTX-HR_LLMNOTEXT | pair history only | +0.0017 [-0.0002, +0.0035] p=0.041 | -0.0005 [-0.0025, +0.0017] p=0.671 |
| HR_LLMCTX-HR_LLMNOTEXT | no pair history | +0.0003 [-0.0000, +0.0008] p=0.038 | +0.0005 [+0.0001, +0.0009] p=0.003 |
| HR_LLMNOTEXT-HR_sub | overall | -0.0002 [-0.0009, +0.0006] p=0.672 | -0.0009 [-0.0017, -0.0001] p=0.989 |
| HR_LLMNOTEXT-HR_sub | exact recurrence, gold in top-50 | +0.0002 [-0.0008, +0.0011] p=0.385 | -0.0002 [-0.0012, +0.0008] p=0.684 |
| HR_LLMNOTEXT-HR_sub | pair history only | -0.0016 [-0.0031, +0.0000] p=0.975 | -0.0038 [-0.0056, -0.0020] p=1.000 |
| HR_LLMNOTEXT-HR_sub | no pair history | -0.0010 [-0.0022, -0.0001] p=0.993 | -0.0023 [-0.0058, -0.0004] p=0.999 |
| HR_LLMSHUF-HR_sub | overall | +0.0002 [-0.0005, +0.0009] p=0.257 | -0.0003 [-0.0011, +0.0005] p=0.778 |
| HR_LLMSHUF-HR_sub | exact recurrence, gold in top-50 | +0.0003 [-0.0006, +0.0012] p=0.246 | +0.0003 [-0.0007, +0.0012] p=0.296 |
| HR_LLMSHUF-HR_sub | pair history only | +0.0001 [-0.0015, +0.0017] p=0.485 | -0.0024 [-0.0043, -0.0006] p=0.995 |
| HR_LLMSHUF-HR_sub | no pair history | -0.0007 [-0.0018, +0.0001] p=0.934 | -0.0026 [-0.0061, -0.0006] p=1.000 |
| HR_CTX_sub-HR_sub | overall | +0.0004 [-0.0011, +0.0020] p=0.309 | +0.0001 [-0.0014, +0.0017] p=0.454 |
| HR_CTX_sub-HR_sub | exact recurrence, gold in top-50 | +0.0005 [-0.0015, +0.0026] p=0.305 | +0.0006 [-0.0014, +0.0027] p=0.277 |
| HR_CTX_sub-HR_sub | pair history only | +0.0000 [-0.0029, +0.0029] p=0.485 | -0.0022 [-0.0050, +0.0006] p=0.940 |
| HR_CTX_sub-HR_sub | no pair history | +0.0011 [-0.0012, +0.0030] p=0.141 | +0.0005 [-0.0017, +0.0021] p=0.268 |
| HR_LLMCTX-HR_CTX_sub | overall | +0.0003 [-0.0012, +0.0019] p=0.340 | -0.0003 [-0.0017, +0.0013] p=0.615 |
| HR_LLMCTX-HR_CTX_sub | exact recurrence, gold in top-50 | +0.0005 [-0.0016, +0.0024] p=0.329 | +0.0002 [-0.0017, +0.0022] p=0.396 |
| HR_LLMCTX-HR_CTX_sub | pair history only | +0.0001 [-0.0031, +0.0032] p=0.481 | -0.0021 [-0.0051, +0.0009] p=0.911 |
| HR_LLMCTX-HR_CTX_sub | no pair history | -0.0018 [-0.0033, -0.0003] p=0.993 | -0.0023 [-0.0040, -0.0009] p=1.000 |
