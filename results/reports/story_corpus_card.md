# Story corpus card — pilot 4 (story-context channel, Task 3)

All corpus numbers below were produced on `serverB` (docker `tkg-gpu:latest`) by `ps_build.py`, `ps_embed.py`, `ps_ctx.py` and `ps_build_gdelt.py`; this card was rendered locally from their JSON summaries.

## 1. Primary benchmark: MidEast-TE (SCTc-TE)

| field | value |
|---|---|
| source | Chen et al., *SCTc-TE: A Comprehensive Formulation and Benchmark for Temporal Event Forecasting* (arXiv:2312.01052); `github.com/yecchen/GDELT-ComplexEvent` (`data/MIDEAST_CE.zip`, `dataset_construction/data_cleaned/doc_{2015..2022}.json`) |
| period | 20150219 – 20220317 (2,584 daily snapshots) |
| events | 455,877 Vicuna-13B-extracted atomic events (complex-event events + outliers), 440,177 unique (s, r, o, t) after de-duplication within split; 2,794 entities, 234 CAMEO relations |
| split (authors' TKG-baseline boundaries, `prepare_data_tkg_EGIRIS.ipynb`) | train ≤ 20200317 (347,449 facts), valid ≤ 20210317 (47,780), test to 20220317 (44,948 facts = 89,896 queries, both directions). Event-row counts 360,190 / 49,125 / 46,562 equal the notebook's printed counts. |
| licence | no LICENSE file in the repository root (the vendored BERTopic sub-folder carries its own MIT licence); the dataset terms are unstated. Article texts are publishers' copyrighted news content redistributed by the authors for research. Used here for research only; no article text is redistributed in any file (prompts with ≤ 40-word snippets exist only on the remote host). |
| raw files (sha256, host serverB) | `MIDEAST_CE.zip` f04379a116be…; `doc_2015.json` b77f4522aff5…; `doc_2016.json` 75cfb190575d…; `doc_2017.json` fba766d08a0a…; `doc_2018.json` 4ba613568f29…; `doc_2019.json` 3bd23742fbe3…; `doc_2020.json` 725754a3f0c0…; `doc_2021.json` 8a0e3418df72…; `doc_2022.json` dcd559b57a55… |

### Text actually available

* The events reference **274,795 source articles** (`Md5_list`, mean 1.09 articles per event row). Every article is attached to events of exactly one day (checked), so an article's date is well defined.
* **Full text** (title + paragraphs) is shipped only for the Egypt sub-corpus: 95,373 JSON documents, of which 39,228 are referenced by events (**14.3%** of articles; 56,145 shipped texts belong to filtered-out events and carry no day). The README states that the full Iran/Israel article dump (`all_vicuna_docs.json`) is on S3 but gives no link; it was not obtainable, and article URLs were **not** crawled.
* For the rest, the richest shipped text is the **headline recovered from the URL slug** (e.g. `.../Palestinian-teen-shot-dead-in-clashes-with-Israel` → 'Palestinian teen shot dead in clashes with Israel'): **65.4%** of articles have ≥ 4 slug words. **20.3%** have neither (numeric/opaque URLs) and are never shown to a reader.
* Reader text per article: full-text articles → title + lead (≤ 600 characters, boilerplate lines dropped); slug articles → slug headline. Full-text share by split (by article day): train 14.2%, valid 15.0%, test 13.8%.
* Embeddings: BAAI/bge-m3 (sentence-transformers, dense, fp16, max_seq 256); 219,080 articles embedded; lexical entity mentions found in 74.7% of all articles (3.72 entities per text article).

## 2. Time masking

* **Boundary = day t−1 (strictly before t).** A query at day t may read only articles dated ≤ t−1. The structural bases and the history re-ranker see exactly the snapshots ≤ t−1 (ground-truth rolling history), so the text channel gets the same information exposure. MidEast-TE has day granularity and GDELT-derived article days; an intra-day boundary is not defined.
* **Gold-event articles are never visible**: an article is dated by its events' (single) day, so the articles attached to the gold event are dated t and fall outside every window by construction (audited below).
* **Story window** C_q: up to 16 most recent text-bearing articles dated in [t−7, t−1] that are attached to an event involving the query subject (either role). LLM prompts use the first 8 (≤ 40 words each, no per-article dates).
* **Shuffled control** C'_q: the same number of articles drawn uniformly from the subject's text articles dated ≤ t−8 (fill: random text articles of any entity in that range); the shuffled context is therefore stale but still time-safe.

## 3. Leakage audit (`ps_ctx.py`, all valid and test queries)

| quantity | valid | test | required |
|---|---|---|---|
| queries | 95,560 | 89,896 | — |
| windows containing an article dated ≥ t | 0.00% | 0.00% | 0 (assert) |
| shuffled contexts containing an article dated ≥ t | 0.00% | 0.00% | 0 (assert) |
| shuffled contexts containing an article from the window [t−7, t−1] | 0.00% | 0.00% | 0 |
| windows (real or shuffled) containing an article of the gold event | 0.00% | 0.00% | 0 (assert) |
| queries with ≥ 1 context article | 86.76% | 86.86% | report |
| mean context articles (if any) | 12.23 | 12.53 | report |
| context articles with full text | 15.68% | 13.84% | report |
| windows whose articles lexically mention the gold object (if context) | 42.93% | 47.06% | report (legitimate signal) |
| … same, shuffled context | 36.01% | 38.77% | report |
| windows with an article attached to an event of the gold object (if context) | 56.56% | 58.68% | report |
| … same, shuffled context | 48.19% | 50.82% | report |

## 4. Contamination status

* MidEast-TE spans 2015-02-19 – 2022-03-17 (test 2021-03-18 – 2022-03-17): **entirely before** the Llama-3.1 knowledge cutoff (December 2023) and the bge-m3 release (January 2024). LLM results on it can mix reading with event memory; the `LLM_NOTEXT` arm (date + query + options, no articles) measures the memory / prior part.
* **Post-cutoff slice built (secondary, descriptive):** GDELT 2.0 15-minute event exports 2025-01-01 – 2025-06-30 from data.gdeltproject.org (official, no crawling; 15,800 of 17,376 files retrieved after 3 retries each — the remaining 1,576 gave no file, leaving 165 days with events). Root events with ≥ 1 Middle-East actor country; entity = actor country + primary role (e.g. 'Israel government'); 1,024 entities, 213 CAMEO relations, 512,489 unique facts (train 368,449 / valid 101,062 / test 42,978; train Jan–Apr, valid May, test June 2025). Text = slug headlines of the shipped SOURCEURL (80.0% of 248,622 URLs), no full text. Events whose URL first appeared on an earlier day were dropped, so every article is dated on its events' day. The schema matches MidEast-TE in relations (CAMEO) and text type (headlines), but actors are GDELT-coded (country × role) rather than Vicuna-extracted names, so absolute numbers are not comparable across the two datasets.
* GDELT-2025 leakage audit (test, n = 85,956): windows with an article dated ≥ t 0.00%, with a gold-event article 0.00%; context coverage 98.9%; gold named in context 26.5% (shuffled 26.3%); gold co-participates 58.6% (shuffled 56.3%).
* bge-m3 is an encoder; its exposure to 2015–2022 news can shape the embedding space but it cannot generate answers. The shuffled-context arms control for subject-level topical similarity.

## 5. Files (host serverB, `$WORK_DIR/tracks/pilot_story/`)

`data/MIDEAST/{train,valid,test}.txt` (s r o t), `corpus/{docs.jsonl, doc_t.npy, doc_ents.npz, doc_quads.npz, ent_docs.npz, quad_docs.npz, doc_mentions.npz, doc_emb.npy, ent_emb.npy}`, `work/ctx/{valid,test}_ctx.npz` (context article ids), and the same tree under `gdelt25/`.
