# Data card: GDELTFW (v79 "fresh window" temporal knowledge graph)

GDELTFW is the temporal knowledge graph that the v79 fresh-window comparison (HRcov vs SA vs the CountTRuCoLa comparator, TiRGN
base) was registered and evaluated on. It is the GDELT25X recipe applied to a window shifted forward by 7 months: the date
arguments change and nothing else does. The code release ships the builder and the SHA-256 of every output file. It does not
ship the data files.

## 1. Source

| item | value |
|---|---|
| raw source | GDELT 2.0 Event Database, 15-minute event export files `http://data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip` (official server, fetched one file per 15-minute stamp, no crawling) |
| relation names | official `CAMEO.eventcodes.txt` (https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt), overridden for some codes by `01_window_build/gdelt25x_relation_names.tsv` (names column only; ids never depend on names) |
| export stamps requested | 2025-08-01 00:00:00 .. 2026-04-30 23:45:00 UTC, i.e. 273 days x 96 = 26,208 files |
| export files retrieved | 26,208 of 26,208 (`missing_export_files.txt` is empty; no skip list was used) |
| builder | `01_window_build/build_gdelt25x.py` (sha256 `3bf22cc72e28ad7bda413d69c69e55761cf469d463f899dd71ee7bf1e9865fbb`), the release builder of GDELT25X, unmodified |
| build command | `python3 -u builder/build_gdelt25x.py --out $FW/data/GDELTFW --raw $FW/raw/gdelt --start 20250801 --end 20260501 --valid 20260201 --test 20260301 --workers 16` (docker image `tkg-gpu:latest`, 16 CPUs; `01_window_build/run_build_gdeltfw.sh`) |
| build run | serverA (serverA), job faac377d-9748-42af-9403-7352ff11995d, 2026-10-16 14:31:48Z to 14:45:39Z (829.7 s builder wall time) |
| output format | RE-GCN / harness format: `train.txt valid.txt test.txt` (tab-separated `s r o t`, integer ids), `stat.txt`, `entity2id.txt`, `relation2id.txt`, `timestamp2id.txt`, `build_summary.json`, `missing_export_files.txt` |
| file checksums | `01_window_build/GDELTFW.sha256`. To verify a rebuild, compare the rebuilt files with this list using `sha256sum -c`. The builder's `--check` option compares against `GDELT25X.sha256`, i.e. the earlier window, so it does not apply here. |

## 2. Date ranges and splits

The time stamp of an event is its **DATEADDED** day: the UTC day the event record was added to GDELT. The SQLDATE event-date
field is not used. Day indices run 0..272 over the days that contain at least one event. Every one of the 273 calendar days
contains events, so the index has no gaps.

| split | calendar days (DATEADDED) | ts ids | days | facts (unique quadruples) | facts per day | queries (both directions) |
|---|---|---|---|---|---|---|
| train | 2025-08-01 .. 2026-01-31 | 0 .. 183 | 184 | 509,717 | 2,770.2 | 1,019,434 |
| valid | 2026-02-01 .. 2026-02-28 | 184 .. 211 | 28 | 76,117 | 2,718.5 | 152,234 |
| test | 2026-03-01 .. 2026-04-30 | 212 .. 272 | 61 | 228,584 | 3,747.3 | 457,168 |
| total | 2025-08-01 .. 2026-04-30 | 0 .. 272 | 273 | 814,418 | | 1,628,836 |

Every fact (s, r, o, t) gives two queries: (s, r, ?, t), and (o, r + R, ?, t) with the inverse relation r + R (R = 215). The
v79 protocol divides validation into a **fit part** (ts 184..202) and a **selection part** (ts 203..211, i.e. from 2026-02-20;
50,192 queries). The cut is `uts[int(0.7 * len(uts))]` over the validation time stamps; the code computes and asserts it. All
Stage-1 fitting and selection used these two parts only. Test was scored once, in Stage 2.

**Uninspected property (v79 requirement).** Before any model was fitted, a read-only scan of the three project hosts and a
search of the project artifact store found that no earlier project experiment had used any fact dated on or after 2025-10-01.
The highest date seen anywhere was 2025-09-30 (`registration/fresh_window_uninspected_check.json`, PASS). Validation and test
(2026-02 .. 2026-04) and train months 2025-10 .. 2026-01 therefore lie in uninspected time. The first two train months
(2025-08 .. 2025-09) overlap the GDELT25X test window, which earlier work had already evaluated on. GDELTFW uses that overlap
for training only.

## 3. Filtering recipe (applied in this order by `build_gdelt25x.py`)

1. Read each export file and keep 13 columns: SQLDATE, Actor1Code/Name/CountryCode/Type1Code, Actor2Code/Name/CountryCode/Type1Code, IsRootEvent, EventCode, DATEADDED, SOURCEURL.
2. Keep **root events** (`IsRootEvent = 1`) where **both actors are coded** (Actor1Code and Actor2Code non-empty).
3. Keep events where **at least one actor's country code is in the Middle-East set ME** = {ISR, PSE, IRN, EGY, SYR, LBN, JOR, SAU, ARE, QAT, TUR, IRQ, YEM, KWT, BHR, OMN, LBY}.
4. Keep events with DATEADDED day in [2025-08-01, 2026-05-01). This leaves **1,552,574 event rows**.
5. **URL first-seen rule**: among the rows kept by steps 2-4, drop an event if its SOURCEURL first appeared on an earlier day. These rows are re-codings of older articles.
6. **Entity** = actor CountryCode + Type1Code (e.g. ISR + GOV). An actor without a country code is represented by its ActorCode. **Relation** = CAMEO EventCode. **Self-loops** (subject entity = object entity) are dropped.
7. Assign splits by day: train < 2026-02-01 <= valid < 2026-03-01 <= test.
8. **Minimum counts on train**: keep entities with >= 10 train occurrences (as subject or object) and relations with >= 5 train events. The kept sets are applied to all three splits, so validation and test contain no unseen entity or relation. After steps 5-8, **1,539,261 event rows** remain.
9. Map entities and relations to ids (sorted keys) and **de-duplicate** identical (s, r, o, t). This gives **814,418 unique quadruples**: 1,209 entities and 215 relations.

Entity labels in `entity2id.txt` come from a fixed country / actor-type name table. If the table has no entry, the label is the
most frequent GDELT actor name for that key. A label that occurs twice is suffixed with its key.

| quantity | value |
|---|---|
| entities | 1,209 |
| relations (CAMEO event codes) | 215 (430 including inverses) |
| time stamps | 273 days, daily granularity, no gaps |
| facts | 814,418 (train 509,717 / valid 76,117 / test 228,584) |
| evaluation queries | valid 152,234, test 457,168 |
| evaluation used in v79 | time-aware filtered rank over all 1,209 entities, ties averaged (py-tgb Evaluator formula); MRR, Hits@1/3/10 |

## 4. Licence and terms of use of the raw data

The raw data are the GDELT Project's event exports. The GDELT Terms of Use (https://www.gdeltproject.org/about.html#termsofuse)
make all GDELT datasets available for unlimited and unrestricted academic, commercial or governmental use without fee. They
allow anyone to redistribute, rehost, republish and mirror the datasets in any form. **Any use or redistribution must cite the
GDELT Project and link to https://www.gdeltproject.org/.** GDELTFW is derived from these exports, so the same attribution
requirement applies to any redistribution of the GDELTFW files. Suggested citation:

> Leetaru, K. and Schrodt, P. A. (2013). GDELT: Global Data on Events, Location and Tone, 1979-2012. International Studies Association (ISA) Annual Convention, San Francisco. https://www.gdeltproject.org/

Event categories follow the CAMEO coding scheme (Schrodt, P. A., CAMEO Conflict and Mediation Event Observations Event and
Actor Codebook). This package contains no GDELT data. It contains only the builder, the relation-name table, and checksums.

## 5. Known limitations

- **Machine-coded events.** GDELT codes events automatically from news text. Some actor and event codes are wrong. The rate of coding errors in this window was not measured.
- **Duplicate reporting.** The URL first-seen rule and quadruple de-duplication reduce duplicates but do not remove them. The same real-world event reported by several outlets on one day can still yield several distinct quadruples.
- **Selection.** The ME actor-country filter, the root-event filter and the minimum-count filter give a region-centred and frequency-biased graph. Results on GDELTFW need not transfer to other regions or to the long tail.
- **Coarse entities.** An entity is a country plus actor-type key, so distinct real-world actors with the same key are merged.
- **Timing.** DATEADDED is the day an event entered GDELT, not the day it happened.
- **Density shift.** Test has more facts per day (3,747) than train (2,770) or validation (2,719).
- **Reproducibility.** The window was downloaded on 2026-10-16. A later rebuild reproduces these files only if the official export files are unchanged. Check a rebuild against `01_window_build/GDELTFW.sha256`.
- **Personal data.** The graph has no individual-level identifier fields. Entity labels are country/role names or GDELT actor-name strings.
