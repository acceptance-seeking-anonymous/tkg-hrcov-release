# Construction specification: GDELT25X and GDELTFW

This is a compact specification for the supplement. Both datasets are built by the same code path:
`build_gdelt25x.py` in the code release, which is identical in logic to `ps_build_gdelt.py`, the script that built GDELT25X.
The two datasets differ only in the four date arguments (§7).

## 1. Raw source

- **Exports.** GDELT 2.0 Event Database, 15-minute event export files from the official server:
  `http://data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip`. There is one file per 15-minute stamp from `start`
  to `end` (exclusive); no crawling is done.
- **Download.** Each file is tried three times with a 60 s timeout. Files that cannot be retrieved are listed in
  `missing_export_files.txt`.
- **Relation names.** Taken from the official `CAMEO.eventcodes.txt`; names column only (§4).
- **Columns read.** Each export is a tab-separated file with 61 columns. The builder reads 13 of them by 0-based index:
  1 SQLDATE (read, not used); 5/6/7/12 Actor1 Code/Name/CountryCode/Type1Code; 15/16/17/22 Actor2 Code/Name/CountryCode/Type1Code;
  25 IsRootEvent; 26 EventCode; 59 DATEADDED; 60 SOURCEURL.
- **Licence.** GDELT terms of use: free use and redistribution, with a citation of the GDELT Project and a link to
  https://www.gdeltproject.org/.

## 2. Actor and entity fields

- An event's two participants are Actor1 (subject) and Actor2 (object). Direction is kept.
- **Entity key.** CountryCode + Type1Code when the actor has a CountryCode (e.g. `ISR`+`GOV`; `ISR` alone if Type1Code is empty).
  Otherwise the entity key is the actor's full CAMEO ActorCode (e.g. international or non-state actor codes).
- **Entity label (display only).** Country name from a fixed country table, plus the actor-type name from a fixed table.
  If a code is not in the tables, the label falls back to the most frequent Actor*Name of that key. A label that occurs
  twice is suffixed with its key. Labels never affect ids.

## 3. Selection rules (in order)

1. Root events only (`IsRootEvent = 1`). Both `Actor1Code` and `Actor2Code` must be non-empty.
2. At least one actor's CountryCode is in the Middle-East set ME = {ISR, PSE, IRN, EGY, SYR, LBN, JOR, SAU, ARE, QAT, TUR, IRQ, YEM, KWT, BHR, OMN, LBY}.
3. The DATEADDED day (first 8 characters, UTC) lies in `[start, end)`.
4. **URL first-seen rule.** Among the rows kept by steps 1–3, an event is dropped if its SOURCEURL first appears on an earlier day. This removes re-codings of older articles.
5. Self-loops are dropped (subject key = object key).
6. **Vocabulary from the training split only.** Keep entities with ≥ 10 occurrences (subject or object) among train events, and relations with ≥ 5 train events. Drop every fact, in any split, that uses another entity or relation. Validation and test therefore contain no unseen entity or relation.

## 4. Relation mapping

- Relation = the CAMEO `EventCode` string at its full level (e.g. `010`, `0211`, `1823`). Codes are not rolled up to base or root codes.
- Relation ids are the lexicographic order of the retained codes.
- Inverse relations are not stored. Queries `(o, r + R, ?, t)` are formed at evaluation time, giving two queries per fact.
- Relation names come from `CAMEO.eventcodes.txt`. For GDELTFW, `gdelt25x_relation_names.tsv` reproduces the names column of the GDELT25X snapshot; ids never depend on names.

## 5. Daily aggregation and deduplication

- Time granularity is one day: the DATEADDED UTC day. All events added on the same day share one timestamp.
- Time ids are the ranks of the distinct days that keep at least one event after step 4. A day with no retained event gets no id.
- Deduplication: identical `(s, r, o, t)` are merged into one fact after mapping. This collapses multiple reports of the same coded event on one day, and repeated codings within a day. No weights or counts are kept.
- Split files are sorted by `(t, s, r, o)`.

## 6. Split construction

Splits are calendar intervals on the DATEADDED day:
- train: `start ≤ day < valid`
- validation: `valid ≤ day < test`
- test: `test ≤ day < end`

Validation is further divided in the evaluation protocol, not by the builder:
- fit part: validation time stamps before the cut `uts[int(0.7·|uts|)]`;
- selection part: the rest.

Output files, in the RE-GCN / harness format: `train.txt valid.txt test.txt` (`s r o t`), `stat.txt`, `entity2id.txt`,
`relation2id.txt`, `timestamp2id.txt`, `build_summary.json`, `missing_export_files.txt`.

## 7. Parameters and resulting datasets

| | GDELT25X | GDELTFW |
|---|---|---|
| `start` / `valid` / `test` / `end` | 2025-01-01 / 2025-07-01 / 2025-08-01 / 2025-10-01 | 2025-08-01 / 2026-02-01 / 2026-03-01 / 2026-05-01 |
| export files retrieved | 24,527 of 26,208 (none for 2025-06-15 … 07-01) | 26,208 of 26,208 |
| rows after steps 1–3 | 1,435,662 | 1,552,574 |
| rows after steps 4–6 | 1,422,918 | 1,539,261 |
| unique facts | 791,436 | 814,418 |
| entities / relations | 1,173 / 218 | 1,209 / 215 |
| days with events (time ids) | 256 | 273 |
| train: days with events, facts | 2025-01-01 … 2025-06-14, 165 days, 514,299 | 2025-08-01 … 2026-01-31, 184 days, 509,717 |
| validation | 2025-07-02 … 2025-07-31, 30 days, 94,778 facts | 2026-02-01 … 2026-02-28, 28 days, 76,117 facts (152,234 queries; selection part from ts 203 = 2026-02-20, 50,192 queries) |
| test | 2025-08-01 … 2025-09-30, 61 days, 182,359 facts (364,718 queries) | 2026-03-01 … 2026-04-30, 61 days, 228,584 facts (457,168 queries) |
| file checksums | `GDELT25X.sha256` (code release) | `01_window_build/GDELTFW.sha256` (code release) |

Evaluation in this study is the time-aware filtered rank over all entities, with ties averaged (py-tgb Evaluator formula),
reporting MRR and Hits@1/3/10.
