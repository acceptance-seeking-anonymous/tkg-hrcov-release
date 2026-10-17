# registration/

Decision-rule and protocol files of the paper. Identifying paths and host names inside them are replaced by neutral labels or placeholders (`<HOST>`, `<SCRATCH>`, `<HOME>`, `anon`, `serverA` / `serverB` / `serverC`); `redaction_map.json` lists the changed fields. The recorded hashes are those of the unredacted originals, which are available to the editor and reviewers on request (`--originals DIR` also reads the unpublished redaction rules provided with them). Byte-identical copies of the seven fresh-window documents are under `../fresh_window/registration/`.

| file | rule | registered (UTC) | content hash |
|---|---|---|---|
| `hardening_decision_rule.json` | main test | 2026-10-12T18:43:20Z | time stamp only |
| `stageA_decision_rule.json` | tkgl-icews Stage A pilot | 2026-10-12T14:54:56Z | time stamp only |
| `stageB_rule.json` | tkgl-icews Stage B | 2026-10-13T05:46:34Z | time stamp only |
| `a_presub_rule.json` | pre-submission rule: TGB-config. RE-GCN, base-free control G1, cluster bootstrap G7, ICEWS18/GDELT controls G2 | 2026-10-13T13:35:00Z | time stamp only |
| `ablation_rule.json` | feature-group ablation A10/A11 | 2026-10-13T19:16:03Z | `11bf117ef6fba8da…` (indent1_sorted) |
| `g8_rule.json` | CognTKE base G8 | 2026-10-13T18:28:20Z | `273be7b98b2ba2e7…` (indent1_sorted) |
| `g8_deviation_B.json` | CognTKE base, deviation B | 2026-10-13T18:43:00Z | `24d818f604e0f882…` (indent1_sorted) |
| `g6_protocol.json` | latency profiling G6 | 2026-10-14T04:23:49Z | `0c0638d4002a9c1b…` (indent1_sorted) |
| `tois_v7_rule.json` | later analyses (A, B, C items) | 2026-10-14T10:13:36Z | `bfef916814902786…` (indent1_sorted) |
| `fh_decision_rule.json` | forward holdout GDELT25X | 2026-10-14T10:28:13Z | `2bb027715962a746…` (indent1_sorted) |
| `c4_protocol.json` | second base seed C4 | 2026-10-14T18:54:49Z | `171774ea1da536fb…` (oneline_sorted) |
| `v76_rule.json` | reviewer-requested analyses (LambdaMART, validation-trained base, cross-fitted RecB, base seeds) | 2026-10-15T12:07:43Z | `edcfca14128e22f6…` (indent1_sorted) |
| `v76_rule_amendment_1.json` | reviewer-requested analyses, amendment 1 | 2026-10-15T12:49:21Z | `bc8cd8f87192ee98…` (indent1_sorted) |
| `v76_rule_registered.json` | reviewer-requested analyses rule, registered canonical serialisation | 2026-10-15T12:07:43Z | `edcfca14128e22f6…` (indent1_sorted) |
| `v76_rule_amendment_1_registered.json` | reviewer-requested analyses, amendment 1, registered canonical serialisation | 2026-10-15T12:49:21Z | `bc8cd8f87192ee98…` (indent1_sorted) |
| `v78_framing_rule.json` | framing rule: standalone ladder, external comparators, reproduction gate | 2026-10-15T17:34:21Z | `8e05c67b7ce8f3a4…` (indent1_sorted) |
| `v78_rule_amendment_1.json` | framing rule, amendment 1 (implementation details, registered before any ladder number) | 2026-10-15T18:54:33Z | `8fffffcaad75b066…` (indent1_sorted) |
| `v78_rule_amendment_1_registered.json` | framing rule, amendment 1, registered canonical serialisation | 2026-10-15T18:54:33Z | `8fffffcaad75b066…` (indent1_sorted) |
| `v78_external_protocol_note.json` | external comparators, protocol note 1 (gate, history boundary, run configuration) | 2026-10-15T17:52:22Z ¹ | `52a12ef8bd3c1e54…` (indent1_sorted) |
| `v78_external_protocol_note_2_ext_adapted.json` | external comparators, protocol note 2 (EXT_adapted CountTRuCoLa; registered-optional run decided after the framing-rule verdict) | 2026-10-16T08:20:08Z ¹ | `ce9ef0ebc845dafd…` (indent1_sorted) |
| `v79_fresh_window_rule.json` | fresh-window rule: registered comparison of HRcov − SA and HRcov − CountTRuCoLa on the uninspected 2026 GDELT window, with outcome labels | 2026-10-16T14:24:12Z | `209bb1b57495aefb…` (indent1_sorted) |
| `v79_fresh_window_rule_amendment_1.json` | fresh-window rule, amendment 1 (scope changes, interpretation of the outcome labels, latency protocol), registered before the window was built | 2026-10-16T14:24:46Z | `207689b83c46d7c8…` (indent1_sorted) |
| `fresh_window_implementation_note.json` | fresh-window implementation note (data, validation parts, base, code, systems, comparator candidates), registered before any model was fitted | 2026-10-16T14:49:29Z | `c99bb4b7f690a84d…` (indent1_sorted) |
| `fresh_window_implementation_note_addendum_1.json` | fresh-window implementation note, addendum 1 (the two CountTRuCoLa learning-window-50 configurations stopped before any rule or score existed) | 2026-10-16T16:08:05Z | `57cab06716278481…` (indent1_sorted) |
| `fresh_window_uninspected_check.json` | fresh-window uninspected check, recorded before any model was fitted | 2026-10-16T14:46:47Z ² | file SHA-256 `bfd93379c66b3914…` ³ |
| `fresh_window_freeze_manifest.json` | fresh-window freeze manifest: SHA-256 of the 201 files frozen before the single test pass, with the chronology | 2026-10-16T16:32:23Z ² | file SHA-256 `ee433c5b87993e3c…` ³ |
| `fresh_window_verdict.json` | fresh-window verdict, outcome R4_OTHER (mixed), decided after the single test pass | 2026-10-16T17:12:18Z ² | file SHA-256 `a96619d3e6cd71a3…` ³ |

`registration_manifest.json` records, per file, the recorded hash and its serialisation convention, the SHA-256 of the original and of the redacted copy, the structure fingerprint, the redacted fields and the files that cite the hash. `redaction_map.json` lists, per redacted file, the changed JSON fields and placeholder counts (no original values).

```bash
python3 registration/verify_registration_hashes.py                  # without originals (all checks that can run)
python3 registration/verify_registration_hashes.py --originals DIR  # with the unredacted originals: recompute every hash
```

Hash convention: `sha256(json.dumps(body, indent=1, sort_keys=True, default=str))` (or the one-line form for `c4_protocol.json`), where `body` is the rule with its `registered_utc` field and without the `sha256` field. The hash is computed before the file is written, and the file is written in insertion order with the `sha256` field embedded, so the SHA-256 of the file bytes differs from the recorded hash. `v76_rule_registered.json`, `v76_rule_amendment_1_registered.json` and `v78_rule_amendment_1_registered.json` are the exact hashed serialisations of their rule files.

¹ The two external-comparator protocol notes carry their time stamp in the field `created_utc` (manifest key `time_field`); like `registered_utc` elsewhere, it is part of the hashed body.

² Time-stamp fields: `recorded_utc` (uninspected check), `created_utc` (freeze manifest), `decided_utc` (verdict) (manifest key `time_field`).

³ These three documents carry no registration-time hash. The manifest records `content_hash_kind = "file_sha256"` and `content_hash` = SHA-256 of the original file bytes, computed at release time; the checker recomputes it from the original (`--originals`) or, for the unredacted verdict, from the bundled copy. They are not counted among the content-hashed registration documents cited in the manuscript (`\NHASHED`).
