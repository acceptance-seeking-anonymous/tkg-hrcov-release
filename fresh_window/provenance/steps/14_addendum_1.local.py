"""Verbatim local notebook cell(s), session-archive index 596 (local python kernel, workspace root as cwd).
Register fresh_window_implementation_note_addendum_1.json (C1/C2 dropped; comparator = C0 alone vs 52 C0 blends).
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# write_prereg: see steps/04_implementation_note.local.py
# ===== cell 596
add = {
 "note_id": "v79_fresh_window_implementation_note_addendum_1",
 "amends": {"file": "fresh_window_implementation_note.json", "version_id": "7ef9ae3a-6fbf-433b-b6a1-536c6512f598"},
 "status_at_registration": "No CountTRuCoLa rule set, ranking or score of any kind exists for any configuration; validation rankings not yet produced. HRcov, SA, BLEND and TiRGN Stage-1 fits are complete (validation only); no test statistic exists.",
 "change": "CountTRuCoLa candidate configurations C1 (tkgl-icews14 entry, LEARN_WINDOW_SIZE 50) and C2 (tkgl-icews18 entry, LEARN_WINDOW_SIZE 50) are dropped; the external comparator is selected between C0 alone and the 52 C0 blends by selection-part MRR (the v79/v78 rule's own procedure).",
 "reason": "Resource infeasibility on the shared serverA, decided from memory/time logs only: after 68 min both runs were in the c-rule example collection at 8% (96 / 1209 x-entities, progress-bar ETA 3 h 48 min) with resident memory 135 GB and 132 GB growing about 1.1 GB/min each (fw_ct_memwatch log), i.e. a projected > 300 GB per run before rule learning and application. C0 (the released default configuration, window 10) runs at 20 GB. Both runs were stopped at 2026-10-16T16:08Z.",
 "consequence": "The comparator is the released default configuration (the configuration the released code applies to a dataset without a DATASET_OVERRIDES entry), alone or validation-tuned blended with TiRGN. This reverts to the single-configuration reading of 'faithful CountTRuCoLa'; the window-50 configurations remain untested on GDELTFW and are reported as such."}
r_ = write_prereg("fw/out/fresh_window_implementation_note_addendum_1.json", add); print(r_)
