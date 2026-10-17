"""Verbatim local notebook cell(s), session-archive index 632 (local python kernel, workspace root as cwd).
Add the UTC chronology to the host manifest -> registration/fresh_window_freeze_manifest.json.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 632
import shutil
M = json.load(open("hpc/117823f0-d9fc-4870-95e1-83bbedc15ba9/h/freeze_manifest_remote.json"))
impl = json.load(open("fw/out/fresh_window_implementation_note.json")); addn = json.load(open("fw/out/fresh_window_implementation_note_addendum_1.json"))
M["chronology_utc"] = [
 ["2026-10-16T14:24:12Z", "v79_fresh_window_rule registered (lead)"], ["2026-10-16T14:24:46Z", "amendment_1 registered (lead)"],
 ["2026-10-16T14:30:28Z", "uninspected scans serverA / serverB / serverC (jobs 24868d2a, 7281237c, 6f3c1306)"],
 ["2026-10-16T14:45:39Z", "GDELTFW build finished (job faac377d; 26,208 export files)"],
 ["2026-10-16T14:46:47Z", "uninspected check recorded, PASS (artifact version f255c8f4)"],
 [impl.get("registered_utc"), "implementation note registered (artifact version 7ef9ae3a)"],
 ["2026-10-16T14:50:42Z", "Stage 1A: TiRGN training on GPU 0 starts (job c2ac489f)"], ["2026-10-16T15:01:08Z", "TiRGN best epoch 11 frozen; validation rows written"],
 ["2026-10-16T14:53:54Z", "CountTRuCoLa first launch failed at dataset construction (relation2id format in the wrapper); no rule learned"],
 ["2026-10-16T14:57:37Z", "CountTRuCoLa C0/C1/C2 relaunched (job e7efbdf3)"],
 ["2026-10-16T15:07:07Z", "HRcov fit + freeze (3 seeds), BLEND selection (job dc39717c)"],
 ["2026-10-16T15:01:49Z", "SA chain first attempt failed at the cut parse (container banner); nothing computed"],
 ["2026-10-16T15:08:38Z", "RecB_xf selection + SA fit + freeze (3 seeds) (job 9ccd0e7d)"],
 ["2026-10-16T16:08:00Z", "CountTRuCoLa C1/C2 stopped for memory/time (no rule or score existed)"], [addn.get("registered_utc"), "addendum 1 registered (artifact version dede10e9)"],
 ["2026-10-16T16:19:13Z", "CountTRuCoLa C0 rules learned on train, applied to validation, scored"],
 ["2026-10-16T16:20:19Z", "external comparator frozen: C0 blend a=0.1 tau=0.5 (selection-part MRR)"],
 ["2026-10-16T16:31:22Z", "latency protocol done (validation slice)"], [M["created_utc"], "SHA-256 freeze manifest written (201 files, 0 missing)"]]
M["note"] = "Stage-1 freeze of the v79 fresh-window experiment; no test-split statistic of any kind exists at creation. Stage 2 verifies every file before the single test pass."
json.dump(M, open("fw/out/fresh_window_freeze_manifest.json", "w"), indent=1)
shutil.copy("hpc/117823f0-d9fc-4870-95e1-83bbedc15ba9/h/fw_code_frozen.tgz", "fw/out/fw_code_frozen.tgz")
print(M["created_utc"], os.path.getsize("fw/out/fw_code_frozen.tgz"))
