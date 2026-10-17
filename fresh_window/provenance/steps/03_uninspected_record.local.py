"""Verbatim local notebook cell(s), session-archive index 360, 362, 364, 366, 368, 370, 372, 376, 438 (local python kernel, workspace root as cwd).
Assemble fresh_window_uninspected_check.json from the three scans, an artifact-store date search and the build summary.
Cell 360 raised at json.load (scan.json carries a leading banner) after defining J; cell 364 re-reads with lj() and raised at
line 10 (print loop) after S was built; cell 366 re-prints. Cell 362 is a bash cell. Output: registration/fresh_window_uninspected_check.json.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 360
import json, os
J = {"serverA":"hpc/24868d2a-00f2-4201-a925-29f33feb31d4","serverB":"hpc/7281237c-9381-4054-a66d-2109569cbc8e","serverC":"hpc/6f3c1306-74ba-414c-ab05-e93115855078"}
S = {}
for h,d in J.items():
    S[h] = json.load(open(f"{d}/scan.json"))
    print(h, open(f"{d}/scan_mode.txt").read().strip(), "| err bytes", os.path.getsize(f"{d}/scan.err"))
    print("  keys", list(S[h].keys()))
print(open(f"{J['serverA']}/scan.err").read()[-800:])
# ===== cell 362 (bash cell; verbatim, commented out so this file stays valid Python)
#| for d in hpc/24868d2a-00f2-4201-a925-29f33feb31d4 hpc/7281237c-9381-4054-a66d-2109569cbc8e hpc/6f3c1306-74ba-414c-ab05-e93115855078; do echo "== $d"; wc -c $d/*; cat $d/scan_mode.txt; head -c 600 $d/scan.json; echo; tail -c 800 $d/scan.err; done
# ===== cell 364
def lj(p):
    t = open(p).read(); i = t.find('{"host"'); return json.loads(t[i:].strip())
S = {h: lj(f"{d}/scan.json") for h,d in J.items()}
for h,s in S.items():
    print("=====", h, s["scanned_utc"], s["roots"])
    print(" gdelt", {k:v for k,v in s["gdelt_files"].items() if k!="examples_ge_cut"})
    print(" ex", s["gdelt_files"]["examples_ge_cut"][:5])
    print(" ts2id n", len(s["timestamp2id"]), "n_ge_cut", s["n_timestamp2id_ge_cut"])
    for t in s["timestamp2id"]:
        if t.get("ge_cut") or (t.get("max_date") or "") >= "2025-10-01": print("   GE", t)
    print(" ts2id max dates", sorted({t.get("max_date") for t in s["timestamp2id"]}, key=str)[-6:])
    print(" builds", [ (b.get("path"), b.get("end") or b.get("max_date")) for b in s["build_summaries"]][:10])
    print(" ccnews", s["ccnews"]); print(" errors", s["errors"][:5])
# ===== cell 366
for h,s in S.items():
    print("=====", h)
    print(" gdelt", {k:v for k,v in s["gdelt_files"].items() if k!="examples_ge_cut"})
    for t in s["timestamp2id"]: print("  ts2id", t)
    for b in s["build_summaries"]: print("  build", str(b)[:300])
    print(" ccnews", s["ccnews"]); print(" errors", s["errors"][:5], "n_ts2id_ge_cut", s["n_timestamp2id_ge_cut"])
# ===== cell 368
hits = {}
for pat in ["202510", "202511", "202512", "202601", "202602", "202603", "202604", "2025-10-", "2025-11-", "2025-12-", "2026-01-", "2026-02-", "2026-03-", "2026-04-"]:
    r = host.artifacts(content=pat, limit=200, project_id="all")
    hits[pat] = [(a["filename"], a["latest_version_id"], a.get("project_id")) for a in r["artifacts"]]
    print(pat, r["count"], [h[0] for h in hits[pat]][:15])
# ===== cell 370
proj = "<id>"
seen = {}
for pat, L in hits.items():
    for fn, vid, pj in L:
        if pj == proj: seen.setdefault((fn, vid), []).append(pat)
for k, v in sorted(seen.items()): print(k, v)
# ===== cell 372
import re
cand = ['ccnews_match_report.md','status_stage0.json','sample_manifest.json','ps_calib_sample.py','ps_gold_sample.py','ps_job_regold.sh','ps_timing_docs.py','leakage_audit.json','leakage_pair_judgements.json','A6_4_prereg.json','A6_6_draw_sealed.json','a66_size_review.md','error_cases.md','error_taxonomy.json','feasibility.py','gs117_reserve.py','gs117_reserve_result.json','annotation_sheet_A.csv','examples_master.csv','smallpedia_pilot_verdict.json','hardening_verdict.json','v78_results.json','wikidata_statement_evidence.jsonl','nan_and_dating_note.md','tgb_nan_search_log.json','evidence.json','provenance_audit_edges.csv','exploration_findings.md','extracted_09.json','jobs_09.json','parse_completion_full.json']
rx = re.compile(r"(2025-1[0-2]-\d\d|2026-0[1-4]-\d\d|20251[0-2]\d\d|20260[1-4]\d\d)")
ctx = {}
for (fn, vid), pats in seen.items():
    if fn not in cand: continue
    p = host.artifact_path(vid); t = open(p, errors="ignore").read()
    ms = [(m.group(0), t[max(0,m.start()-70):m.end()+40].replace("\n"," ")) for m in rx.finditer(t)]
    vals = sorted({m[0] for m in ms})
    ctx[fn] = dict(vid=vid, n=len(ms), vals=vals[:8], ex=ms[0][1] if ms else None)
    print(fn, len(ms), vals[:6], "|", (ms[0][1][:170] if ms else ""))
# ===== cell 376
import hashlib, datetime
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
rec = dict(
  check="v79 fresh-window uninspected check (rule: no fact dated >= 2025-10-01 used by any earlier experiment), recorded BEFORE any fitting",
  rule_refs=dict(v79_fresh_window_rule="e10ab60d-9569-49b6-bd1e-6cf415138a78", amendment_1="6debf06f-fc19-49ac-be4e-44b87bbb47da"),
  cut_date="2025-10-01", fresh_window=dict(train="2025-08-01..2026-01-31 (with the GDELT25X min-count filters applied on train)", valid="2026-02-01..2026-02-28", test="2026-03-01..2026-04-30"),
  recorded_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  scanner=dict(code="fw_uninspected_scan.py", sha256=sha("fw/code/fw_uninspected_scan.py"),
               what="(a) every GDELT 2.0 export/mentions/gkg file name under the roots, stamp >= 20251001; (b) every timestamp2id.txt (dates parsed as yyyymmdd/unix/year), max date >= 2025-10-01; (c) every *build*summary*.json 'source' date range; (d) CC-NEWS WARC day references in the status-pilot tree"),
  hosts={})
jobs = {"serverA":"24868d2a-00f2-4201-a925-29f33feb31d4","serverB":"7281237c-9381-4054-a66d-2109569cbc8e","serverC":"6f3c1306-74ba-414c-ab05-e93115855078"}
for h,s in S.items():
    rec["hosts"][s["host"]] = dict(job_id=jobs[h], scanned_utc=s["scanned_utc"], mode=open(f"{J[h]}/scan_mode.txt").read().strip(), roots=s["roots"],
        gdelt_files=dict(n_total=s["gdelt_files"]["n_total"], n_stamp_ge_20251001=s["gdelt_files"]["n_ge_cut"], max_stamp=s["gdelt_files"]["max_stamp"], dirs=s["gdelt_files"]["dirs"]),
        timestamp2id=[dict(path=t["path"], n=t["n"], first=t["first"], last=t["last"], max_date=t["max_date"], any_ge_cut=t["any_ge_cut"]) for t in s["timestamp2id"]],
        gdelt_build_summaries=[b for b in s["build_summaries"] if b.get("source")],
        n_other_build_summaries_without_gdelt_source=sum(1 for b in s["build_summaries"] if not b.get("source")),
        ccnews=s["ccnews"], n_timestamp2id_ge_cut=s["n_timestamp2id_ge_cut"], scan_errors=s["errors"])
rec["artifact_store_check"] = dict(
  method="content substring search over all artifacts of project <id> for 2025-10..2026-04 date strings (yyyymm and yyyy-mm- forms); every TKG-related hit opened and classified",
  n_project_artifacts_with_any_hit=len(seen),
  classification={fn: dict(version_id=v["vid"], values=v["vals"]) for fn, v in ctx.items()},
  verdict="no hit is a GDELT event date: hits are RNG seeds (20251013..20251016), model identifiers (<LLM model id>, <LLM model id>), Wikidata retrieval dates / Wikipedia revision timestamps (smallpedia / extraction work), PTAB decision dates (other sub-project), numeric substrings in metric values, git dates, and the v79 rule itself")
rec["findings"] = dict(
  max_gdelt_export_stamp_any_host=max(s["gdelt_files"]["max_stamp"] or 0 for s in S.values()),
  max_timestamp2id_date_any_host=max(t["max_date"] for s in S.values() for t in s["timestamp2id"]),
  max_gdelt_build_end_exclusive="20251001 (GDELT25X; status pilot and forward holdout)",
  max_ccnews_day=max(s["ccnews"]["max_day"] or 0 for s in S.values()),
  n_files_or_datasets_ge_cut=0)
rec["conclusion"] = "PASS: no GDELT export file, built dataset, build summary or CC-NEWS reference dated on or after 2025-10-01 exists on the three project compute hosts, and no project artifact contains GDELT facts in that range. The fresh-window test period (2026-03-01..2026-04-30) and its validation/train months after 2025-09-30 are uninspected by earlier experiments."
rec["caveats"] = [
  "serverC was scanned as the host user (no docker root), so directories not readable by that user were not covered; the project's GDELT work ran on serverB and serverA, both scanned with read-only root access.",
  "Scan roots = the user's home directory on each host; storage outside it and the lead's local machine are covered only through the artifact-store search.",
  "The serverA scan (14:30:28Z) preceded this experiment's own downloads into tracks/phase2_fresh_window/raw/gdelt (count of files with stamp >= 20251001 = 0 at scan time); those downloads are the fresh window itself and are excluded by definition.",
  "The train period of the fresh window overlaps GDELT25X's valid/test months 2025-08-01..2025-09-30 (by design of the +7-month shift); the rule's 'uninspected' requirement concerns facts dated >= 2025-10-01, which cover all of valid and test."]
os.makedirs("fw/out", exist_ok=True)
json.dump(rec, open("fw/out/fresh_window_uninspected_check.json","w"), indent=1)
print(rec["findings"], rec["recorded_utc"])
# ===== cell 438
d = "hpc/faac377d-9748-42af-9403-7352ff11995d/out"
bs = json.load(open(f"{d}/build_summary.json"))
rec["build"] = dict(job_id="faac377d-9748-42af-9403-7352ff11995d", host="serverA", dataset="GDELTFW",
    path="tracks/phase2_fresh_window/data/GDELTFW", builder="manuscriptA_code_release/data/build_gdelt25x.py (release builder; logic of the GDELT25X builder), run with --start 20250801 --end 20260501 --valid 20260201 --test 20260301 (GDELT25X dates + 7 months), no skip list",
    builder_sha256={l.split()[1].split('/')[-1]: l.split()[0] for l in open(f"{d}/builder_sha256.txt")},
    summary=bs, data_sha256={l.split()[1].split('/')[-1]: l.split()[0] for l in open(f"{d}/data_sha256.txt")},
    timestamp_range=dict(first="20250801 (id 0)", train_last="20260131 (id 183)", valid="20260201..20260228 (ids 184..211)", test="20260301..20260430 (ids 212..272)", n_days=273, gaps=0),
    sanity=dict(missing_export_files=0, entities_in_train=1209, relations_in_train=215, self_loops=0))
rec["build_check"] = "Builder output checked once: every fact of the GDELTFW valid (2026-02) and test (2026-03..04) splits and the train months 2025-10..2026-01 lies on or after 2025-10-01, i.e. in the range no earlier project experiment has used (scan above). Check recorded before any model is fitted."
rec["recorded_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
json.dump(rec, open("fw/out/fresh_window_uninspected_check.json","w"), indent=1)
print(rec["recorded_utc"], sha("fw/out/fresh_window_uninspected_check.json"))
