#!/usr/bin/env python3
"""fresh_window / fw_uninspected_scan.py -- v79 'uninspected' check (stdlib only, read-only).
Scans one host's project scratch for any evidence that data dated on or after 2025-10-01 entered an earlier experiment:
  (a) GDELT 2.0 export / mentions / gkg files whose 14-digit stamp is >= 20251001000000 (anywhere under the roots)
  (b) every timestamp2id.txt (harness / RE-GCN format datasets): first/last key, interpreted as YYYYMMDD (8 digits),
      unix seconds (9-10 digits), or a year (4 digits); flags any key >= 2025-10-01
  (c) every *build*summary*.json: its 'source' field (GDELT window strings)
  (d) CC-NEWS WARC names referenced in *.log / *.json / *.txt files under pilot-story tracks (days >= 2025-10-01)
Prints one JSON object.  Usage: python3 fw_uninspected_scan.py <host label> <root> [<root> ...]
"""
import os, sys, re, json, datetime

CUT = 20251001
host = sys.argv[1]; roots = sys.argv[2:]
STAMP = re.compile(r"(20\d{12})\.(export|mentions|gkg)\.(CSV|csv)")
WARC = re.compile(r"CC-NEWS-(20\d{2})(\d{2})(\d{2})")
out = dict(host=host, roots=roots, scanned_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           gdelt_files=dict(n_total=0, n_ge_cut=0, max_stamp=None, examples_ge_cut=[], dirs={}),
           timestamp2id=[], build_summaries=[], ccnews=dict(n_refs=0, max_day=None, n_ge_cut=0, files_with_ge_cut=[]), errors=[])


def interp(k):
    k = k.strip()
    if not k.lstrip("-").isdigit():
        return None, "non-numeric"
    v = int(k)
    if len(k) == 8 and 19000101 <= v <= 20991231:
        return v, "yyyymmdd"
    if 9 <= len(k) <= 10:
        d = datetime.datetime.fromtimestamp(v, datetime.timezone.utc)
        return int(d.strftime("%Y%m%d")), "unix"
    if len(k) == 4:
        return v * 10000 + 101, "year"
    return None, "other"


for root in roots:
    for dp, dns, fns in os.walk(root, onerror=lambda e: out["errors"].append(str(e)[:200])):
        dns[:] = [d for d in dns if d not in ("proc", "sys", "__pycache__", ".git", "hf_cache", "site-packages", "node_modules")]
        for fn in fns:
            m = STAMP.search(fn)
            if m:
                st = int(m.group(1)); g = out["gdelt_files"]; g["n_total"] += 1
                g["dirs"][dp] = g["dirs"].get(dp, 0) + 1
                if g["max_stamp"] is None or st > g["max_stamp"]: g["max_stamp"] = st
                if st >= CUT * 1000000:
                    g["n_ge_cut"] += 1
                    if len(g["examples_ge_cut"]) < 20: g["examples_ge_cut"].append(os.path.join(dp, fn))
                continue
            p = os.path.join(dp, fn)
            if fn == "timestamp2id.txt":
                try:
                    keys = [ln.split()[0] for ln in open(p) if ln.strip()]
                    vals = [interp(k) for k in keys]
                    dated = [v for v, kind in vals if v is not None]
                    kinds = sorted({kind for _, kind in vals})
                    out["timestamp2id"].append(dict(path=p, n=len(keys), first=keys[0] if keys else None, last=keys[-1] if keys else None,
                                                    kinds=kinds, max_date=max(dated) if dated else None,
                                                    any_ge_cut=bool(dated and max(dated) >= CUT)))
                except Exception as e:
                    out["errors"].append(f"{p}: {e}"[:200])
            elif re.search(r"build.*summary.*\.json$", fn):
                try:
                    d = json.load(open(p)); out["build_summaries"].append(dict(path=p, source=d.get("source")))
                except Exception as e:
                    out["errors"].append(f"{p}: {e}"[:200])
            elif "pilot_story" in dp and fn.endswith((".log", ".json", ".txt")) and os.path.getsize(p) < 50_000_000:
                try:
                    txt = open(p, errors="ignore").read()
                    for y, mo, dd in WARC.findall(txt):
                        day = int(y + mo + dd); c = out["ccnews"]; c["n_refs"] += 1
                        if c["max_day"] is None or day > c["max_day"]: c["max_day"] = day
                        if day >= CUT:
                            c["n_ge_cut"] += 1
                            if p not in c["files_with_ge_cut"] and len(c["files_with_ge_cut"]) < 20: c["files_with_ge_cut"].append(p)
                except Exception as e:
                    out["errors"].append(f"{p}: {e}"[:200])
g = out["gdelt_files"]; g["dirs"] = dict(sorted(g["dirs"].items(), key=lambda x: -x[1])[:15])
out["n_timestamp2id_ge_cut"] = sum(1 for t in out["timestamp2id"] if t["any_ge_cut"])
out["errors"] = out["errors"][:30]
print(json.dumps(out))
