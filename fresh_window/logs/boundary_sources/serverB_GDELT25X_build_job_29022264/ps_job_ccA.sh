#!/bin/bash
# status pilot job A (host serverB, CPU + network): GDELT 2025-07..09 extension + retry of missing Jan-Jun files -> GDELT25X slice;
# CC-NEWS 2025-01..09 WARC listings; target URL set; 2-file scan probe.
set +e
W=$PWD; P=/home/[REDACTED:~ANON]/cs_scratch/open_ideas_harness/tracks/pilot_story; X=$P/gdelt25x; CC=$P/ccnews
mkdir -p $X/raw $X/data $X/corpus $X/dumps $X/work $CC/paths $CC/probe
cp $W/ps_build_gdelt.py $W/ps_ccnews_scan.py $P/code/
[ -e $X/raw/gdelt ] || ln -s $P/gdelt25/raw/gdelt $X/raw/gdelt
cp -n $P/gdelt25/raw/CAMEO.eventcodes.txt $X/raw/ 2>/dev/null
ls $P/gdelt25/raw/gdelt | wc -l > gdelt_files_before.txt
docker run --rm -v /home/[REDACTED:~ANON]/cs_scratch:/home/[REDACTED:~ANON]/cs_scratch -w $W -e PS_HOST=serverB -e PYTHONDONTWRITEBYTECODE=1 \
  -e PS_DS=GDELT25X -e PS_P=$X -e PS_G_START=20250101 -e PS_G_END=20251001 -e PS_G_VALID=20250701 -e PS_G_TEST=20250801 \
  tkg-gpu:latest python3 $P/code/ps_build_gdelt.py > build_gdelt25x.log 2>&1; echo "build exit $?" >> build_gdelt25x.log
ls $P/gdelt25/raw/gdelt | wc -l > gdelt_files_after.txt
cp $X/corpus/build_summary.json build_gdelt25x_summary.json; cp $X/corpus/missing_export_files.txt . 2>/dev/null
python3 - <<'PY'
import json, gzip
P="/home/[REDACTED:~ANON]/cs_scratch/open_ideas_harness/tracks/pilot_story"
urls = {}
for ds in ("gdelt25", "gdelt25x"):
    for ln in open(f"{P}/{ds}/corpus/docs.jsonl"):
        d = json.loads(ln); urls.setdefault(d["url"], set()).add(ds)
with gzip.open(f"{P}/ccnews/targets.txt.gz", "wt") as f:
    for u in urls: f.write(u + "\n")
json.dump(dict(n_targets=len(urls), n_gdelt25=sum("gdelt25" in v for v in urls.values()), n_gdelt25x=sum("gdelt25x" in v for v in urls.values())), open("targets_summary.json", "w"))
PY
for m in 01 02 03 04 05 06 07 08 09; do curl -sL -m 60 https://data.commoncrawl.org/crawl-data/CC-NEWS/2025/$m/warc.paths.gz | zcat > $CC/paths/paths_2025$m.txt; wc -l $CC/paths/paths_2025$m.txt; done > warc_counts.txt
sed -n '100,101p' $CC/paths/paths_202503.txt > $CC/probe/paths.txt
( time python3 $P/code/ps_ccnews_scan.py --paths $CC/probe/paths.txt --targets $CC/targets.txt.gz --out $CC/probe --workers 2 ) > probe.log 2>&1
cat $CC/probe/scan_log.jsonl >> probe.log
