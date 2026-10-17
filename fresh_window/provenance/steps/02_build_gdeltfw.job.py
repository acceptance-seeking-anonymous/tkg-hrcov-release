"""Verbatim orchestration cell, session-archive index 309 (remote job submission).
Build GDELTFW from the official GDELT 2.0 15-minute exports (job faac377d). Same command as 01_window_build/run_build_gdeltfw.sh.
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 309
cmd = r'''set +e
FW=/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window
REL=/home/anon/cs_scratch/code_release_verify/repo/manuscriptA_code_release/data
mkdir -p $FW/builder $FW/raw/gdelt $FW/data $FW/logs
cp $REL/build_gdelt25x.py $REL/gdelt25x_relation_names.tsv $REL/GDELT25X.sha256 $FW/builder/ 2>&1
sha256sum $FW/builder/* > $FW/logs/builder_sha256.txt
echo "[build] start $(date -u +%FT%TZ)" > $FW/logs/build.log
uptime >> $FW/logs/build.log
docker run --rm --cpus 16 -e OMP_NUM_THREADS=1 -v /home/anon/cs_scratch:/home/anon/cs_scratch -w $FW tkg-gpu:latest \
  python3 -u builder/build_gdelt25x.py --out $FW/data/GDELTFW --raw $FW/raw/gdelt --start 20250801 --end 20260501 --valid 20260201 --test 20260301 --workers 16 >> $FW/logs/build.log 2>&1
echo "[build] exit $? $(date -u +%FT%TZ)" >> $FW/logs/build.log
docker run --rm -v /home/anon/cs_scratch:/home/anon/cs_scratch tkg-gpu:latest chmod -R a+rX $FW/data $FW/raw
ls $FW/raw/gdelt | wc -l >> $FW/logs/build.log
mkdir -p out; cp $FW/logs/build.log $FW/logs/builder_sha256.txt $FW/data/GDELTFW/build_summary.json $FW/data/GDELTFW/stat.txt $FW/data/GDELTFW/timestamp2id.txt out/ 2>/dev/null
wc -l $FW/data/GDELTFW/*.txt > out/wc.txt 2>&1; sha256sum $FW/data/GDELTFW/*.txt > out/data_sha256.txt 2>&1
wc -l $FW/data/GDELTFW/missing_export_files.txt >> out/wc.txt 2>&1
'''
jb = c_serverA.submit_job(command=cmd, intent="Build the v79 fresh GDELT window (GDELT25X builder, all dates +7 months: 2025-08-01..2026-04-30) on serverA CPU; downloads ~26k official 15-min exports",
                     outputs=["out/*"], run_timeout_s=8*3600)
print(jb)
