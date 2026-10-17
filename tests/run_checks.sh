#!/bin/bash
# tests/run_checks.sh -- CPU-only checks that need nothing but this repository:
#   (a) tables / figure data / v7.6 numbers of the manuscript regenerated from results/ and compared with the printed values
#   (b) registration files: integrity, embedded hashes, cross-references, redaction structure (bundle-only mode)
#   (c) fresh-window package: released files vs FILES.md, freeze manifest and registration copies; release-v4 additions vs
#       ADDITIONS.tsv and the log bundle's SHA256SUMS (bundle-only)
#   (d) every Python file compiles, every shell script parses
#   (e) every file of the repository matches SHA256SUMS
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R"
rc=0
echo "== (a) manuscript numbers"; TABLES_OUT=${TABLES_OUT:-$R/results/tables} python3 analysis/check_manuscript_numbers.py || rc=1
echo "== (b) registration hashes (bundle-only)"; python3 registration/verify_registration_hashes.py || rc=1
echo "== (c) fresh-window package (bundle-only)"; python3 fresh_window/verify_fresh_window.py || rc=1
echo "== (d) syntax"
python3 - <<'PY' || rc=1
import os, py_compile, subprocess, sys, tempfile
bad = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d != ".git"]
    for f in files:
        p = os.path.join(root, f)
        if f.endswith(".py"):
            try: py_compile.compile(p, doraise=True, cfile=os.path.join(tempfile.gettempdir(), "_chk.pyc"))
            except Exception as e: bad.append((p, str(e)[:120]))
        elif f.endswith(".sh"):
            r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
            if r.returncode: bad.append((p, r.stderr[:120]))
print("syntax errors:", len(bad)); [print("  ", b) for b in bad]
sys.exit(1 if bad else 0)
PY
echo "== (e) SHA256SUMS"
python3 - <<'PY' || rc=1
import hashlib, os, sys
bad = []; listed = set()
for line in open("SHA256SUMS"):
    h, p = line.rstrip("\n").split("  ", 1); listed.add(p)
    if not os.path.isfile(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != h: bad.append(p)
extra = [os.path.relpath(os.path.join(r, f)) for r, _, fs in os.walk(".") for f in fs]
extra = [p for p in extra if p != "SHA256SUMS" and "__pycache__" not in p and not p.startswith(("results/tables/", "results/figures/", ".git/")) and p not in listed]
print("files listed:", len(listed), " mismatched or missing:", len(bad), " unlisted:", len(extra)); [print("  ", b) for b in (bad + extra)[:20]]
sys.exit(1 if bad else 0)
PY
echo "== overall: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"; exit $rc
