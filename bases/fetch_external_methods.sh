#!/bin/bash
# bases/fetch_external_methods.sh -- fetch the two external learned-history methods of the v7.8 comparison and of the 2026 GDELT
# window at the commits used in the paper, into the layout read by src/v78_external/ (B = $WORK_DIR/tracks/v78_external) and by
# fresh_window/ (its B, default $H/tracks/v78_external).  Nothing is vendored; both are used unmodified.
#   source configs/env.sh && bash bases/fetch_external_methods.sh [--venvs]
# --venvs also creates the two virtual environments of the original runs from the recorded pip freezes:
#   $B/env/er  (EdgeReMIND: torch 2.8.0+cpu, py-tgb 2.3.0, the EdgeReMIND clone installed editable)  src/v78_external/env/er_freeze.txt
#   $B/env/ct  (CountTRuCoLa: the repository's requirements, ray 2.59.0)                             src/v78_external/env/ct_freeze.txt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
B="${B:-$WORK_DIR/tracks/v78_external}"; mkdir -p "$B/repos" "$B/env"

fetch() {   # name url commit
  local d=$B/repos/$1
  if [ ! -d "$d/.git" ]; then git clone -q "$2" "$d"; fi
  git -C "$d" fetch -q origin "$3" 2>/dev/null || true
  git -C "$d" checkout -q "$3"
  [ "$(git -C "$d" rev-parse HEAD)" = "$3" ] || { echo "commit mismatch for $1"; exit 2; }
  echo "fetched $1 @ $3"
}
fetch EdgeReMIND              https://github.com/BryantPollard/EdgeReMIND           41c0db2df31bf4ab60e6e4fec8176932314e3ec8
fetch counttrucola_submission https://github.com/JuliaGast/counttrucola_submission  5d9aedf4106926d13befa141e0855d684f3f5f9d

if [ "${1:-}" = "--venvs" ]; then
  python3 -m venv "$B/env/er"
  grep -v '^-e ' "$HERE/src/v78_external/env/er_freeze.txt" > "$B/env/er_requirements.txt"   # torch==2.8.0+cpu from the CPU index
  "$B/env/er/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r "$B/env/er_requirements.txt"
  "$B/env/er/bin/pip" install -q -e "$B/repos/EdgeReMIND"
  python3 -m venv "$B/env/ct"
  "$B/env/ct/bin/pip" install -q -r "$HERE/src/v78_external/env/ct_freeze.txt"
  echo "venvs in $B/env (compare with src/v78_external/env/*_freeze.txt: pip freeze)"
fi
