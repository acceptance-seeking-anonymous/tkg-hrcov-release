# configs/env.sh -- source from the repository root:  source configs/env.sh
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_REPO/configs/paths.env" ]; then . "$_REPO/configs/paths.env"; else . "$_REPO/configs/paths.env.example"; fi
export SRC="$_REPO/src" WORK_DIR EXT_DIR RESULTS_DIR BASES_DIR STAGE_DIR GPU
# module search path shared by all pipelines (a script's own directory always comes first); src/forward_holdout and
# src/pilot_polecat/timesafe are frozen self-contained copies and are deliberately not on this path
_PP=""
for d in harness residual pilot_polecat pilot_smallpedia pilot_icews hardening presub v7 g8_cogntke v76 v78 common latency_g6 ablations c4 datasets; do
  _PP="$_PP:$SRC/$d"
done
export PYTHONPATH="${_PP#:}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK_DIR/tracks" "$EXT_DIR" "$RESULTS_DIR" 2>/dev/null || true
