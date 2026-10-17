#!/bin/bash
# analysis/stage_v78_inputs.sh -- stage the inputs of src/v78/v78_adapted_agg.py and src/fitpop/fitpop_{agg,diag_agg}.py
# into the flat directory $RESULTS_DIR/v78 and check the SHA-256 of the large per-query files against results/LARGE_FILES.tsv.
# Usage (repository root):  source configs/env.sh; bash analysis/stage_v78_inputs.sh [DIR_WITH_DOWNLOADED_NPZ]
# The optional argument is a directory holding the downloaded per-query .npz files (flat); they are copied into place.
set -u
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RESULTS_DIR:?source configs/env.sh first}"
D="$RESULTS_DIR/v78"; mkdir -p "$D"
cp "$_REPO"/results/v78/runs/*.json "$_REPO"/results/fitpop/fitpop_counts.json "$_REPO"/results/fitpop/fitint_S*.npy "$D"/
if [ $# -ge 1 ]; then cp "$1"/*.npz "$D"/ 2>/dev/null || true; fi
command -v sha256sum >/dev/null 2>&1 && SHA="sha256sum" || SHA="shasum -a 256"
miss=0; bad=0; ok=0
while IFS=$'\t' read -r tier cat path size sha srv; do
  case "$path" in '$RESULTS_DIR/v78/'*) ;; *) continue ;; esac
  f="$D/${path#\$RESULTS_DIR/v78/}"
  if [ ! -f "$f" ]; then echo "MISSING $f"; miss=$((miss+1)); continue; fi
  h=$($SHA "$f" | cut -d' ' -f1)
  if [ "$h" = "$sha" ]; then ok=$((ok+1)); else echo "SHA MISMATCH $f"; bad=$((bad+1)); fi
done < "$_REPO/results/LARGE_FILES.tsv"
echo "staged in $D: $ok large files verified, $miss missing, $bad mismatched"
[ $bad -eq 0 ]
