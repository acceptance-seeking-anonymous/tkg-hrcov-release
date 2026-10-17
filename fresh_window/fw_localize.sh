#!/bin/bash
# fresh_window/fw_localize.sh DEST -- copy the runnable parts of the fresh-window package to DEST and point their hard-coded paths
# at $H, $FW and $B (source fw_env.sh first). The released files are not changed; only the copies under DEST are rewritten.
# The frozen scripts set H=<root>, FW=$H/tracks/phase2_fresh_window and B=$H/tracks/v78_external themselves; the longest prefixes
# are replaced first, so a custom FW or B outside $H also works.
set -euo pipefail
DEST="${1:?usage: bash fw_localize.sh DEST}"; : "${H:?source fw_env.sh first}"; : "${FW:?}"; : "${B:?}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT0="/home/anon/cs_scratch/open_ideas_harness"
mkdir -p "$DEST"
for d in 00_env 01_window_build 02_uninspected_check 03_setup code code_sa code_ct tests registration; do cp -R "$SRC_DIR/$d" "$DEST/"; done
grep -rlF "$ROOT0" "$DEST" | while read -r f; do
  sed -i.bak -e "s#$ROOT0/tracks/phase2_fresh_window#$FW#g" -e "s#$ROOT0/tracks/v78_external#$B#g" -e "s#$ROOT0#$H#g" "$f" && rm -f "$f.bak"
done
echo "localized copy in $DEST (H=$H, FW=$FW, B=$B); run the steps of FILES.md from there"
