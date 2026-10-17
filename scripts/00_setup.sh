#!/bin/bash
# scripts/00_setup.sh -- third-party code (pinned commits + patches), TGB 2.0 datasets, G8 libraries.  About 20 min + downloads.
#   bash scripts/00_setup.sh [datasets...]   (default: tkgl-polecat tkgl-smallpedia tkgl-icews)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -e
bash "$R/bases/fetch_upstream.sh"
python3 "$R/data/download_tgb.py" --names ${@:-tkgl-polecat tkgl-smallpedia tkgl-icews}
cp -n "$WORK_DIR/tracks/datasets/raw/tkgl-smallpedia.zip" "$STAGE_DIR/" 2>/dev/null || true
bash "$SRC/g8_cogntke/g8_setup.sh"
echo "ICEWS18 / GDELT (RE-GCN release) and GDELT25X are prepared separately: see data/README.md"
