#!/bin/bash
# bases/fetch_upstream.sh -- fetch the third-party code at the pinned commits, apply this repository's patches, and place the trees
# where the experiment scripts expect them.  Nothing here is vendored: TiRGN and CognTKE carry no licence file upstream, so they
# are fetched from their authors' repositories; TGB (MIT) is fetched as well.  See THIRD_PARTY.md.
#   source configs/env.sh && bash bases/fetch_upstream.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${EXT_DIR:=$WORK_DIR/ext}"
BASES_DIR=${BASES_DIR:-$WORK_DIR/bases}; STAGE_DIR=${STAGE_DIR:-$WORK_DIR/stage_inputs}
mkdir -p "$BASES_DIR" "$STAGE_DIR"

fetch() {   # name url commit
  local name=$1 url=$2 sha=$3 d=$BASES_DIR/$1
  if [ ! -d "$d/.git" ]; then git clone -q "$url" "$d"; fi
  git -C "$d" fetch -q origin "$sha" 2>/dev/null || true
  git -C "$d" checkout -q "$sha"
  [ "$(git -C "$d" rev-parse HEAD)" = "$sha" ] || { echo "commit mismatch for $name"; exit 2; }
  echo "fetched $name @ $sha"
}
fetch TiRGN   https://github.com/Liyyy2122/TiRGN          4b74c98b24b200a138e22b7e69f948f0f8d0232a
fetch CognTKE https://github.com/WeiChen3690/CognTKE      4f30966ec2c4775bad0a9addee6e22bdd1118452
fetch TGB     https://github.com/shenyangHuang/TGB        3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb

patched() {   # source-tree patch target-dir md5-list   (an existing target is kept: it may hold converted data and checkpoints)
  local src=$BASES_DIR/$1 patch=$HERE/patches/$2 dst=$3 md5=$HERE/patches/$4
  if [ -d "$dst" ]; then echo "kept existing $dst"; return; fi
  mkdir -p "$dst"
  ( cd "$src" && git archive --format=tar HEAD ) | tar xf - -C "$dst"
  patch -s -p1 -d "$dst" -i "$patch"
  ( cd "$dst" && md5sum -c --quiet "$md5" ) && echo "patched $dst: all .py files match $4"
}
T=$WORK_DIR/tracks
# TiRGN with the pure-torch DGL replacement, the harness-order score dump and the training-loop changes; one copy per track,
# as in the original runs (each copy receives its own converted data/ and checkpoints)
patched TiRGN tirgn_stage.patch      "$T/pilot_smallpedia/stage/TiRGN"       tirgn_stage.md5   # also provides dgl_shim.py for RE-GCN (TGB)
patched TiRGN tirgn_stage.patch      "$T/hardening/stage/TiRGN"              tirgn_stage.md5
patched TiRGN tirgn_stage.patch      "$T/phase2_forward_holdout/stage/TiRGN" tirgn_stage.md5
# TiRGN as used for the ICEWS18 / GDELT score dumps (same base changes plus the dump helpers)
patched TiRGN tirgn_gdelt_dump.patch "$EXT_DIR/gdelt_tirgn_dump/TiRGN"       tirgn_gdelt_dump.md5

link() { mkdir -p "$(dirname "$2")"; ln -sfn "$1" "$2"; echo "  $2 -> $1"; }
link "$BASES_DIR/TGB"              "$T/pilot_icews/stage/tgb_modules"       # RecB predictor (modules/recurrencybaseline_predictor.py)
link "$BASES_DIR/TGB"              "$T/pilot_polecat/stage/tgb_repo"        # official TGB RecB example script (pc_recb_official.sh)
# TGB 2.0 RE-GCN example scripts (G3 / v7.6 M6): a source copy without datasets, unpacked per run by the scripts
( cd "$BASES_DIR/TGB" && git archive --format=tar HEAD ) | gzip -n > "$STAGE_DIR/g3_tgb_bits.tgz"
echo "  $STAGE_DIR/g3_tgb_bits.tgz"
# CognTKE: used unmodified (the empty diff is recorded in bases/README.md); data layout and libraries: src/g8_cogntke/g8_setup.sh
echo "done"
