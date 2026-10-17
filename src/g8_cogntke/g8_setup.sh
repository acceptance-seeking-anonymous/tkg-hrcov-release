#!/bin/bash
# G8 setup: CognTKE (fetched by bases/fetch_upstream.sh at commit 4f30966, unmodified) + tkgl-polecat in its data layout + the
# three small library directories the G8 runs used (put first on PYTHONPATH by g8_train.sh / g8_chain.sh):
#   pylib     loguru (imported by the CognTKE code; not in requirements.txt)
#   pylib_ts  torch_scatter 2.1.2 wheel for torch 2.8.0 + CUDA 12.8 (the CognTKE code imports torch_scatter; a pure-torch
#             fallback with the same scatter() signature is in src/g8_cogntke/shim/ and was checked against it, g8_scatter_eq.py)
#   pylib_era numpy 1.26.4 + scipy 1.11.4, used ONLY by g8_rankcheck_cpu.py (released ranker under the authors'-era stack)
# Usage: bash g8_setup.sh
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
BASES_DIR=${BASES_DIR:-$WORK_DIR/bases}
set -e
G=$WORK_DIR/tracks/hardening/g8_cogntke; D=$WORK_DIR/tracks/datasets/harness_data/tkgl-polecat
mkdir -p $G
[ -e $G/CognTKE ] || ln -s $BASES_DIR/CognTKE $G/CognTKE
mkdir -p $G/CognTKE/data/POLECAT
for f in entity2id relation2id train valid test; do cp $D/$f.txt $G/CognTKE/data/POLECAT/$f.txt; done
( cd $G/CognTKE/data/POLECAT && sha256sum *.txt ) > g8_data_sha256.txt
[ -d $G/pylib ] || python3 -m pip install -q --target $G/pylib loguru
[ -d $G/pylib_ts ] || python3 -m pip install -q --no-deps --only-binary :all: --target $G/pylib_ts torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
[ -d $G/pylib_era ] || python3 -m pip install -q --only-binary :all: --target $G/pylib_era numpy==1.26.4 scipy==1.11.4
echo "G8 setup done: $G"
