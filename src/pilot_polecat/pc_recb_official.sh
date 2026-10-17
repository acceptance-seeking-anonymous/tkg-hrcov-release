#!/bin/bash
# Official TGB RecurrencyBaseline (Gastinger et al. IJCAI'24) on tkgl-polecat, TGB repo example script, run unmodified.
# usage: bash pc_recb_official.sh <num_processes> <train_flag True|False>
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -e
NP=$1; TR=$2
python3 - <<'PY'
import os, zipfile
from tgb.utils.info import PROJ_DIR
root = os.path.join(PROJ_DIR, "datasets", "tkgl_polecat"); os.makedirs(root, exist_ok=True)
with zipfile.ZipFile(os.path.expandvars("$WORK_DIR/tracks/datasets/raw/tkgl-polecat.zip")) as z:   # WORK_DIR exported by configs/env.sh
    for n in z.namelist():
        if not os.path.exists(os.path.join(root, n)): z.extract(n, root)
print("staged", os.listdir(root))
PY
cd $WORK_DIR/tracks/pilot_polecat/stage/tgb_repo/examples/linkproppred/tkgl-polecat
git -C ../../.. rev-parse HEAD > ./tgb_repo_commit.txt 2>/dev/null || true
python3 -u recurrencybaseline.py -d tkgl-polecat -p $NP -tr $TR -w 0 2>&1 | tee ./recb_official_train${TR}.log
cp saved_results/*.json ./ 2>/dev/null || true
cp best_config.json ./recb_best_config.json 2>/dev/null || true
