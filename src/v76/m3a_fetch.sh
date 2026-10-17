#!/bin/bash
# M3a: finiteness scan of the validation-trained base outputs (dumps, full rows, controls) + copy of the official files (read-only).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; H=$WORK_DIR/tracks/hardening; V=$H/v76; M=$V/m3a
cp $M/official_test_smallpedia_tirgn_valfit.npz $M/blend_test_smallpedia_tirgn_valfit.json $M/blend_valid_smallpedia_tirgn_valfit.json $M/sel_smallpedia_valfit.json $J/
cp $M/sp/dumps/tirgn_SMALLPEDIA_dump_log.json $J/m3a_tirgn_valfit_dump_log.json
mkdir -p $J/ctl && cp $M/official_test_smallpedia_tirgn_valfit.npz $M/blend_valid_smallpedia_tirgn_valfit.npz $J/ctl/
env OMP_NUM_THREADS=1 \
  python3 -u $SRC/common/finite_scan.py --workers 6 --out ./finite_scan_m3a.json --roots "m3a_valbase_dumps=$M/sp/dumps" "m3a_controls=./ctl" > scan.log 2>&1
echo "rc=$?" >> scan.log; rm -rf $J/ctl
