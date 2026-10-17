#!/bin/bash
# v7 scan on serverA GPU 1 only (GPU 0 = G8). Guard: abort if GPU 1 has < 34 GB free. Usage: bash v7_scan.sh [jobs]
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
JOBS=${1:-"pc_test pc_valid sp_test sp_valid ic_test gd_test"}
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; mkdir -p $V/code $V/scan
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 34000 ]; }
RUN="env CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1"
PCA="--ds polecat --source tirgn --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --feat $T/pc_tirgn/work/feat_tirgn --lists $T/pc_tirgn/work/lists_tirgn --layout pc --recb_dir $H/work/recb --sel_K 100 --sel_var def --valid_rows $T/pc_tirgn/work/valid_sample_idx.npy"
SPA="--ds smallpedia --source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32 --feat $T/pilot_smallpedia/work/feat_tirgn --layout sp --recb_dir $H/work/recb --sel_K 100 --sel_var tr"
FS=$T/retriever/fullscores
ICA="--ds icews18 --source fullrows --fullrows $FS/tirgn_ICEWS18_valid_scores_f32.bin,$FS/tirgn_ICEWS18_test_scores_f32.bin --feat $H/presub_g2/ICEWS18/work/feat_tirgn --lists $H/presub_g2/ICEWS18/work/lists_tirgn --layout pc --recb_dir $H/presub_g2/recb --sel_K 100 --sel_var def --b_items 0"
GDA="--ds gdelt --source fullrows --fullrows $FS/tirgn_GDELT_valid_scores_f32.bin,$FS/tirgn_GDELT_test_scores_f32.bin --feat $H/presub_g2/GDELT/work/feat_tirgn --lists $H/presub_g2/GDELT/work/lists_tirgn --layout pc --recb_dir $H/presub_g2/recb --sel_K 100 --sel_var tr --b_items 0"
# smoke (2000 rows) first
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
$RUN python3 -u $SRC/v7/v7_scan.py $PCA --split test --smoke 2000 --out $J/smoke_pc > smoke_pc.log 2>&1; tm smoke_pc $?
$RUN python3 -u $SRC/v7/v7_scan.py $SPA --split test --smoke 2000 --out $J/smoke_sp > smoke_sp.log 2>&1; tm smoke_sp $?
tail -2 smoke_pc.log smoke_sp.log >> chain.log
grep -q DONE smoke_pc.log && grep -q DONE smoke_sp.log || { echo "smoke failed" >> chain.log; exit 4; }
for jb in $JOBS; do
  guard || { echo "GPU1 busy before $jb -> skip" >> chain.log; continue; }
  case $jb in
    pc_test) A="$PCA --split test --out $V/scan/polecat_tirgn";; pc_valid) A="$PCA --split valid --out $V/scan/polecat_tirgn";;
    sp_test) A="$SPA --split test --out $V/scan/smallpedia_tirgn";; sp_valid) A="$SPA --split valid --out $V/scan/smallpedia_tirgn";;
    ic_test) A="$ICA --split test --out $V/scan/icews18_tirgn";; gd_test) A="$GDA --split test --out $V/scan/gdelt_tirgn";;
  esac
  $RUN python3 -u $SRC/v7/v7_scan.py $A > scan_$jb.log 2>&1; tm $jb $?; tail -1 scan_$jb.log >> chain.log
done
mkdir -p $J/sum; for d in $V/scan/*; do b=$(basename $d); for f in $d/*_scan_summary.json $d/*_scan.npz; do [ -f $f ] && cp $f $J/sum/${b}_$(basename $f); done; done
echo ALLDONE >> chain.log
