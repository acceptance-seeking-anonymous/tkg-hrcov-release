#!/bin/bash
# v7: official py-tgb scoring of the B/C/A5 promoted lists (TiRGN replay for polecat, float32 full rows for smallpedia), serverA GPU 1 only.
# Usage: bash v7_official.sh   (systems discovered from $V/b/train_* promlists; split into groups, one process per group)
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; mkdir -p $V/official $V/code
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
guard() { F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU1 free MiB $F at $(date -u)" >> chain.log; [ "$F" -gt 20000 ]; }
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
mk() { ds=$1; ng=$2; ls $V/b/train_${ds}_*/promlist_*.npy | grep -v "EV_PT_fused\|EV_RB_fused" | sort > lists_$ds.txt; n=$(wc -l < lists_$ds.txt); per=$(( (n + ng - 1) / ng )); split -l $per -d lists_$ds.txt grp_${ds}_; }
mk polecat_tirgn 3; mk smallpedia_tirgn 2
guard || { echo "GPU1 busy -> abort" >> chain.log; exit 3; }
for g in grp_polecat_tirgn_* grp_smallpedia_tirgn_*; do
  SYS=$(awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}' $g)
  tag=${g#grp_}
  if [[ $g == grp_polecat* ]]; then
    A="--ds polecat --base v7_$tag --stage test --source tirgn --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --bs 256"
  else
    A="--ds smallpedia --base v7_$tag --stage test --source fullrows --fullrows $T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32 --recb_dir $H/work/recb --bs 256"
  fi
  ( $GPU bash -c "python3 -u $SRC/hardening/hd_blend.py $A --no_controls --systems $SYS --out $V/official" > off_$tag.log 2>&1; tm off_$tag $? ) &
  sleep 20
done
wait
cp $V/official/blend_test_*.json $J/ 2>/dev/null
echo ALLDONE >> chain.log
