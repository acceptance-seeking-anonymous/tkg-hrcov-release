#!/bin/bash
# v7: re-run of the polecat official-scoring group 02 (container was stopped to respect the GPU-1 cap).
# Holds the C2 chain (its waiter polls for 'base v7_polecat_tirgn_0' containers) until group 02 is done, so that
# the C2 GPU stages do not overlap with this rerun. Group 02 is split into two halves (02a, 02b) run in parallel.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7
SRC=$WORK_DIR/runs/grp_polecat_tirgn_02
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
bash -c 'echo base v7_polecat_tirgn_0hold > /dev/null; sleep 40000' >> chain.log 2>&1
echo "hold started $(date -u)" >> chain.log
# (orchestration wait of the original multi-job run removed)
for i in $(seq 1 120); do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0}); [ "$F" -gt 26000 ] && break; sleep 60; done
echo "GPU1 free MiB $F at $(date -u)" >> chain.log
cp $SRC grp_all; head -n 8 grp_all > grp_02a; tail -n +9 grp_all > grp_02b; wc -l grp_02a grp_02b >> chain.log
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
for g in 02a 02b; do
  SYS=$(awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}' grp_$g)
  A="--ds polecat --base v7_polecat_tirgn_$g --stage test --source tirgn --tirgn_tree $T/pilot_smallpedia/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --bs 256"
  ( $GPU bash -c "python3 -u $SRC/hardening/hd_blend.py $A --no_controls --systems $SYS --out $V/official" > off_polecat_tirgn_$g.log 2>&1; tm off_polecat_tirgn_$g $? ) &
  sleep 30
done
wait
# (orchestration wait of the original multi-job run removed)
ls -la $V/official/ >> chain.log
echo ALLDONE >> chain.log
