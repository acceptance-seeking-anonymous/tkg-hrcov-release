#!/bin/bash
# presub / hcpat_smallpedia_regcn_retuned.sh -- definitional re-run: HRCOV_PAT on smallpedia RE-GCN (re-tuned) (train + official scoring) and official
# scoring of the polecat TiRGN HRCOV_PAT lists trained on serverA, on the serverB RTX 4090.  The TiRGN replay waits until the G3 chain
# (job) has finished so the two do not compete for the 24 GB card.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD; H=$WORK_DIR/tracks/hardening; M=$WORK_DIR; G=$H/presub_hcpat; mkdir -p $G/code $G/pc_lists
# (code staging step of the original run removed: code is read from $SRC)
S=$WORK_DIR; SCP="cp"; SSH="bash -c"
G_A=$WORK_DIR/tracks/hardening/presub_hcpat
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > gpu_at_start.txt
ENVS="env PC_HARNESS=$M PYTHONDONTWRITEBYTECODE=1 HD_HAR=$M HD_HOST=local G2_THREADS=8"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS"
# ---- smallpedia RE-GCN re-tuned
TS=$G/train_smallpedia_regcn_tuned
$GPU python3 -u $SRC/hardening/hcpat_train.py --layout sp --feat $M/tracks/sp_new/work/feat_regcn --out $TS --label smallpedia_regcn_tuned > train_smallpedia_regcn_tuned.log 2>&1; tm train_sp $?
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base regcn_tuned_hcpat --stage test --source fullrows --fullrows $H/sp_new/dumps/regcn_valid_logp.f32,$H/sp_new/dumps/regcn_test_logp.f32 --recb_dir $H/work/recb --no_controls --systems $SYS --out $G" > official_sp_regcn_tuned.log 2>&1; tm official_sp $?
cp $G/official_test_smallpedia_regcn_tuned_hcpat.npz $G/blend_test_smallpedia_regcn_tuned_hcpat.json $W/ 2>/dev/null
# ---- polecat TiRGN: wait for the serverA lists, then for the G3 chain to finish
n=0
until $SSH "test -f $G_A/train_polecat_tirgn/train_hcpat.json && ls $G_A/train_polecat_tirgn/promlist_HRCOV_PAT_s2.npy" > /dev/null 2>&1; do sleep 120; n=$((n+1)); [ $n -ge 240 ] && { echo "serverA lists never appeared" >> chain.log; exit 5; }; done
sleep 30
for s in 0 1 2; do $SCP $G_A/train_polecat_tirgn/promlist_HRCOV_PAT_s$s.npy $G/pc_lists/; done
$SCP $G_A/train_polecat_tirgn/train_hcpat.json $W/train_hcpat_polecat_tirgn.json; tm fetch_pc_lists $?
( cd $G/pc_lists && md5sum *.npy ) >> chain.log; $SSH "cd $G_A/train_polecat_tirgn && md5sum promlist_HRCOV_PAT_s*.npy" >> chain.log
G3LOG=$WORK_DIR/runs/chain.log
# (orchestration wait of the original multi-job run removed)
tm wait_g3 0; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv >> gpu_at_start.txt
SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$G/pc_lists/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
$GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base tirgn_hcpat --stage test --source tirgn --tirgn_tree $H/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt --recb_dir $H/work/recb --no_controls --systems $SYS --out $G" > official_pc_tirgn.log 2>&1; tm official_pc $?
cp $G/official_test_polecat_tirgn_hcpat.npz $G/blend_test_polecat_tirgn_hcpat.json $W/ 2>/dev/null
cp $TS/train_hcpat.json $W/train_hcpat_smallpedia_regcn_tuned.json 2>/dev/null
echo ALLDONE >> chain.log
