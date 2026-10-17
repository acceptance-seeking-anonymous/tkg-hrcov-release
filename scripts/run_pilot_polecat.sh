#!/bin/bash
# scripts/run_pilot_polecat.sh -- pilot 1 on tkgl-polecat (RE-GCN and LogCL bases): base training + harness-order dumps, TGB id
# bridge, base-independent prep, lists -> exact lookup + official base evaluation -> features -> re-rankers (3 seeds) ->
# official evaluation of the systems; official TGB RecB; aggregation under registration/... (pilot rule, see README).
# Commands and arguments follow results/reports/pilot_pipeline_notes.md section 4 (the original runs were launched one stage
# at a time; this wrapper strings the documented commands together).  Wall clock on one RTX PRO 6000: about 4 h per base.
#   bash scripts/run_pilot_polecat.sh [bases]      (default: "regcn logcl")
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BASES=${1:-regcn logcl}
P=$WORK_DIR/tracks/pilot_polecat; DD=$WORK_DIR/tracks/datasets/harness_data/tkgl-polecat
mkdir -p $P/dumps $P/work $P/stage; rundir pilot_polecat
need $DD/stat.txt
python3 -u $SRC/pilot_polecat/pc_tgb_verify.py 2>&1 | tee tgb_verify.log
python3 -u $SRC/pilot_polecat/pc_prep.py --workers 24 2>&1 | tee prep.log
for B in $BASES; do
  [ -f $P/dumps/${B}_scores_POLECAT_traintest.npz ] || env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 \
    python3 -u $SRC/harness/base_dump.py --model $B --data_dir $DD --dataset POLECAT --out_dir $P/dumps --epochs 12 --seed 42 2>&1 | tee base_$B.log
  bash $SRC/pilot_polecat/pc_chain.sh $B $GPU 12 0 2>&1 | tee chain_$B.log
  env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/pilot_polecat/pc_lookup.py --base $B --mode systems --systems \
    HR_s0=$P/work/train_$B/promlist_HR_s0.npy,HRCOV_s0=$P/work/train_$B/promlist_HRCOV_s0.npy 2>&1 | tee systems_$B.log
done
bash $SRC/pilot_polecat/pc_recb_official.sh 16 False 2>&1 | tee recb_official.log     # needs requirements-optional.txt (ray)
echo "aggregation: python3 $SRC/pilot_polecat/pc_agg.py --root <dir with bundle_<base>/ from pc_chain.sh> --rule $R/registration/pilots/polecat_decision_rule.json --out polecat_pilot_verdict.json"
