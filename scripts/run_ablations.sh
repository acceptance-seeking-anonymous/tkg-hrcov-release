#!/bin/bash
# scripts/run_ablations.sh -- leave-one-feature-group-out ablation (post-hoc; registration/ablation_rule.json) on the four
# decision bases: HRCOV_PAT refits with the FULL / A10_noBase / A11_noEvid / A11_noCov column sets (ablations/abl_train.py,
# 3 seeds), one official test pass per base (hardening/hd_blend.py --no_controls), aggregation (ablations/abl_agg.py with
# configs/aggregation/abl_cfg.json).  Needs the feature directories of the hardening / pilot runs.  About 6 h on one GPU.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
T=$WORK_DIR/tracks; H=$T/hardening; A=$H/ablations_A; mkdir -p $A/results; rundir ablations
GPUE="env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 HD_HAR=$WORK_DIR HD_HOST=local"
declare -A LA=( [polecat_tirgn]="--layout pc --feat $T/pc_tirgn/work/feat_tirgn --lists $T/pc_tirgn/work/lists_tirgn"
                [polecat_logcl]="--layout pc --feat $T/pilot_polecat/work/feat_logcl --lists $T/pilot_polecat/work/lists_logcl"
                [smallpedia_tirgn]="--layout sp --feat $T/pilot_smallpedia/work/feat_tirgn"
                [smallpedia_regcn_tgb]="--layout sp --feat $A/inputs/sp_regcn_tgb_feat" )
declare -A BL=( [polecat_tirgn]="--ds polecat --source tirgn --tirgn_tree $H/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt"
                [polecat_logcl]="--ds polecat --source replay --ckpt $T/pilot_polecat/dumps/logcl_POLECAT_seed42.pt --L 10"
                [smallpedia_tirgn]="--ds smallpedia --source fullrows --fullrows none,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32"
                [smallpedia_regcn_tgb]="--ds smallpedia --source fullrows --fullrows none,$A/inputs/sp_regcn_tgb_rows/regcn_test_logp.f32" )
for B in polecat_tirgn polecat_logcl smallpedia_tirgn smallpedia_regcn_tgb; do
  $GPUE python3 -u $SRC/ablations/abl_train.py ${LA[$B]} --out $A/results/train_$B --label $B 2>&1 | tee abl_$B.log
  SYS=""; for a in FULL A10_noBase A11_noEvid A11_noCov; do for s in 0 1 2; do SYS="$SYS,${a}_s$s=$A/results/train_$B/promlist_${a}_s$s.npy"; done; done
  $GPUE python3 -u $SRC/hardening/hd_blend.py ${BL[$B]} --base abl_${B#*_} --stage test --no_controls --recb_dir $H/work/recb \
    --systems ${SYS#,} --out $A/results 2>&1 | tee official_$B.log
done
python3 $R/configs/aggregation/expand_cfg.py $R/configs/aggregation/abl_cfg.json abl_cfg.json
python3 -u $SRC/ablations/abl_agg.py --cfg abl_cfg.json --out ablations_A.json --csv ablations_A_table.csv
