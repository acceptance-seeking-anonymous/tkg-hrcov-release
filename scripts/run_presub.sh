#!/bin/bash
# scripts/run_presub.sh <item> [smoke_n] -- registration/a_presub_rule.json items.
#   g1      base-free learned ranker over RecB candidates (presub/g1_run.sh; RTX 4090 + CPU, ~6 h)
#   g2      ICEWS18 / GDELT with three frozen bases (presub/g2_run.sh, then g2_verify.sh; CPU, ~1 day)
#   g3      TGB 2.0 RE-GCN configuration on tkgl-smallpedia (presub/g3_run.sh; RTX 4090, ~3 h)
#   hcpat   bridge-free HRCOV_PAT on every decision base (hcpat_polecat_tirgn.sh, hcpat_smallpedia_regcn_*.sh, hcpat2.sh)
#   g7      cluster bootstraps (whole timestamps / whole facts) over the hardening outputs (CPU, ~1 h)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
I=$1; SM=${2:-0}; rundir presub_$I
case "$I" in
  g1) bash $SRC/presub/g1_run.sh $SM ;;
  g2) declare -A DDIR=( [ICEWS18]=$EXT_DIR/logcl_redun/data/ICEWS18 [GDELT]=$EXT_DIR/gdelt_armAD/data/GDELT )
      for DS in ICEWS18 GDELT; do for M in regcn logcl; do      # frozen RE-GCN / LogCL bases (harness defaults) + full-score rows
        [ -f $WORK_DIR/dumps/${M}_${DS}_seed42.pt ] || env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/harness/base_dump.py --model $M \
          --data_dir ${DDIR[$DS]} --dataset $DS --out_dir $WORK_DIR/dumps --seed 42
        env CUDA_VISIBLE_DEVICES=$GPU python3 -u $SRC/presub/fulldump_regcn_logcl.py --model $M --dataset $DS --out $WORK_DIR/tracks/retriever/fullscores
      done; done
      # TiRGN on ICEWS18 / GDELT: score dumps of the TiRGN checkpoints (EXT_DIR/logcl_redun, EXT_DIR/gdelt_tirgn_dump; data/README.md)
      bash $SRC/presub/g2_run.sh $SM; bash $SRC/presub/g2_verify.sh
      python3 $SRC/presub/g2_agg_cfg.py; python3 $SRC/presub/g2_agg.py --cfg g2_agg_cfg.json --rule $R/registration/a_presub_rule.json --out g2_verdict.json ;;
  g3) cp -n $STAGE_DIR/g3_tgb_bits.tgz $WORK_DIR/tracks/datasets/raw/tkgl-smallpedia.zip . ; bash $SRC/presub/g3_run.sh $([ "$SM" -gt 0 ] && echo 1 || echo 0) ;;
  hcpat) bash $SRC/presub/hcpat_polecat_tirgn.sh; bash $SRC/presub/hcpat_smallpedia_regcn_retuned.sh
         bash $SRC/presub/hcpat_smallpedia_regcn_tgb.sh; bash $SRC/presub/hcpat2.sh ;;
  g7) need $WORK_DIR/runs/hardening_aggregate/hardening_verdict_raw.json
      python3 $R/configs/aggregation/expand_cfg.py $R/configs/aggregation/hardening_agg_cfg.json agg_cfg.json
      python3 $SRC/presub/g7_cluster_bootstrap.py --cfg agg_cfg.json --verdict $WORK_DIR/runs/hardening_aggregate/hardening_verdict_raw.json --out g7_cluster_bootstrap.json ;;
  *) echo "usage: $0 g1|g2|g3|hcpat|g7 [smoke_n]"; exit 2 ;;
esac
