#!/bin/bash
# scripts/run_hardening.sh <step> -- the main test (registration/hardening_decision_rule.json): extra bases, RecB and the
# BASE(+)RecB / RecB-POOL controls, HR / HRCOV on every base, official scoring, aggregation.
#   tirgn_polecat   TiRGN on tkgl-polecat (authors' configuration, L = 10) + lists/lookup/features/HR/HRCOV + controls
#                   (hardening/tirgn_pc.sh, tirgn_lookup.sh, tirgn_pc_feat.sh, tirgn_pc_blend.sh, pc_tirgn_chain.sh; ~10 h)
#   sp_new          new tkgl-smallpedia bases (re-tuned RE-GCN / LogCL grid, hd_base_train.py) + pipeline (sp_new_chain*.sh)
#   recb            RecB per-relation grid on validation and test-stage blend inputs (hd_recb.py; polecat ~3.5 h, 16 CPUs)
#   aggregate       hd_agg.py over configs/aggregation/hardening_agg_cfg.json -> hardening_verdict_raw.json (~10 min, CPU)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
H=$WORK_DIR/tracks/hardening; mkdir -p $H/work/recb; rundir hardening_$1
case "$1" in
  tirgn_polecat) bash $SRC/hardening/tirgn_pc.sh; bash $SRC/hardening/tirgn_lookup.sh; bash $SRC/hardening/tirgn_pc_feat.sh
                 bash $SRC/hardening/tirgn_pc_blend.sh; bash $SRC/hardening/pc_tirgn_chain.sh ;;
  sp_new)        bash $SRC/hardening/sp_new_chain_part1.sh; bash $SRC/hardening/sp_new_chain.sh; bash $SRC/hardening/sp_new_chain_logcl.sh ;;
  recb)          for DS in tkgl-polecat tkgl-smallpedia; do for ST in valid test; do
                   python3 -u $SRC/hardening/hd_recb.py --data_dir $WORK_DIR/tracks/datasets/harness_data/$DS --name $DS --stage $ST --workers 16 --P 16 --out $H/work/recb
                 done; done ;;
  aggregate)     python3 $R/configs/aggregation/expand_cfg.py $R/configs/aggregation/hardening_agg_cfg.json agg_cfg.json
                 python3 $SRC/hardening/hd_agg.py --cfg agg_cfg.json --rule $R/registration/hardening_decision_rule.json --out hardening_verdict_raw.json ;;
  *) echo "usage: $0 tirgn_polecat|sp_new|recb|aggregate"; exit 2 ;;
esac
