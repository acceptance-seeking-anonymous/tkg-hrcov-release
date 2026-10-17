#!/bin/bash
# scripts/run_g8_cogntke.sh -- CognTKE base on tkgl-polecat (registration/g8_rule.json + g8_deviation_B.json): setup, training
# in the authors' released configuration (20 epochs, valid-best; ~2 days on one RTX PRO 6000, ~51 GB GPU memory), dump,
# pilot-1 pipeline on base 'cogn', controls, one official test pass, aggregation.  Needs the TiRGN polecat track
# (run_hardening.sh tirgn_polecat) for the shared prep files.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir g8_cogntke
bash $SRC/g8_cogntke/g8_setup.sh
bash $SRC/g8_cogntke/g8_train.sh B 20
bash $SRC/g8_cogntke/g8_chain.sh dump
python3 $R/configs/aggregation/expand_cfg.py $R/configs/aggregation/g8_agg_cfg.json g8_agg_cfg.json
python3 $SRC/g8_cogntke/g8_agg.py --cfg g8_agg_cfg.json --out g8_verdict.json --csv g8_table.csv
