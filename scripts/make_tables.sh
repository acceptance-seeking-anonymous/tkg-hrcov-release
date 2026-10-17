#!/bin/bash
# scripts/make_tables.sh -- regenerate the main table, the base-strength figure data, the v7.6 numbers and the figures from
# results/ and compare the numbers with the manuscript (CPU, < 1 min).  Outputs: results/tables/, results/figures/.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$R"
python3 analysis/check_manuscript_numbers.py || exit 1
for f in analysis/figures/fig_*.py analysis/figures/figA3_absolute_mrr.py; do python3 "$f" || exit 1; done
echo "tables: results/tables/  figures: results/figures/"
