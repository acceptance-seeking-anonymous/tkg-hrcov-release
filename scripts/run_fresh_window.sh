#!/bin/bash
# scripts/run_fresh_window.sh <item> -- registered comparison on the uninspected 2026 GDELT window (fresh_window/).
#   verify      check the released package against FILES.md, the freeze manifest and the registration copies
#   localize D  copy the runnable parts to D with the hard-coded paths pointed at $H, $FW, $B (fresh_window/fw_env.sh)
# The runs themselves follow fresh_window/FILES.md, "Order of commands to reproduce", from the localized copy.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
case "$1" in
  verify) python3 $R/fresh_window/verify_fresh_window.py ;;
  localize) . $R/fresh_window/fw_env.sh && bash $R/fresh_window/fw_localize.sh "${2:?destination directory}" ;;
  *) echo "usage: $0 verify | localize <dir>"; exit 2 ;;
esac
