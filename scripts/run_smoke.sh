#!/bin/bash
# scripts/run_smoke.sh [N] -- end-to-end smoke test (tests/smoke/run_smoke.sh); ~12 min on one RTX PRO 6000 (needs > 24 GB GPU memory).
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "$R/tests/smoke/run_smoke.sh" "${1:-3000}"
