#!/bin/bash
# scripts/run_finiteness.sh -- read-only NaN / inf scan of every stored score and probability array (common/finite_scan.sh,
# g8_finite.sh, c4_finite.sh) and the rescoring check (common/rescore.sh); CPU, ~1 h.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rundir finiteness
bash $SRC/common/finite_scan.sh; bash $SRC/common/g8_finite.sh; bash $SRC/common/c4_finite.sh; bash $SRC/common/rescore.sh
