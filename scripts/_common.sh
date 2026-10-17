# scripts/_common.sh -- sourced by every entry point
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$R/configs/env.sh"
set -o pipefail
rundir() { mkdir -p "$WORK_DIR/runs/$1" && cd "$WORK_DIR/runs/$1" && echo "run directory: $PWD"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing input: $f (see README, step order)"; exit 2; }; done; }
