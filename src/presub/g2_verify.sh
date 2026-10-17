#!/bin/bash
# presub / g2_verify.sh -- pre-registered full-row verification (all valid + test rows) for the 6 G2 bases, 1 CPU each.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD; H=$WORK_DIR; HD=$H/tracks/hardening; G=$HD/presub_g2_verify; mkdir -p $G/code
# (code staging step of the original run removed: code is read from $SRC)
uptime > load_at_start.txt
for DS in ICEWS18 GDELT; do for B in tirgn regcn logcl; do
  env OMP_NUM_THREADS=1 G2_DS=$DS PC_P=$G/$DS PYTHONDONTWRITEBYTECODE=1 HD_HOST=local \
    \
    python3 -u $SRC/presub/g2_verify.py --base $B --out verify_${DS}_$B.json > verify_${DS}_$B.log 2>&1 &
done; done
wait
echo ALLDONE > done.txt
