#!/bin/bash
# presub / g3_resume.sh -- after the dump OOM: smoke-check the row-blocked g3_dump.py, then re-run the full G3 chain from the dump
# stage with the already-trained TGB checkpoint (dumps/tgb_regcn_ckpt.pt is kept; TGB training is not repeated).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
bash g3_run.sh 1 0
OK=$(python3 -c "
import json
r=json.load(open('g3_tgb_run_smoke.json'))['test_mrr']; c=json.load(open('blend_test_smallpedia_regcn_tgb_check_smoke.json'))['metrics']['base']['official']['MRR']
d=json.load(open('g3_dump_log_smoke.json')); v=json.load(open('g3_tgb_run_smoke.json'))['val_mrr']
print(int(abs(r-c) < 1e-5 and abs(d['valid_filt']['MRR']-v) < 1e-5 and max(x['max_abs_logit_diff'] for x in d['per_query_path_check'].values()) < 1e-4))" 2>/dev/null | tail -1)   # the image prints a CUDA banner first
echo "smoke_check_ok=$OK" >> chain.log
[ "$OK" = "1" ] && bash g3_run.sh 0 1
echo RESUME_DONE >> chain.log
