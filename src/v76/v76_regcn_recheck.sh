#!/bin/bash
# v76 clarification check (2026-10-15): re-run the tkgl-smallpedia RE-GCN (TGB) official scoring passes with the unused
# validation slot replaced by a non-existent path (/nonexistent/UNUSED_IN_TEST_STAGE) and compare every output array with the
# original run byte for byte.  Waits for phase 2b (its lm2 pass) to finish.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; H=$WORK_DIR/tracks/hardening; V=$H/v76; OUT=$V/official_recheck; mkdir -p $OUT
# (orchestration wait of the original multi-job run removed)
RT=/nonexistent/UNUSED_IN_TEST_STAGE,$H/ablations_A/inputs/sp_regcn_tgb_rows/regcn_test_logp.f32
G="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local"
sysof() { awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}'; }
B=smallpedia_regcn_tgb
L=$(ls $V/m2b/bt_$B/promlist_*_fused_s*.npy $V/m2b/lm_$B/promlist_LMART_PAT_fused_s*.npy 2>/dev/null | sort); n=$(echo "$L" | wc -l); half=$(( (n + 1) / 2 ))
S1=$(echo "$L" | head -n $half | sysof); S2=$(echo "$L" | tail -n +$((half + 1)) | sysof); S3=$(ls $V/m2b/lm2_$B/promlist_LMART_PAT_fused_s*.npy | sort | sysof)
for g in g1 g2 lm2; do
  case $g in g1) SYS=$S1;; g2) SYS=$S2;; lm2) SYS=$S3;; esac
  $G bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --source fullrows --fullrows $RT --base v76_${B}_$g --stage test --no_controls --recb_dir $H/work/recb --bs 256 --systems $SYS --out $OUT" > recheck_$g.log 2>&1
  echo "rerun $g rc=$?" >> recheck.log
done
python3 - >> recheck.log <<PY
import numpy as np, hashlib, json
res = {}
for g in ("g1", "g2", "lm2"):
    a = "$V/official/official_test_smallpedia_v76_smallpedia_regcn_tgb_%s.npz" % g; b = "$OUT/official_test_smallpedia_v76_smallpedia_regcn_tgb_%s.npz" % g
    try:
        A = np.load(a); Bz = np.load(b)
        keys = sorted(set(A.files) | set(Bz.files))
        same = {k: bool(k in A.files and k in Bz.files and A[k].dtype == Bz[k].dtype and A[k].tobytes() == Bz[k].tobytes()) for k in keys}
        res[g] = dict(n_arrays=len(keys), all_identical=all(same.values()), differing=[k for k, v in same.items() if not v],
                      sha_rr_base=hashlib.sha256(Bz["rr_base"].astype(np.float32).tobytes()).hexdigest())
    except Exception as e:
        res[g] = dict(error=str(e))
json.dump(res, open("regcn_recheck.json", "w"), indent=1); print(json.dumps(res))
PY
echo done >> recheck.log
