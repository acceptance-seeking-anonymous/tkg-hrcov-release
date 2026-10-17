#!/bin/bash
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v7; mkdir -p $V/code $V/extract
# (code staging step of the original run removed: code is read from $SRC)
env OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  python3 -u $SRC/v7/v7_extract.py --out $J/x > extract.log 2>&1; echo "extract rc=$?" >> chain.log
mkdir -p $J/f
cpy() { s=$(stat -c %s "$1" 2>/dev/null || echo 0); if [ "$s" -gt 0 ] && [ "$s" -lt 99000000 ]; then cp "$1" "$J/f/$2"; else echo "SKIP $1 $s" >> chain.log; fi; }
cpy $H/work/blend/official_test_smallpedia_tirgn.npz off_smallpedia_tirgn_ctrl.npz
cpy $H/work/blend/sel_smallpedia.json sel_smallpedia.json
cpy $H/work/blend/blend_test_smallpedia_tirgn.json bt_smallpedia_tirgn_ctrl.json
cpy $H/presub_hcpat2/official_test_smallpedia_tirgn_hcpat.npz off_smallpedia_tirgn_hcpat.npz
cpy $H/presub_hcpat2/official_test_polecat_logcl_hcpat.npz off_polecat_logcl_hcpat.npz
for d in icews18 gdelt; do for b in tirgn regcn logcl; do cpy $H/presub_g2/official_test_${d}_${b}.npz off_${d}_${b}.npz; cpy $H/presub_g2/blend_test_${d}_${b}.json bt_${d}_${b}.json; done; cpy $H/presub_g2/sel_${d}.json sel_${d}.json; done
cpy $H/presub_hcpat/train_polecat_tirgn/ranks_test.npz rk_polecat_tirgn_hcpat_test.npz; cpy $H/presub_hcpat/train_polecat_tirgn/ranks_select.npz rk_polecat_tirgn_hcpat_select.npz; cpy $H/presub_hcpat/train_polecat_tirgn/train_hcpat.json tr_polecat_tirgn_hcpat.json
for x in polecat_logcl smallpedia_tirgn; do cpy $H/presub_hcpat2/train_$x/ranks_test.npz rk_${x}_hcpat_test.npz; cpy $H/presub_hcpat2/train_$x/ranks_select.npz rk_${x}_hcpat_select.npz; cpy $H/presub_hcpat2/train_$x/train_hcpat.json tr_${x}_hcpat.json; done
cpy $T/pc_tirgn/work/train_tirgn/ranks_test.npz rk_polecat_tirgn_r1_test.npz; cpy $T/pc_tirgn/work/train_tirgn/train_tirgn.json tr_polecat_tirgn_r1.json
cpy $T/pilot_smallpedia/work/train_tirgn/ranks_test.npz rk_smallpedia_tirgn_r1_test.npz; cpy $T/pilot_smallpedia/work/train_tirgn/train_tirgn.json tr_smallpedia_tirgn_r1.json
for D in ICEWS18 GDELT; do for b in tirgn regcn logcl; do cpy $H/presub_g2/$D/work/train_$b/ranks_test.npz rk_${D}_${b}_test.npz; cpy $H/presub_g2/$D/work/train_$b/train_g2.json tr_${D}_${b}.json; ls $H/presub_g2/$D/work/train_$b/ >> chain.log; done; done
cpy $H/work/recb/recb_test_ranks_tkgl-polecat.npz recb_rk_polecat.npz; cpy $H/work/recb/recb_test_ranks_tkgl-smallpedia.npz recb_rk_smallpedia.npz
cpy $H/presub_g2/recb/recb_test_ranks_ICEWS18.npz recb_rk_icews18.npz; cpy $H/presub_g2/recb/recb_test_ranks_GDELT.npz recb_rk_gdelt.npz
ls -la $J/x $J/f >> chain.log
echo ALLDONE >> chain.log
