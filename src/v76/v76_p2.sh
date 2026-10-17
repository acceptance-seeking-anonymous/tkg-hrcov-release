#!/bin/bash
# v76 phase 2 on serverA (v76_rule.json caab7495).  GPU 0 only (GPU 1 belongs to others); CPU <= ~16.
#  Track A (GPU0): M3a validation-trained TiRGN smallpedia (smoke, full train + dump), controls selected on validation-select,
#                  official test pass of base / BLEND / POOL.
#  Track B (GPU0, 8 thr): M4 G1_xf features + train (polecat, smallpedia), official G1_xf pass (RecB_xf rows).
#  Track C: v76_btrain arms on the four M2b bases (GPU0), then LambdaMART per base (CPU, 16 threads, sequential), each
#           followed by its official pass (GPU0, background).  Then M3a secondary (HRCOV_PAT on the validation-trained base).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v76
mkdir -p $V/code $V/m3a/sp/dumps $V/m4 $V/m2b $V/official
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
GPU1T="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 V7_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
TGB=""
LGB=""
DDS=$T/datasets/harness_data/tkgl-smallpedia
SPE="env PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"
PCF=$T/pc_tirgn/work/feat_tirgn; PCL=$T/pc_tirgn/work/lists_tirgn; SPF=$T/pilot_smallpedia/work/feat_tirgn
CGF=$H/g8_cogntke/pc/work/feat_cogn; CGL=$H/g8_cogntke/pc/work/lists_cogn; RTF=$H/ablations_A/inputs/sp_regcn_tgb_feat
SPROWS=$T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32
RTROWS=/nonexistent/UNUSED_IN_TEST_STAGE,$H/ablations_A/inputs/sp_regcn_tgb_rows/regcn_test_logp.f32   # valid slot is never opened in --stage test (hd_blend.py:280,333; fullrow_source 229-241); any read fails loudly
# ------------------------------------------------------------------ Track A: M3a
(
  M=$V/m3a; P=$M/sp
  env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PC_P=$M/smoke TIRGN_TREE=$T/pilot_smallpedia/stage/TiRGN $SPE $MNT \
    bash -c "mkdir -p $M/smoke/dumps && cd $M/smoke && python3 -u $SRC/v76/sp_tirgn_valfit.py --smoke 3 --fit_cut 105" > m3a_smoke.log 2>&1; tm m3a_smoke $?
  if ! grep -q "ALL DONE" m3a_smoke.log; then echo "M3a BLOCKED at smoke" >> chain.log; tail -20 m3a_smoke.log >> chain.log; exit 0; fi
  rm -f $M/smoke/dumps/*.f32
  env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PC_P=$P TIRGN_TREE=$T/pilot_smallpedia/stage/TiRGN $SPE $MNT \
    bash -c "cd $P && python3 -u $SRC/v76/sp_tirgn_valfit.py --fit_cut 105" > m3a_train.log 2>&1; tm m3a_train $?
  cp $P/dumps/tirgn_SMALLPEDIA_dump_log.json $J/m3a_tirgn_valfit_dump_log.json 2>/dev/null
  if ! grep -q "ALL DONE" m3a_train.log; then echo "M3a BLOCKED at train (recipe/memory)" >> chain.log; tail -30 m3a_train.log >> chain.log; exit 0; fi
  VR=$P/dumps/tirgn_valid_logp.f32,$P/dumps/tirgn_test_logp.f32
  $GPU1T python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_valfit --stage valid --source fullrows --fullrows $VR --recb_dir $H/work/recb --out $M > m3a_blend_valid.log 2>&1; tm m3a_blend_valid $?
  $GPU1T python3 -u $SRC/v76/v76_select_sub.py --dd $DDS --ds smallpedia --base tirgn_valfit --dir $M --cut 105 --out $M/sel_smallpedia_valfit.json > m3a_select.log 2>&1; tm m3a_select $?
  $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_valfit --stage test --source fullrows --fullrows $VR --recb_dir $H/work/recb --sel $M/sel_smallpedia_valfit.json --out $M" > m3a_blend_test.log 2>&1; tm m3a_blend_test $?
  cp $M/blend_valid_smallpedia_tirgn_valfit.json $M/sel_smallpedia_valfit.json $M/blend_test_smallpedia_tirgn_valfit.json $M/official_test_smallpedia_tirgn_valfit.npz $J/ 2>/dev/null
  echo "TRACK_A_DONE" >> chain.log
) &
PA=$!
# ------------------------------------------------------------------ Track B: M4 G1_xf
(
  for DS in polecat smallpedia; do
    if [ $DS = polecat ]; then E="PC_P=$T/pilot_polecat PYTHONPATH=$SRC/pilot_polecat:$PYTHONPATH"; CM=pc_common; else E="PC_P=$T/pilot_smallpedia PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"; CM=sp_common; fi
    env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $E $MNT \
      python3 -u $SRC/presub/g1_features.py --ds $DS --common $CM --recb_dir $V/recb_xf --out $V/m4/g1f_$DS --threads 8 > m4_g1f_$DS.log 2>&1; tm m4_g1f_$DS $?
    cp $V/m4/g1f_$DS/g1_features_summary.json $J/m4_g1_features_summary_$DS.json 2>/dev/null
    $GPU python3 -u $SRC/presub/g1_train.py --feat $V/m4/g1f_$DS --out $V/m4/g1t_$DS > m4_g1t_$DS.log 2>&1; tm m4_g1t_$DS $?
    cp $V/m4/g1t_$DS/train_g1.json $J/m4_train_g1xf_$DS.json 2>/dev/null
  done
  for DS in polecat smallpedia; do
    R=$V/m4/g1t_$DS; SYS=""; for arm in G1_K50 G1_K150; do for s in 0 1 2; do SYS="$SYS,${arm}_s$s=$R/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
    $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds $DS --base recb_g1xf --stage test --source recb --no_controls --recb_dir $V/recb_xf --systems $SYS --out $V/m4" > m4_official_$DS.log 2>&1; tm m4_official_$DS $?
    cp $V/m4/official_test_${DS}_recb_g1xf.npz $V/m4/blend_test_${DS}_recb_g1xf.json $J/ 2>/dev/null
  done
  echo "TRACK_B_DONE" >> chain.log
) &
PB=$!
# ------------------------------------------------------------------ Track C: M2b (+ M4 RecB-feature arms) re-ranker fits
M4ARMS=MLP64_PAT,LIN_PAT,HR19,HR19_RBtr,HR19_RBxf,PAT_RBtr,PAT_RBxf
$GPU python3 -u $SRC/v76/v76_btrain.py --layout pc --feat $PCF --lists $PCL --rbtr $V/rbf/polecat_tirgn_tr --rbxf $V/rbf/polecat_tirgn_xf --out $V/m2b/bt_polecat_tirgn --label polecat_tirgn --arms $M4ARMS > bt_polecat_tirgn.log 2>&1; tm bt_polecat_tirgn $?
$GPU python3 -u $SRC/v76/v76_btrain.py --layout sp --feat $SPF --rbtr $V/rbf/smallpedia_tirgn_tr --rbxf $V/rbf/smallpedia_tirgn_xf --out $V/m2b/bt_smallpedia_tirgn --label smallpedia_tirgn --arms $M4ARMS > bt_smallpedia_tirgn.log 2>&1; tm bt_smallpedia_tirgn $?
$GPU python3 -u $SRC/v76/v76_btrain.py --layout pc --feat $CGF --lists $CGL --out $V/m2b/bt_polecat_cogn --label polecat_cogn --arms MLP64_PAT,LIN_PAT > bt_polecat_cogn.log 2>&1; tm bt_polecat_cogn $?
$GPU python3 -u $SRC/v76/v76_btrain.py --layout sp --feat $RTF --out $V/m2b/bt_smallpedia_regcn_tgb --label smallpedia_regcn_tgb --arms MLP64_PAT,LIN_PAT > bt_smallpedia_regcn_tgb.log 2>&1; tm bt_smallpedia_regcn_tgb $?
cp $V/m2b/bt_*/train_v7.json $J/ 2>/dev/null; for d in $V/m2b/bt_*; do cp $d/train_v7.json $J/train_v76_$(basename $d).json 2>/dev/null; done
wait $PB    # CPU cap: G1 feature threads finish before LambdaMART takes 16 threads
offpass() {   # $1 base key, $2 dir list of promlist globs ; official pass(es) in background
  local B=$1; shift; local L=$(ls "$@" 2>/dev/null | sort); local n=$(echo "$L" | wc -l); local half=$(( (n + 1) / 2 ))
  local g=0
  for chunk in "$(echo "$L" | head -n $half)" "$(echo "$L" | tail -n +$((half + 1)))"; do
    [ -z "$chunk" ] && continue; g=$((g + 1))
    SYS=$(echo "$chunk" | awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}')
    case $B in
      polecat_tirgn) A="--ds polecat --source tirgn --tirgn_tree $H/stage/TiRGN --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt";;
      polecat_cogn) A="--ds polecat --source cogn --ckpt $H/g8_cogntke/dumps";;
      smallpedia_tirgn) A="--ds smallpedia --source fullrows --fullrows $SPROWS";;
      smallpedia_regcn_tgb) A="--ds smallpedia --source fullrows --fullrows $RTROWS";;
    esac
    ( $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py $A --base v76_${B}_g$g --stage test --no_controls --recb_dir $H/work/recb --bs 256 --systems $SYS --out $V/official" > off_${B}_g$g.log 2>&1; tm off_${B}_g$g $?
      cp $V/official/blend_test_*_v76_${B}_g$g.json $V/official/official_test_*_v76_${B}_g$g.npz $J/ 2>/dev/null ) &
    sleep 30
  done
}
for B in polecat_tirgn smallpedia_tirgn polecat_cogn smallpedia_regcn_tgb; do
  case $B in polecat_tirgn) LA="--layout pc --feat $PCF --lists $PCL";; smallpedia_tirgn) LA="--layout sp --feat $SPF";;
             polecat_cogn) LA="--layout pc --feat $CGF --lists $CGL";; smallpedia_regcn_tgb) LA="--layout sp --feat $RTF";; esac
  env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT \
    bash -c "$LGB python3 -u $SRC/v76/v76_lmart.py $LA --out $V/m2b/lm_$B --label $B --threads 16 --mlp_model $V/m2b/bt_$B/models/MLP64_PAT_s0.pt" > lm_$B.log 2>&1; tm lm_$B $?
  cp $V/m2b/lm_$B/train_lmart.json $J/train_lmart_$B.json 2>/dev/null
  offpass $B $V/m2b/bt_$B/promlist_*_fused_s*.npy $V/m2b/lm_$B/promlist_LMART_PAT_fused_s*.npy
done
# ------------------------------------------------------------------ M3a secondary: HRCOV_PAT on the validation-trained base
wait $PA
if grep -q TRACK_A_DONE chain.log; then
  P=$V/m3a/sp; mkdir -p $P/work $P/stage
  for f in prep_summary.json valid_sample_idx.npy te_SMALLPEDIA.npz ste_SMALLPEDIA.npz strata_valid.npz strata_test.npz recb_valid_rank.npy recb_test_rank.npy reach_sample.npz; do [ -f $P/work/$f ] || cp $T/pilot_smallpedia/work/$f $P/work/; done
  [ -d $P/work/tgb_ns ] || cp -r $T/pilot_smallpedia/work/tgb_ns $P/work/
  for f in $T/pilot_smallpedia/stage/*.npz $T/pilot_smallpedia/stage/*.json; do [ -f $f ] && cp $f $P/stage/; done
  PEN="env PC_P=$P PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1"
  env OMP_NUM_THREADS=1 $PEN $MNT python3 -u $SRC/pilot_smallpedia/sp_lists.py --base tirgn --workers 12 > m3a_lists.log 2>&1; tm m3a_lists $?
  env OMP_NUM_THREADS=4 $PEN $MNT python3 -u $SRC/pilot_smallpedia/sp_features.py --base tirgn > m3a_feat.log 2>&1; tm m3a_feat $?
  $GPU python3 -u $SRC/v76/v76_btrain.py --layout sp --feat $P/work/feat_tirgn --out $V/m3a/bt_valbase --label smallpedia_tirgn_valbase --arms MLP64_PAT > m3a_bt.log 2>&1; tm m3a_bt $?
  cp $V/m3a/bt_valbase/train_v7.json $J/train_v76_m3a_valbase.json 2>/dev/null
  SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_valbase_s$s=$V/m3a/bt_valbase/promlist_MLP64_PAT_fused_s$s.npy"; done; SYS=${SYS#,}
  $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_valfit_hcpat --stage test --source fullrows --fullrows $P/dumps/tirgn_valid_logp.f32,$P/dumps/tirgn_test_logp.f32 --no_controls --recb_dir $H/work/recb --systems $SYS --out $V/m3a" > m3a_off_hcpat.log 2>&1; tm m3a_off_hcpat $?
  cp $V/m3a/official_test_smallpedia_tirgn_valfit_hcpat.npz $V/m3a/blend_test_smallpedia_tirgn_valfit_hcpat.json $J/ 2>/dev/null
fi
wait
for f in off_*.log m3a_*.log m4_*.log lm_*.log bt_*.log; do echo "== $f" >> chain.log; tail -2 $f >> chain.log; done
echo ALLDONE >> chain.log
