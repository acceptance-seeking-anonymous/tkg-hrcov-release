#!/bin/bash
# v76 phase 2b on serverA (v76_rule.json caab7495 + amendment 1 498d56ba).  Re-runs after phase 2 (job, cancelled):
#  B1 (GPU0, 8 thr): M4 G1_xf features + train + official (re-run after CUDA OOM).
#  B2 (GPU0): official passes of the v76_btrain arms on polecat TiRGN (corrected TiRGN tree path).
#  C  (CPU 16 thr, after B1): LambdaMART v2 (tie-exact statistic) on the four M2b bases -> lm2_<base>; each followed by an
#     official pass of its fused lists (base tag v76_<base>_lm2).
#  D  (after C): M3a secondary, HRCOV_PAT on the validation-trained base (sp lists / features, v7 MLP64_PAT recipe, official).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v76
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $MNT"
GPU1T="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2 V7_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $MNT"
TGB=""
LGB=""
TREE=$T/pilot_smallpedia/stage/TiRGN
PCF=$T/pc_tirgn/work/feat_tirgn; PCL=$T/pc_tirgn/work/lists_tirgn; SPF=$T/pilot_smallpedia/work/feat_tirgn
CGF=$H/g8_cogntke/pc/work/feat_cogn; CGL=$H/g8_cogntke/pc/work/lists_cogn; RTF=$H/ablations_A/inputs/sp_regcn_tgb_feat
SPROWS=$T/pilot_smallpedia/dumps/tirgn_valid_logp.f32,$T/pilot_smallpedia/dumps/tirgn_test_logp.f32
RTROWS=/nonexistent/UNUSED_IN_TEST_STAGE,$H/ablations_A/inputs/sp_regcn_tgb_rows/regcn_test_logp.f32   # valid slot is never opened in --stage test (hd_blend.py:280,333; fullrow_source 229-241); any read fails loudly
srcargs() { case $1 in
  polecat_tirgn) echo "--ds polecat --source tirgn --tirgn_tree $TREE --ckpt $H/tirgn_polecat/tirgn_POLECAT_seed42.pt";;
  polecat_cogn) echo "--ds polecat --source cogn --ckpt $H/g8_cogntke/dumps";;
  smallpedia_tirgn) echo "--ds smallpedia --source fullrows --fullrows $SPROWS";;
  smallpedia_regcn_tgb) echo "--ds smallpedia --source fullrows --fullrows $RTROWS";; esac; }
sysarg() { ls "$@" 2>/dev/null | sort | awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}'; }
# ------------------------------------------------------------------ B1: G1_xf
(
  for DS in polecat smallpedia; do
    if [ $DS = polecat ]; then E="PC_P=$T/pilot_polecat PYTHONPATH=$SRC/pilot_polecat:$PYTHONPATH"; CM=pc_common; else E="PC_P=$T/pilot_smallpedia PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"; CM=sp_common; fi
    env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $E $MNT \
      python3 -u $SRC/presub/g1_features.py --ds $DS --common $CM --recb_dir $V/recb_xf --out $V/m4/g1f_$DS --threads 8 > m4_g1f_$DS.log 2>&1; tm m4_g1f_$DS $?
    cp $V/m4/g1f_$DS/g1_features_summary.json $J/m4_g1_features_summary_$DS.json 2>/dev/null
    $GPU python3 -u $SRC/presub/g1_train.py --feat $V/m4/g1f_$DS --out $V/m4/g1t_$DS > m4_g1t_$DS.log 2>&1; tm m4_g1t_$DS $?
    cp $V/m4/g1t_$DS/train_g1.json $J/m4_train_g1xf_$DS.json 2>/dev/null
  done
  echo "G1XF_FEATURES_TRAIN_DONE" >> chain.log
  for DS in polecat smallpedia; do
    R=$V/m4/g1t_$DS; SYS=""; for arm in G1_K50 G1_K150; do for s in 0 1 2; do SYS="$SYS,${arm}_s$s=$R/promlist_${arm}_s$s.npy"; done; done; SYS=${SYS#,}
    $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds $DS --base recb_g1xf --stage test --source recb --no_controls --recb_dir $V/recb_xf --systems $SYS --out $V/m4" > m4_official_$DS.log 2>&1; tm m4_official_$DS $?
    cp $V/m4/official_test_${DS}_recb_g1xf.npz $V/m4/blend_test_${DS}_recb_g1xf.json $J/ 2>/dev/null
  done
  echo "TRACK_B1_DONE" >> chain.log
) &
PB1=$!
# ------------------------------------------------------------------ B2: polecat TiRGN official passes of the v76_btrain arms
(
  L=$(ls $V/m2b/bt_polecat_tirgn/promlist_*_fused_s*.npy | sort); n=$(echo "$L" | wc -l); half=$(( (n + 1) / 2 ))
  g=0
  for chunk in "$(echo "$L" | head -n $half)" "$(echo "$L" | tail -n +$((half + 1)))"; do
    g=$((g + 1)); SYS=$(echo "$chunk" | awk '{n=split($0,a,"/"); f=a[n]; sub(/^promlist_/,"",f); sub(/\.npy$/,"",f); printf "%s%s=%s", (NR>1?",":""), f, $0}')
    ( $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py $(srcargs polecat_tirgn) --base v76_polecat_tirgn_g$g --stage test --no_controls --recb_dir $H/work/recb --bs 256 --systems $SYS --out $V/official" > off_polecat_tirgn_g$g.log 2>&1; tm off_polecat_tirgn_g$g $? ) &
    sleep 60
  done
  wait; echo "TRACK_B2_DONE" >> chain.log
) &
PB2=$!
# ------------------------------------------------------------------ C: LambdaMART v2 (after B1's CPU-heavy part)
until grep -q "G1XF_FEATURES_TRAIN_DONE" chain.log 2>/dev/null; do sleep 30; done
for B in polecat_tirgn smallpedia_tirgn polecat_cogn smallpedia_regcn_tgb; do
  case $B in polecat_tirgn) LA="--layout pc --feat $PCF --lists $PCL";; smallpedia_tirgn) LA="--layout sp --feat $SPF";;
             polecat_cogn) LA="--layout pc --feat $CGF --lists $CGL";; smallpedia_regcn_tgb) LA="--layout sp --feat $RTF";; esac
  env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $MNT \
    bash -c "$LGB python3 -u $SRC/v76/v76_lmart.py $LA --out $V/m2b/lm2_$B --label $B --threads 16 --mlp_model $V/m2b/bt_$B/models/MLP64_PAT_s0.pt" > lm2_$B.log 2>&1; tm lm2_$B $?
  cp $V/m2b/lm2_$B/train_lmart.json $J/train_lmart2_$B.json 2>/dev/null
  SYS=$(sysarg $V/m2b/lm2_$B/promlist_LMART_PAT_fused_s*.npy)
  ( $GPU1T bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py $(srcargs $B) --base v76_${B}_lm2 --stage test --no_controls --recb_dir $H/work/recb --bs 256 --systems $SYS --out $V/official" > off_${B}_lm2.log 2>&1; tm off_${B}_lm2 $? ) &
  sleep 20
done
# ------------------------------------------------------------------ D: M3a secondary
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
wait
# ------------------------------------------------------------------ collect
cp $V/official/*.npz $V/official/*.json $J/ 2>/dev/null
cp $V/m3a/*.json $V/m3a/official_test_*.npz $J/ 2>/dev/null; cp $V/m3a/sp/dumps/tirgn_SMALLPEDIA_dump_log.json $J/m3a_tirgn_valfit_dump_log.json 2>/dev/null
for d in $V/m2b/bt_*; do cp $d/train_v7.json $J/train_v76_$(basename $d).json; cp $d/ranks_test.npz $J/ranks_test_$(basename $d).npz; done
for d in $V/m2b/lm2_*; do cp $d/ranks_test.npz $J/ranks_test_$(basename $d).npz; done
cp $V/m4/g1t_*/train_g1.json $J/ 2>/dev/null; cp $V/recb_xf/recb_selection_*.json $V/recb_xf/recb_test_meta_*.json $J/ 2>/dev/null
for f in off_*.log m3a_*.log m4_*.log lm2_*.log; do echo "== $f" >> chain.log; tail -2 $f >> chain.log; done
echo ALLDONE >> chain.log
