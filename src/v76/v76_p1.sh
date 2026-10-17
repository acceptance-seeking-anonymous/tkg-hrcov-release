#!/bin/bash
# v76 phase 1 on serverA (v76_rule.json caab7495): M4 RecB_xf selection + RecB_xf rows, RecB features (tr / xf) on the PT pools,
# then smoke tests of every new code path (v76_btrain, v76_lmart, sp_tirgn_valfit, g1 on RecB_xf).  GPU 0 only; CPU <= 16.
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v76; mkdir -p $V/code $V/recb_xf $V/rbf $V/smoke
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
MNT=""
CPU="env OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
GPU="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 V7_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $MNT"
DDP=$T/datasets/harness_data/tkgl-polecat; DDS=$T/datasets/harness_data/tkgl-smallpedia
# ---- M4: RecB_xf selection (stored exact valid ranks) and RecB_xf rows / CSR
$CPU python3 -u $SRC/v76/v76_recb_xf_select.py --data_dir $DDP --cut 1423 --orig_dir $H/work/recb --out $V/recb_xf > xfsel_polecat.log 2>&1; tm xfsel_polecat $?
$CPU python3 -u $SRC/v76/v76_recb_xf_select.py --data_dir $DDS --cut 105 --orig_dir $H/work/recb --out $V/recb_xf > xfsel_smallpedia.log 2>&1; tm xfsel_smallpedia $?
$CPU python3 -u $SRC/hardening/hd_recb.py --data_dir $DDP --stage test --workers 16 --P 8 --out $V/recb_xf > recb_xf_polecat.log 2>&1; tm recb_xf_polecat $?
$CPU python3 -u $SRC/hardening/hd_recb.py --data_dir $DDS --stage test --workers 16 --P 16 --out $V/recb_xf > recb_xf_smallpedia.log 2>&1; tm recb_xf_smallpedia $?
cp $V/recb_xf/recb_selection_*.json $V/recb_xf/recb_test_meta_*.json $J/ 2>/dev/null
# P of the stored runs (hd_recb test stage used P=8 for polecat? record what the stored meta says)
grep -h '"P_emulated"' $H/work/recb/recb_test_meta_*.json >> chain.log
# ---- RecB features (rb_log_q, rb_log_psi, rb_log_xi) on the stored PT pools, from RecB-tr and RecB_xf
PCF=$T/pc_tirgn/work/feat_tirgn; SPF=$T/pilot_smallpedia/work/feat_tirgn
for v in tr xf; do
  RD=$H/work/recb; [ $v = xf ] && RD=$V/recb_xf
  for sp in valid test; do
    mkdir -p $V/rbf/polecat_tirgn_$v $V/rbf/smallpedia_tirgn_$v
    $CPU python3 -u $SRC/v76/v76_rbfeat.py --ds polecat --split $sp --cand $PCF/${sp}_cand.npy --q $PCF/${sp}_q.npz --W 100 --recb_dir $RD --out $V/rbf/polecat_tirgn_$v/${sp}_RB.npy > rbf_pc_${v}_$sp.log 2>&1 &
    $CPU python3 -u $SRC/v76/v76_rbfeat.py --ds smallpedia --split $sp --cand $SPF/${sp}_A_cand.npy --q $SPF/${sp}_q.npz --W 100 --recb_dir $RD --out $V/rbf/smallpedia_tirgn_$v/${sp}_RB.npy > rbf_sp_${v}_$sp.log 2>&1 &
  done
done
wait; tm rbfeat 0
cp $V/rbf/*/*_RB.json $J/ 2>/dev/null; for d in $V/rbf/*; do for f in $d/*_RB.json; do cp $f $J/rbf_$(basename $d)_$(basename $f); done; done
# ---- smoke tests (code paths only; no test number is read)
PCA="--layout pc --feat $PCF --lists $T/pc_tirgn/work/lists_tirgn"; SPA="--layout sp --feat $SPF"
$GPU python3 -u $SRC/v76/v76_btrain.py $PCA --rbtr $V/rbf/polecat_tirgn_tr --rbxf $V/rbf/polecat_tirgn_xf --out $V/smoke/bt_pc --label smoke_pc --arms MLP64_PAT,LIN_PAT,HR19,HR19_RBxf,PAT_RBtr --seeds 0 --smoke 3000 > smoke_bt_pc.log 2>&1; tm smoke_bt_pc $?
$GPU python3 -u $SRC/v76/v76_btrain.py $SPA --rbtr $V/rbf/smallpedia_tirgn_tr --rbxf $V/rbf/smallpedia_tirgn_xf --out $V/smoke/bt_sp --label smoke_sp --arms MLP64_PAT,HR19_RBtr,PAT_RBxf --seeds 0 --smoke 3000 > smoke_bt_sp.log 2>&1; tm smoke_bt_sp $?
$GPU bash -c "python3 -u $SRC/v76/v76_lmart.py $PCA --out $V/smoke/lm_pc --label smoke_pc --seeds 0,1 --smoke 3000 --threads 16 --mlp_model $V/smoke/bt_pc/models/MLP64_PAT_s0.pt" > smoke_lm_pc.log 2>&1; tm smoke_lm_pc $?
$GPU bash -c "python3 -u $SRC/v76/v76_lmart.py --layout sp --feat $H/ablations_A/inputs/sp_regcn_tgb_feat --out $V/smoke/lm_sprt --label smoke_sprt --seeds 0 --smoke 3000 --threads 16" > smoke_lm_sprt.log 2>&1; tm smoke_lm_sprt $?
mkdir -p $V/smoke/valfit/dumps
$GPU bash -c "cd $V/smoke/valfit && PC_P=$V/smoke/valfit TIRGN_TREE=$T/pilot_smallpedia/stage/TiRGN python3 -u $SRC/v76/sp_tirgn_valfit.py --smoke 3 --fit_cut 105" > smoke_valfit.log 2>&1; tm smoke_valfit $?
PCE="PC_P=$T/pilot_polecat PYTHONPATH=$SRC/pilot_polecat:$PYTHONPATH"; SPE="PC_P=$T/pilot_smallpedia PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH"
env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 HD_HOST=local $PCE $MNT \
  python3 -u $SRC/presub/g1_features.py --ds polecat --common pc_common --recb_dir $V/recb_xf --out $V/smoke/g1f_pc --smoke 2000 > smoke_g1f_pc.log 2>&1; tm smoke_g1f_pc $?
$GPU python3 -u $SRC/presub/g1_train.py --feat $V/smoke/g1f_pc --out $V/smoke/g1t_pc --smoke 1 > smoke_g1t_pc.log 2>&1; tm smoke_g1t_pc $?
cp $V/smoke/*/train_*.json $J/ 2>/dev/null; cp train_*smoke*.json $J/ 2>/dev/null
for f in smoke_*.log; do echo "== $f" >> chain.log; tail -3 $f >> chain.log; done
echo ALLDONE >> chain.log
