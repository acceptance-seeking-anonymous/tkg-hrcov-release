#!/bin/bash
# G8 post-training chain (one GPU; CPU stages <= 8 workers); run after g8_setup.sh and g8_train.sh:
# dump (valid-best CognTKE) -> pilot-1 pc pipeline on base 'cogn' (lists, exact lookup, features, HR/HRCOV, HRCOV_PAT) ->
# controls (valid grid, valid-only selection) -> one official test pass (base, BLEND, POOL, HR, HRCOV, HRCOV_PAT) ->
# -inf base sensitivity. Usage: bash g8_chain.sh [start_stage]  (stages: dump lists lookup features train hcpat blendv blendt ninf)
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
START=${1:-dump}
J=$PWD; OIH=$WORK_DIR; T=$OIH/tracks; H=$T/hardening; G=$H/g8_cogntke; P=$G/pc
DD=$T/datasets/harness_data/tkgl-polecat
mkdir -p $P/work $P/code $P/stage $P/dumps $G/blend $G/code $G/dumps
# (code staging step of the original run removed: code is read from $SRC)
for f in te_POLECAT.npz strata_valid.npz strata_test.npz prep_summary.json recb_valid_rank.npy recb_test_rank.npy valid_sample_idx.npy; do [ -f $P/work/$f ] || cp $T/pc_tirgn/work/$f $P/work/; done
[ -d $P/work/tgb_ns ] || cp -r $T/pc_tirgn/work/tgb_ns $P/work/
cp $T/pc_tirgn/stage/*.npy $P/stage/
# (base registry entry already present in src/pilot_polecat/pc_common.py; the original run added it in place)
grep -n '"cogn"' $SRC/pilot_polecat/pc_common.py > $J/pc_common_patch.txt
ln -sf $G/dumps/cogn_scores_POLECAT_traintest.npz $P/dumps/cogn_scores_POLECAT_traintest.npz
ENVS="env PC_HARNESS=$OIH PC_P=$P PC_DD=$DD PYTHONDONTWRITEBYTECODE=1 HD_HAR=$OIH PYTHONPATH=$G/pylib_ts:$G/pylib:$PYTHONPATH"
GPUF="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local"
gpu() { $GPUF $ENVS "$@" ; }
gpur() { ( cd $G/CognTKE && $GPUF $ENVS "$@" ) ; }   # CognTKE code reads ./data/POLECAT from its repository root
cpu() { env OMP_NUM_THREADS=1 HD_HOST=local $ENVS "$@" ; }
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date)" >> chain.log; }
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}); echo "GPU used MiB at start: $U; start=$START" >> chain.log
go=0; st() { [ "$START" == "$1" ] && go=1; [ $go -eq 1 ]; }
if st dump; then gpur python3 -u $SRC/g8_cogntke/g8_dump.py --state $G/run_B/best_state.pt --out $G/dumps > dump.log 2>&1; tm dump $?; fi
if st lists; then cpu python3 -u $SRC/pilot_polecat/pc_lists.py --base cogn --workers 8 > lists_cogn.log 2>&1; tm lists $?; fi
if st lookup; then gpu python3 -u $SRC/g8_cogntke/hd_lookup.py --base cogn --source cogn --ckpt $G/dumps --work $P/work > lookup_cogn.log 2>&1; tm lookup $?; fi
if st features; then cpu python3 -u $SRC/pilot_polecat/pc_features.py --base cogn > feat_cogn.log 2>&1; tm features $?; fi
if st train; then gpu python3 -u $SRC/pilot_polecat/pc_train.py --base cogn --arms HR,HRCOV > train_cogn.log 2>&1; tm train $?; fi
if st hcpat; then gpu python3 -u $SRC/hardening/hcpat_train.py --layout pc --feat $P/work/feat_cogn --lists $P/work/lists_cogn --out $P/work/hcpat_cogn --label polecat_cogn > hcpat_cogn.log 2>&1; tm hcpat $?; fi
if st blendv; then
  gpu python3 -u $SRC/hardening/hd_blend.py --ds polecat --base cogn --stage valid --source cogn --ckpt $G/dumps --recb_dir $H/work/recb --out $G/blend > blend_valid_cogn.log 2>&1; tm blend_valid $?
  gpu python3 $SRC/hardening/hd_select.py --ds polecat --bases cogn --dir $G/blend --out $G/blend/sel_polecat_cogn.json > select_cogn.log 2>&1; tm select $?
fi
TD=$P/work/train_cogn; HP=$P/work/hcpat_cogn
SYS=""; for s in 0 1 2; do SYS="$SYS,HR_s$s=$TD/promlist_HR_s$s.npy,HRCOV_s$s=$TD/promlist_HRCOV_s$s.npy,HRCOV_PAT_s$s=$HP/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
if st blendt; then gpu bash -c "python3 -u $SRC/hardening/hd_blend.py --ds polecat --base cogn --stage test --source cogn --ckpt $G/dumps --recb_dir $H/work/recb --sel $G/blend/sel_polecat_cogn.json --systems $SYS --out $G/blend" > blend_test_cogn.log 2>&1; tm blend_test $?; fi
if st ninf; then gpu bash -c "G8_NINF=1 python3 -u $SRC/hardening/hd_blend.py --ds polecat --base cogn_ninf --stage test --source cogn --ckpt $G/dumps --recb_dir $H/work/recb --no_controls --out $G/blend" > blend_test_cogn_ninf.log 2>&1; tm ninf $?; fi
cp $G/dumps/dump_summary.json $G/dumps/reach_test.npy $G/dumps/reach_valid.npy $J/ 2>/dev/null
cp $G/blend/blend_*_polecat_cogn*.json $G/blend/official_test_polecat_cogn*.npz $G/blend/sel_polecat_cogn.json $J/ 2>/dev/null
cp $TD/train_cogn.json $HP/train_hcpat.json $J/ 2>/dev/null; cp lists_summary_cogn.json feat_summary_cogn.json lookup_cogn.json $J/ 2>/dev/null
cp $G/run_B/epoch_log.json $J/ 2>/dev/null
echo ALLDONE >> chain.log
