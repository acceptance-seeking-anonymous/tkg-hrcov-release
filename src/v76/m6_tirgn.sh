#!/bin/bash
# v76 M6 (lowest priority; v76_rule.json caab7495): TiRGN on tkgl-smallpedia, new base seeds 43 and 44, on serverA GPU 0 ONLY after the
# items M2b/M3a/M4 have finished their GPU work there (waits for the phase-2b ALLDONE marker and an idle GPU 0).
# Per seed = sp_tirgn.py unchanged (--seed S) -> sp pipeline (sp_lists, sp_features: HRCOV_PAT pool) -> hcpat_train.py (stored
# recipe) -> BLEND / POOL valid grid on the full validation set + hd_select -> one official test pass -> finiteness scan.
# Stop rule: no seed is started after 2026-10-19T10:00Z (window = registration 2026-10-15T12:07:43Z + 96 h, minus ~2 h per seed).
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
SEEDS=${1:-"43 44"}
J=$PWD; O=$WORK_DIR; T=$O/tracks; H=$T/hardening; V=$H/v76; M6=$V/m6; mkdir -p $M6
# (code staging step of the original run removed: code is read from $SRC)
T0=$(date +%s); tm() { echo "[stage-time] $1 rc=$2 $(( $(date +%s) - T0 ))s $(date -u)" >> chain.log; }
DEADLINE=$(date -u -d "2026-10-19T10:00:00Z" +%s)
echo "waiting for $WAITJOB ALLDONE and idle GPU 0 since $(date -u)" >> chain.log
# (orchestration wait of the original multi-job run removed)
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}) -lt 4000 ]; do sleep 120; done
echo "GO at $(date -u)" >> chain.log
MNT=""
TGB=""
for S in $SEEDS; do
  [ $(date -u +%s) -gt $DEADLINE ] && { echo "seed $S not started: past the M6 window" >> chain.log; continue; }
  P=$M6/sp_tirgn_s$S; TS=$M6/hcpat_tirgn_s$S; mkdir -p $P/dumps $P/work $P/stage
  PEN="env PC_P=$P PYTHONPATH=$SRC/pilot_smallpedia:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1 TIRGN_TREE=$T/pilot_smallpedia/stage/TiRGN"
  G0="env CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HOST=local $PEN $MNT"
  if [ ! -f $P/dumps/tirgn_SMALLPEDIA_dump_log.json ]; then
    $G0 bash -c "cd $P && python3 -u $SRC/pilot_smallpedia/sp_tirgn.py --seed $S" > m6_tirgn_train_s$S.log 2>&1; tm m6_tirgn_train_s$S $?
  fi
  cp $P/dumps/tirgn_SMALLPEDIA_dump_log.json $J/m6_tirgn_dump_log_s$S.json 2>/dev/null
  grep -q "ALL DONE" m6_tirgn_train_s$S.log 2>/dev/null || [ -f $P/dumps/tirgn_SMALLPEDIA_dump_log.json ] || { echo "seed $S BLOCKED at training"; tail -20 m6_tirgn_train_s$S.log; } >> chain.log
  for f in prep_summary.json valid_sample_idx.npy te_SMALLPEDIA.npz ste_SMALLPEDIA.npz strata_valid.npz strata_test.npz recb_valid_rank.npy recb_test_rank.npy reach_sample.npz; do [ -f $P/work/$f ] || cp $T/pilot_smallpedia/work/$f $P/work/; done
  [ -d $P/work/tgb_ns ] || cp -r $T/pilot_smallpedia/work/tgb_ns $P/work/
  for f in $T/pilot_smallpedia/stage/*.npz $T/pilot_smallpedia/stage/*.json; do [ -f $f ] && cp $f $P/stage/; done
  env OMP_NUM_THREADS=1 $PEN $MNT python3 -u $SRC/pilot_smallpedia/sp_lists.py --base tirgn --workers 12 > m6_tirgn_lists_s$S.log 2>&1; tm m6_tirgn_lists_s$S $?
  env OMP_NUM_THREADS=4 $PEN $MNT python3 -u $SRC/pilot_smallpedia/sp_features.py --base tirgn > m6_tirgn_feat_s$S.log 2>&1; tm m6_tirgn_feat_s$S $?
  $G0 python3 -u $SRC/hardening/hcpat_train.py --layout sp --feat $P/work/feat_tirgn --out $TS --label smallpedia_tirgn_s$S > m6_tirgn_hcpat_s$S.log 2>&1; tm m6_tirgn_hcpat_s$S $?
  cp $TS/train_hcpat.json $J/train_hcpat_tirgn_s$S.json 2>/dev/null
  VR=$P/dumps/tirgn_valid_logp.f32,$P/dumps/tirgn_test_logp.f32
  $G0 python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_s$S --stage valid --source fullrows --fullrows $VR --recb_dir $H/work/recb --out $M6 > m6_tirgn_bvalid_s$S.log 2>&1; tm m6_tirgn_bvalid_s$S $?
  $G0 python3 $SRC/hardening/hd_select.py --ds smallpedia --bases tirgn_s$S --dir $M6 --out $M6/sel_smallpedia_tirgn_s$S.json > m6_tirgn_select_s$S.log 2>&1; tm m6_tirgn_select_s$S $?
  SYS=""; for s in 0 1 2; do SYS="$SYS,HRCOV_PAT_s$s=$TS/promlist_HRCOV_PAT_s$s.npy"; done; SYS=${SYS#,}
  $G0 bash -c "$TGB python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base tirgn_s$S --stage test --source fullrows --fullrows $VR --recb_dir $H/work/recb --sel $M6/sel_smallpedia_tirgn_s$S.json --systems $SYS --out $M6" > m6_tirgn_btest_s$S.log 2>&1; tm m6_tirgn_btest_s$S $?
  cp $M6/blend_valid_smallpedia_tirgn_s$S.json $M6/blend_test_smallpedia_tirgn_s$S.json $M6/official_test_smallpedia_tirgn_s$S.npz $M6/sel_smallpedia_tirgn_s$S.json $J/ 2>/dev/null
  mkdir -p ctl_s$S && cp $M6/official_test_smallpedia_tirgn_s$S.npz $M6/blend_valid_smallpedia_tirgn_s$S.npz ctl_s$S/ 2>/dev/null
  env OMP_NUM_THREADS=1 \
    python3 -u $SRC/common/finite_scan.py --workers 8 --out ./finite_scan_m6_tirgn_s$S.json --roots "m6_tirgn_dumps_s$S=$P/dumps" "m6_tirgn_pipeline_s$S=$P/work" "m6_tirgn_hcpat_s$S=$TS" "m6_tirgn_controls_s$S=./ctl_s$S" > m6_scan_s$S.log 2>&1; tm m6_scan_s$S $?
  rm -rf ctl_s$S
  echo "seed $S done" >> chain.log
done
echo ALLDONE >> chain.log
