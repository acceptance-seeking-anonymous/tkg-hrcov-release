SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
W=$PWD
H=$WORK_DIR/tracks/hardening
M=$WORK_DIR
S=$WORK_DIR   # one machine: the tree of the second server is the same $WORK_DIR
SCP="cp"
P=$M/tracks/sp_new
mkdir -p $P/work $P/stage $P/code $M/tracks/pilot_smallpedia
# ---- mirror code (harness root, retriever, residual, pilot code) from the serverA snapshot + base-independent smallpedia inputs
# (source snapshot of the original run removed: code is read from $SRC)
for f in prep_summary.json valid_sample_idx.npy te_SMALLPEDIA.npz ste_SMALLPEDIA.npz strata_valid.npz strata_test.npz recb_valid_rank.npy recb_test_rank.npy reach_sample.npz; do [ -f $P/work/$f ] || $SCP $S/tracks/pilot_smallpedia/work/$f $P/work/; done
[ -d $P/work/tgb_ns ] || $SCP -r $S/tracks/pilot_smallpedia/work/tgb_ns $P/work/
for f in static_edges.npz static_names.json h2t_entity.npy t2h_entity.npy tsraw2id.json; do [ -f $P/stage/$f ] || $SCP $S/tracks/pilot_smallpedia/stage/$f $P/stage/; done
mkdir -p $M/tracks/pilot_smallpedia/work $M/tracks/pilot_smallpedia/stage
[ -d $M/tracks/pilot_smallpedia/work/tgb_ns ] || cp -r $P/work/tgb_ns $M/tracks/pilot_smallpedia/work/
cp $P/stage/h2t_entity.npy $P/stage/t2h_entity.npy $M/tracks/pilot_smallpedia/stage/
# (code staging step of the original run removed: code is read from $SRC)
export SP_SAVE_LISTS_ALL_SEEDS=1   # sp_train.py saves the promoted lists of every seed (the original run edited the file in place)
echo "SP_SAVE_LISTS_ALL_SEEDS=$SP_SAVE_LISTS_ALL_SEEDS" > $W/sp_train_patch.txt
[ -L $P/dumps ] || ln -s $H/sp_new/dumps $P/dumps
# (code staging step of the original run removed: code is read from $SRC)
# ---- wait for the base grid
while pgrep -f hd_base_train.py > /dev/null; do sleep 60; done
cp $H/sp_new/dumps/*grid_log.json $H/sp_new/dumps/*dump_log.json $W/ 2>/dev/null
ENVS="env PC_HARNESS=$M PC_P=$P PC_DD=$M/tracks/datasets/harness_data/tkgl-smallpedia PYTHONDONTWRITEBYTECODE=1"
CPU="env OMP_NUM_THREADS=1 $ENVS"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HAR=$M HD_HOST=local $ENVS"
BASES=""
for B in regcn logcl; do
  TH=$( [ $B = regcn ] && echo 0.4900 || echo 0.5091 )
  OK=$(python3 -c "import json;d=json.load(open('$H/sp_new/dumps/${B}_SMALLPEDIA_grid_log.json'));print(int(d['selected']['valid_MRR']>$TH))" 2>/dev/null)
  echo "$B grid-best beats pilot: $OK" >> chain.log
  [ "$OK" = "1" ] && BASES="$BASES $B"
done
for B in $BASES; do
  T0=$(date +%s)
  $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers 12 > lists_$B.log 2>&1; echo "lists $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B > feat_$B.log 2>&1; echo "features $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $GPU python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B --arms HR,HRCOV,HRCOV_ST > train_$B.log 2>&1; echo "train $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $GPU python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base $B --stage valid --source fullrows --fullrows $P/dumps/${B}_valid_logp.f32,$P/dumps/${B}_test_logp.f32 --recb_dir $H/work/recb --out $H/work/blend > blend_valid_spnew_$B.log 2>&1; echo "blend_valid $B $? $(( $(date +%s)-T0 ))s" >> chain.log
done
