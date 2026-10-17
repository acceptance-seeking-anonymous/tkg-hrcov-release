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
# logcl_tuned only (grid-best L3/lr3e-4/time-ordered, epoch 12, valid 0.5332 > pilot 0.5091); mirror already in place
# (code staging step of the original run removed: code is read from $SRC)
ENVS="env PC_HARNESS=$M PC_P=$P PC_DD=$M/tracks/datasets/harness_data/tkgl-smallpedia PYTHONDONTWRITEBYTECODE=1"
CPU="env OMP_NUM_THREADS=1 $ENVS"
GPU="env OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HD_HAR=$M HD_HOST=local $ENVS"
BASES="logcl"
echo "logcl grid-best beats pilot: 1 (0.5332 > 0.5091)" > chain.log
for B in $BASES; do
  T0=$(date +%s)
  $CPU python3 -u $SRC/pilot_smallpedia/sp_lists.py --base $B --workers 12 > lists_$B.log 2>&1; echo "lists $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $CPU python3 -u $SRC/pilot_smallpedia/sp_features.py --base $B > feat_$B.log 2>&1; echo "features $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $GPU python3 -u $SRC/pilot_smallpedia/sp_train.py --base $B --arms HR,HRCOV,HRCOV_ST > train_$B.log 2>&1; echo "train $B $? $(( $(date +%s)-T0 ))s" >> chain.log
  $GPU python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base ${B}_tuned --stage valid --source fullrows --fullrows $P/dumps/${B}_valid_logp.f32,$P/dumps/${B}_test_logp.f32 --recb_dir $H/work/recb --out $H/work/blend > blend_valid_spnew_$B.log 2>&1; echo "blend_valid $B $? $(( $(date +%s)-T0 ))s" >> chain.log
done

SEL=""
for B in $BASES; do SEL="$SEL,${B}_tuned"; done
SEL=${SEL#,}
if [ -n "$SEL" ]; then
  $GPU python3 $SRC/hardening/hd_select.py --ds smallpedia --bases $SEL --dir $H/work/blend --out $H/work/blend/sel_smallpedia_tuned.json > select_spnew.log 2>&1
  for B in $BASES; do
    TD=$P/work/train_$B; SYS=""
    for a in HR HRCOV HRCOV_ST; do for s in 0 1 2; do SYS="$SYS,${a}_s$s=$TD/promlist_${a}_s$s.npy"; done; done
    SYS=${SYS#,}
    $GPU bash -c "python3 -u $SRC/hardening/hd_blend.py --ds smallpedia --base ${B}_tuned --stage test --source fullrows --fullrows $P/dumps/${B}_valid_logp.f32,$P/dumps/${B}_test_logp.f32 --recb_dir $H/work/recb --sel $H/work/blend/sel_smallpedia_tuned.json --systems $SYS --out $H/work/blend" > blend_test_spnew_$B.log 2>&1
    echo "blend_test $B $?" >> chain.log
  done
fi
for B in $BASES; do cp $P/work/train_$B/train_$B.json $P/work/train_$B/ranks_test.npz $W/ 2>/dev/null; mv $W/ranks_test.npz $W/ranks_test_spnew_$B.npz 2>/dev/null; cp $P/work/feat_$B/test_q.npz $W/test_q_spnew_$B.npz 2>/dev/null; done
cp lists_summary_*.json feat_summary_*.json $W/ 2>/dev/null
cp $H/work/blend/*smallpedia*tuned* $H/work/blend/sel_smallpedia_tuned.json $W/ 2>/dev/null
