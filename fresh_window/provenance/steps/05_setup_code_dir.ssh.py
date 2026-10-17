"""Verbatim orchestration cell, session-archive index 456 (call_command on serverA).
Create the fresh-window code directory (same script as 03_setup/fw_setup_code.sh).
Runs in the authors' analysis environment: `c_serverA`/`c_serverB`/`c_serverC` are host.compute handles for serverA/serverB/serverC;
submit_job stages `inputs` flat into a fresh job directory on the host and runs `command` there; outputs come back under
hpc/<job_id>/. The plain-shell equivalent is given in FILES.md. Only change: user name restored (/home/anon).
"""
# ===== cell 456
setup = r'''
set -e
H=/home/anon/cs_scratch/open_ideas_harness/tracks
FH=$H/phase2_forward_holdout; FW=$H/phase2_fresh_window
mkdir -p $FW/code $FW/stage $FW/logs $FW/GDELTFW/work $FW/freeze
cp -p $FH/code/*.py $FH/code/*.sh $FW/code/
rm -f $FW/code/fh_agg.py $FW/code/fh_stage2_test.sh $FW/code/fh_stage1_chain.sh
[ -d $FW/stage/TiRGN ] || cp -rp $FH/stage/TiRGN $FW/stage/TiRGN
rm -rf $FW/stage/TiRGN/__pycache__ $FW/stage/TiRGN/src/__pycache__
cd $FW/code
sed -i -e 's/GDELT25X/GDELTFW/g' -e 's/gdelt25x/gdeltfw/g' -e 's/phase2_forward_holdout/phase2_fresh_window/g' *.py *.sh
sed -i -e 's/n_ent=1173, n_rel=218/n_ent=1209, n_rel=215/' fh_common.py
sed -i -e 's/N=1173, R=218/N=1209, R=215/' fh_blend.py
sed -i -e 's/--n_ent 1173 --n_rel 218/--n_ent 1209 --n_rel 215/' fh_tirgn_train.py
grep -n "1173\|n_rel=218\|R=218\|GDELT25X\|gdelt25x\|forward_holdout" *.py *.sh || echo "no residual constants"
cd $FW/code && for f in *.py *.sh; do if [ -f $FH/code/$f ]; then diff $FH/code/$f $f > /dev/null && s=same || s=changed; echo "$f $s"; fi; done
diff -r $FH/stage/TiRGN $FW/stage/TiRGN -x __pycache__ > /dev/null && echo "TiRGN tree identical"
'''
r = c_serverA.call_command(setup, intent="Create fresh-window code dir as substituted copy of forward-holdout code")
print(r.exit_code, r.stdout[-3000:], r.stderr[-1500:])
