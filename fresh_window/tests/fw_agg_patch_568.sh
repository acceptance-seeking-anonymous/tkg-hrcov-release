#!/bin/bash
# Verbatim local bash cell, session-archive index 568 (already applied: code/fw_agg.py contains this change).
cd fw/code_fw && python3 - <<'EOF'
s = open("fw_agg.py").read()
s = s.replace('days=f"{days[m].min()}..{days[m].max()}"', 'days=f"{min(days[m])}..{max(days[m])}"')
open("fw_agg.py", "w").write(s)
EOF
python3 fw_agg.py --tirgn /tmp/aggt/t.npz --sa /tmp/aggt/s.npz --ext /tmp/aggt/e.npz --ts2id /tmp/aggt/ts2id.txt --comparator /tmp/aggt/c.json --out /tmp/aggt/v.json --results /tmp/aggt/r.json && python3 -c "
import json; v=json.load(open('/tmp/aggt/v.json')); print(v['margins']['HRcov-SA']); print(v['metrics_official']['HRcov']); print(list(json.load(open('/tmp/aggt/r.json'))['weekly_deltas'].items())[:2])"
