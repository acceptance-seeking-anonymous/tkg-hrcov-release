#!/bin/bash
# Verbatim local bash cell, session-archive index 566: fw_agg.py on synthetic score files (/tmp/aggt), run before
# any real test number existed. The first run raised in the weekly-delta block; cell 568 (tests/fw_agg_patch_568.sh) patched
# fw_agg.py and the check was re-run (outcome on synthetic data: R1_BEATS_BOTH). code/fw_agg.py is the patched version.
cd fw/code_fw && mkdir -p /tmp/aggt && python3 - <<'EOF'
import numpy as np, json
rng = np.random.default_rng(1); n = 5000; ts = np.sort(rng.integers(212, 273, n))
def arm(shift):
    rk = np.maximum(1, rng.geometric(0.35 + shift, n)).astype(np.float32); return rk
T = {"rows": np.arange(n)}
for nm, sh in (("base", 0), ("BLEND", 0.01), ("POOL", 0.01), ("HRCOV_PAT_s0", 0.05), ("HRCOV_PAT_s1", 0.05), ("HRCOV_PAT_s2", 0.05)):
    rk = arm(sh); T[f"rr_{nm}"] = 1 / rk; T[f"rank_{nm}"] = rk; T[f"hrank_{nm}"] = rk
np.savez("/tmp/aggt/t.npz", **T)
S = {"rows": np.arange(n), "ts": ts}
for nm, sh in (("HIST", 0), ("SA_W100_s0", 0.02), ("SA_W100_s1", 0.02), ("SA_W100_s2", 0.02)):
    rk = arm(sh); S[f"rr_{nm}"] = 1 / rk; S[f"rank_{nm}"] = rk; S[f"hrank_{nm}"] = rk
np.savez("/tmp/aggt/s.npz", **S)
E = {"rows": np.arange(n), "ts": ts, "rr_base": T["rr_base"], "rank_base": T["rank_base"], "hrank_base": T["hrank_base"]}
rk = arm(0.0); E["rr_EXT"] = 1 / rk; E["rank_EXT"] = rk; E["hrank_EXT"] = rk
np.savez("/tmp/aggt/e.npz", **E)
open("/tmp/aggt/ts2id.txt", "w").write("".join(f"2026{3 if i < 31 else 4:02d}{(i % 31) + 1 if i < 31 else i - 30:02d}\t{212 + i}\n" for i in range(61)))
json.dump(dict(config="C0", variant="blend", a=0.3, tau=1.0), open("/tmp/aggt/c.json", "w"))
EOF
python3 fw_agg.py --tirgn /tmp/aggt/t.npz --sa /tmp/aggt/s.npz --ext /tmp/aggt/e.npz --ts2id /tmp/aggt/ts2id.txt --comparator /tmp/aggt/c.json --out /tmp/aggt/v.json --results /tmp/aggt/r.json && python3 -c "
import json; v=json.load(open('/tmp/aggt/v.json')); print(v['margins']['HRcov-SA']); print(list(json.load(open('/tmp/aggt/r.json'))['weekly_deltas'].items())[:2])"
