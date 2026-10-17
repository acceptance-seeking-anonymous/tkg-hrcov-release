#!/usr/bin/env python3
"""phase2_fresh_window / fw_ct_latency.py -- v79 latency protocol, CountTRuCoLa rule application (released code, frozen Stage-1
rules of the selected configuration) on the fixed validation slice (first --n harness validation queries of the first
selection-part timestamp).  Run with cwd = that configuration's repository copy, this file + fw_ct_main.py in rule_based/,
the repo venv; threads pinned.  Stages:
  precompute : predictor.get_z_rule_stats + get_f_rule_stats for the split (per-split work; reported amortised over the
               152,234 validation queries and as wall time)
  apply      : predictor.apply on the slice index only (the split statistics cached), NUM_CPUS=1 ('batch 1': queries one at
               a time) and NUM_CPUS=--nc (production Ray buckets); one warm-up call each, then --reps timed calls, median.
  python3 rule_based/fw_ct_latency.py --fw_dd D --fw_root ROOT --n 2000 --reps 5 --nc 10 --out lat_ct.json --params K=V ...
"""
import os, sys, json, time, statistics
import numpy as np

def _pop(flag, default=None):
    if flag not in sys.argv: return default
    i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
N_ = int(_pop("--n", "2000")); REPS = int(_pop("--reps", "5")); NC = int(_pop("--nc", "10")); OUT = _pop("--out", "lat_ct.json")
DD = sys.argv[sys.argv.index("--fw_dd") + 1]
import fw_ct_main as W          # pops --fw_dd/--fw_root, patches the dataset class and evaluate()   # noqa: E402
import rule_dataset, predictor, utils                                                              # noqa: E402
from options import Options                                                                         # noqa: E402
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:8.1f}s]", *x, flush=True)
options = Options(config_file_name="config-default.yaml").options
rd = rule_dataset.RuleDataset(name=options["DATASET_NAME"], large_data_hardcode_flag=options["LARGE_DATA_HARDCODE_FLAG"],
                              very_large_data_hardcode_flag=options["VERY_LARGE_DATA_HARDCODE_FLAG"])
if rd.large_data_flag and not rd.very_large_data_flag and "C_X_COUNT" not in options: options["C_X_COUNT"] = 20
_, _, _, _, _, _, path_rankings_val, _, _, _, path_rules, all_false = utils.sort_out_pathnames(options, options["DATASET_NAME"])
rules_file = path_rules.format("ids"); assert os.path.exists(rules_file), rules_file
# slice: harness validation order (per ts: forward block in file order, then inverses), first N_ rows of ts = cut
st = open(f"{DD}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
q = np.loadtxt(f"{DD}/valid.txt", dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4); q = q[np.argsort(q[:, 3], kind="stable")]
blocks = []
for u in np.unique(q[:, 3]):
    m = q[q[:, 3] == u]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R; blocks += [m, inv]
sro = np.concatenate(blocks); uts = np.unique(sro[:, 3]); cut = int(uts[int(0.7 * len(uts))])
s0 = int(np.flatnonzero(sro[:, 3] == cut)[0]); SL = sro[s0:s0 + N_]
idx = {}
for s_, r_, o_, t_ in SL: idx.setdefault(int(s_), {}).setdefault(int(r_), set()).add(int(t_))
n_ct_queries = sum(len(v2) for v in idx.values() for v2 in v.values())
predictor.setOptions(options)
times = {}
tz = []; tf = []
for _ in range(max(1, min(REPS, 3))):
    t = time.perf_counter(); z = predictor.get_z_rule_stats(rd, "val"); tz.append(time.perf_counter() - t)
    t = time.perf_counter(); f = predictor.get_f_rule_stats(rd, "val", rd.very_large_data_flag); tf.append(time.perf_counter() - t)
log("precompute", tz, tf)
_z, _f = z, f
predictor.get_z_rule_stats = lambda *a_, **k_: _z
predictor.get_f_rule_stats = lambda *a_, **k_: _f
rd.head_rel_ts["val"] = idx; rd.val_data = SL
res = dict(n=int(N_), slice_ts=cut, slice_first_row=s0, n_countrucola_queries=int(n_ct_queries), rules=rules_file,
           precompute=dict(z_rule_stats_s=tz, f_rule_stats_s=tf, median_total_s=statistics.median(tz) + statistics.median(tf),
                           amortised_s_per_1000_valid_queries=(statistics.median(tz) + statistics.median(tf)) / len(sro) * 1000.0),
           apply={}, options={k: options[k] for k in ("LEARN_WINDOW_SIZE", "RULE_TYPE_C", "AGGREGATION_DECAY", "NUM_TOP_RULES", "C_X_COUNT") if k in options})
for mode, nc in (("batch1", 1), ("batched", NC)):
    options["NUM_CPUS"] = nc; predictor.setOptions(options)
    outp = f"/tmp/fw_ct_lat_{os.getpid()}_{mode}.txt"
    predictor.apply(rd, rules_file, outp, 0.5, evaluation_mode="val", all_rule_types_false=all_false)       # warm-up (Ray start)
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); predictor.apply(rd, rules_file, outp, 0.5, evaluation_mode="val", all_rule_types_false=all_false); ts.append(time.perf_counter() - t)
    nl = sum(1 for _ in open(outp)) // 2
    res["apply"][mode] = dict(num_cpus=nc, raw_s=ts, median_s=statistics.median(ts), s_per_1000=statistics.median(ts) / N_ * 1000.0, ranking_lines=nl)
    log(mode, res["apply"][mode])
res["elapsed_s"] = round(time.time() - T0, 1)
json.dump(res, open(OUT, "w"), indent=1)
log("DONE")
