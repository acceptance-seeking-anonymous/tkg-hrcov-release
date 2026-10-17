#!/usr/bin/env python3
"""configs/aggregation/expand_cfg.py IN.json OUT.json -- expand $WORK_DIR / $EXT_DIR / $RESULTS_DIR in an aggregation config
(the configs in this directory list the per-query files each aggregator reads, with the original locations mapped to these
variables) and report which inputs are missing."""
import json, os, sys


def expand(o):
    if isinstance(o, dict): return {k: expand(v) for k, v in o.items()}
    if isinstance(o, list): return [expand(v) for v in o]
    if isinstance(o, str) and "$" in o: return os.path.expandvars(o)
    return o


def paths(o):
    if isinstance(o, dict):
        for v in o.values(): yield from paths(v)
    elif isinstance(o, list):
        for v in o: yield from paths(v)
    elif isinstance(o, str) and o.startswith("/"): yield o


if __name__ == "__main__":
    cfg = expand(json.load(open(sys.argv[1])))
    json.dump(cfg, open(sys.argv[2], "w"), indent=1)
    miss = [p for p in paths(cfg) if not os.path.exists(p)]
    print(f"{len(miss)} missing input paths"); [print("  ", p) for p in miss[:50]]
