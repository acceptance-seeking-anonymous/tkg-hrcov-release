#!/usr/bin/env python3
"""phase2_forward_holdout / fh_tirgn_train.py -- runs the harness TiRGN trainer hd_tirgn.py (byte-identical copy in this
code dir) as __main__ after capping this process's GPU memory (env FH_GPU_CAP_GB, default 30 GB; the GPU-sharing rule).
All arguments are passed through to hd_tirgn.py.  Stage 1 always passes --no_dump (no test rows are scored by training).
  python3 fh_tirgn_train.py --data_dir D --n_ent 1173 --n_rel 218 --tree T --ds GDELT25X --L 7 --hr 0.3 --lr 0.001 \
      --epochs 30 --patience 5 --max_train_hours 10 --seed 42 --no_dump --out OUT
"""
import os, sys, runpy
import torch

assert "--no_dump" in sys.argv, "Stage 1 must not dump (the dump scores the test split)"
if torch.cuda.is_available():
    tot = torch.cuda.get_device_properties(0).total_memory
    cap = float(os.environ.get("FH_GPU_CAP_GB", "30")) * 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot), 0)
    print(f"[fh] GPU memory cap {cap / 2**30:.1f} GiB of {tot / 2**30:.1f} GiB ({torch.cuda.get_device_name(0)})", flush=True)
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [f"{here}/hd_tirgn.py"] + sys.argv[1:]
runpy.run_path(f"{here}/hd_tirgn.py", run_name="__main__")
