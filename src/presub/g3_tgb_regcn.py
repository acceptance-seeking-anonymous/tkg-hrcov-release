#!/usr/bin/env python3
"""presub / g3_tgb_regcn.py -- G3: run TGB 2.0's own examples/linkproppred/tkgl-smallpedia/regcn.py UNCHANGED
(defaults of modules.tkg_utils.get_args_regcn: uvrgcn encoder, 2 layers, h 200, 100 bases, layer norm, self loop,
train/test history 3, ConvTransE, dropout 0.2, lr 1e-3, grad-norm 1, task weight 0.7, 10 epochs with the script's
valid-best checkpoint, seed 1, run 1).  The only deviation: DGL is replaced by the project's pure-torch DGL shim
(the tkg-gpu image has no DGL wheel for torch 2.8), installed into sys.modules before the repo imports `dgl`.

  python3 g3_tgb_regcn.py --repo <tgb repo copy> --shim_dir <dir with dgl_shim.py> --zip <tkgl-smallpedia.zip> --out <dir>
                          [--smoke 1] [extra TGB args ...]

The raw TGB zip is extracted into the installed py-tgb package's dataset dir (no download), as in sp_verify.py.
Smoke (--smoke 1, code check only): --n-epochs 2 (epoch 0 is never validated by the script, epoch 1 is); every
training epoch visits 8 random snapshots; valid and test lists are cut to their FIRST snapshot (the TGB history
protocol is otherwise unchanged, so g3_dump.py --smoke 1 reproduces the same evaluation rows).
Writes <out>/tgb_regcn_ckpt.pt (the script's saved valid-best checkpoint), <out>/tgb_regcn_results.json (the
script's own results json) and <out>/g3_tgb_run.json (versions, argv, timing, best epoch).
"""
import os, sys, json, time, zipfile, random, argparse, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--shim_dir", required=True)
ap.add_argument("--zip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", type=int, default=0)
a, extra = ap.parse_known_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.shim_dir)
import dgl_shim  # noqa: F401  -- registers fake dgl / dgl.function / dgl.nn before the repo imports them
assert getattr(sys.modules["dgl"], "_IS_TORCH_SHIM", False)
import numpy as np
import torch
import tgb
from tgb.utils.info import PROJ_DIR
import importlib.metadata as im

root = os.path.join(PROJ_DIR, "datasets", "tkgl_smallpedia")
os.makedirs(root, exist_ok=True)
with zipfile.ZipFile(a.zip) as z:
    for n in z.namelist():
        if not os.path.exists(os.path.join(root, n)):
            z.extract(n, root)

SMOKE = {}
if a.smoke:
    import tgb.utils.utils as TU
    _sbt = TU.split_by_time; calls = [0]

    def split_by_time_smoke(data):                     # call order in the script: train, valid, test
        out = _sbt(data); calls[0] += 1
        return out if calls[0] == 1 else out[:1]
    TU.split_by_time = split_by_time_smoke
    _shuffle = random.shuffle

    def shuffle_smoke(x, *aa, **kw):                   # the script's only shuffle: the training snapshot index list
        _shuffle(x, *aa, **kw)
        if isinstance(x, list) and len(x) > 8 and all(isinstance(v, int) for v in x[:3]):
            del x[8:]
    random.shuffle = shuffle_smoke
    extra = ["--n-epochs", "2"] + extra
    SMOKE = dict(n_epochs=2, train_snapshots_per_epoch=8, valid_snapshots=1, test_snapshots=1)

ex_dir = os.path.join(a.repo, "examples", "linkproppred", "tkgl-smallpedia")
script = os.path.join(ex_dir, "regcn.py")
sys.argv = [script] + extra
os.chdir(ex_dir)
print("argv", sys.argv, "smoke", SMOKE, flush=True)
g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
try:                                                   # the script ends with sys.exit(); keep its globals
    exec(compile(open(script).read(), script, "exec"), g)
except SystemExit as e:
    print("script exited", e.code, flush=True)
    assert e.code in (None, 0), e.code

ck_src = os.path.join(ex_dir, "saved_models", f"REGCN_tkgl-smallpedia_{g['SEED']}_{g['args'].run_nr}")
res_src = os.path.join(ex_dir, "saved_results", "REGCN_tkgl-smallpedia_results.json")
shutil.copy(ck_src, os.path.join(a.out, "tgb_regcn_ckpt.pt"))
if os.path.exists(res_src):
    shutil.copy(res_src, os.path.join(a.out, "tgb_regcn_results.json"))
ck = torch.load(ck_src, map_location="cpu", weights_only=False)
args = vars(g["args"])
RUN = dict(script="TGB examples/linkproppred/tkgl-smallpedia/regcn.py (unchanged)", deviation="DGL -> pure-torch dgl_shim",
           argv=sys.argv, smoke=SMOKE, args={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in args.items()},
           best_epoch=int(ck["epoch"]), val_mrr=float(g["val_mrr"]), test_mrr=float(g["test_mrr"]), test_hits10=float(g["test_hits10"]),
           num_nodes=int(g["num_nodes"]), num_rels_tgb=int(g["num_rels"]),
           n_snapshots=dict(train=len(g["train_list"]), valid=len(g["valid_list"]), test=len(g["test_list"])),
           n_queries=dict(valid=int(sum(len(x) for x in g["valid_list"])), test=int(sum(len(x) for x in g["test_list"]))),
           versions=dict(torch=torch.__version__, py_tgb=im.version("py-tgb"), dgl=sys.modules["dgl"].__version__, numpy=np.__version__),
           gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), host=os.environ.get("HD_HOST", ""),
           total_s=round(time.time() - t0, 1))
json.dump(RUN, open(os.path.join(a.out, "g3_tgb_run.json"), "w"), indent=1)
print(json.dumps({k: RUN[k] for k in ("best_epoch", "val_mrr", "test_mrr", "n_queries", "total_s")}), flush=True)
