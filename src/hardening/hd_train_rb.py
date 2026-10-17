#!/usr/bin/env python3
"""hardening / hd_train_rb.py -- HR / HRCOV with RecB score features appended (arms HR_RB, HRCOV_RB), pilot recipe unchanged.

Runs the pilot trainer (polecat: pilot_polecat/code/pc_train.py; smallpedia: pilot_smallpedia/code/sp_train.py) with its
feature tensor X replaced by [X | RB] (RB = hd_rbfeat.py, 3 columns), two extra arms, and all-seed promoted lists.
Work dir: PC_P must point to a hardening copy whose work/feat_<b> symlinks the pilot feature dir and whose
work/feat_<b>/{split}[_A]_RB.npy... are given with --rb_dir.  Nothing in the pilot directories is written.
  PC_P=<hd copy> python3 hd_train_rb.py --ds polecat --base regcn --rb_dir <dir>
"""
import os, sys, argparse
import numpy as np
import torch
_snt = torch.set_num_threads
torch.set_num_threads = lambda n: _snt(int(os.environ.get('HD_THREADS', '8')))
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--base", required=True); ap.add_argument("--rb_dir", required=True)
ap.add_argument("--arms", default="HR_RB,HRCOV_RB")
a, rest = ap.parse_known_args()
TR = os.path.expandvars("$WORK_DIR/tracks")


class CatX:
    """[X | RB] along the last axis, materialised per access (X memmap [n, W, d], RB memmap [n, W, 3])."""
    def __init__(self, X, RB):
        assert X.shape[:2] == RB.shape[:2], (X.shape, RB.shape)
        self.X, self.RB = X, RB; self.shape = X.shape[:2] + (X.shape[2] + RB.shape[2],); self.dtype = X.dtype

    def __len__(self): return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            first, rest_ = key[0], key[1:]
        else:
            first, rest_ = key, ()
        A = np.asarray(self.X[first]); B = np.asarray(self.RB[first]).astype(A.dtype)
        C = np.concatenate([A, B], -1)
        return C[(slice(None),) + rest_] if rest_ else C


if a.ds == "polecat":
    sys.path.insert(0, f"{TR}/pilot_polecat/code")
    import pc_train as T
    from pc_common import KC, W_COV
    T.COLS["hist_rb"] = list(range(0, 19)) + [30, 31, 32]; T.COLS["all_rb"] = list(range(0, 33))
    T.ARMS["HR_RB"] = ("stand", KC, "hist_rb"); T.ARMS["HRCOV_RB"] = ("stand", W_COV, "all_rb")
    T.SAVE_LISTS = ("HR", "HRCOV", "HR_RB", "HRCOV_RB")
    _ls = T.load_split

    def load_split(fd, split):
        d = _ls(fd, split)
        d["X"] = CatX(d["X"], np.load(f"{a.rb_dir}/{split}_RB.npy", mmap_mode="r"))
        return d
    T.load_split = load_split
else:
    sys.path.insert(0, f"{TR}/pilot_smallpedia/code")
    import sp_train as T
    H_, C_ = list(range(0, 19)), list(range(19, 30))
    T.ARMS["HR_RB"] = ("A", list(range(0, 50)), H_ + [42, 43, 44], None)
    T.ARMS["HRCOV_RB"] = ("A", list(range(0, 150)), H_ + C_ + [42, 43, 44], None)
    T.SAVE_LISTS = ("HR", "HRCOV", "HRCOV_ST", "HR_ST", "HR_RB", "HRCOV_RB")
    _ls = T.load_split

    def load_split(fd, split):
        d = _ls(fd, split)
        d["A"]["X"] = CatX(d["A"]["X"], np.load(f"{a.rb_dir}/{split}_A_RB.npy", mmap_mode="r"))
        return d
    T.load_split = load_split
    src = open(T.__file__).read()
    assert "if arm in SAVE_LISTS and sd_ == seeds[0]:" in src
sys.argv = [sys.argv[0], "--base", a.base, "--arms", a.arms] + rest
if a.ds == "smallpedia":
    # all-seed promoted lists: re-exec the trainer source with the seed restriction removed (same module globals)
    code = open(T.__file__).read().replace("if arm in SAVE_LISTS and sd_ == seeds[0]:", "if arm in SAVE_LISTS:")
    g = dict(T.__dict__); g["__name__"] = "sp_train_rb"
    exec(compile(code.replace('if __name__ == "__main__":\n    main()', ''), T.__file__, "exec"), g)
    g["ARMS"].update(T.ARMS); g["SAVE_LISTS"] = T.SAVE_LISTS; g["load_split"] = load_split
    g["main"]()
else:
    T.main()
