"""
harness / bridge_cache.py -- read-only API over the bridge feature caches written by
build_bridge_cache.py.  Arrays are memory-mapped, so opening a cache is instant and cheap.

    from bridge_cache import BridgeCache
    bc = BridgeCache("ICEWS18", "g12")            # base defaults to "tirgn"
    q = bc.query(j)                                # j = 0..bc.nq-1
    q["cands"], q["X16"], q["casc"], q["lm"], q["lm_gated"], q["hard"], q["gold_pos"], q["row"]
    X = bc.arm_features(j, "arm5")                 # exactly the G12 arm5 input matrix (19 cols)

Feature reconstruction (identical to stage2_ranker.build_query output):
  X16 = [pm_top1, pm_subj, n_interm, idf, r2_recur, reached_both, cand_logfreq, imin, imean,
         hop_min(=2), q_entropy, q_top1, q_margin12, q_subj_logfreq, q_rel_logfreq, q_cset_logsize]
  casc = [tight>0, log1p(tight)];  lm_gated = lm * (1 - 1/rolecount(cand))
"""
import os, json
import numpy as np

HARNESS = os.path.expandvars("$WORK_DIR")


class BridgeCache:
    def __init__(self, dataset, cohort, base="tirgn", root=None):
        self.dir = f"{root or HARNESS}/cache/{dataset}_{base}_{cohort}"
        self.meta = json.load(open(f"{self.dir}/meta.json"))
        mm = lambda n: np.load(f"{self.dir}/{n}.npy", mmap_mode="r")
        self.qrow = mm("qrow"); self.qsplit = mm("qsplit"); self.built = mm("built")
        self.reached = mm("reached"); self.gold_pos = mm("gold_pos"); self.lm_ok = mm("lm_ok")
        self.qfeat = mm("qfeat"); self.offsets = mm("offsets"); self.cands = mm("cands")
        self.cfeat = mm("cfeat"); self.lm = mm("lm"); self.hard = mm("hard")
        self.nq = len(self.qrow)
        from oih_common import DATASETS
        cfg = DATASETS[dataset]
        v = np.load(cfg["lm_vocab"], allow_pickle=True)
        self.ent_role = v["ent_role"].astype(np.int64)
        self.role_count = np.bincount(self.ent_role, minlength=int(self.ent_role.max()) + 1)
        self._logfreq = None
        self.dataset = dataset

    def logfreq(self):
        if self._logfreq is None:
            from oih_common import Splits
            S = Splits(self.dataset)
            self._logfreq = np.log1p(S.freq.astype(np.float32))
        return self._logfreq

    def reuse_gate(self, cands):
        rc = self.role_count[self.ent_role[np.asarray(cands, np.int64)]].astype(np.float64)
        return (1.0 - 1.0 / np.maximum(rc, 1.0)).astype(np.float32)

    def query(self, j):
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        c = np.asarray(self.cands[a:b]); cf = np.asarray(self.cfeat[a:b]); qf = np.asarray(self.qfeat[j])
        n = b - a
        X = np.zeros((n, 16), np.float32)
        if n:
            X[:, 0] = cf[:, 0]; X[:, 1] = cf[:, 1]; X[:, 2] = cf[:, 2]; X[:, 3] = cf[:, 3]; X[:, 4] = cf[:, 4]
            X[:, 5] = ((cf[:, 0] > 0) & (cf[:, 1] > 0)).astype(np.float32)
            X[:, 6] = self.logfreq()[c]; X[:, 7] = cf[:, 5]; X[:, 8] = cf[:, 6]; X[:, 9] = 2.0
            X[:, 10:16] = qf[None, :]
        tight = cf[:, 7] if n else np.zeros(0, np.float32)
        casc = np.stack([(tight > 0).astype(np.float32), np.log1p(tight).astype(np.float32)], 1) if n else np.zeros((0, 2), np.float32)
        lm = np.asarray(self.lm[a:b])
        return dict(row=int(self.qrow[j]), split=int(self.qsplit[j]), built=bool(self.built[j]),
                    reached=bool(self.reached[j]), gold_pos=int(self.gold_pos[j]), cands=c, X16=X,
                    casc=casc, lm=lm, lm_gated=lm * self.reuse_gate(c) if n else lm,
                    hard=np.asarray(self.hard[a:b]), lm_ok=bool(self.lm_ok[j]))

    def arm_features(self, j, arm, q=None):
        q = q or self.query(j)
        X = q["X16"]
        if arm in ("arm3",):
            X = np.concatenate([X, q["lm"][:, None]], 1)
        elif arm in ("arm4", "arm5"):
            X = np.concatenate([X, q["lm_gated"][:, None]], 1)
        if arm == "arm5":
            X = np.concatenate([X, q["casc"]], 1)
        return X.astype(np.float32)
