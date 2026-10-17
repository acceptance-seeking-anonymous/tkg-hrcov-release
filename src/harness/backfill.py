"""
harness / backfill.py -- the pluggable BACKFILL interface used by fulltest_eval.py, and the
reference implementation (GraphBridge G12 arm5 over the relation-expanded bridge).

Interface (anything that produces this npz can be evaluated):
    backfill_<name>_<DS>_<split>.npz
      rows    [n]     int64  dump row ids the backfill scored (queries it was asked about)
      dump    []      str    path of the base dump those row ids index
      ids     [n,B]   int32  backfill candidates, best first, -1 padded (may overlap the base top-K)
      scores  [n,B]   f32    backfill scores (higher = better; only the ORDER is used by the fusion rule)
      wide_gold_rank [n] int32  (optional diagnostic) gold rank in the backfill's full list, 0 = absent
      wide_size [n]   int32  (optional) size of the backfill's full candidate list
Programmatic form:
    class Backfill:   name; def candidates(self, rows) -> (ids[n,B], scores[n,B])
The fusion rule (ConfidenceSwitch in fulltest_eval.py) decides WHICH queries get backfilled (gold-free
base confidence) and WHERE the block is inserted; the backfill only supplies ranked candidates.
"""
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
from bridge_cache import BridgeCache
from stage2_arms import make_net
from oih_common import HARNESS, jdump


class Backfill:
    name = "abstract"

    def candidates(self, rows):
        raise NotImplementedError


class G12Arm5Backfill(Backfill):
    """Scores a bridge cache (valid_pool / test_pool) with a saved arm5 stage-2 ranker."""
    name = "g12arm5"

    def __init__(self, dataset, cohort, model_path, base="tirgn", top_b=200):
        self.bc = BridgeCache(dataset, cohort, base=base)
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.net = make_net(ck["nf"]); self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.mu = ck["mu"]; self.sd = ck["sd"]; self.top_b = top_b
        self.model_path = model_path
        self.row2j = {int(r): j for j, r in enumerate(np.asarray(self.bc.qrow))}

    def score_query(self, j):
        q = self.bc.query(j)
        if not q["built"] or len(q["cands"]) == 0:
            return q, None
        X = self.bc.arm_features(j, "arm5", q)
        with torch.no_grad():
            s = self.net(torch.tensor((X - self.mu) / self.sd)).squeeze(1).numpy()
        return q, s

    def run_all(self, out_path):
        bc = self.bc; n = bc.nq; B = self.top_b
        ids = np.full((n, B), -1, np.int32); sc = np.full((n, B), -np.inf, np.float32)
        wgr = np.zeros(n, np.int32); wsz = np.zeros(n, np.int32)
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
        for j in range(n):
            q, s = self.score_query(j)
            if s is None:
                continue
            order = np.argsort(-s); c = q["cands"][order]
            k = min(B, len(c)); ids[j, :k] = c[:k]; sc[j, :k] = s[order][:k]
            wsz[j] = len(c)
            if q["reached"]:
                wgr[j] = int(np.where(order == q["gold_pos"])[0][0]) + 1
        np.savez(out_path, rows=np.asarray(bc.qrow).astype(np.int64), dump=np.array(bc.meta["dump_rows_from"]),
                 ids=ids, scores=sc, wide_gold_rank=wgr, wide_size=wsz, model=np.array(self.model_path))
        return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cohorts", default="valid_pool,test_pool")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    mp_ = f"{HARNESS}/results/arm5_{a.dataset}_seed{a.seed}.pt"
    out = {}
    for coh in a.cohorts.split(","):
        bf = G12Arm5Backfill(a.dataset, coh, mp_)
        p = f"{HARNESS}/results/backfill_g12arm5_{a.dataset}_{coh}.npz"
        bf.run_all(p)
        d = np.load(p)
        out[coh] = dict(path=p, n=int(len(d["rows"])), wide_reach=round(float((d["wide_gold_rank"] > 0).mean()), 4),
                        median_wide_size=int(np.median(d["wide_size"][d["wide_size"] > 0])) if (d["wide_size"] > 0).any() else 0,
                        elapsed_s=round(time.time() - t0, 1))
        print(coh, json.dumps(out[coh]), flush=True)
    jdump(out, f"backfill_{a.dataset}.json")
