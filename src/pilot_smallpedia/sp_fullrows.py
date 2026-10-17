#!/usr/bin/env python3
"""pilot_smallpedia / sp_fullrows.py -- replay a frozen RE-GCN / LogCL base exactly as harness base_dump.py does
(ground-truth online history, last 10 snapshots; LogCL global copy-history rolled over every earlier snapshot) and
write the FULL 47,433-entity log-prob row of every valid and test query to a float32 memmap (rows in split_queries
order).  Replaces pilot-1's per-list GPU lookup (pc_lookup.py --mode lookup): with N = 47k all downstream stages read
exact base log-probs from these rows.  Also re-derives the filtered gold rank and compares it with the dump.
  python3 sp_fullrows.py --base regcn
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from sp_common import (H, DS, DD, DUMPS, CKPT, FULLROWS, N_ENT, Splits, split_queries, jdump, log_fn)
from regcn import read_quads, group_by_time, make_edges, REGCN
from logcl import LogCL, GlobalHist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    t0 = time.time(); log = log_fn(t0)
    dev = "cuda"
    S = Splits(DS); N, R = S.n_ent, S.R
    ck = torch.load(CKPT[a.base], map_location="cpu", weights_only=False)
    L = 10
    model = (REGCN if a.base == "regcn" else LogCL)(N, R, 200, 2, 0.2, L).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    pairs = {sp: group_by_time(read_quads(os.path.join(DD, f"{sp}.txt"))) for sp in ("train", "valid", "test")}
    F_ = S.filter_index()
    D = np.load(DUMPS[a.base])
    dsp = D["split"]; d_gs = D["gold_score"]; d_gf = D["gold_rank_filt"]; d_sro = D["sro"]
    rows_sp = {sp: np.flatnonzero(dsp == c) for sp, c in (("valid", 1), ("test", 2))}
    MM = {sp: np.memmap(FULLROWS[a.base][sp], dtype=np.float32, mode="w+", shape=(len(rows_sp[sp]), N)) for sp in ("valid", "test")}
    stats = {sp: dict(n=0, n_rank_mismatch=0, max_abs_rank_diff=0, max_abs_gold_lp_diff=0.0) for sp in ("valid", "test")}
    for sp in ("valid", "test"):
        assert (d_sro[rows_sp[sp]].astype(np.int64) == split_queries(S, sp)).all()
    hist_edges = []; gh = GlobalHist(N) if a.base == "logcl" else None
    with torch.no_grad():
        for sp in ("train", "valid", "test"):
            pos0 = 0
            for k, (ts, triples) in enumerate(pairs[sp]):
                if not len(triples):
                    continue
                nq = 2 * len(triples)
                if sp != "train":
                    s_all = np.concatenate([triples[:, 0], triples[:, 2]]); r_all = np.concatenate([triples[:, 1], triples[:, 1] + R])
                    o_all = np.concatenate([triples[:, 2], triples[:, 0]])
                    hist = hist_edges[-L:]
                    for b0 in range(0, nq, 2048):
                        bb = slice(b0, min(nq, b0 + 2048))
                        st = torch.tensor(s_all[bb], device=dev); rt = torch.tensor(r_all[bb], device=dev)
                        if a.base == "regcn":
                            sc, _, _ = model(hist, st, rt)
                        else:
                            sc, _, _, _, _ = model(hist, st, rt, gh.freq_vec(s_all[bb], r_all[bb], dev))
                        lp = F.log_softmax(sc, dim=1)
                        gold = torch.tensor(o_all[bb], device=dev)
                        glp = lp.gather(1, gold.unsqueeze(1)).squeeze(1)
                        fk = S.fkey(s_all[bb], r_all[bb], np.full(bb.stop - bb.start, ts))
                        ri, ci = [], []
                        i0 = np.searchsorted(F_["keys"], fk)
                        for qq, (key, j_) in enumerate(zip(fk, i0)):
                            if j_ < len(F_["keys"]) and F_["keys"][j_] == key:
                                for oo in F_["objs"][F_["start"][j_]:F_["end"][j_]]:
                                    if int(oo) != int(o_all[bb][qq]):
                                        ri.append(qq); ci.append(int(oo))
                        lf = lp.clone()
                        if ri:
                            lf[torch.tensor(ri, device=dev), torch.tensor(ci, device=dev)] = float("-inf")
                        rk = ((lf > glp.unsqueeze(1)).sum(1) + 1).cpu().numpy()
                        lo_, hi_ = pos0 + bb.start, pos0 + bb.stop
                        MM[sp][lo_:hi_] = lp.cpu().numpy()
                        dr = rows_sp[sp][lo_:hi_]
                        assert (d_sro[dr, 0] == s_all[bb]).all() and (d_sro[dr, 1] == r_all[bb]).all() and (d_sro[dr, 2] == o_all[bb]).all()
                        st_ = stats[sp]; st_["n"] += len(rk)
                        diff = np.abs(rk.astype(np.int64) - d_gf[dr].astype(np.int64))
                        st_["n_rank_mismatch"] += int((diff > 0).sum()); st_["max_abs_rank_diff"] = max(st_["max_abs_rank_diff"], int(diff.max()))
                        st_["max_abs_gold_lp_diff"] = max(st_["max_abs_gold_lp_diff"], float(np.abs(glp.cpu().numpy() - d_gs[dr]).max()))
                        del lp, lf, sc
                    pos0 += nq
                hist_edges.append(make_edges(triples, R, dev))
                if gh is not None:
                    gh.update(triples, R)
            if sp != "train":
                MM[sp].flush(); log(sp, stats[sp])
    for sp in stats:
        stats[sp]["frac_rank_equal"] = 1 - stats[sp]["n_rank_mismatch"] / max(stats[sp]["n"], 1)
    res = dict(base=a.base, host="serverA (GPU, tkg-gpu:latest)", gpu=torch.cuda.get_device_name(0), stats=stats,
               files={sp: FULLROWS[a.base][sp] for sp in MM}, elapsed_s=round(time.time() - t0, 1))
    jdump(res, f"fullrows_{a.base}.json")
    os.system(f"chmod a+r {FULLROWS[a.base]['valid']} {FULLROWS[a.base]['test']}")
    log("ALL DONE", json.dumps(stats))


if __name__ == "__main__":
    main()
