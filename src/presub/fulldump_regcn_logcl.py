"""
retriever track / fulldump_regcn_logcl.py -- INFERENCE-ONLY full-score dump (valid + test) for the frozen
harness RE-GCN / LogCL checkpoints (dumps/<model>_<DS>_seed42.pt), replicating base_dump.py's history
protocol and row order exactly (train snapshots roll history unscored; valid rows use train + rolling valid;
test rows use train + valid + rolling test; per snapshot rows = forward then inverse).

Writes <out>/<model>_<DS>_<split>_scores_f32.bin (row-major log-softmax [n, N]) + _meta.npz (sro, gold_score).
Imports the harness model code read-only.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
H = os.path.expandvars("$WORK_DIR")
sys.path.insert(0, H)
from regcn import read_quads, load_stat, group_by_time, make_edges, get_history, REGCN   # noqa
from logcl import LogCL, GlobalHist                                                        # noqa
from oih_common import DATASETS                                                            # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["regcn", "logcl"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    t0 = time.time()
    dd = DATASETS[a.dataset]["dd"]
    ck = torch.load(f"{H}/dumps/{a.model}_{a.dataset}_seed42.pt", map_location=dev, weights_only=False)
    num_ent, num_rel = ck["num_ent"], ck["num_rel"]
    h_dim, n_layers, L, dropout = 200, 2, 10, 0.2
    model = (REGCN if a.model == "regcn" else LogCL)(num_ent, num_rel, h_dim, n_layers, dropout, L).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    train = read_quads(os.path.join(dd, "train.txt")); valid = read_quads(os.path.join(dd, "valid.txt"))
    test = read_quads(os.path.join(dd, "test.txt"))
    tr_pairs = group_by_time(train); va_pairs = group_by_time(valid); te_pairs = group_by_time(test)
    os.makedirs(a.out, exist_ok=True)
    gh = GlobalHist(num_ent) if a.model == "logcl" else None
    hist_edges = []
    for (ts, triples) in tr_pairs:
        if len(triples) == 0:
            continue
        hist_edges.append(make_edges(triples, num_rel, dev))
        if gh is not None: gh.update(triples, num_rel)
    log = dict(model=a.model, dataset=a.dataset, device=dev)

    @torch.no_grad()
    def run(pairs, split):
        base = f"{a.out}/{a.model}_{a.dataset}_{split}"
        f = open(base + "_scores_f32.bin", "wb"); sro = []; gsl = []; n = 0
        for (ts, triples) in pairs:
            if len(triples) == 0:
                continue
            s_all = np.concatenate([triples[:, 0], triples[:, 2]])
            rel_all = np.concatenate([triples[:, 1], triples[:, 1] + num_rel])
            obj_all = np.concatenate([triples[:, 2], triples[:, 0]])
            s = torch.tensor(s_all, device=dev); rel = torch.tensor(rel_all, device=dev)
            if a.model == "regcn":
                scores, _, _ = model(hist_edges[-L:], s, rel)
            else:
                scores, _, _, _, _ = model(hist_edges[-L:], s, rel, gh.freq_vec(s_all, rel_all, dev))
            logp = F.log_softmax(scores, dim=1).float()
            gold = torch.tensor(obj_all, device=dev)
            gsl.append(logp.gather(1, gold.unsqueeze(1)).squeeze(1).cpu().numpy())
            sro.append(np.stack([s_all, rel_all, obj_all, np.full(len(s_all), ts)], 1).astype(np.int32))
            f.write(np.ascontiguousarray(logp.cpu().numpy()).tobytes()); n += len(s_all)
            hist_edges.append(make_edges(triples, num_rel, dev))
            if gh is not None: gh.update(triples, num_rel)
        f.close()
        np.savez(base + "_meta.npz", sro=np.concatenate(sro), gold_score=np.concatenate(gsl), n_rows=np.int64(n), n_ent=np.int64(num_ent))
        log[split] = dict(rows=int(n), s=round(time.time() - t0, 1))
        print(split, log[split], flush=True)
    run(va_pairs, "valid"); run(te_pairs, "test")
    json.dump(log, open(f"fulldump_{a.model}_{a.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
