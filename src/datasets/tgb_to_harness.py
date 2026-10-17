#!/usr/bin/env python3
"""Convert a TGB 2.0 tkgl-*.zip into the RE-GCN/TiRGN-style data directory read by the
harness (base_dump.py: read_quads/load_stat):
    train.txt / valid.txt / test.txt   "s<TAB>r<TAB>o<TAB>t"   (t = 0-based index of the distinct timestamp)
    stat.txt                            "num_entities<TAB>num_relations<TAB>0"
    entity2id.txt / relation2id.txt     "<original id><TAB><int id>"
    timestamp2id.txt                    "<original ts><TAB><int id>"
    conversion_log.json                 split rule, counts, checks
Split = TGB generate_splits (np.quantile(ts,[0.70,0.85]); ts<=q70 train, q70<ts<=q85 valid, ts>q85 test),
verified against arXiv:2509.09474 Tab.3 counts for smallpedia/polecat/icews/wikidata.
Inverse relations are NOT written: the harness adds r+R for head queries itself (both directions).
Optional --drop-self-loops writes a filtered copy (tkgl-polecat has 10.4% s==o test facts); default keeps them
so that numbers stay comparable with the TGB leaderboard.
NOTE: entity ids follow first appearance in the CSV, not py-tgb's internal ids; use py-tgb's Evaluator with
its own id mapping for leaderboard-official numbers.
"""
import argparse, os, json, sys
import numpy as np
def _ts(v):
    try:
        return int(float(v))
    except ValueError:
        return int(v.replace('-', '')[:8])   # YYYY-MM-DD -> YYYYMMDD (order-preserving)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zipfile, io, csv

EXPECTED = {"tkgl-smallpedia": (387757, 81033, 81586), "tkgl-polecat": (1246556, 266736, 266318),
            "tkgl-icews": (10861600, 2326157, 2325689), "tkgl-wikidata": (6982503, 1434950, 1438750)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--drop-self-loops", action="store_true")
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    name = [n for n in z.namelist() if n.endswith(".csv") and "edgelist" in n and "static" not in n][0]
    ent, rel, S, R, O, T = {}, {}, [], [], [], []
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8")); h = [x.strip().lower() for x in next(rd)]
        it = [h.index(c) for c in ("ts", "timestamp", "date", "time") if c in h][0]; ih, itl, ir = h.index("head"), h.index("tail"), h.index("relation_type")
        for row in rd:
            S.append(ent.setdefault(row[ih], len(ent))); O.append(ent.setdefault(row[itl], len(ent)))
            R.append(rel.setdefault(row[ir], len(rel))); T.append(_ts(row[it]))
    S, R, O, T = map(np.array, (S, R, O, T))
    q70, q85 = np.quantile(T, [0.70, 0.85])
    lab = np.where(T <= q70, 0, np.where(T <= q85, 1, 2))
    tsv, tid = np.unique(T, return_inverse=True)
    keep = np.ones(len(S), bool)
    if a.drop_self_loops:
        keep = S != O
    os.makedirs(a.out, exist_ok=True)
    counts = []
    for k, fn in enumerate(["train", "valid", "test"]):
        m = (lab == k) & keep
        idx = np.flatnonzero(m); idx = idx[np.argsort(tid[idx], kind="stable")]
        np.savetxt(os.path.join(a.out, fn + ".txt"), np.stack([S[idx], R[idx], O[idx], tid[idx]], 1), fmt="%d", delimiter="\t")
        counts.append(int(m.sum()))
    open(os.path.join(a.out, "stat.txt"), "w").write(f"{len(ent)}\t{len(rel)}\t0\n")
    with open(os.path.join(a.out, "entity2id.txt"), "w") as f:
        for k, v in ent.items(): f.write(f"{k}\t{v}\n")
    with open(os.path.join(a.out, "relation2id.txt"), "w") as f:
        for k, v in rel.items(): f.write(f"{k}\t{v}\n")
    with open(os.path.join(a.out, "timestamp2id.txt"), "w") as f:
        for i, v in enumerate(tsv.tolist()): f.write(f"{v}\t{i}\n")
    exp = EXPECTED.get(a.name)
    log = dict(name=a.name, edgelist=name, n_entities=len(ent), n_relations=len(rel), n_timestamps=int(len(tsv)),
               split_rule="TGB generate_splits quantiles 0.70/0.85", q70=float(q70), q85=float(q85), counts=counts,
               expected_counts=exp, counts_match=(None if exp is None or a.drop_self_loops else list(exp) == counts),
               self_loops_dropped=bool(a.drop_self_loops), n_self_loops=int((S == O).sum()))
    json.dump(log, open(os.path.join(a.out, "conversion_log.json"), "w"), indent=1); print(json.dumps(log))

if __name__ == "__main__":
    main()
