#!/usr/bin/env python3
"""phase2_fresh_window / fh_freeze.py -- sha256 manifest of every Stage-1 frozen file (stdlib only; host python3).
  python3 fh_freeze.py --fh <track dir> --out <manifest.json>              # write
  python3 fh_freeze.py --fh <track dir> --verify <manifest.json>           # Stage 2: exit 3 on any mismatch / missing file
Groups: rule, code (feature / training / scoring / decision code incl. the TiRGN model tree), data (GDELTFW splits),
base_model (TiRGN checkpoint + epoch log), valid_base_rows, prep (valid sample, TE tables, valid strata),
controls (RecB selection + valid CSR, blend/pool valid grids, selection), features_valid, reranker (6 frozen models,
their select scores and ranks, training json).  Times are UTC (file mtime and manifest creation).
"""
import os, sys, json, glob, hashlib, argparse, datetime


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b: break
            h.update(b)
    return h.hexdigest()


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def groups(FH):
    W = f"{FH}/GDELTFW/work"
    g = {
        "rule": [f"{FH}/freeze/fh_decision_rule.json"],
        "code": sorted(glob.glob(f"{FH}/code/*.py") + glob.glob(f"{FH}/code/*.sh")) +
                sorted(p for p in glob.glob(f"{FH}/stage/TiRGN/**/*.py", recursive=True) if "__pycache__" not in p),
        "data": sorted(glob.glob(f"{FH}/data/GDELTFW/*.txt")),
        "base_model": [f"{W}/base_tirgn/tirgn_GDELTFW_seed42.pt", f"{W}/base_tirgn/tirgn_GDELTFW_epoch_log.json"],
        "valid_base_rows": [f"{W}/fullrows/tirgn_GDELTFW_valid_scores_f32.bin", f"{W}/base_tirgn/tirgn_scores_GDELTFW_valid.npz",
                            f"{W}/base_tirgn/rows_valid_log.json"],
        "prep": [f"{W}/valid_sample_idx.npy", f"{W}/te_GDELTFW.npz", f"{W}/strata_valid.npz", f"{W}/prep_summary_stage1.json"],
        "controls": [f"{W}/recb/recb_selection_GDELTFW.json", f"{W}/recb/recb_csr_valid_GDELTFW.npz",
                     f"{W}/controls/blend_valid_gdeltfw_tirgn.json", f"{W}/controls/blend_valid_gdeltfw_tirgn.npz",
                     f"{W}/controls/sel_gdeltfw.json"],
        "features_valid": sorted(glob.glob(f"{W}/lists_tirgn/valid_*")) + sorted(glob.glob(f"{W}/feat_tirgn/valid_*")) +
                          [f"{W}/feat_tirgn/meta.json"],
        "reranker": sorted(glob.glob(f"{W}/train_tirgn/models/*")) + [f"{W}/train_tirgn/ranks_select.npz", f"{W}/train_tirgn/train_fh.json"],
    }
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fh", required=True); ap.add_argument("--out", default=""); ap.add_argument("--verify", default="")
    a = ap.parse_args()
    if a.verify:
        M = json.load(open(a.verify)); bad = []
        for grp, items in M["groups"].items():
            for it in items:
                p = it["path"]
                if it.get("missing"):
                    bad.append(("missing_at_freeze", p)); continue
                if not os.path.exists(p): bad.append(("missing", p)); continue
                if sha256(p) != it["sha256"]: bad.append(("sha256_mismatch", p))
        out = dict(verified_utc=utc(datetime.datetime.now().timestamp()), n_checked=sum(len(v) for v in M["groups"].values()), problems=bad)
        print(json.dumps(out, indent=1))
        sys.exit(3 if bad else 0)
    M = dict(name="freeze_manifest_remote", created_utc=utc(datetime.datetime.now().timestamp()), host="serverA",
             track_dir=a.fh, groups={})
    for grp, paths in groups(a.fh).items():
        rows = []
        for p in paths:
            if not os.path.exists(p):
                rows.append(dict(path=p, missing=True)); continue
            st = os.stat(p)
            rows.append(dict(path=p, sha256=sha256(p), bytes=st.st_size, mtime_utc=utc(st.st_mtime)))
        M["groups"][grp] = rows
    M["n_files"] = sum(len(v) for v in M["groups"].values())
    M["n_missing"] = sum(1 for v in M["groups"].values() for r in v if r.get("missing"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(M, open(a.out, "w"), indent=1)
    print(json.dumps(dict(n_files=M["n_files"], n_missing=M["n_missing"], created_utc=M["created_utc"])))


if __name__ == "__main__":
    main()
