#!/usr/bin/env python3
"""data/download_tgb.py -- download the TGB 2.0 tkgl datasets with the py-tgb 2.3.0 URL table and convert them to the harness layout.

  python3 data/download_tgb.py --names tkgl-polecat tkgl-smallpedia [tkgl-icews] [--root $WORK_DIR/tracks/datasets]

For each name: <root>/raw/<name>.zip (the archive py-tgb downloads; it also holds the official test negatives <name>_test_ns.pkl
that the evaluation scripts extract) and <root>/harness_data/<name>/ (train/valid/test/stat/entity2id/relation2id/timestamp2id,
written by src/datasets/tgb_to_harness.py; split = TGB generate_splits, counts checked against the TGB 2.0 paper).
tkgl-smallpedia also needs its static edge list (inside the same zip; read directly by src/pilot_smallpedia).
Sizes: tkgl-smallpedia ~0.1 GB, tkgl-polecat ~0.3 GB, tkgl-icews ~3 GB (zip).
"""
import argparse, hashlib, os, subprocess, sys
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.join(os.path.dirname(HERE), "src", "datasets", "tgb_to_harness.py")


def url_of(name):
    from tgb.utils.info import DATA_URL_DICT      # py-tgb==2.3.0
    return DATA_URL_DICT[name]


def fetch(url, dst):
    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    os.replace(tmp, dst)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["tkgl-polecat", "tkgl-smallpedia"])
    ap.add_argument("--root", default=os.path.join(os.environ.get("WORK_DIR", "."), "tracks", "datasets"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.root, "raw"), exist_ok=True)
    for name in a.names:
        z = os.path.join(a.root, "raw", f"{name}.zip")
        if not os.path.exists(z):
            print(f"downloading {name}", flush=True); fetch(url_of(name), z)
        print(f"{name}.zip sha256 {sha256(z)}", flush=True)
        out = os.path.join(a.root, "harness_data", name)
        if not os.path.exists(os.path.join(out, "stat.txt")):
            subprocess.run([sys.executable, CONVERT, "--zip", z, "--name", name, "--out", out], check=True)
        print(f"{name}: {out}", flush=True)


if __name__ == "__main__":
    main()
