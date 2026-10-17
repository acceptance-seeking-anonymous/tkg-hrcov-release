#!/usr/bin/env python3
"""finite_scan.py -- count non-finite values (NaN, +inf, -inf) in every stored floating-point array under the given roots.
Covers .npy (memory-mapped, chunked), .f32 (raw float32 row dumps, chunked), .npz (stored members memory-mapped at their
zip offset; compressed members read whole) and .pt (torch tensors in state dicts / dicts, incl. mu/sd). Integer and bool
arrays are listed as 'not floating' and not counted. Read-only.
  python3 finite_scan.py --roots group=dir[,dir...] [group=dir ...] --out finite_scan_host.json --workers 8
"""
import os, sys, json, time, zipfile, argparse
import numpy as np
from multiprocessing import Pool

CH = 1 << 26  # elements per chunk (~256 MB float32)


def count_arr(a):
    """a: ndarray or memmap (any shape). Chunked over the flattened view."""
    if not np.issubdtype(a.dtype, np.floating):
        return dict(dtype=str(a.dtype), shape=list(a.shape), floating=False)
    flat = a.reshape(-1); n = flat.shape[0]; nan = pinf = ninf = 0
    for lo in range(0, n, CH):
        x = np.asarray(flat[lo:lo + CH])
        if x.dtype == np.float16: x = x.astype(np.float32)
        fin = np.isfinite(x)
        if not fin.all():
            bad = x[~fin]; nan += int(np.isnan(bad).sum()); pinf += int((bad == np.inf).sum()); ninf += int((bad == -np.inf).sum())
    return dict(dtype=str(a.dtype), shape=list(a.shape), floating=True, n=int(n), nan=nan, posinf=pinf, neginf=ninf)


def npz_members(path):
    out = {}
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    fh = open(path, "rb")
    for zi in infos:
        name = zi.filename[:-4] if zi.filename.endswith(".npy") else zi.filename
        try:
            if zi.compress_type == zipfile.ZIP_STORED:
                fh.seek(zi.header_offset); hdr = fh.read(30)
                nlen = int.from_bytes(hdr[26:28], "little"); elen = int.from_bytes(hdr[28:30], "little")
                start = zi.header_offset + 30 + nlen + elen
                fh.seek(start); ver = np.lib.format.read_magic(fh)
                rh = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = rh(fh)
                off = fh.tell()
                if dtype.hasobject: out[name] = dict(error="object array"); continue
                cnt = int(np.prod(shape)) if len(shape) else 1
                a = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=(cnt,))
                r = count_arr(a); r["shape"] = list(shape); out[name] = r
            else:
                with zipfile.ZipFile(path) as zf:
                    with zf.open(zi) as f:
                        a = np.lib.format.read_array(f, allow_pickle=False)
                out[name] = count_arr(a)
        except Exception as e:  # noqa: BLE001
            out[name] = dict(error=f"{type(e).__name__}: {e}"[:300])
    fh.close()
    return out


def pt_members(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out = {}

    def walk(o, pre):
        if isinstance(o, torch.Tensor):
            t = o.detach()
            out[pre] = count_arr(t.float().numpy() if t.dtype in (torch.float16, torch.bfloat16) else t.numpy())
        elif isinstance(o, np.ndarray):
            out[pre] = count_arr(o)
        elif isinstance(o, dict):
            for k, v in o.items(): walk(v, f"{pre}.{k}" if pre else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o): walk(v, f"{pre}[{i}]")
        elif isinstance(o, float):
            out[pre] = dict(dtype="python float", floating=True, n=1, nan=int(o != o), posinf=int(o == float("inf")), neginf=int(o == float("-inf")))
    walk(obj, "")
    return out


def scan_file(path):
    t0 = time.time()
    try:
        sz = os.path.getsize(path)
        if path.endswith(".npy"):
            a = np.load(path, mmap_mode="r", allow_pickle=False); m = {"": count_arr(a)}
        elif path.endswith(".f32"):
            a = np.memmap(path, dtype=np.float32, mode="r"); m = {"": count_arr(a)}
        elif path.endswith(".npz"):
            m = npz_members(path)
        elif path.endswith(".pt"):
            m = pt_members(path)
        else:
            return None
        return dict(path=path, bytes=int(sz), members=m, seconds=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        return dict(path=path, error=f"{type(e).__name__}: {e}"[:300])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--min_bytes", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    jobs = []
    for spec in a.roots:
        g, dirs = spec.split("=", 1)
        for d in dirs.split(","):
            if not os.path.exists(d): print("missing root", d, flush=True); continue
            if os.path.isfile(d): jobs.append((g, d)); continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith((".npy", ".npz", ".f32", ".pt")):
                        p = os.path.join(root, f)
                        try:
                            if os.path.getsize(p) >= a.min_bytes: jobs.append((g, p))
                        except OSError:
                            pass
    seen = {}; uj = []
    for g, p in jobs:
        rp = os.path.realpath(p)
        if rp in seen or not os.path.exists(rp): continue
        seen[rp] = g; uj.append((g, p))
    uj.sort(key=lambda x: -os.path.getsize(x[1]))
    print(f"{len(uj)} files, {sum(os.path.getsize(p) for _, p in uj) / 1e9:.1f} GB", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_file, [p for _, p in uj], chunksize=1)):
            if r is None: continue
            res.append(r)
            if i % 100 == 0: print(f"[{time.time() - t0:7.1f}s] {i}/{len(uj)}", flush=True)
    grp = {os.path.realpath(p): g for g, p in uj}
    for r in res: r["group"] = grp.get(os.path.realpath(r["path"]), "?")
    summ = {}
    for r in res:
        s = summ.setdefault(r["group"], dict(files=0, bytes=0, float_arrays=0, float_values=0, nan=0, posinf=0, neginf=0, errors=0,
                                             files_with_nan_or_posinf=[], files_with_neginf=[], files_with_errors=[]))
        s["files"] += 1; s["bytes"] += r.get("bytes", 0)
        if "error" in r: s["errors"] += 1; s["files_with_errors"].append(r["path"]); continue
        nn = pi = ni = 0
        for k, m in r["members"].items():
            if "error" in m: s["errors"] += 1; s["files_with_errors"].append(f"{r['path']}::{k}"); continue
            if not m.get("floating"): continue
            s["float_arrays"] += 1; s["float_values"] += m["n"]; nn += m["nan"]; pi += m["posinf"]; ni += m["neginf"]
        s["nan"] += nn; s["posinf"] += pi; s["neginf"] += ni
        if nn or pi: s["files_with_nan_or_posinf"].append(r["path"])
        if ni: s["files_with_neginf"].append(r["path"])
    out = dict(name="finite_scan", host=os.environ.get("HD_HOST", ""), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
               seconds=round(time.time() - t0, 1), roots=a.roots, summary=summ, files=sorted(res, key=lambda r: r["path"]))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({g: {k: v for k, v in s.items() if not isinstance(v, list)} for g, s in summ.items()}, indent=1))


if __name__ == "__main__":
    main()
