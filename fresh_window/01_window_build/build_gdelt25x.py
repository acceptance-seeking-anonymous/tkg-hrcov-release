#!/usr/bin/env python3
"""build_gdelt25x.py -- build the GDELT25X forward-holdout dataset from the public GDELT 2.0 event exports.

Source: http://data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip (15-minute exports, official server; no crawling)
and the official CAMEO.eventcodes.txt.  Recipe (identical to the build used in the paper):
  * window 2025-01-01 .. 2025-09-30 (DATEADDED day); train < 2025-07-01, valid July 2025, test August-September 2025
  * root events (IsRootEvent = 1) with both actors coded and at least one actor country in the Middle-East set ME
  * entity = CountryCode + Type1Code of the actor (actors without a country code use their ActorCode)
  * relation = CAMEO EventCode; time = DATEADDED day index (day 0 = first day with events)
  * events whose SOURCEURL first appeared on an EARLIER day are dropped (re-codings of older articles)
  * entities with < 10 train events and relations with < 5 train events are dropped; self-loops dropped
  * output in the RE-GCN / harness format: train.txt valid.txt test.txt (s r o t), stat.txt, entity2id.txt,
    relation2id.txt, timestamp2id.txt

Reproducing the released snapshot exactly: some 15-minute export files were not retrievable when the dataset was built
(1,681 of 26,208).  Pass --skip-list gdelt25x_missing_export_files.txt so that the same files are left out; the
SHA-256 of the resulting files must then equal GDELT25X.sha256 (python3 build_gdelt25x.py --check --out DIR).  The names
column of relation2id.txt in the snapshot came from an earlier relation table rather than CAMEO.eventcodes.txt for some codes;
gdelt25x_relation_names.tsv (read by default) carries those names, so that relation2id.txt can match the snapshot byte for
byte when the rebuilt relation set is the same (ids never depend on names).  This builder was not re-run end to end for the code
release; --check reports which files match.

  python3 build_gdelt25x.py --out $DATA_DIR/GDELT25X --raw $DATA_DIR/raw/gdelt --skip-list gdelt25x_missing_export_files.txt
"""
import os, sys, json, re, time, io, zipfile, datetime, argparse, hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ME = {"ISR", "PSE", "IRN", "EGY", "SYR", "LBN", "JOR", "SAU", "ARE", "QAT", "TUR", "IRQ", "YEM", "KWT", "BHR", "OMN", "LBY"}
CN = {"ISR": "Israel", "PSE": "Palestine", "IRN": "Iran", "EGY": "Egypt", "SYR": "Syria", "LBN": "Lebanon", "JOR": "Jordan",
      "SAU": "Saudi Arabia", "ARE": "United Arab Emirates", "QAT": "Qatar", "TUR": "Turkey", "IRQ": "Iraq", "YEM": "Yemen",
      "KWT": "Kuwait", "BHR": "Bahrain", "OMN": "Oman", "LBY": "Libya", "USA": "United States", "RUS": "Russia", "CHN": "China",
      "GBR": "United Kingdom", "FRA": "France", "DEU": "Germany", "UKR": "Ukraine", "IND": "India", "PAK": "Pakistan",
      "EUR": "Europe", "AFG": "Afghanistan", "ITA": "Italy", "ESP": "Spain", "CAN": "Canada", "AUS": "Australia", "JPN": "Japan",
      "KOR": "South Korea", "PRK": "North Korea", "SDN": "Sudan", "TUN": "Tunisia", "DZA": "Algeria", "MAR": "Morocco",
      "CYP": "Cyprus", "GRC": "Greece", "AZE": "Azerbaijan", "ARM": "Armenia", "BRA": "Brazil", "ZAF": "South Africa",
      "NLD": "Netherlands", "BEL": "Belgium", "IRL": "Ireland", "NOR": "Norway", "SWE": "Sweden", "CHE": "Switzerland",
      "AUT": "Austria", "POL": "Poland", "HUN": "Hungary", "SOM": "Somalia", "ETH": "Ethiopia", "NGA": "Nigeria", "KEN": "Kenya",
      "MEA": "Middle East", "AFR": "Africa", "ASA": "Asia", "WST": "the West", "SAS": "South Asia", "MDE": "Middle East"}
TN = {"GOV": "government", "MIL": "military", "OPP": "opposition", "REB": "rebels", "BUS": "business", "CVL": "civilians",
      "MED": "media", "COP": "police", "JUD": "judiciary", "LEG": "legislature", "EDU": "education", "HLH": "health sector",
      "REF": "refugees", "IGO": "intergovernmental organisation", "NGO": "NGO", "SPY": "intelligence", "LAB": "labour",
      "ELI": "elites", "CRM": "criminals", "UAF": "armed group", "AGR": "agriculture", "ENV": "environmental group",
      "MOD": "moderates", "RAD": "radicals", "INS": "insurgents", "SEP": "separatists", "REL": "religious group", "NGM": "NGO",
      "HRI": "human-rights group", "DEV": "development group", "SET": "settlers", "ACT": "activists", "MNC": "multinational"}
COLS = {1: "sqldate", 5: "a1code", 6: "a1name", 7: "a1cc", 12: "a1t1", 15: "a2code", 16: "a2name", 17: "a2cc", 22: "a2t1",
        25: "root", 26: "code", 59: "added", 60: "url"}


def fetch(url, fn):
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return True
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                b = r.read()
            open(fn, "wb").write(b); return True
        except Exception:
            time.sleep(2 * (k + 1))
    return False


def read_one(fn):
    try:
        with zipfile.ZipFile(fn) as z:
            raw = z.read(z.namelist()[0])
    except Exception:
        return None
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, keep_default_na=False, usecols=list(COLS), quoting=3,
                     on_bad_lines="skip")
    df.columns = [COLS[c] for c in df.columns]
    df = df[(df.root == "1") & (df.a1code != "") & (df.a2code != "")]
    df = df[df.a1cc.isin(ME) | df.a2cc.isin(ME)]
    return df


def ekey(code, cc, t1):
    if cc:
        return cc + (t1 if t1 else "")
    return code


def ename(key, cc, t1, freqname):
    if cc:
        base = CN.get(cc, freqname.get(cc, cc).title())
        return base + (" " + TN.get(t1, t1.lower()) if t1 else "")
    return freqname.get(key, key).title()




def log_fn(t0):
    return lambda *x: print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "GDELT25X"))
    ap.add_argument("--raw", default=os.path.join(os.environ.get("DATA_DIR", "data"), "raw", "gdelt"))
    ap.add_argument("--start", default="20250101"); ap.add_argument("--end", default="20251001")       # [start, end)
    ap.add_argument("--valid", default="20250701"); ap.add_argument("--test", default="20250801")      # valid / test start days
    ap.add_argument("--skip-list", default=None, help="export URLs to leave out (snapshot reproduction)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check", action="store_true", help="only compare the files in --out with GDELT25X.sha256")
    ap.add_argument("--relation-names", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdelt25x_relation_names.tsv"),
                    help="relation-name table of the released snapshot (code, id, name); default: the copy next to this script")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.check:
        ok = True
        for ln in open(os.path.join(here, "GDELT25X.sha256")):
            h, fn = ln.split()
            got = hashlib.sha256(open(os.path.join(a.out, fn), "rb").read()).hexdigest()
            print(fn, "OK" if got == h else "MISMATCH"); ok &= got == h
        sys.exit(0 if ok else 1)
    t0 = time.time(); log = log_fn(t0)
    gd = a.raw; DD = a.out; os.makedirs(gd, exist_ok=True); os.makedirs(DD, exist_ok=True)
    G0, G1, GV, GT = a.start, a.end, a.valid, a.test
    d0 = datetime.datetime.strptime(G0, "%Y%m%d"); d1 = datetime.datetime.strptime(G1, "%Y%m%d")
    stamps = []
    t = d0
    while t < d1:
        stamps.append(t.strftime("%Y%m%d%H%M%S")); t += datetime.timedelta(minutes=15)
    jobs = [(f"http://data.gdeltproject.org/gdeltv2/{s}.export.CSV.zip", f"{gd}/{s}.export.CSV.zip") for s in stamps]
    skip = set()
    if a.skip_list:
        skip = {ln.strip() for ln in open(a.skip_list) if ln.strip()}
        jobs = [j for j in jobs if j[0] not in skip]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda x: fetch(*x), jobs))
    log("downloaded", sum(ok), "/", len(ok), "(skipped by list:", len(skip), ")")
    open(f"{DD}/missing_export_files.txt", "w").write("\n".join(j[0] for j, k in zip(jobs, ok) if not k) + "\n")
    cam_fn = os.path.join(os.path.dirname(gd), "CAMEO.eventcodes.txt")
    for u in ("https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt",
              "http://data.gdeltproject.org/documentation/CAMEO.eventcodes.txt"):
        if fetch(u, cam_fn):
            break
    cname = {}
    if os.path.exists(cam_fn):
        cam = pd.read_csv(cam_fn, sep="\t", dtype=str, keep_default_na=False)
        for c_, nm in zip(cam.iloc[:, 0], cam.iloc[:, 1]):
            cname.setdefault(c_, nm.strip().capitalize())
    if a.relation_names and os.path.exists(a.relation_names):   # names column of the released relation2id.txt (ids are unaffected)
        for ln in open(a.relation_names, encoding="utf-8"):
            c_, _, nm = ln.rstrip("\n").split("\t", 2); cname[c_] = nm
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(read_one, [j[1] for j in jobs if os.path.exists(j[1])]) if p is not None and len(p)]
    df = pd.concat(parts, ignore_index=True); del parts
    df["day"] = df.added.str[:8]
    df = df[(df.day >= G0) & (df.day < G1)]
    log("ME root events", len(df))
    first = df.groupby("url").day.min()
    df = df[df.day.values == first.loc[df.url].values]
    days = sorted(df.day.unique()); d2i = {d: i for i, d in enumerate(days)}
    df["t"] = df.day.map(d2i)
    df["k1"] = [ekey(c, cc, t1) for c, cc, t1 in zip(df.a1code, df.a1cc, df.a1t1)]
    df["k2"] = [ekey(c, cc, t1) for c, cc, t1 in zip(df.a2code, df.a2cc, df.a2t1)]
    df = df[df.k1 != df.k2]
    df["split"] = np.where(df.day < GV, 0, np.where(df.day < GT, 1, 2))
    tr = df[df.split == 0]
    ef = pd.concat([tr.k1, tr.k2]).value_counts(); keep_e = set(ef[ef >= 10].index)
    rf = tr.code.value_counts(); keep_r = set(rf[rf >= 5].index)
    df = df[df.k1.isin(keep_e) & df.k2.isin(keep_e) & df.code.isin(keep_r)]
    freqname = {}
    for k, nm in pd.concat([pd.DataFrame({"k": df.k1, "n": df.a1name}), pd.DataFrame({"k": df.k2, "n": df.a2name})]).groupby("k").n:
        vc = nm[nm != ""].value_counts(); freqname[k] = vc.index[0] if len(vc) else k
    ents = sorted(set(df.k1) | set(df.k2)); e2i = {e: i for i, e in enumerate(ents)}
    rels = sorted(set(df.code)); r2i = {r: i for i, r in enumerate(rels)}
    meta = {}
    for k, cc, t1 in zip(pd.concat([df.k1, df.k2]), pd.concat([df.a1cc, df.a2cc]), pd.concat([df.a1t1, df.a2t1])):
        meta.setdefault(k, (cc, t1))
    names = []
    seen = {}
    for e in ents:
        cc, t1 = meta[e]
        nm = ename(e, cc, t1, freqname)
        if nm in seen:
            nm = f"{nm} ({e})"
        seen[nm] = 1; names.append(nm)
    df["s"] = df.k1.map(e2i); df["o"] = df.k2.map(e2i); df["r"] = df.code.map(r2i)
    q_all = df[["s", "r", "o", "t", "split"]].to_numpy().astype(np.int64)
    uq = np.unique(q_all, axis=0)
    for k, nm in {0: "train", 1: "valid", 2: "test"}.items():
        q = uq[uq[:, 4] == k][:, :4]
        q = q[np.lexsort((q[:, 2], q[:, 1], q[:, 0], q[:, 3]))]
        np.savetxt(f"{DD}/{nm}.txt", q, fmt="%d", delimiter="\t")
    open(f"{DD}/stat.txt", "w").write(f"{len(ents)}\t{len(rels)}\t0\n")
    with open(f"{DD}/entity2id.txt", "w") as f:
        for i, nm in enumerate(names):
            f.write(f"{nm}\t{i}\n")
    with open(f"{DD}/relation2id.txt", "w") as f:
        for c, i in r2i.items():
            f.write(f"{c}\t{i}\t{cname.get(c, c)}\n")
    with open(f"{DD}/timestamp2id.txt", "w") as f:
        for d, i in d2i.items():
            f.write(f"{d}\t{i}\n")
    spl = {nm: dict(n_facts=int((uq[:, 4] == k).sum()), t_range=[int(uq[uq[:, 4] == k][:, 3].min()), int(uq[uq[:, 4] == k][:, 3].max())])
           for k, nm in {0: "train", 1: "valid", 2: "test"}.items()}
    summ = dict(source=f"GDELT 2.0 event exports (15-min), {G0}..{G1} (end exclusive), valid from {GV}, test from {GT}",
                n_files=int(sum(ok)), n_files_requested=len(ok), n_skipped_by_list=len(skip), n_event_rows=int(len(df)),
                n_unique_quads=int(len(uq)), n_entities=len(ents), n_relations=len(rels), n_days=len(days), splits=spl,
                filters="root events; >=1 actor country in ME; URL first-seen day only; entity >=10 / relation >=5 train events",
                elapsed_s=round(time.time() - t0, 1))
    json.dump(summ, open(f"{DD}/build_summary.json", "w"), indent=1)
    log(json.dumps(summ))


if __name__ == "__main__":
    main()
