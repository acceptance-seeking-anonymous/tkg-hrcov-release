#!/usr/bin/env python3
"""phase2_memtest / ps2_build_gdelt.py = pilot_story/ps_build_gdelt.py + optional matched key space (PS2_KEYS_OUT / PS2_KEYS_IN) + entity_keys.txt.
Original docstring:
pilot_story / ps_build_gdelt.py -- POST-CUTOFF contamination slice: GDELT 2.0 events 2025-01-01..2025-06-30 (CPU).

Official source only (data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip, 15-minute event exports; no crawling).
  * root events (IsRootEvent = 1) with both actors coded and at least one actor's country in the Middle-East set ME
  * entity = (CountryCode, Type1Code) of the actor (e.g. ISR+GOV -> "Israel government"); actors without a country code
    use their ActorCode (e.g. IGO groups) with the most frequent ActorName as display name
  * relation = CAMEO EventCode (names from the official CAMEO.eventcodes.txt)
  * time = DATEADDED day (day 0 = 2025-01-01); text = headline words of the shipped SOURCEURL slug (no full text)
  * events whose SOURCEURL first appeared on an EARLIER day are dropped (re-codings of old articles), so every document is
    dated on the day of all its events and the gold event's own article is never visible (dated t)
  * entities with < 10 train events and relations with < 5 train events are dropped
  * splits: train Jan 1 - Apr 30, valid May 1 - May 31, test Jun 1 - Jun 30 (all after the Llama-3.1 cutoff, Dec 2023)
Outputs: same files as ps_build.py under $PS_P (data/GDELT25, corpus/).
"""
import os, sys, json, re, time, io, zipfile, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from ps_common import P, DD, RAW, CORPUS, CSR, jdump, log_fn
from ps_build import slug_text

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
    try:                                                      # phase2_memtest addition 3 of 3 (robustness): some 2022/2023 exports are empty
        df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, keep_default_na=False, usecols=list(COLS), quoting=3,
                         on_bad_lines="skip")
    except pd.errors.EmptyDataError:                          # an empty export holds no events: same result as a skipped file
        return None
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


def main():
    t0 = time.time(); log = log_fn(t0)
    gd = f"{RAW}/gdelt"; os.makedirs(gd, exist_ok=True); os.makedirs(DD, exist_ok=True); os.makedirs(CORPUS, exist_ok=True)
    G0, G1 = os.environ.get("PS_G_START", "20250101"), os.environ.get("PS_G_END", "20250701")       # [start, end)
    GV, GT = os.environ.get("PS_G_VALID", "20250501"), os.environ.get("PS_G_TEST", "20250601")       # valid / test start days
    d0 = datetime.datetime.strptime(G0, "%Y%m%d"); d1 = datetime.datetime.strptime(G1, "%Y%m%d")
    stamps = []
    t = d0
    while t < d1:
        stamps.append(t.strftime("%Y%m%d%H%M%S")); t += datetime.timedelta(minutes=15)
    jobs = [(f"http://data.gdeltproject.org/gdeltv2/{s}.export.CSV.zip", f"{gd}/{s}.export.CSV.zip") for s in stamps]
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda x: fetch(*x), jobs))
    log("downloaded", sum(ok), "/", len(ok))
    open(f"{CORPUS}/missing_export_files.txt", "w").write("\n".join(j[0] for j, k in zip(jobs, ok) if not k) + "\n")
    for u in ("https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt",
              "http://data.gdeltproject.org/documentation/CAMEO.eventcodes.txt"):
        if fetch(u, f"{RAW}/CAMEO.eventcodes.txt"):
            break
    cname = {}
    mid = os.environ.get("PS_MIDEAST_REL", "/home/[REDACTED:~ANON]/cs_scratch/open_ideas_harness/tracks/pilot_story/data/MIDEAST/relation2id.txt")
    if os.path.exists(mid):                                   # MidEast-TE relation names (same CAMEO codes) as fallback
        for ln in open(mid):
            c_, _, nm = ln.rstrip("\n").split("\t"); cname[c_] = nm
    if os.path.exists(f"{RAW}/CAMEO.eventcodes.txt"):
        cam = pd.read_csv(f"{RAW}/CAMEO.eventcodes.txt", sep="\t", dtype=str, keep_default_na=False)
        for c_, nm in zip(cam.iloc[:, 0], cam.iloc[:, 1]):
            cname.setdefault(c_, nm.strip().capitalize())
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
    # ---- phase2_memtest addition 1 of 3: optional matched entity/relation space
    if os.environ.get("PS2_KEYS_OUT"):                       # pass 1: record the window's own filtered key sets
        json.dump(dict(ents=sorted(keep_e), rels=sorted(keep_r)), open(os.environ["PS2_KEYS_OUT"], "w"))
    if os.environ.get("PS2_KEYS_IN"):                        # pass 2: restrict to the keys shared by both windows
        _K = json.load(open(os.environ["PS2_KEYS_IN"]))
        keep_e = keep_e & set(_K["ents"]); keep_r = keep_r & set(_K["rels"])
        log("matched key space", len(keep_e), "entities", len(keep_r), "relations")
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
    uq, inv = np.unique(q_all, axis=0, return_inverse=True); inv = inv.ravel()
    for k, nm in {0: "train", 1: "valid", 2: "test"}.items():
        a = uq[uq[:, 4] == k][:, :4]
        a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0], a[:, 3]))]
        np.savetxt(f"{DD}/{nm}.txt", a, fmt="%d", delimiter="\t")
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
    with open(f"{DD}/entity_keys.txt", "w") as f:            # phase2_memtest addition 2 of 3: actor key per entity id
        for i, e in enumerate(ents):
            f.write(f"{e}\t{i}\n")
    np.save(f"{CORPUS}/quads.npy", uq)
    urls = df.groupby("url").t.min().sort_values(kind="stable")
    md = urls.index.to_numpy(); d_t = urls.to_numpy().astype(np.int64); u2i = {u: i for i, u in enumerate(md)}
    n_src = dict(fulltext=0, slug=0, none=0)
    with open(f"{CORPUS}/docs.jsonl", "w") as f:
        for i, u in enumerate(md):
            txt = slug_text(u); src = "slug" if len(txt.split()) >= 4 else "none"
            if src == "none":
                txt = ""
            n_src[src] += 1
            f.write(json.dumps(dict(i=i, md5=u, t=int(d_t[i]), src=src, text=txt, url=u)) + "\n")
    dq = pd.DataFrame({"d": df.url.map(u2i).values.astype(np.int64), "q": inv}).drop_duplicates()
    o = np.argsort(dq.d.values, kind="stable"); dd_, qq_ = dq.d.values[o], dq.q.values[o]
    CSR(np.searchsorted(dd_, np.arange(len(md) + 1)), qq_).save(f"{CORPUS}/doc_quads.npz")
    o = np.lexsort((dq.d.values, dq.q.values)); q2, d2 = dq.q.values[o], dq.d.values[o]
    CSR(np.searchsorted(q2, np.arange(len(uq) + 1)), d2).save(f"{CORPUS}/quad_docs.npz")
    de = pd.DataFrame({"d": np.concatenate([dd_, dd_]), "e": np.concatenate([uq[qq_, 0], uq[qq_, 2]])}).drop_duplicates().sort_values(["d", "e"])
    CSR(np.searchsorted(de.d.values, np.arange(len(md) + 1)), de.e.values).save(f"{CORPUS}/doc_ents.npz")
    de2 = de.sort_values(["e", "d"], kind="stable")
    CSR(np.searchsorted(de2.e.values, np.arange(len(ents) + 1)), de2.d.values).save(f"{CORPUS}/ent_docs.npz")
    np.save(f"{CORPUS}/doc_t.npy", d_t)
    spl = {nm: dict(n_facts=int((uq[:, 4] == k).sum()), t_range=[int(uq[uq[:, 4] == k][:, 3].min()), int(uq[uq[:, 4] == k][:, 3].max())])
           for k, nm in {0: "train", 1: "valid", 2: "test"}.items()}
    summ = dict(host=os.environ.get("PS_HOST", "?"), source=f"GDELT 2.0 event exports (15-min), {G0}..{G1} (end exclusive), valid from {GV}, test from {GT}",
                n_files=int(sum(ok)), n_files_expected=len(ok), n_event_rows=int(len(df)), n_unique_quads=int(len(uq)),
                n_entities=len(ents), n_relations=len(rels), n_days=len(days), splits=spl,
                docs=dict(n_referenced=int(len(md)), src_counts=n_src, frac_slug=n_src["slug"] / len(md)),
                filters="root events; >=1 actor country in ME; URL first-seen day only; entity >=10 / relation >=5 train events",
                elapsed_s=round(time.time() - t0, 1))
    jdump(summ, "build_gdelt_summary.json"); jdump(summ, f"{CORPUS}/build_summary.json")
    log(json.dumps(summ))


if __name__ == "__main__":
    main()
