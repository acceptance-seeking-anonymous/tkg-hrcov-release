#!/usr/bin/env python3
"""pilot_icews / pi_relnorm.py -- relation-normalised (RELNORM) id map for tkgl-icews + per-merge evidence (CPU).

From 2020-01-01 the TGB edge list uses 119 new relation types; 117 of them are an existing CAMEO code with its
leading zero(s) stripped (e.g. '42' vs '042').  For every relation type that first appears on/after 2020-01-01 this
script looks for an original code o with o.lstrip('0') == new code and o != new code, and records the evidence:
  (1) string rule (zero-stripped identity),
  (2) temporal complementarity: last use of o, first use of the new code, number of days on which both are used,
  (3) volume continuity: mean facts / month of o in 2018-01..2019-12 vs the new code in 2020-01..2021-12,
  (4) usage similarity: cosine between the (subject, object) entity-count profiles of o (2018-2019) and of the new
      code (2020-2021), and the rank of o among ALL codes active in 2018-2019 by that cosine (1 = most similar).
A merge is accepted iff (1) holds with a unique o AND o is not used on/after 2020-01-01 on any day the new code is
used (overlap_days == 0).  Unmatched new codes are kept as their own relations.
Outputs: $P/stage/relmap.npy ([391] new id -> original id, identity elsewhere), relnorm_mapping.csv, relnorm_summary.json
  python3 pi_relnorm.py
"""
import os, sys, json, time, zipfile, io, csv, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pi_common import P, DD, ZIP, N_ENT, N_REL, Splits, DS, jdump, log_fn

t0 = time.time(); log = log_fn(t0)
os.makedirs(f"{P}/stage", exist_ok=True)
# harness relation id -> TGB relation_type value -> CAMEO code string
h2raw = {}
for ln in open(f"{DD}/relation2id.txt"):
    k, v = ln.rstrip("\n").split("\t"); h2raw[int(v)] = k
with zipfile.ZipFile(ZIP) as z:
    mapn = [n for n in z.namelist() if "edgemapping" in n][0]
    with z.open(mapn) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8")); hdr = next(rd)
        type2code = {}
        for row in rd:
            if len(row) >= 2:
                type2code[row[1].strip()] = row[0].strip()
code = {h: type2code[h2raw[h]] for h in range(N_REL)}
assert len(set(code.values())) == N_REL, "edge mapping not injective"
ident = all(h2raw[h] == str(h) for h in range(N_REL))
day2raw = {}
for ln in open(f"{DD}/timestamp2id.txt"):
    k, v = ln.split(); day2raw[int(v)] = int(k)
def ymd(day):
    return datetime.datetime.utcfromtimestamp(day2raw[int(day)]).strftime("%Y-%m-%d")
def day_of(datestr):
    ts = datetime.datetime.strptime(datestr, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    days = np.array(sorted(day2raw)); raws = np.array([day2raw[d] for d in days])
    return int(days[np.searchsorted(raws, ts - 12 * 3600)])     # TGB day stamps are local-midnight (05:00 UTC)
S = Splits(DS)
q = np.concatenate([S.q["train"], S.q["valid"], S.q["test"]])
split = np.concatenate([np.full(len(S.q[k]), i, np.int8) for i, k in enumerate(("train", "valid", "test"))])
s, r, o, t = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
D2020 = day_of("2020-01-01"); D2018 = day_of("2018-01-01"); D2022 = day_of("2022-01-01")
log("days", D2018, D2020, D2022, ymd(D2020))
first = np.full(N_REL, 10 ** 9); last = np.full(N_REL, -1)
np.minimum.at(first, r, t); np.maximum.at(last, r, t)
cnt = np.bincount(r, minlength=N_REL); cnt_sp = np.stack([np.bincount(r[split == k], minlength=N_REL) for k in range(3)], 1)
new_ids = [h for h in range(N_REL) if first[h] >= D2020]
# usage days per relation (as sets of day ids)
od = np.lexsort((t, r)); r_s = r[od]; t_s = t[od]
bnd = np.searchsorted(r_s, np.arange(N_REL + 1))
days_of = [np.unique(t_s[bnd[h]:bnd[h + 1]]) for h in range(N_REL)]
# entity profiles: old period 2018-2019, new period 2020-2021 (subject and object counts concatenated)
def profile(mask):
    rr = r[mask]; ss = s[mask]; oo = o[mask]
    return rr, ss, oo
mo = (t >= D2018) & (t < D2020); mn = (t >= D2020) & (t < D2022)
def prof_mat(mask, rels):
    out = {}
    rr, ss, oo = r[mask], s[mask], o[mask]
    for h in rels:
        m = rr == h
        v = np.concatenate([np.bincount(ss[m], minlength=N_ENT), np.bincount(oo[m], minlength=N_ENT)]).astype(np.float64)
        nrm = np.linalg.norm(v)
        out[h] = v / nrm if nrm > 0 else v
    return out
active_old = [h for h in range(N_REL) if h not in new_ids and ((r[mo] == h).any())]
P_old = prof_mat(mo, active_old); P_new = prof_mat(mn, new_ids)
A = np.stack([P_old[h] for h in active_old])
months_old = 24.0; months_new = 24.0
rows = []; relmap = np.arange(N_REL)
for h in sorted(new_ids, key=lambda x: -cnt[x]):
    c_new = code[h]
    cands = [g for g in range(N_REL) if g != h and code[g] != c_new and code[g].lstrip("0") == c_new and code[g].startswith("0")]
    rec = dict(new_id=h, new_code=c_new, new_first=ymd(first[h]), new_last=ymd(last[h]), new_n_facts=int(cnt[h]),
               new_n_test_facts=int(cnt_sp[h, 2]), n_string_candidates=len(cands))
    sims = A @ P_new[h] if h in P_new else np.zeros(len(active_old))
    order = np.argsort(-sims)
    rec["best_old_by_cosine_code"] = code[active_old[order[0]]] if len(order) else ""
    rec["best_old_by_cosine"] = round(float(sims[order[0]]), 4) if len(order) else None
    if len(cands) == 1:
        g = cands[0]
        ov = int(len(np.intersect1d(days_of[g], days_of[h])))
        n_old_post = int(((r == g) & (t >= D2020)).sum())
        mo_old = float(((r == g) & mo).sum()) / months_old; mo_new = float(((r == h) & mn).sum()) / months_new
        if g in active_old:
            j = active_old.index(g); cs = float(sims[j]); rank = int(np.flatnonzero(order == j)[0]) + 1
        else:
            cs, rank = None, None
        accept = ov == 0
        rec.update(old_id=g, old_code=code[g], old_first=ymd(first[g]), old_last=ymd(last[g]), old_n_facts=int(cnt[g]),
                   old_n_facts_on_or_after_2020=n_old_post, overlap_days=ov, old_facts_per_month_2018_19=round(mo_old, 1),
                   new_facts_per_month_2020_21=round(mo_new, 1), volume_ratio_new_over_old=(round(mo_new / mo_old, 3) if mo_old > 0 else None),
                   cosine_profile_old_vs_new=(round(cs, 4) if cs is not None else None), cosine_rank_of_old_among_active_codes=rank,
                   n_active_old_codes=len(active_old), merged=bool(accept),
                   match_rule="zero-stripped CAMEO code identity, unique candidate, no shared usage day")
        if accept:
            relmap[h] = g
    else:
        rec.update(old_id=None, old_code=None, merged=False,
                   match_rule=("no zero-padded original code" if not cands else "ambiguous: %d candidates" % len(cands)))
    rows.append(rec)
np.save(f"{P}/stage/relmap.npy", relmap)
cols = ["new_id", "new_code", "old_id", "old_code", "merged", "match_rule", "n_string_candidates", "new_first", "new_last", "new_n_facts",
        "new_n_test_facts", "old_first", "old_last", "old_n_facts", "old_n_facts_on_or_after_2020", "overlap_days",
        "old_facts_per_month_2018_19", "new_facts_per_month_2020_21", "volume_ratio_new_over_old", "cosine_profile_old_vs_new",
        "cosine_rank_of_old_among_active_codes", "n_active_old_codes", "best_old_by_cosine_code", "best_old_by_cosine"]
with open("relnorm_mapping.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for rec in rows:
        w.writerow({k: rec.get(k, "") for k in cols})
os.system(f"cp relnorm_mapping.csv {P}/stage/")
merged = [x for x in rows if x.get("merged")]
tr_rels = set(np.unique(S.q["train"][:, 1]).tolist())
test_r = S.q["test"][:, 1]
summ = dict(host="serverA (CPU, tkg-gpu:latest)", relation_ids_equal_tgb_types=ident, n_new_relation_types=len(new_ids),
            first_day_2020=ymd(D2020), n_merged=len(merged), unmatched=[(x["new_id"], x["new_code"], x["match_rule"]) for x in rows if not x.get("merged")],
            relmap_identity_on_train_relations=bool(all(relmap[h] == h for h in tr_rels)),
            n_merged_old_used_after_2020=int(sum(1 for x in merged if x["old_n_facts_on_or_after_2020"] > 0)),
            cosine_rank1_share=float(np.mean([x["cosine_rank_of_old_among_active_codes"] == 1 for x in merged])) if merged else None,
            cosine_rank_le3_share=float(np.mean([(x["cosine_rank_of_old_among_active_codes"] or 99) <= 3 for x in merged])) if merged else None,
            median_cosine=float(np.median([x["cosine_profile_old_vs_new"] for x in merged if x["cosine_profile_old_vs_new"] is not None])) if merged else None,
            median_volume_ratio=float(np.median([x["volume_ratio_new_over_old"] for x in merged if x["volume_ratio_new_over_old"]])) if merged else None,
            test_facts_relation_unseen_in_train=float(np.mean(~np.isin(test_r, list(tr_rels)))),
            test_facts_relation_unseen_in_train_after_relnorm=float(np.mean(~np.isin(relmap[test_r], list(tr_rels)))),
            test_facts_on_merged_codes=float(np.mean(relmap[test_r] != test_r)),
            elapsed_s=round(time.time() - t0, 1))
jdump(summ, "relnorm_summary.json"); jdump(summ, f"{P}/stage/relnorm_summary.json")
os.system(f"chmod -R a+rX {P}/stage")
log(json.dumps(summ)[:1500])
