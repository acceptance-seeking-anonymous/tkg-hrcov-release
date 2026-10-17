#!/usr/bin/env python3
"""pilot_smallpedia / sp_leakage_audit.py -- direct-static-edge leakage audit (addendum; local, CPU).

Population: test queries whose (s, o) pair is a direct static Wikidata edge (either direction, any property).
Unit: (static property, direction relative to the canonical temporal fact, temporal relation); inverse queries (r >= 283) are
mapped to the forward fact (o, r - 283, s).  Classes SAME / INVERSE / IMPLIES / RELATED / UNRELATED:
  * rule first (Wikidata metadata fetched from wikidata.org wbgetentities: identical property, P1696 inverse property,
    symmetric-property class Q18647518 / self-inverse);
  * otherwise an LLM judge (platform reasoning model) given English labels + descriptions; a second pass of the same judge
    and a smaller model are kept as reliability checks (agreement reported).  Judgements are cached in
    leakage_pair_judgements.json (the LLM step is not re-run here).
Query-level class = strongest class over its static edges (SAME > INVERSE > IMPLIES > RELATED > UNRELATED).
Outputs: leakage_audit.json (fractions, top-20 pairs, HRCOV_ST - HRCOV contribution by class per base, reliability)
         smallpedia_leakage_relation_pairs.csv (all pairs, for human check)
  python3 sp_leakage_audit.py --root aggroot --out leakage_audit.json --csv smallpedia_leakage_relation_pairs.csv
"""
import os, json, csv, argparse, collections
import numpy as np

N_ENT = 47433
CL = ["SAME", "INVERSE", "IMPLIES", "RELATED", "UNRELATED"]
ORDER = {c: i for i, c in enumerate(CL)}
RESTATE = ("SAME", "INVERSE", "IMPLIES")
BASES = ("regcn", "logcl", "tirgn")
OUTCOME_PAIRS = [("P131", "fwd", "P1366"), ("P3842", "fwd", "P1366"), ("P131", "rev", "P1365"), ("P3842", "rev", "P1365"),
                 ("P54", "fwd", "P647")]   # static present-day state that records the outcome of the forecast event (post-hoc)


def rr_mean(R, arm):
    ks = [f"{arm}_s{s}" for s in (0, 1, 2) if f"{arm}_s{s}" in R]
    return np.mean([1.0 / R[k].astype(np.float64) for k in ks], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default="leakage_audit.json")
    ap.add_argument("--csv", default="smallpedia_leakage_relation_pairs.csv")
    a = ap.parse_args()
    lab = json.load(open(f"{a.root}/wikidata_property_labels.json"))
    J = json.load(open(f"{a.root}/leakage_pair_judgements.json"))
    prel = json.load(open(f"{a.root}/static_names.json"))["prel_names"]
    relt = [ln.split("\t")[0] for ln in open(f"{a.root}/relation2id.txt")]; NR = len(relt)
    L = lambda pid: lab.get(pid, {}).get("label") or pid
    z = np.load(f"{a.root}/static_edges.npz"); h, t_, p = z["h"].astype(np.int64), z["t"].astype(np.int64), z["p"].astype(np.int64)
    m = (h < N_ENT) & (t_ < N_ENT); key = h[m] * N_ENT + t_[m]; pp = p[m]; o_ = np.argsort(key, kind="stable"); key = key[o_]; pp = pp[o_]

    def edges(x, y):
        k = x * N_ENT + y; i = np.searchsorted(key, k); j = np.searchsorted(key, k, side="right"); return pp[i:j]
    q = dict(np.load(f"{a.root}/bundle_tirgn/test_q.npz")); sro = q["sro"].astype(np.int64); n = len(sro)
    fwd = sro[:, 1] < NR
    S_ = np.where(fwd, sro[:, 0], sro[:, 2]); O_ = np.where(fwd, sro[:, 2], sro[:, 0]); R_ = np.where(fwd, sro[:, 1], sro[:, 1] - NR)
    qedges = {}
    for i in range(n):
        e1 = edges(S_[i], O_[i]); e2 = edges(O_[i], S_[i])
        if len(e1) or len(e2):
            qedges[i] = [(prel[int(x)], "fwd") for x in e1] + [(prel[int(x)], "rev") for x in e2]
    direct = np.zeros(n, bool); direct[list(qedges)] = True
    assert (direct == q["st_S"].astype(bool)).all()
    jd = {(r["static_property"], r["direction"], r["temporal_relation"]): r for r in J["judgements"]}
    pairq = collections.defaultdict(list)
    for i, es in qedges.items():
        for (ps, d) in es:
            pairq[(ps, d, relt[R_[i]])].append(i)
    assert set(pairq) == set(jd), "pair set differs from cached judgements"
    for k, r in jd.items():
        r["final"] = r["rule"].upper() if r["rule"] else r["reasoning"]
    nd_ = len(qedges)

    def qclass(field):
        out = {}
        for i, es in qedges.items():
            cs = []
            for (ps, d) in es:
                r = jd[(ps, d, relt[R_[i]])]
                cs.append(r["rule"].upper() if r["rule"] else r[field])
            out[i] = min(cs, key=lambda c: ORDER[c])
        return out
    qc = qclass("reasoning"); qc2 = qclass("reasoning_pass2"); qcu = qclass("utility")
    frac = lambda d: {c: round(sum(v == c for v in d.values()) / nd_, 4) for c in CL}
    A = dict(population=dict(n_test=int(n), n_direct_edge_queries=int(nd_), share=round(nd_ / n, 4), n_relation_pairs=len(jd),
                             n_temporal_relations=len({k[2] for k in jd}), n_static_properties=len({k[0] for k in jd})))
    A["query_class_fraction"] = dict(primary=frac(qc), primary_pass2=frac(qc2), secondary_smaller_model=frac(qcu))
    A["restating_fraction"] = {nm: round(sum(v in RESTATE for v in d.values()) / nd_, 4) for nm, d in (("primary", qc), ("primary_pass2", qc2), ("secondary_smaller_model", qcu))}
    w = {k: len(v) for k, v in pairq.items()}; W = sum(w.values())
    ag = lambda f1, f2, b=False: round(sum(w[k] * ((jd[k][f1] in RESTATE) == (jd[k][f2] in RESTATE) if b else jd[k][f1] == jd[k][f2]) for k in jd) / W, 3)
    A["judge_reliability"] = dict(primary_model=J["judge_model_primary"], secondary_model=J["judge_model_secondary"],
                                  pass1_vs_pass2=dict(pair=round(np.mean([r["reasoning"] == r["reasoning_pass2"] for r in jd.values()]), 3), query_weighted=ag("reasoning", "reasoning_pass2"),
                                                      query_weighted_restating_binary=ag("reasoning", "reasoning_pass2", True)),
                                  primary_vs_secondary=dict(pair=round(np.mean([r["reasoning"] == r["utility"] for r in jd.values()]), 3), query_weighted=ag("reasoning", "utility"),
                                                            query_weighted_restating_binary=ag("reasoning", "utility", True)),
                                  rule_labelled_pairs=int(sum(1 for r in jd.values() if r["rule"])),
                                  rule_vs_primary_agreement=round(np.mean([r["rule"].upper() == r["reasoning"] for r in jd.values() if r["rule"]]), 3))
    # contributions per base
    cls_arr = np.array(["none"] * n, dtype=object)
    for i, c in qc.items():
        cls_arr[i] = c
    restate = np.isin(cls_arr, list(RESTATE))
    outc = np.zeros(n, bool)
    for k in OUTCOME_PAIRS:
        outc[pairq.get(k, [])] = True
    G = {}; RR = {}
    for b in BASES:
        R = dict(np.load(f"{a.root}/bundle_{b}/ranks_test.npz"))
        x = rr_mean(R, "HRCOV"); y = rr_mean(R, "HRCOV_ST"); d = y - x; RR[b] = (x, y)
        e = dict(total_HRCOV_ST_minus_HRCOV=round(float(d.mean()), 6),
                 direct_edge_contribution=round(float(d[direct].sum() / n), 6), no_direct_edge_contribution=round(float(d[~direct].sum() / n), 6))
        for c in CL:
            mk = cls_arr == c
            e[c] = dict(n=int(mk.sum()), contribution=round(float(d[mk].sum() / n), 6), MRR_HRCOV=round(float(x[mk].mean()), 4), MRR_HRCOV_ST=round(float(y[mk].mean()), 4))
        e["restating (SAME+INVERSE+IMPLIES)"] = dict(n=int(restate.sum()), contribution=round(float(d[restate].sum() / n), 6),
                                                     MRR_HRCOV=round(float(x[restate].mean()), 4), MRR_HRCOV_ST=round(float(y[restate].mean()), 4))
        e["outcome-recording pairs (post-hoc)"] = dict(n=int(outc.sum()), contribution=round(float(d[outc].sum() / n), 6),
                                                      MRR_HRCOV=round(float(x[outc].mean()), 4), MRR_HRCOV_ST=round(float(y[outc].mean()), 4))
        G[b] = e
    A["HRCOV_ST_gain_by_class"] = G
    A["outcome_recording_pairs_post_hoc"] = dict(pairs=[dict(static=k[0], static_label=L(k[0]), direction=k[1], temporal=k[2], temporal_label=L(k[2]), n=len(pairq.get(k, [])))
                                                        for k in OUTCOME_PAIRS if k in pairq],
                                                 note="static present-day state that records the outcome of the forecast event (e.g. a municipality 'located in' the "
                                                      "municipality it was merged into vs temporal 'replaced by'; a player 'member of' the team vs temporal 'drafted by'); "
                                                      "semantically judged UNRELATED/RELATED, but empirically the strongest lift; identified post-hoc from the top-20 table")
    rows = []
    xt, yt = RR["tirgn"]
    for k in sorted(jd, key=lambda k: (-len(pairq[k]), k)):
        r = jd[k]; idx = np.array(pairq[k])
        rows.append(dict(static_property=k[0], static_label=L(k[0]), direction=("s' -p-> o'" if k[1] == "fwd" else "o' -p-> s'"),
                         temporal_relation=k[2], temporal_label=L(k[2]), n_test_queries=int(len(idx)), share_of_direct=round(len(idx) / nd_, 4),
                         class_final=r["final"], class_source=("rule (Wikidata identity / inverse property)" if r["rule"] else "LLM judge (reasoning model)"),
                         judge_pass2=r["reasoning_pass2"], judge_smaller_model=r["utility"],
                         judge_reason=(r["reasoning_text"] or "").split("|", 1)[-1].strip()[:160],
                         MRR_HRCOV_tirgn=round(float(xt[idx].mean()), 4), MRR_HRCOV_ST_tirgn=round(float(yt[idx].mean()), 4),
                         lift_tirgn=round(float((yt[idx] - xt[idx]).mean()), 4)))
    A["top20_relation_pairs"] = rows[:20]
    A["notes"] = ["s' / o' = subject / object of the canonical (forward) temporal fact; direction is that of the static edge relative to it",
                  "labels/descriptions fetched from www.wikidata.org (wbgetentities) on 2026-10-12; property semantics may have changed since the TGB snapshot",
                  "LLM classification of non-rule pairs is a judgement aid; the top-20 table is provided for human check"]
    with open(a.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(A, open(a.out, "w"), indent=1)
    print(json.dumps(dict(n_direct=nd_, frac=A["query_class_fraction"]["primary"], restating=A["restating_fraction"],
                          gain={b: (G[b]["direct_edge_contribution"], G[b]["restating (SAME+INVERSE+IMPLIES)"]["contribution"]) for b in BASES})))


if __name__ == "__main__":
    main()
