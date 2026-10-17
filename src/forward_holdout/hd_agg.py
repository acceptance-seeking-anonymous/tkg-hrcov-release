#!/usr/bin/env python3
"""hardening / hd_agg.py -- aggregation, paired bootstraps and the pre-registered decision -> hardening_verdict.json.

Inputs (config json, --cfg): per dataset {name, dd, strata, recb_ranks, recb_sel, recb_meta, published{}, bases{label:
{official: npz, test_json, valid_json, arch, valid_MRR(optional override), host, rb_ranks(optional)}}}.
All MRRs of claim-bearing arms are the official py-tgb Evaluator per-query RRs (rr_<arm> in the official npz);
re-rankers are averaged over seeds 0-2 per query.  Harness ranks (hrank_<arm>) are reported alongside.
"""
import os, sys, json, argparse
import numpy as np

MARGIN = 0.005


def boot(d, B=2000, seed=0):
    """Poisson(1) paired bootstrap of mean(d) (harness paired_bootstrap method); returns stats incl. the draws."""
    d = np.asarray(d, np.float64); n = len(d); rng = np.random.default_rng(seed)
    nz = np.flatnonzero(d); dn = d[nz]; bs = np.empty(B)
    for b0 in range(0, B, 100):
        bb = min(100, B - b0)
        w = rng.poisson(1.0, size=(bb, len(nz)))
        bs[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), CI95=[float(lo), float(hi)], p_one_sided=float((bs <= 0).mean()),
                p_le_margin=float((bs <= MARGIN).mean()), B=B, n=int(n), n_changed=int(len(nz)))


def mets(rr, rank):
    rk = np.asarray(rank, np.float64)
    return dict(MRR=float(np.mean(rr)), H1=float(np.mean(rk <= 1)), H3=float(np.mean(rk <= 3)), H10=float(np.mean(rk <= 10)))


def hmets(hr):
    hr = np.asarray(hr, np.float64)
    return dict(MRR=float(np.mean(1 / hr)), H1=float(np.mean(hr <= 1)), H3=float(np.mean(hr <= 3)), H10=float(np.mean(hr <= 10)))


def load_quads(fp):
    return np.loadtxt(fp, dtype=np.int64, usecols=(0, 1, 2, 3)).reshape(-1, 4)


def split_queries(q, R):
    order = np.argsort(q[:, 3], kind="stable"); q = q[order]
    uts, st = np.unique(q[:, 3], return_index=True); en = np.append(st[1:], len(q)); out = []
    for u, b, e in zip(uts, st, en):
        m = q[b:e]; inv = m[:, [2, 1, 0, 3]].copy(); inv[:, 1] += R; out += [m.copy(), inv]
    return np.concatenate(out)


def system_block(Z, prefix, seeds=(0, 1, 2)):
    names = [f"{prefix}_s{s}" for s in seeds if f"rr_{prefix}_s{s}" in Z]
    if not names: return None
    per = {nm: mets(Z[f"rr_{nm}"], Z[f"rank_{nm}"]) for nm in names}
    perh = {nm: hmets(Z[f"hrank_{nm}"]) for nm in names}
    out = dict(seeds=len(names), official={k: float(np.mean([per[n][k] for n in names])) for k in ("MRR", "H1", "H3", "H10")},
               official_sd={k: float(np.std([per[n][k] for n in names], ddof=1)) if len(names) > 1 else 0.0 for k in ("MRR", "H1", "H3", "H10")},
               harness={k: float(np.mean([perh[n][k] for n in names])) for k in ("MRR", "H1", "H3", "H10")},
               per_seed_official_MRR=[per[n]["MRR"] for n in names])
    rr = np.mean([np.asarray(Z[f"rr_{n}"], np.float64) for n in names], 0)
    return out, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--rule", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.load(open(a.cfg)); rule = json.load(open(a.rule))
    V = dict(rule_file=os.path.basename(a.rule), rule_registered_utc=rule["registered_utc"], datasets={})
    for ds, D in C.items():
        Q = {sp: load_quads(f"{D['dd']}/{sp}.txt") for sp in ("train", "test")}
        st = open(f"{D['dd']}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
        sro = split_queries(Q["test"], R)
        freq = np.bincount(Q["train"][:, 0], minlength=N) + np.bincount(Q["train"][:, 2], minlength=N)
        S = np.load(D["strata"]); assert (S["sro"] == sro).all(), "strata row order"
        gf = freq[sro[:, 2]]
        strata = {"head (gold train freq >= 500)": gf >= 500, "mid (50-499)": (gf >= 50) & (gf < 500), "cold (< 50)": gf < 50,
                  "no-pair-history": ~S["dyad"], "pair-history": S["dyad"], "first-seen (s or o)": S["first_seen"],
                  "exact recurrence (s,r,o) before t": S["trip"]}
        out = dict(name=D["name"], n_test=int(len(sro)), published=D.get("published", {}), bases={}, rows=[])
        # ---------- RecB reference rows (official formula, exact; hd_recb.py)
        RB = np.load(D["recb_ranks"]); rsel = json.load(open(D["recb_sel"]))
        recb_rows = {}
        for k, lab in (("faith_def", "RecB default (TGB script, faithful)"), ("faith_tr", "RecB trained (TGB train mode, faithful)"),
                       ("clean_def", "RecB default (clean psi reset)"), ("clean_tr", "RecB trained (clean psi reset)")):
            rk = RB[k].astype(np.float64); rr = (1.0 / RB[k]).astype(np.float32).astype(np.float64)
            recb_rows[k] = dict(label=lab, **mets(rr, rk))
            out["rows"].append(dict(base="-", arm=lab, evaluator="official formula (exact, hd_recb.py)", seeds="-", **{k2: recb_rows[k][k2] for k2 in ("MRR", "H1", "H3", "H10")},
                                    host=D.get("recb_host", "serverA (CPU)")))
        out["recb"] = dict(rows=recb_rows, valid=dict(faith_tr=rsel["faith"]["valid_MRR_trained"], clean_tr=rsel["clean"]["valid_MRR_trained"],
                                                      faith_def=rsel["faith"]["valid_MRR_default"], clean_def=rsel["clean"]["valid_MRR_default"]))
        rr_recb_tr = (1.0 / RB["clean_tr"]).astype(np.float64)
        # ---------- per base
        blends = {}
        for b, B in D["bases"].items():
            if not os.path.exists(B["official"]):
                out["bases"][b] = dict(missing=B["official"]); continue
            Z = dict(np.load(B["official"]))
            tj = json.load(open(B["test_json"])); vj = json.load(open(B["valid_json"]))
            assert len(Z["rr_base"]) == len(sro)
            rec = dict(arch=B["arch"], host=B.get("host", tj.get("host")), valid_MRR_base=B.get("valid_MRR", vj["valid_MRR_base"]),
                       selected=tj.get("selected"), official_vs_formula_max_batch_dev=tj.get("official_vs_formula_max_batch_dev"),
                       valid_MRR_blend_selected=max(vj["valid_MRR_blend"]), valid_MRR_pool_selected=max(vj["valid_MRR_pool"]),
                       selected_epoch=B.get("selected_epoch"), arms={})
            rr = {}
            for arm in ("base", "BLEND", "POOL"):
                rec["arms"][arm] = dict(official=mets(Z[f"rr_{arm}"], Z[f"rank_{arm}"]), harness=hmets(Z[f"hrank_{arm}"]))
                rr[arm] = np.asarray(Z[f"rr_{arm}"], np.float64)
            for pre in ("HR", "HRCOV", "HRCOV_ST", "HRCOV_ST_ND"):
                sb = system_block(Z, pre)
                if sb:
                    rec["arms"][pre] = sb[0]; rr[pre] = sb[1]
            # harness-only secondary arms (RecB features)
            if B.get("rb_ranks") and os.path.exists(B["rb_ranks"]):
                RBK = np.load(B["rb_ranks"])
                for pre in ("HR_RB", "HRCOV_RB"):
                    ks = [k for k in RBK.files if k.startswith(pre + "_s")]
                    if ks:
                        hm = [hmets(RBK[k]) for k in ks]
                        rec["arms"][pre] = dict(seeds=len(ks), harness={m: float(np.mean([h[m] for h in hm])) for m in ("MRR", "H1", "H3", "H10")},
                                                harness_sd={m: float(np.std([h[m] for h in hm], ddof=1)) for m in ("MRR", "H1", "H3", "H10")})
                        rr[pre + "_harness"] = np.mean([1.0 / RBK[k].astype(np.float64) for k in ks], 0)
                        if "HR" in rr and pre == "HR_RB":
                            rec["HR_RB_minus_HR_harness"] = boot(rr[pre + "_harness"] - np.mean([1.0 / Z[f"hrank_HR_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0))
                        if "HRCOV" in rr and pre == "HRCOV_RB":
                            rec["HRCOV_RB_minus_HRCOV_harness"] = boot(rr[pre + "_harness"] - np.mean([1.0 / Z[f"hrank_HRCOV_s{s}"].astype(np.float64) for s in (0, 1, 2)], 0))
            if "HR" in rr and "HRCOV" in rr:
                rec["HRCOV_minus_HR_official"] = boot(rr["HRCOV"] - rr["HR"])
                rec["HRCOV_minus_better_control_official"] = boot(rr["HRCOV"] - (rr["BLEND"] if rr["BLEND"].mean() >= rr["POOL"].mean() else rr["POOL"]))
            rec["_rr"] = rr; rec["_hrank_base"] = Z["hrank_base"]
            blends[b] = rr["BLEND"]
            out["bases"][b] = rec
        # ---------- architecture selection + two strongest (valid MRR, HRCOV available)
        arch_best = {}
        for b, rec in out["bases"].items():
            if "missing" in rec or "HRCOV" not in rec["arms"]: continue
            ar = rec["arch"]
            if ar not in arch_best or rec["valid_MRR_base"] > out["bases"][arch_best[ar]]["valid_MRR_base"]:
                arch_best[ar] = b
        ranked = sorted(arch_best.values(), key=lambda b: -out["bases"][b]["valid_MRR_base"])
        two = ranked[:2]
        out["architecture_selection"] = {ar: dict(selected=b, valid_MRR=out["bases"][b]["valid_MRR_base"]) for ar, b in arch_best.items()}
        out["two_strongest"] = two
        # strongest blend overall (all bases with a blend), by test MRR (conservative)
        bb = max(blends, key=lambda b: blends[b].mean()) if blends else None
        out["strongest_blend_overall"] = dict(base=bb, MRR=float(blends[bb].mean()) if bb else None)
        dec = {}
        for b in two:
            rec = out["bases"][b]; rr = rec["_rr"]
            better = "BLEND" if rr["BLEND"].mean() >= rr["POOL"].mean() else "POOL"
            c1 = {c: boot(rr["HRCOV"] - rr[c]) for c in ("BLEND", "POOL")}
            c1_pass = all(v["delta"] >= MARGIN and v["p_one_sided"] < 0.05 for v in c1.values())
            c2 = boot(rr["HRCOV"] - blends[bb])
            c2_pass = c2["delta"] > 0 and c2["p_one_sided"] < 0.05
            hm = strata["head (gold train freq >= 500)"]
            c3 = boot(rr["HRCOV"][hm] - rr[better][hm], B=1000)
            damage = (c3["CI95"][1] < 0) and (abs(c3["delta"]) >= 0.001)
            dec[b] = dict(better_comparator=better, C1=c1, C1_pass=bool(c1_pass), C2_vs=f"{bb}(+)RecB", C2=c2, C2_pass=bool(c2_pass),
                          C3_head=c3, C3_damage=bool(damage), PASS=bool(c1_pass and c2_pass and not damage))
            # secondary contrasts
            sec = {}
            for arm in ("HR", "HRCOV_ST", "HRCOV_ST_ND"):
                if arm in rr:
                    sec[f"{arm}_minus_{better}"] = boot(rr[arm] - rr[better])
            sec["HRCOV_minus_RecB_trained_clean"] = boot(rr["HRCOV"] - rr_recb_tr)
            if "HR" in rr:   # pool-widening gain on this base (re-measured on valid-best bases)
                sec["HRCOV_minus_HR (pool-widening gain)"] = boot(rr["HRCOV"] - rr["HR"])
            sec["BLEND_minus_base"] = boot(rr["BLEND"] - rr["base"])
            dec[b]["secondary"] = sec
            # strata
            pa = rec["_hrank_base"] > 50
            strat = dict(strata)
            strat["pool-absent (gold outside base top-50)"] = pa; strat["gold in base top-50"] = ~pa
            strat["R_rec_out (recurrence, gold outside base top-50)"] = S["trip"] & pa
            strat["R_rec_in (recurrence, gold in base top-50)"] = S["trip"] & ~pa
            srows = {}
            for nm, m in strat.items():
                if m.sum() == 0: continue
                bs = boot(rr["HRCOV"][m] - rr[better][m], B=1000)
                srows[nm] = dict(share=float(m.mean()), n=int(m.sum()), MRR_base=float(rr["base"][m].mean()), MRR_RecB_tr=float(rr_recb_tr[m].mean()),
                                 MRR_BLEND=float(rr["BLEND"][m].mean()), MRR_POOL=float(rr["POOL"][m].mean()), MRR_HR=float(rr["HR"][m].mean()) if "HR" in rr else None,
                                 MRR_HRCOV=float(rr["HRCOV"][m].mean()), delta_HRCOV_minus_better=bs["delta"], CI95=bs["CI95"],
                                 contribution_to_full_delta=float((rr["HRCOV"][m] - rr[better][m]).sum() / len(sro)))
            dec[b]["strata"] = srows
            # oracle ceilings (gold-in-set fractions = MRR ceiling of a perfect re-ranker over that set)
            hb = rec["_hrank_base"].astype(np.float64); rbk = RB["clean_tr"].astype(np.float64)
            oc = {f"base_top{K}": float((hb <= K).mean()) for K in (10, 20, 50, 100)}
            oc.update({f"RecB_trained_top{K}": float((rbk <= K).mean()) for K in (20, 50, 100)})
            oc.update({f"base_top{K}_union_RecB_top{K}": float(((hb <= K) | (rbk <= K)).mean()) for K in (20, 50, 100)})
            oc["note"] = ("fraction of test queries whose gold is inside the set; = MRR of an oracle that places gold at rank 1 whenever "
                          "it is in the set. RecB top-K uses the clean trained RecB filtered rank (ties broken as in hd_recb.py); "
                          "base top-K uses the harness exact filtered base rank.")
            dec[b]["oracle_ceilings"] = oc
        outcome = "PRACTICAL_ADVANCE" if two and len(two) == 2 and all(dec[b]["PASS"] for b in two) else (
            "HEAD_DAMAGE" if two and all(dec[b]["C1_pass"] and dec[b]["C2_pass"] for b in two) else "RECURRENCY_DONE_WELL")
        if len(two) < 2: outcome += " (PROVISIONAL: fewer than two bases with HRCOV)"
        out["decision"] = dict(per_base=dec, outcome=outcome, claim_survives=outcome == "PRACTICAL_ADVANCE")
        # ---------- table rows
        for b, rec in out["bases"].items():
            if "missing" in rec: continue
            for arm, v in rec["arms"].items():
                row = dict(base=b, arm=arm, host=rec["host"])
                if "official" in v:
                    row.update(evaluator="official py-tgb Evaluator", seeds=v.get("seeds", "-"), **v["official"])
                    if "official_sd" in v: row["sd"] = v["official_sd"]
                    row["harness_MRR"] = v["harness"]["MRR"]
                else:
                    row.update(evaluator="harness exact filtered (secondary arm)", seeds=v["seeds"], **v["harness"], sd=v["harness_sd"])
                out["rows"].append(row)
            rec.pop("_rr"); rec.pop("_hrank_base")
        for k, v in D.get("published", {}).items():
            out["rows"].append(dict(base="-", arm=f"published: {k}", evaluator="published row (not re-run)", MRR=v, host="cited source (see arm label)"))
        V["datasets"][ds] = out
    ok = [V["datasets"][d]["decision"]["claim_survives"] for d in V["datasets"]]
    V["overall"] = "survives on both datasets" if all(ok) else ("survives on one dataset" if any(ok) else "does not survive")
    json.dump(V, open(a.out, "w"), indent=1)
    print(json.dumps({d: V["datasets"][d]["decision"]["outcome"] for d in V["datasets"]}), V["overall"])


if __name__ == "__main__":
    main()
