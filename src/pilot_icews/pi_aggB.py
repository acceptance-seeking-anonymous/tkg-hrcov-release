#!/usr/bin/env python3
"""pilot_icews / pi_aggB.py -- Stage B aggregation on host serverA (CPU): full-test rows, paired bootstraps, the pre-registered
stageB_rule.json decision, strata, relation-code artefact block, Stage A -> Stage B shift, leaderboard-comparable rows.

  python3 pi_aggB.py --rule stageB_rule.json --out icews_stageB_verdict.json --workers 16
Inputs (all under work_stageB unless noted): strata_test.npz; RecB-lite harness ranks recb_test_sample_rank.npy (= full test)
and official recblite_official.json / recblite_official_test_sample_rank.npy; official TGB RecB recb_official_w100.json /
recb_official_w100_test_sample_rr.npy; hardening RecB recb/recb_test_ranks_tkgl-icews.npz + recb_test_meta / selection;
per base b: feat_<b>_orig/test_q.npz, train_<b>/{train_<b>.json, ranks_test.npz}, blend/{blend_valid,blend_test}_icews_<b>.json,
blend/official_test_icews_<b>.npz, blend/sel_icews_<b>.json, dumps/base_<b>_stageB.json; RE-GCN-RN: dumps/regcnrn_stageB_scores.npz
(harness) + blend/official_test_icews_regcnrn.npz (official, --base_only).
Bootstrap = harness oih_common.paired_bootstrap (Poisson(1) paired over queries, seed 0; B = 2000 decision + head guardrail,
B = 1000 strata / secondary); one-sided p = share of replicates <= 0.  HRCOV per-query RR = mean over seeds 0,1,2.
"""
import os, sys, json, argparse, time
import numpy as np
import multiprocessing as mp

P = os.environ.get("PI_P", os.path.expandvars("$WORK_DIR/tracks/pilot_icews"))
W = os.environ.get("PI_W", f"{P}/work_stageB")
G = {}


def paired_bootstrap(rr_a, rr_b, B=2000, seed=0, chunk=20):
    """harness oih_common.paired_bootstrap (Poisson(1) paired bootstrap over queries); chunk only bounds memory."""
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64)
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    nz = np.nonzero(d)[0]; d0 = float(d.mean())
    if len(nz) == 0:
        return {"delta": 0.0, "CI95": [0.0, 0.0], "p_delta_gt0": 0.0, "p_delta_le_0005": 1.0, "excludes_zero": False, "n": n, "n_changed": 0}
    boots = np.empty(B); dn = d[nz]
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        w = rng.poisson(1.0, size=(bb, len(nz)))
        boots[b0:b0 + bb] = (w * dn).sum(1) / n
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": round(d0, 6), "CI95": [round(float(lo), 6), round(float(hi), 6)], "p_delta_gt0": round(float((boots > 0).mean()), 4),
            "p_one_sided": round(float((boots <= 0).mean()), 4), "p_delta_le_0005": round(float((boots <= 0.005).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(n), "n_changed": int(len(nz)), "method": "Poisson(1) paired bootstrap over queries, B=%d" % B}


def _bs(task):
    key, a, b, mask, B = task
    m = G["masks"][mask] if mask else slice(None)
    return key, paired_bootstrap(G["RR"][a][m], G["RR"][b][m], B=B)


def mets(rr_list, n=None):
    if not isinstance(rr_list, list): rr_list = [rr_list]
    rk = [1.0 / np.asarray(r, np.float64) for r in rr_list]
    out = dict(MRR=float(np.mean(np.mean(rr_list, 0))), n=int(len(rr_list[0])))
    for k in (1, 3, 10): out[f"H{k}"] = float(np.mean([np.mean(x <= k + 1e-9) for x in rk]))
    if len(rr_list) > 1: out["seed_sd_MRR"] = float(np.std([np.mean(r) for r in rr_list]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bases", default="regcn,logcl"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stageA_verdict", default="")
    a = ap.parse_args()
    t0 = time.time()
    rule = json.load(open(a.rule))
    J = lambda p: json.load(open(p))
    q = dict(np.load(f"{W}/strata_test.npz")); n = len(q["sro"])
    RR = {}; rows = []; notes = []
    RR["RecB-lite_h"] = 1.0 / np.load(f"{W}/recb_test_sample_rank.npy").astype(np.float64)
    if os.path.exists(f"{W}/recblite_official_test_sample_rank.npy"):
        rlo = J(f"{W}/recblite_official.json"); RR["RecB-lite"] = 1.0 / np.load(f"{W}/recblite_official_test_sample_rank.npy").astype(np.float64)
        rows.append(dict(base="-", system="RecB-lite (full history, lambda 0.99)", evaluator="official py-tgb", **mets(RR["RecB-lite"]),
                         frac_rank_equal_harness=rlo.get("frac_rank_equal")))
    else:
        RR["RecB-lite"] = RR["RecB-lite_h"]; notes.append("RecB-lite official ranks missing -> harness ranks used in the decision")
    rows.append(dict(base="-", system="RecB-lite (full history, lambda 0.99)", evaluator="harness", **mets(RR["RecB-lite_h"])))
    if os.path.exists(f"{W}/recb_official_w100_test_sample_rr.npy"):
        RR["RecB_TGB_w100"] = np.load(f"{W}/recb_official_w100_test_sample_rr.npy").astype(np.float64)
        rows.append(dict(base="-", system="RecB, official TGB predictor (default lambda 0.1 alpha 0.99, window 100)", evaluator="official py-tgb (TGB script)",
                         **mets(RR["RecB_TGB_w100"])))
    hdr = f"{W}/recb/recb_test_ranks_tkgl-icews.npz"
    if os.path.exists(hdr):
        z = np.load(hdr)
        for k in ("clean_def", "clean_tr", "faith_def", "faith_tr"):
            RR[f"RecB_hd_{k}"] = (1.0 / z[k].astype(np.float32)).astype(np.float64)
            rows.append(dict(base="-", system=f"RecB (hardening re-implementation, window 0, {k})", evaluator="official formula (exact, tie 1/2)", **mets(RR[f"RecB_hd_{k}"])))
    if os.path.exists(f"{W}/recbRN_test_sample_rank.npy"):
        RR["RecB-lite-RN"] = 1.0 / np.load(f"{W}/recbRN_test_sample_rank.npy").astype(np.float64)
    head = q["freq_stratum"] == 2; un = q["rel_unseen_train"].astype(bool)
    masks = {"head": head, "unseen": un, "seen": ~un}
    V = dict(stage="B", pilot="3 of 4 (tkgl-icews)", rule_file=os.path.basename(a.rule), rule_registered_utc=rule["registered_utc"], n_test=int(n),
             hosts=dict(compute="serverA (2x RTX PRO 6000 Blackwell Max-Q, docker tkg-gpu:latest, torch 2.8.0+cu128, py-tgb 2.3.0)",
                        aggregation="serverA CPU (pi_aggB.py)"), rows=rows, decision={}, strata={}, secondary={}, base_training={}, notes=notes)
    per = {}
    for b in a.bases.split(","):
        fq = dict(np.load(f"{W}/feat_{b}_orig/test_q.npz")); assert (fq["sro"] == q["sro"]).all()
        rk = dict(np.load(f"{W}/train_{b}/ranks_test.npz")); tr = J(f"{W}/train_{b}/train_{b}.json")
        basej = J(f"{P}/dumps/base_{b}_stageB.json")
        off = dict(np.load(f"{W}/blend/official_test_icews_{b}.npz")); bt = J(f"{W}/blend/blend_test_icews_{b}.json")
        bv = J(f"{W}/blend/blend_valid_icews_{b}.json"); sel = J(f"{W}/blend/sel_icews_{b}.json")[b]
        assert (off["rows"] == np.arange(n)).all()
        fs = f"{W}/blend/official_test_icews_{b}_sys.npz"            # split official pass (systems), if used
        if os.path.exists(fs):
            o2 = dict(np.load(fs)); assert (o2["rows"] == np.arange(n)).all()
            notes.append(f"{b}: official systems scored in a separate pass; |rr_base pass1 - pass2| max = "
                         f"{float(np.abs(o2['rr_base'].astype(np.float64) - off['rr_base'].astype(np.float64)).max()):.2e}")
            off.update({k: v for k, v in o2.items() if "HRCOV" in k or "HR_" in k})
        Bs = tr["selection"]["B_star"]; A = f"HRCOV{Bs}"; seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in rk if "_s" in k})
        hr = lambda nm: [1.0 / rk[f"{nm}_s{s}"].astype(np.float64) for s in seeds]
        RR[f"{b}:base"] = off["rr_base"].astype(np.float64); RR[f"{b}:base_h"] = 1.0 / off["hrank_base"].astype(np.float64)
        RR[f"{b}:BLEND"] = off["rr_BLEND"].astype(np.float64); RR[f"{b}:POOL"] = off["rr_POOL"].astype(np.float64)
        offs = [off[f"rr_{A}_s{s}"].astype(np.float64) for s in seeds]
        RR[f"{b}:HRCOV"] = np.mean(offs, 0)
        for nm in ["HR"] + [f"HRCOV{x}" for x in (150, 200, 300)] + [f"{A}_RN"]:
            RR[f"{b}:{nm}_h"] = np.mean(hr(nm), 0)
        rows.append(dict(base=b, system=f"{b} base (all train facts, epoch {basej['selected_epoch']})", evaluator="official py-tgb", **mets(RR[f"{b}:base"]),
                         harness_MRR=float(RR[f"{b}:base_h"].mean())))
        for nm, lab in (("BLEND", f"{b}(+)RecB blend {bt['selected']['blend']}"), ("POOL", f"RecB-POOL_{b} K={bt['selected']['pool'][0]} {bt['selected']['pool'][1]}")):
            rows.append(dict(base=b, system=lab, evaluator="official py-tgb", **mets(RR[f"{b}:{nm}"]), harness_MRR=float(np.mean(1.0 / off[f"hrank_{nm}"].astype(np.float64))),
                             valid_MRR=(sel["valid_MRR_blend"] if nm == "BLEND" else sel["valid_MRR_pool"])))
        rows.append(dict(base=b, system=f"HRCOV{Bs} (valid-chosen B*; claim system)", evaluator="official py-tgb (3 seeds)", **mets(offs),
                         harness_MRR=float(np.mean(np.mean(hr(A), 0))), select_MRR_mean=tr["arms"][A]["select_MRR_mean"]))
        for nm in ["HR"] + [f"HRCOV{x}" for x in (150, 200, 300)] + [f"{A}_RN"]:
            rows.append(dict(base=b, system=nm + (" (B*)" if nm == A else ""), evaluator="harness (3 seeds)", **mets(hr(nm)), select_MRR_mean=tr["arms"][nm]["select_MRR_mean"]))
        per[b] = dict(A=A, Bs=Bs, seeds=seeds, fq=fq, tr=tr, basej=basej, bt=bt, bv=bv, sel=sel)
        V["base_training"][b] = dict(grid=basej["grid"], selected_epoch=basej["selected_epoch"], selected_valid_MRR=basej["selected_valid_MRR"],
                                     train_s=basej.get("train_s"), official_vs_formula_max_batch_dev=bt.get("official_vs_formula_max_batch_dev"))
    # RE-GCN-RN reference row
    if os.path.exists(f"{W}/blend/official_test_icews_regcnrn.npz"):
        o = dict(np.load(f"{W}/blend/official_test_icews_regcnrn.npz")); RR["regcnrn:base"] = o["rr_base"].astype(np.float64)
        bj = J(f"{P}/dumps/base_regcnrn_stageB.json")
        rows.append(dict(base="regcnrn", system=f"RE-GCN-RN base (relation-normalised inputs, epoch {bj['selected_epoch']})", evaluator="official py-tgb",
                         **mets(RR["regcnrn:base"]), harness_MRR=float(np.mean(1.0 / o["hrank_base"].astype(np.float64)))))
        V["base_training"]["regcnrn"] = dict(grid=bj["grid"], selected_epoch=bj["selected_epoch"], selected_valid_MRR=bj["selected_valid_MRR"])
    # ---------------- bootstraps (parallel)
    tasks = []
    for b in per:
        for C in ("BLEND", "POOL", "RecB-lite"):
            ck = f"{b}:{C}" if C != "RecB-lite" else "RecB-lite"
            tasks.append((f"dec|{b}|{C}", ck, f"{b}:HRCOV", None, 2000))
    G["RR"] = RR; G["masks"] = masks
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = dict(pool.map(_bs, tasks, chunksize=1))
    # C* = best comparator on test
    t2 = []
    for b in per:
        cm = {C: float(RR[f"{b}:{C}" if C != "RecB-lite" else "RecB-lite"].mean()) for C in ("BLEND", "POOL", "RecB-lite")}
        Cs = max(cm, key=cm.get); per[b]["Cs"] = Cs; per[b]["cm"] = cm
        cs_key = f"{b}:{Cs}" if Cs != "RecB-lite" else "RecB-lite"; per[b]["cs_key"] = cs_key
        t2.append((f"head|{b}", cs_key, f"{b}:HRCOV", "head", 2000))
        # secondary
        t2 += [(f"sec|{b}|HRCOV_vs_base", f"{b}:base", f"{b}:HRCOV", None, 1000), (f"sec|{b}|HRCOV_vs_HR_h", f"{b}:HR_h", f"{b}:{per[b]['A']}_h", None, 1000),
               (f"sec|{b}|base_vs_RecB-lite", "RecB-lite", f"{b}:base", None, 1000), (f"sec|{b}|BLEND_vs_base", f"{b}:base", f"{b}:BLEND", None, 1000),
               (f"sec|{b}|POOL_vs_base", f"{b}:base", f"{b}:POOL", None, 1000)]
        A = per[b]["A"]
        for mk in (None, "unseen", "seen"):
            t2.append((f"rn|{b}|{mk or 'all'}", f"{b}:{A}_h", f"{b}:{A}_RN_h", mk, 1000))
        if "RecB_TGB_w100" in RR:
            t2.append((f"sec|{b}|HRCOV_vs_RecB_TGB_w100", "RecB_TGB_w100", f"{b}:HRCOV", None, 1000))
    if "regcnrn:base" in RR:
        for mk in (None, "unseen", "seen"):
            t2.append((f"rnbase|{mk or 'all'}", "regcn:base", "regcnrn:base", mk, 1000))
    if "RecB-lite-RN" in RR:
        for mk in (None, "unseen", "seen"):
            t2.append((f"rnrecb|{mk or 'all'}", "RecB-lite_h", "RecB-lite-RN", mk, 1000))
    # strata
    rb_of = {b: per[b]["fq"]["r_base"] for b in per}
    SD = {"all": np.ones(n, bool), "exact recurrence (s,r',o) before t": q["trip"].astype(bool), "non-recurrent": ~q["trip"].astype(bool),
          "pair history (any (s,o) event before t)": q["dyad"].astype(bool), "no pair history": ~q["dyad"].astype(bool),
          "relation unseen in train (post-2020 zero-stripped code)": un, "relation seen in train": ~un, "cold gold (train freq < 50)": q["freq_stratum"] == 0,
          "mid gold": q["freq_stratum"] == 1, "head gold (train freq >= 500)": head, "first-seen entity": q["first_seen"].astype(bool),
          "hist=no (no train+valid co-occurrence)": ~q["in_hist"].astype(bool)}
    for b in per:
        SD[f"pool-absent ({b}: gold outside base top-50)"] = rb_of[b] > 50; SD[f"gold in {b} top-50"] = rb_of[b] <= 50
    for k, m in SD.items(): masks[k] = m
    for b in per:
        for k in SD:
            if (b == "regcn" and "logcl" in k) or (b == "logcl" and "regcn" in k): continue
            if SD[k].sum() == 0: continue
            t2.append((f"st|{b}|{k}|vsC", per[b]["cs_key"], f"{b}:HRCOV", k, 1000))
            t2.append((f"st|{b}|{k}|vsBase", f"{b}:base", f"{b}:HRCOV", k, 1000))
    G["masks"] = masks
    with mp.get_context("fork").Pool(a.workers) as pool:
        res.update(dict(pool.map(_bs, t2, chunksize=1)))
    # ---------------- decision
    dmg = lambda h: bool(h["CI95"][1] < 0 and abs(h["delta"]) >= 0.001)
    labels = {}
    for b in per:
        Cs = per[b]["Cs"]; dC = {C: res[f"dec|{b}|{C}"] for C in ("BLEND", "POOL", "RecB-lite")}
        c1_each = {C: bool(d["delta"] >= 0.005 and d["p_one_sided"] < 0.05) for C, d in dC.items()}
        c1 = all(c1_each.values()); h = res[f"head|{b}"]; c2 = not dmg(h)
        V["decision"][b] = dict(B_star=per[b]["Bs"], claim_system=per[b]["A"], comparator_test_MRR=per[b]["cm"], best_comparator=Cs,
                                HRCOV_test_MRR=float(RR[f"{b}:HRCOV"].mean()), C1_margin=dict(vs=dC, pass_each=c1_each, pass_=bool(c1)),
                                C2_head_guardrail=dict(head_share=float(head.mean()), vs_best_comparator=dict(**h, DAMAGE=dmg(h)), pass_=bool(c2)),
                                PASS=bool(c1 and c2), selection=per[b]["tr"]["selection"], blend_selection=per[b]["sel"])
        labels[b] = (c1, c2)
    allp = all(c1 and c2 for c1, c2 in labels.values())
    if allp: outcome = "PRACTICAL_ADVANCE"
    elif not all(c1 for c1, _ in labels.values()): outcome = "RECURRENCY_DONE_WELL"
    else: outcome = "HEAD_DAMAGE"
    V["outcome"] = outcome
    # ---------------- secondary
    for b in per:
        A = per[b]["A"]
        V["secondary"][b] = dict({k.split("|", 2)[2]: v for k, v in res.items() if k.startswith(f"sec|{b}|")},
                                 relnorm_HRCOV=dict({k.split("|")[2]: v for k, v in res.items() if k.startswith(f"rn|{b}|")},
                                                    MRR_by_rel_seen={nm: dict(unseen=float(RR[k][un].mean()), seen=float(RR[k][~un].mean()))
                                                                     for nm, k in (("base", f"{b}:base"), ("RecB-lite", "RecB-lite"), ("BLEND", f"{b}:BLEND"),
                                                                                   ("POOL", f"{b}:POOL"), ("HR", f"{b}:HR_h"), (A, f"{b}:HRCOV"), (f"{A}_RN", f"{b}:{A}_RN_h"))}),
                                 ceilings=dict(gold_in_top50=float((rb_of[b] <= 50).mean()),
                                               gold_in_Ccov_Bstar=float(((per[b]["fq"]["gold_pos"] >= 0) & (per[b]["fq"]["gold_pos"] < 50 + per[b]["Bs"])).mean()),
                                               MRR_top50_at1=float(np.mean(np.where(rb_of[b] <= 50, 1.0, RR[f"{b}:base_h"]))),
                                               MRR_Ccov_Bstar_at1=float(np.mean(np.where((per[b]["fq"]["gold_pos"] >= 0) & (per[b]["fq"]["gold_pos"] < 50 + per[b]["Bs"]), 1.0, RR[f"{b}:base_h"]))),
                                               realised_HRCOV=float(RR[f"{b}:HRCOV"].mean())),
                                 evaluator_agreement=dict(base=abs(float(RR[f"{b}:base"].mean() - RR[f"{b}:base_h"].mean())),
                                                          HRCOV=abs(float(RR[f"{b}:HRCOV"].mean() - RR[f"{b}:{A}_h"].mean()))))
    V["secondary"]["relation_code_artefact"] = dict(
        share_rel_unseen_in_train=float(un.mean()),
        regcnrn_minus_regcn_base={k.split("|")[1]: v for k, v in res.items() if k.startswith("rnbase|")},
        recblite_rn_minus_recblite_POSTHOC={k.split("|")[1]: v for k, v in res.items() if k.startswith("rnrecb|")},
        MRR_by_rel_seen={nm: dict(unseen=float(RR[nm][un].mean()), seen=float(RR[nm][~un].mean())) for nm in RR if not nm.endswith("_h") or nm == "RecB-lite_h"})
    for b in per:
        st = {}
        for k in SD:
            if f"st|{b}|{k}|vsC" not in res: continue
            m = SD[k]
            row = dict(share=float(m.mean()), n=int(m.sum()))
            for nm, kk in (("base", f"{b}:base"), ("RecB-lite", "RecB-lite"), ("BLEND", f"{b}:BLEND"), ("POOL", f"{b}:POOL"), ("HR_h", f"{b}:HR_h"),
                           ("HRCOV", f"{b}:HRCOV"), ("RN_h", f"{b}:{per[b]['A']}_RN_h")):
                row[f"MRR_{nm}"] = float(RR[kk][m].mean())
            row["d_HRCOV_vs_best_comparator"] = res[f"st|{b}|{k}|vsC"]; row["d_HRCOV_vs_base"] = res[f"st|{b}|{k}|vsBase"]
            st[k] = row
        V["strata"][b] = st
    # Stage A -> Stage B on the identical 10% sample rows
    tA = np.load(f"{P}/work/test_sample_idx.npy")
    V["stageA_sample_rows"] = {nm: float(RR[nm][tA].mean()) for nm in RR}
    V["leaderboard_reference"] = dict(source="TGB 2.0 tkgl-icews leaderboard as recorded in Stage A (pilot record)", rows={"RE-GCN": 0.182, "CEN": 0.187, "RecB (train mode)": 0.211, "EdgeReMIND": 0.269})
    V["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(V, open(a.out, "w"), indent=1)
    # small per-query arrays for figures (seen/unseen split + per-stratum already in json)
    print(json.dumps(dict(outcome=outcome, decision={b: dict(Cs=V["decision"][b]["best_comparator"], HRCOV=V["decision"][b]["HRCOV_test_MRR"],
                                                             comp=V["decision"][b]["comparator_test_MRR"], PASS=V["decision"][b]["PASS"]) for b in per},
                          elapsed=V["elapsed_s"]), indent=1))


if __name__ == "__main__":
    main()
