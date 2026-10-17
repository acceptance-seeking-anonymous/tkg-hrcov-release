#!/usr/bin/env python3
"""
Residual track -- aggregation: full-test rows, strata, batched paired Poisson bootstrap (B=2000, harness
convention: delta = sum_i w_i d_i / n), head-damage guardrail, strikingness-weighted MRR, decision rule.
Seed aggregation: per-query reciprocal rank averaged over the 3 training seeds (MRR of the average == mean of
seed MRRs); per-seed deltas are reported alongside.
"""
import os, sys, json, time, argparse
H = os.path.expandvars("$WORK_DIR"); TRACK = f"{H}/tracks/residual"
sys.path.insert(0, H)
import numpy as np
from oih_common import metrics_from_ranks, jdump

B_BOOT = 2000
STRIK_B = 0.1


def m5(rk):
    return metrics_from_ranks(rk, ks=(1, 3, 10, 50))


def batched_boot(cols, n, B=B_BOOT, seed=0, chunk=100):
    """cols: dict name -> d vector [n] already scaled so that delta = mean(d).  Returns name -> (delta, lo, hi, p_le0)."""
    names = list(cols); Dm = np.stack([cols[k] for k in names], 1).astype(np.float32)
    nz = np.where(np.abs(Dm).sum(1) > 0)[0]; Dz = Dm[nz]
    rng = np.random.default_rng(seed); boots = np.zeros((B, len(names)), np.float64)
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        w = rng.poisson(1.0, size=(bb, len(nz))).astype(np.float32)
        boots[b0:b0 + bb] = (w @ Dz) / n
    out = {}
    for j, k in enumerate(names):
        d0 = float(Dm[:, j].sum() / n); lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        out[k] = dict(delta=round(d0, 6), CI95=[round(float(lo), 6), round(float(hi), 6)],
                      p_le0=round(float((boots[:, j] <= 0).mean()), 4), p_ge0=round(float((boots[:, j] >= 0).mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="ICEWS18:tirgn,ICEWS18:logcl,ICEWS18:regcn,GDELT:tirgn,GDELT:logcl,GDELT:regcn")
    a = ap.parse_args()
    t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
    allres = dict(host="serverA (CPU, tkg-gpu:latest)", B=B_BOOT, strik_bias=STRIK_B, combos={})
    for combo in a.combos.split(","):
        DS, base = combo.split(":")
        rp = f"{TRACK}/results/{DS}_{base}/ranks_test.npz"
        if not os.path.exists(rp):
            log("missing", rp); continue
        Rk = dict(np.load(rp)); Q = dict(np.load(f"{TRACK}/feat/{DS}_{base}/test_q.npz"))
        TJ = json.load(open(f"{TRACK}/results/{DS}_{base}/train_{DS}_{base}.json"))
        cp = f"{TRACK}/results/C/ranksC_{DS}_{base}.npz"
        if os.path.exists(cp):
            Cz = dict(np.load(cp)); assert np.array_equal(Cz["rows"], Rk["rows"]), "C rows misaligned"
            assert np.array_equal(Cz["base"], Rk["base"]), "C base ranks differ"
            for k, v in Cz.items():
                if k.startswith("C_s"):
                    Rk[k] = v
        n = len(Rk["base"]); rr0 = 1.0 / Rk["base"].astype(np.float64)
        seeds = sorted({int(k.split("_s")[1]) for k in Rk if k.startswith("R0_s")})
        arms = {}
        for k in Rk:
            if k in ("base", "rows"):
                continue
            nm = k.split("_s")[0] if "_s" in k else k
            arms.setdefault(nm, []).append(k)
        # R gate-variant selection on VALID-select (seed-mean MRR)
        selm = {}
        for v_ in ("R0", "Rreuse", "Rconf"):
            if v_ in TJ["arms"]:
                selm[v_] = float(np.mean([s["select"]["MRR"] for s in TJ["arms"][v_]["seeds"].values()]))
        Rsel = max(selm, key=selm.get)
        rows = []; rrbar = {}
        for nm, keys in sorted(arms.items()):
            rr = np.stack([1.0 / Rk[k].astype(np.float64) for k in sorted(keys)])
            rrbar[nm] = rr.mean(0)
            for k in sorted(keys):
                sd_ = int(k.split("_s")[1]) if "_s" in k else None
                rows.append(dict(dataset=DS, base=base, arm=nm, seed=sd_, **m5(Rk[k]),
                                 dMRR=round(float((1.0 / Rk[k] - rr0).mean()), 6)))
        strat = dict(cold=Q["stratum"] == 0, mid=Q["stratum"] == 1, head=Q["stratum"] == 2, hist_no=~Q["in_hist"],
                     hist_yes=Q["in_hist"], nonrepeat=Q["rec_nonrepeat"], repeat=~Q["rec_nonrepeat"], rec_top1_miss=Q["rec_top1_miss"],
                     no_dyad_window=~Q["gold_dyad_win"], dyad_window=Q["gold_dyad_win"], in_top50=Q["in_top50"], out_top50=~Q["in_top50"])
        if Q["is_pool"].any():
            strat["trigger_pool"] = Q["is_pool"]
        for pool in (True, False):
            for hist in (True, False):
                strat[f"pool={'in' if pool else 'out'}|hist={'yes' if hist else 'no'}"] = (Q["in_top50"] == pool) & (Q["in_hist"] == hist)
        sk = Q["strik"].astype(np.float64); skf = Q["strik_found"]
        wts = np.where(skf, sk + STRIK_B, 0.0)
        for lo_, hi_ in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)):
            strat[f"strik[{lo_},{min(hi_,1.0)})"] = skf & (sk >= lo_) & (sk < hi_)
        cols = {}
        key_strata = ("cold", "mid", "head", "hist_no", "nonrepeat", "rec_top1_miss", "no_dyad_window")
        for nm, rb in rrbar.items():
            d = rb - rr0
            cols[f"{nm}|ALL"] = d
            for st in key_strata:
                m = strat[st]; cols[f"{nm}|{st}"] = np.where(m, d, 0.0) * (n / max(m.sum(), 1))
            cols[f"{nm}|WMRR"] = wts * d * (n / wts.sum())
        # per-seed R_sel / comparator deltas vs base
        for k in Rk:
            if k.startswith(Rsel + "_s") or k.startswith("Amix_s") or k.startswith("Acal_s") or k.startswith("C_s"):
                cols[f"seed:{k}|ALL"] = 1.0 / Rk[k] - rr0
        # pairwise: R_sel vs comparators (seed-averaged)
        comps = [c for c in ("Amix", "Acal", "C", "D", "COPY", "Aalone") if c in rrbar]
        for cmp_ in comps:
            cols[f"pair:{Rsel}-{cmp_}|ALL"] = rrbar[Rsel] - rrbar[cmp_]
            cols[f"pair:{Rsel}-{cmp_}|WMRR"] = wts * (rrbar[Rsel] - rrbar[cmp_]) * (n / wts.sum())
            for st in ("hist_no", "nonrepeat", "head"):
                m = strat[st]; cols[f"pair:{Rsel}-{cmp_}|{st}"] = np.where(m, rrbar[Rsel] - rrbar[cmp_], 0.0) * (n / max(m.sum(), 1))
        for a_, b_ in (("R0ev", "Aevmix"), ("R0ev", "Aevcal"), ("R0", "Amix"), ("R0", "Acal"), ("Rconf", "Amix"), ("R0ev", "D")):
            if a_ in rrbar and b_ in rrbar:
                cols[f"pair:{a_}-{b_}|ALL"] = rrbar[a_] - rrbar[b_]
        log(DS, base, "bootstrapping", len(cols), "columns over n =", n)
        BT = batched_boot(cols, n)
        # strata tables (point estimates for every stratum, CI for key strata)
        stab = {}
        for nm, rb in rrbar.items():
            t = {}
            for st, m in strat.items():
                if m.sum() == 0:
                    continue
                t[st] = dict(n=int(m.sum()), base_MRR=round(float(rr0[m].mean()), 6), arm_MRR=round(float(rb[m].mean()), 6),
                             dMRR=round(float((rb - rr0)[m].mean()), 6))
                if f"{nm}|{st}" in BT:
                    t[st].update(CI95=BT[f"{nm}|{st}"]["CI95"], p_le0=BT[f"{nm}|{st}"]["p_le0"])
            t["WMRR"] = dict(base=round(float((wts * rr0).sum() / wts.sum()), 6), arm=round(float((wts * rb).sum() / wts.sum()), 6),
                             **BT[f"{nm}|WMRR"])
            hd = BT[f"{nm}|head"]
            flag = ("DAMAGE" if (hd["CI95"][1] < 0 and abs(hd["delta"]) >= 0.001) else
                    ("neg-negligible" if hd["CI95"][1] < 0 else ("HELP" if hd["CI95"][0] > 0 and hd["delta"] >= 0.001 else
                     ("pos-negligible" if hd["CI95"][0] > 0 else "neutral"))))
            t["head_damage"] = dict(head=hd, flag=flag, guardrail_pass=bool(flag != "DAMAGE"))
            t["vs_base"] = BT[f"{nm}|ALL"]
            stab[nm] = t
        pairs = {k.split("pair:")[1]: v for k, v in BT.items() if k.startswith("pair:")}
        perseed = {k.split("seed:")[1]: v for k, v in BT.items() if k.startswith("seed:")}
        # decision rule for this (DS, base)
        def beats(c):
            p = pairs.get(f"{Rsel}-{c}|ALL")
            return None if p is None else bool(p["delta"] > 0 and p["p_le0"] < 0.05)
        vb = stab[Rsel]["vs_base"]
        dr = dict(R_selected=Rsel, R_selection_valid_MRR=selm,
                  improves_base=bool(vb["delta"] > 0 and vb["p_le0"] < 0.05), vs_base=vb,
                  beats_A=beats("Amix"), beats_B=beats("Acal"), beats_C=beats("C"), beats_D=beats("D"),
                  guardrail_pass=stab[Rsel]["head_damage"]["guardrail_pass"])
        conds = [dr["improves_base"], dr["beats_A"], dr["beats_B"], dr["beats_C"], dr["beats_D"], dr["guardrail_pass"]]
        dr["C_available"] = dr["beats_C"] is not None
        dr["all_conditions"] = bool(all(c is True for c in conds))
        allres["combos"][f"{DS}:{base}"] = dict(rows=rows, strata=stab, pairs=pairs, per_seed_vs_base=perseed, decision=dr,
                                                 n=int(n), strik_found=float(skf.mean()), seeds=seeds,
                                                 train_selected=dict(
                                                     A=[TJ["arms"]["A"]["seeds"][str(s)]["mix_selected"] for s in seeds] if "A" in TJ["arms"] else None,
                                                     B=[(TJ["arms"]["A"]["seeds"][str(s)]["cal_alpha"], TJ["arms"]["A"]["seeds"][str(s)]["cal_beta"]) for s in seeds] if "A" in TJ["arms"] else None,
                                                     D=TJ["arms"]["D"]["selected"], COPY=TJ["arms"]["COPY"]["selected_lam"]))
        log(DS, base, "decision", json.dumps({k: dr[k] for k in ("R_selected", "improves_base", "beats_A", "beats_B", "beats_C", "beats_D", "guardrail_pass", "all_conditions")}))
    # overall rule: R passes iff on >= 2 bases AND both datasets every condition holds
    per_ds = {}
    for k, v in allres["combos"].items():
        DS, base = k.split(":"); per_ds.setdefault(DS, []).append((base, v["decision"]["all_conditions"]))
    passing = {DS: [b for b, ok in lst if ok] for DS, lst in per_ds.items()}
    bases_both = set(passing.get("ICEWS18", [])) & set(passing.get("GDELT", []))
    allres["decision_rule"] = dict(rule="R passes iff on >=2 frozen bases and on BOTH datasets it improves full-test MRR over the base "
                                        "(paired bootstrap one-sided p<0.05), beats A, B, C and D at matched features (p<0.05) and passes the head-damage guardrail",
                                   passing_bases_by_dataset=passing, bases_passing_both=sorted(bases_both),
                                   PASS=bool(len(bases_both) >= 2))
    allres["elapsed_s"] = round(time.time() - t0, 1)
    jdump(allres, "residual_aggregate.json"); jdump(allres, f"{TRACK}/results/residual_aggregate.json")
    log("PASS" if allres["decision_rule"]["PASS"] else "FAIL", json.dumps(allres["decision_rule"]))


if __name__ == "__main__":
    main()
