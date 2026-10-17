#!/usr/bin/env python3
"""v76 / v76_recb_xf_select.py -- M4 cross-fitted RecB (v76_rule.json): per-relation (lambda_r, alpha_r) selected with the
TGB train() procedure of hd_recb.py, from the STORED exact validation ranks of all 14 x 13 configurations, restricted to the
validation-select queries (harness ts id >= cut).  Relations without validation-select queries keep the TGB defaults.
Writes <out>/recb_selection_<name>.json in hd_recb.py's format (clean + faith), so `hd_recb.py --stage test --out <out>`
produces the RecB_xf CSR files and test ranks.  First reproduces the stored selection from all validation queries (assert).
  python3 v76_recb_xf_select.py --data_dir D --name tkgl-polecat --cut 1423 --orig_dir hardening/work/recb --out OUT
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_recb as HR

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--name", required=True); ap.add_argument("--cut", type=int, required=True)
ap.add_argument("--orig_dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LAMS, ALPHAS = HR.LAMS, HR.ALPHAS
st = open(f"{a.data_dir}/stat.txt").read().split(); N, R = int(st[0]), int(st[1])
Qv = HR.load_quads(f"{a.data_dir}/valid.txt")
sro = HR.split_queries(Qv, R, "valid")                          # identical order to the stored valid ranks
orig = json.load(open(f"{a.orig_dir}/recb_selection_{a.name}.json"))


def select(RK, rows):
    rr_ = (1.0 / np.asarray(RK[rows])).astype(np.float32).astype(np.float64)
    best_l, best_m = LAMS.index(0.1), 0.0
    mL = rr_[:, :, len(ALPHAS) - 1].mean(0)
    for li in range(len(LAMS)):
        if mL[li] > best_m: best_m, best_l = mL[li], li
    best_a, best_ma = ALPHAS.index(0.99), 0.0
    mA = rr_[:, best_l, :].mean(0)
    for ai in range(len(ALPHAS)):
        if mA[ai] > best_ma: best_ma, best_a = mA[ai], ai
    return best_l, best_a, float(best_m), float(best_ma)


sel = {}
for v in ("clean", "faith"):
    RK = np.load(f"{a.orig_dir}/recb_valid_ranks_{v}_{a.name}.npy", mmap_mode="r")
    assert RK.shape[0] == len(sro) and RK.shape[1:] == (len(LAMS), len(ALPHAS)), RK.shape
    Lr = np.full(2 * R, HR.I_TRDEF_L); Ar = np.full(2 * R, HR.I_TRDEF_A); detail = {}
    Lo = np.full(2 * R, HR.I_TRDEF_L); Ao = np.full(2 * R, HR.I_TRDEF_A)
    rel_all = sro[:, 1]; selm = sro[:, 3] >= a.cut
    n_sel_rel = np.zeros(2 * R, np.int64)
    for rel in range(2 * R):
        rows_all = np.flatnonzero(rel_all == rel)
        if len(rows_all):
            Lo[rel], Ao[rel], _, _ = select(RK, rows_all)
        rows = np.flatnonzero((rel_all == rel) & selm); n_sel_rel[rel] = len(rows)
        if len(rows) == 0:
            continue
        bl, ba, mlam, malp = select(RK, rows)
        Lr[rel] = bl; Ar[rel] = ba
        detail[rel] = dict(n=int(len(rows)), lam=LAMS[bl], alpha=ALPHAS[ba], mrr_lam=mlam, mrr_alpha=malp)
    ident = bool((Lo == np.array(orig[v]["lam_idx"])).all() and (Ao == np.array(orig[v]["alpha_idx"])).all())
    assert ident, f"procedure does not reproduce the stored {v} selection"
    nrow = len(sro); RKa = np.asarray(RK)
    ranks_xf = RKa[np.arange(nrow), Lr[rel_all], Ar[rel_all]]
    ranks_or = RKa[np.arange(nrow), Lo[rel_all], Ao[rel_all]]
    m_ = lambda rk, msk: float(np.mean((1.0 / rk[msk]).astype(np.float32).astype(np.float64))) if msk.any() else None
    has_any = np.bincount(rel_all, minlength=2 * R) > 0
    chg = (Lr != Lo) | (Ar != Ao)
    sel[v] = dict(lam_idx=Lr.tolist(), alpha_idx=Ar.tolist(), lam=[LAMS[i] for i in Lr], alpha=[ALPHAS[i] for i in Ar], detail=detail,
                  cut=a.cut, selection_rows="valid ts >= cut", n_select_queries=int(selm.sum()), n_valid_queries=int(nrow),
                  reproduces_original_selection_from_all_valid=ident,
                  n_rel_with_valid=int(has_any.sum()), n_rel_with_select=int((n_sel_rel > 0).sum()),
                  n_rel_valid_but_no_select_default=int((has_any & (n_sel_rel == 0)).sum()),
                  n_rel_changed_vs_original=int(chg.sum()), n_rel_changed_among_with_select=int((chg & (n_sel_rel > 0)).sum()),
                  share_valid_queries_in_changed_rel=float(chg[rel_all].mean()),
                  valid_MRR_xf_fit_rows=m_(ranks_xf, ~selm), valid_MRR_orig_fit_rows=m_(ranks_or, ~selm),
                  valid_MRR_xf_select_rows=m_(ranks_xf, selm), valid_MRR_orig_select_rows=m_(ranks_or, selm))
sel["meta"] = dict(orig_meta=orig.get("meta"), rule="v76_rule.json (caab7495) M4", note="xf selection on validation-select only")
json.dump(sel, open(f"{a.out}/recb_selection_{a.name}.json", "w"), indent=1)
print(json.dumps({v: {k: sel[v][k] for k in sel[v] if k not in ("lam_idx", "alpha_idx", "lam", "alpha", "detail")} for v in ("clean", "faith")}, indent=1))
