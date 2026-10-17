import os
# official per-query rr/rank of the CountTRuCoLa dumps (faithful and adapted) on the harness validation SELECTION rows
import numpy as np, json
T = os.path.expandvars("$WORK_DIR/tracks"); B = f"{T}/v78_external/dumps"
CFG = {"tkgl-smallpedia": dict(cut=105, rows=None), "tkgl-polecat": dict(cut=1423, rows=f"{T}/pc_tirgn/work/valid_sample_idx.npy")}
out = {}
for ds, c in CFG.items():
    for m in ("counttrucola", "counttrucola_adapted"):
        D = f"{B}/{m}/{ds}/val"
        keys = np.load(f"{D}/keys_harness.npy"); rr = np.load(f"{D}/rr_official.npy"); rk = np.load(f"{D}/rank_official.npy")
        hr = np.load(c["rows"]).astype(np.int64) if c["rows"] else np.arange(len(keys))
        sel_h = hr[keys[hr, 3] >= c["cut"]]; sel_all = np.flatnonzero(keys[:, 3] >= c["cut"])
        f = lambda r: dict(n=int(len(r)), MRR=float(rr[r].mean()), H1=float((rk[r] <= 1).mean()), H3=float((rk[r] <= 3).mean()), H10=float((rk[r] <= 10).mean()))
        out[f"{m}/{ds}"] = dict(selection_harness_rows=f(sel_h), selection_all_rows=f(sel_all), any_nan_on_selection=bool(np.isnan(rr[sel_all]).any()))
print(json.dumps(out, indent=1))
