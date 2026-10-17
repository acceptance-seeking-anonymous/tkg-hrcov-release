"""Verbatim local notebook cell(s), session-archive index 636 (local python kernel, workspace root as cwd).
Latency table (fresh_window_latency_table.csv) from latency_systems.json + latency_ct.json.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 636
import pandas as pd
L = json.load(open(f"{hlat}/latency_systems.json")); CL = json.load(open(f"{hlat}/latency_ct.json"))
st = L["stages"]; g = lambda s, k, m: st[s][k][m]["s_per_1000"]
pre = CL["precompute"]["amortised_s_per_1000_valid_queries"]
rows_ = []
for mode in ("batch1", "batched"):
    base = g("base", "inference", mode); recb = g("RecB", "psi_xi", mode)
    ct_apply = CL["apply"][mode]["s_per_1000"]
    spec = {
      "TiRGN base": (base, 0.0, 0.0, "TiRGN encoder (once per timestamp, amortised) + decoder"),
      "BLEND (TiRGN + RecB)": (base, recb, g("BLEND", "blend_scoring", mode), "RecB psi/xi; MIX 0.3 blend + ordering"),
      "HRcov": (base, g("HRcov", "lists", mode) + g("HRcov", "features", mode), g("HRcov", "ranker", mode), "pattern lists + 25 features; 2x64 MLP + fusion + promoted list"),
      "SA": (0.0, recb + g("SA", "history_order", mode) + g("SA", "lists", mode) + g("SA", "features", mode), g("SA", "ranker", mode), "RecB_xf psi/xi + strict history order + conf_hist lists + 16 features; 2x64 MLP + promoted list"),
      "CountTRuCoLa comparator (C0 blend)": (base, ct_apply + pre, g("EXT", "blend_scoring", mode), "rule application (frozen rules; split statistics amortised) ; z-softmax blend with TiRGN + ordering"),
    }
    for sysn, (b, c, s_, desc) in spec.items():
        rows_.append(dict(system=sysn, setting="batch size 1" if mode == "batch1" else "batched (production)", base_inference_s_per_1k=b,
                          candidates_features_s_per_1k=c, ranker_or_blend_s_per_1k=s_, end_to_end_s_per_1k=b + c + s_, stages=desc))
LT = pd.DataFrame(rows_)
LT.to_csv("fw/out/fresh_window_latency_table.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(LT.drop(columns=["stages"]).round(4).to_string(index=False))
