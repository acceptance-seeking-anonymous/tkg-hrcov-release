"""Verbatim local notebook cell(s), session-archive index 626 (local python kernel, workspace root as cwd).
Stage-1 validation summary (fresh_window_stage1_valid.json) from the job outputs.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 626
h1a = "hpc/c2ac489f-80cf-4350-bce1-72129f851218/h"; h1b = "hpc/dc39717c-1a99-4ff3-a9d1-5b5771f5dc65/h"; h1s = "hpc/9ccd0e7d-7bb9-47d1-8c34-6b0b173cb117/h"
hct = "hpc/e7efbdf3-008e-443f-a884-76fc2e78cdac/h"; hext = "hpc/953e4300-193e-4b59-b2ef-2d81e8efb2bb/h"; hlat = "hpc/c46772b5-49f4-49ba-bcd9-150ea7d41d98/h"
J_ = lambda p: json.load(open(p))
ep = J_(f"{h1a}/tirgn_GDELTFW_epoch_log.json")["epoch_log"]; best = max(ep, key=lambda x: x["valid_filt_MRR"])
tr = J_(f"{h1b}/train_fh.json"); sa = J_(f"{h1s}/train_fw_sa.json")["arms"]["SA_W100"]; sel = J_(f"{h1b}/sel_gdeltfw.json")["tirgn"]
rs = J_(f"{h1a}/recb_selection_GDELTFW.json"); xf = J_(f"{h1s}/recb_selection_GDELTFW.json"); hist = J_(f"{h1s}/hist_summary_valid.json")
ctm = J_(f"{hct}/C0/meta.json"); ctmem = J_(f"{hct}/C0/mem_ct_stage1.json"); ext = J_(f"{hext}/sel_ext_gdeltfw.json")
stage1 = dict(
  note="Stage-1 (train/validation only) record of the v79 fresh-window experiment on GDELTFW. Every number here is a validation number; no test statistic existed when this file was written.",
  base_tirgn=dict(epochs_run=len(ep), best_epoch=best["epoch"], valid_filtered_MRR_best=best["valid_filt_MRR"], epoch_log=ep, gpu_wall_s_stage1=626, gpu="serverA GPU 0 (RTX PRO 6000 Blackwell)"),
  base_selection_part_MRR_strict=tr["base_select"],
  recb=dict(trained_valid_MRR_clean=rs["clean"]["valid_MRR_trained"], default_valid_MRR_clean=rs["clean"]["valid_MRR_default"]),
  blend_selection=sel,
  hrcov=dict(valid_cut_ts=tr["valid_cut_ts"], n_fit_rows=tr["n_fit_rows"], n_select=tr["n_select"],
             seeds={s: dict(best_epoch=r["best_epoch"], select_MRR=r["select"]["MRR"], cal_alpha=r.get("cal_alpha"), cal_beta=r.get("cal_beta")) for s, r in tr["arms"]["HRCOV_PAT"]["seeds"].items()}),
  recb_xf=dict({k: xf["clean"][k] for k in xf["clean"] if k not in ("lam_idx", "alpha_idx", "lam", "alpha", "detail")}),
  sa=dict({k: sa[k] for k in sa if k not in ("seeds", "cols")}, seeds={s: dict(best_epoch=r["best_epoch"], select_MRR=r["select"]["MRR"]) for s, r in sa["seeds"].items()}),
  history_order_valid=hist["valid"],
  countrucola_C0=dict(valid_all_queries={k: ctm[k] for k in ("MRR", "H1", "H3", "H10", "n", "gold_in_topk_frac", "n_cand_median", "queries_without_ranking_line", "ranking_lines_unmatched", "evaluator_check")},
                      main_wall_s=ctmem["wall_s"], peak_pss_gb=ctmem["peak_pss_gb"], learn_s=985.7, apply_valid_s=646.0,
                      C1_C2="stopped for memory/time before any rule or score existed (addendum 1)"),
  ext_selection=dict(alone_select_MRR=ext["C0"]["alone_select_MRR"], best_blend=ext["C0"], base_select_MRR_ties_averaged=ext["base_select_MRR"], n_select=ext["n_select"]),
  external_comparator=json.load(open("fw/out/ext_comparator.json")))
json.dump(stage1, open("fw/out/fresh_window_stage1_valid.json", "w"), indent=1)
print(json.dumps({k: stage1[k] for k in ("hrcov", "sa")}, indent=0)[:1500])
