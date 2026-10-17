"""Verbatim local notebook cell(s), session-archive index 610 (local python kernel, workspace root as cwd).
Select the external comparator from sel_ext_gdeltfw.json -> registration/ext_comparator.json.
Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
are the job output directories copied back from serverA. Only change: user name restored.
"""
# ===== cell 610
s = json.load(open("hpc/953e4300-193e-4b59-b2ef-2d81e8efb2bb/h/sel_ext_gdeltfw.json"))
c0 = s["C0"]
variant = "blend" if c0["blend_select_MRR"] > c0["alone_select_MRR"] else "alone"
comp = dict(config="C0", variant=variant, a=c0["a"] if variant == "blend" else None, tau=c0["tau"] if variant == "blend" else None,
            selection_MRR=c0["blend_select_MRR"] if variant == "blend" else c0["alone_select_MRR"],
            alone_selection_MRR=c0["alone_select_MRR"], best_blend_selection_MRR=c0["blend_select_MRR"], base_selection_MRR=s["base_select_MRR"],
            n_selection_queries=s["n_select"], cut_ts=s["cut"], candidates="EXT_alone(C0) + 52 EXT_blend(C0) (a in 13 BLEND betas x tau in {0.5,1,2,4})",
            rule="external_comparator = argmax selection-part MRR (ties averaged) over the CountTRuCoLa alone and blend candidates (v79 amendment 1; implementation note + addendum 1)",
            ct_params="LEARN_WINDOW_SIZE=10 DATAPOINT_THRESHOLD_MULTI=0 RULE_UNSEEN_NEGATIVES=30 AGGREGATION_DECAY=0.8 NUM_TOP_RULES=10 RULE_TYPE_C=True F_UNSEEN_NEGATIVES=10 Z_RULES_FACTOR=0.1",
            ct_release="JuliaGast/counttrucola_submission @ 5d9aedf4106926d13befa141e0855d684f3f5f9d (config-default.yaml top level)",
            selected_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
json.dump(comp, open("fw/out/ext_comparator.json", "w"), indent=1)
print(comp)
