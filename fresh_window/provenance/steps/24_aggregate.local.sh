#!/bin/bash
# Verbatim local notebook cell(s), session-archive index 648 (local bash kernel, workspace root as cwd).
# Aggregation, paired query bootstrap and timestamp-block bootstrap, outcome R1-R4: code/fw_agg.py
# -> registration/fresh_window_verdict.json + fresh_window_results.json.
# Cells of one file ran in sequence in the same kernel; variables from earlier cells are reused. Paths hpc/<job_id>/...
# are the job output directories copied back from serverA. Only change: user name restored.
# ===== cell 648
d=hpc/7349980c-57f6-49ec-a9d5-65fb82f265fb/h; cd fw/code_fw && python3 fw_agg.py --tirgn ../../$d/official_test_gdeltfw_tirgn.npz --sa ../../$d/official_test_gdeltfw_sa.npz --ext ../../$d/official_test_gdeltfw_ext.npz --ts2id ../../hpc/faac377d-9748-42af-9403-7352ff11995d/out/timestamp2id.txt --comparator ../out/ext_comparator.json --out ../out/fresh_window_verdict.json --results ../out/fresh_window_results.json
