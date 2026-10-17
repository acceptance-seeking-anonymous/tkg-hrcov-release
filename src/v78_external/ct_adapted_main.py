#!/usr/bin/env python3
"""v78_external / ct_adapted_main.py -- EXT_adapted for CountTRuCoLa (registered-optional run after the v78 verdict).

Runs the RELEASED rule_based/main.py (JuliaGast/counttrucola_submission @ 5d9aedf, unchanged code and config) with one
in-memory change: validation facts with timestamp < cut (harness 70% cut of the validation timestamps, both query
directions) are moved from the TGB val mask to the train mask before RuleDataset indexes the data, so rule examples,
rules and confidence parameters are learned on train + validation fit window. The val split then contains only the
selection window. Nothing is copied (each fact stays in exactly one split), so the application history is unchanged.
Placed in rule_based/ of a COPY of the repository and run from its root (so ray workers resolve rule_based modules
exactly as for the released entry point):  python rule_based/ct_adapted_main.py --params DATASET_NAME=<ds> NUM_CPUS=<n>
env: V78_HARNESS_DD (harness_data/<ds>), V78_CUT_TS_ID (harness ts id of the cut), V78_SPLIT_LOG (json path)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rule_based/
import rule_dataset                                    # imports tgb.linkproppred.dataset.LinkPropPredDataset (vendored)

DD = os.environ["V78_HARNESS_DD"]; CUT_ID = int(os.environ["V78_CUT_TS_ID"]); LOGP = os.environ["V78_SPLIT_LOG"]
id2raw = {}
for ln in open(f"{DD}/timestamp2id.txt"):
    k, v = ln.split(); id2raw[int(v)] = int(float(k))
CUT_RAW = id2raw[CUT_ID]
_Base = rule_dataset.LinkPropPredDataset

class AdaptedLinkPropPredDataset(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        ts = np.asarray(self.full_data["timestamps"])
        tr, va, te = np.asarray(self.train_mask), np.asarray(self.val_mask), np.asarray(self.test_mask)
        fit = va & (ts < CUT_RAW)
        info = dict(dataset=self.name, cut_ts_id=CUT_ID, cut_raw=int(CUT_RAW), n_train_orig=int(tr.sum()), n_val_orig=int(va.sum()),
                    n_fit_moved=int(fit.sum()), n_val_selection=int((va & ~fit).sum()), n_test=int(te.sum()),
                    val_min_raw=int(ts[va].min()), val_max_raw=int(ts[va].max()), fit_max_raw=int(ts[fit].max()) if fit.any() else None,
                    selection_min_raw=int(ts[va & ~fit].min()))
        assert not (tr & va).any() and fit.any() and (va & ~fit).any()
        self._train_mask = tr | fit
        self._val_mask = va & ~fit
        assert int(self._train_mask.sum()) + int(self._val_mask.sum()) + int(te.sum()) == len(ts)
        json.dump(info, open(LOGP, "w"), indent=1)
        print("[v78 EXT_adapted] split moved:", json.dumps(info), flush=True)

rule_dataset.LinkPropPredDataset = AdaptedLinkPropPredDataset
import main as ct_main                                  # rule_based/main.py (released)
ct_main.main()
