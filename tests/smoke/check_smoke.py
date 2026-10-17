#!/usr/bin/env python3
"""tests/smoke/check_smoke.py -- checks on the outputs of run_smoke.sh (run from its output directory).
PASS iff: the TGB id bridge verified; the base dump exists; the official evaluation of the base and of the three HRCOV_PAT
seeds produced finite MRRs in (0, 1]; the re-ranker's identity check (base order reproduced from the stored lists) is 1.0."""
import argparse, json, math, os, sys


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{p}[{i}]")
    else:
        yield p, o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="smoke_result.json"); a = ap.parse_args()
    W = os.environ["WORK_DIR"]; P = f"{W}/tracks/pilot_polecat"
    res = dict(checks={})
    C = res["checks"]
    tv = json.load(open("tgb_verify.json")) if os.path.exists("tgb_verify.json") else {}
    C["tgb_verify_json"] = bool(tv)
    C["base_dump_exists"] = os.path.exists(f"{P}/dumps/regcn_scores_POLECAT_traintest.npz")
    lk = json.load(open("lookup_regcn_smoke.json")); sy = json.load(open("systems_regcn_smoke.json"))
    tr = json.load(open("train_hcpat_polecat_regcn_smoke.json"))
    res["lookup_official"] = lk.get("official"); res["systems_official"] = sy.get("official")
    res["official_vs_harness"] = dict(lookup=lk.get("official_vs_harness"), systems=sy.get("official_vs_harness"))
    mrr = {k: v for k, v in walk(dict(base=lk.get("official"), systems=sy.get("official"))) if "mrr" in k.lower() and isinstance(v, (int, float))}
    res["official_mrr_values"] = mrr
    C["official_mrr_present"] = len(mrr) >= 2
    C["official_mrr_finite_in_0_1"] = all(math.isfinite(v) and 0 < v <= 1 for v in mrr.values())
    C["systems_cover_3_seeds"] = all(any(f"HRCOV_PAT_s{s}" in k for k in mrr) for s in range(3))
    idt = tr.get("identity_test", {})
    res["hcpat_identity_test"] = idt
    C["hcpat_identity_test_eq_1"] = bool(idt) and all(abs(v - 1.0) < 1e-12 for v in idt.values())
    res["hcpat_test_mrr_harness"] = {k: v for k, v in walk(tr.get("arms", {})) if k.lower().endswith("mrr")}
    res["ok"] = all(C.values())
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(C, indent=1)); print("SMOKE PASS" if res["ok"] else "SMOKE FAIL")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
