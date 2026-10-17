#!/usr/bin/env python3
"""analysis/check_manuscript_numbers.py -- regenerate the main table, the base-strength figure data, the v7.6 numbers and the
tables added in v7.9 (v7.8 ladder and external methods, fit-population audit, 2026 GDELT window) from results/ and compare them
with the values printed in the manuscript (tests/expected/: tab_main, base_strength and v76_fragments were extracted from
v7.6.1 and are unchanged in v7.9; v79_fragments was extracted from v7.9; v80text_fragments from the v8.0-text main text in
progress, where the new ladder table tab:ladder is Table 9 and the fresh-window table tab:fresh, Table 9 in v7.9, is Table 10).  Exit code 0 iff everything matches.

  python3 analysis/check_manuscript_numbers.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
EXP = os.path.join(os.path.dirname(HERE), "tests", "expected")
import tab_main, fig_basestrength_data, v76_tables, v79_tables
from common import f4, write_json


def check_main():
    exp = json.load(open(os.path.join(EXP, "tab_main_manuscript.json")))["rows"]
    got = []
    for blk, b, c, r, p, s, i, mark in tab_main.build():
        cells = []
        for x in (p, s, i):
            cells += ["." if x is None else f4(x[0]), "." if x is None else f4(x[1])]
        got.append(cells)
    n = sum(1 for e in exp for v in e["cells"] if v != ".")
    bad = [(e["label"], e["cells"], g) for e, g in zip(exp, got) if e["cells"] != g]
    return dict(item="tab:main", n_rows=len(exp), n_numbers=n, mismatches=bad, ok=(len(exp) == len(got) and not bad))


def check_basestrength():
    exp = json.load(open(os.path.join(EXP, "base_strength_manuscript.json")))["base_strength"]
    got = fig_basestrength_data.build()
    bad = [(e, g) for e, g in zip(exp, got) if e != g]
    return dict(item="fig:basestrength data", n_runs=len(exp), n_numbers=4 * len(exp), exact_float_equality=True, mismatches=bad,
                ok=(len(exp) == len(got) and not bad))


def check_v76():
    exp = json.load(open(os.path.join(EXP, "v76_fragments_manuscript.json")))["fragments"]
    got = v76_tables.build()
    bad = [(k, exp[k], got.get(k)) for k in exp if got.get(k) != exp[k]]
    return dict(item="v7.6 numbers (tab:mechanism, tab:g1, tab:rules, text)", n_fragments=len(exp), mismatches=bad, ok=not bad)


def check_v79():
    exp = json.load(open(os.path.join(EXP, "v79_fragments_manuscript.json")))["tables"]
    bad = v79_tables.compare(exp, v79_tables.build())
    n = sum(len(r["nums"]) for rows in exp.values() for r in rows)
    return dict(item="v7.9 tables (tab:standalone, tab:fresh; supplement v7.8, fit-population, fresh-window tables)",
                n_table_parts=len(exp), n_numbers=n, mismatches=bad, ok=not bad)


def check_v80text():
    exp = json.load(open(os.path.join(EXP, "v80text_fragments_manuscript.json")))["tables"]
    got = v79_tables.build(); got.update(v79_tables.build_v80())
    bad = v79_tables.compare(exp, got)
    n = sum(len(r["nums"]) for rows in exp.values() for r in rows)
    nl = sum(len(r["nums"]) for k, rows in exp.items() if k.startswith("tab:ladder") for r in rows)
    return dict(item="v8.0-text (in progress): tab:ladder (new, from fitpop_audit.json), tab:fresh and tab:standalone (unchanged)",
                n_table_parts=len(exp), n_numbers=n, n_numbers_tab_ladder=nl, mismatches=bad, ok=not bad)


def main():
    res = [check_main(), check_basestrength(), check_v76(), check_v79(), check_v80text()]
    for r in res:
        print(("PASS " if r["ok"] else "FAIL ") + r["item"] + "  " + json.dumps({k: v for k, v in r.items() if k not in ("item", "ok", "mismatches")}))
        for m in r["mismatches"][:20]:
            print("   mismatch:", m)
    write_json(res, "check_manuscript_numbers.json")
    sys.exit(0 if all(r["ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
