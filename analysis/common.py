"""analysis/common.py -- shared helpers for the table/figure scripts (read result files under RESULTS_DIR)."""
import csv, json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("RESULTS_DIR_IN", os.path.join(REPO, "results"))      # released result files (inputs)
OUT = os.environ.get("TABLES_OUT", os.path.join(REPO, "results", "tables"))      # generated tables / figure data


def res(name):
    """path of a released result file, searched by file name under results/ (names are unique)."""
    for root, _, files in os.walk(RESULTS):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {RESULTS}")


def load(name):
    return json.load(open(res(name), encoding="utf-8"))


def text(name):
    return open(res(name), encoding="utf-8").read()


def md_tables(doc):
    """markdown tables of a report: list of (line index, header cells, rows)."""
    L = doc.splitlines(); out = []; i = 0
    while i < len(L) - 1:
        if L[i].startswith("|") and re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", L[i + 1]):
            hdr = [c.strip() for c in L[i].strip().strip("|").split("|")]; rows = []; j = i + 2
            while j < len(L) and L[j].startswith("|"):
                rows.append([c.strip() for c in L[j].strip().strip("|").split("|")]); j += 1
            out.append((i, hdr, rows)); i = j
        else:
            i += 1
    return out


def f4(x):
    return f"{x:.4f}"


def mt(x, d=4):
    """signed number in the manuscript's math-mode convention ($+$0.0123 / $-$0.0123; -0.0000 printed as $+$0.0000)."""
    v = round(x, d); v = 0.0 if v == 0 else v
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.{d}f}"


def mci(c, d=4):
    return f"[{mt(c[0], d)}, {mt(c[1], d)}]"


def write_json(obj, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name); json.dump(obj, open(p, "w", encoding="utf-8"), indent=1); return p


def write_csv(rows, header, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    return p
