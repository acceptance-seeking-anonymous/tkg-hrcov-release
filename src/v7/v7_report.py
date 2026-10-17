"""v7 / v7_report.py -- helpers to render B/C item results (dicts from v7_bagg_run.run_ds and v7_b4c3_agg) to markdown + CSV rows."""
import csv


def f5(x): return f"{x:+.5f}"
def ci(c): return f"[{c[0]:+.5f}, {c[1]:+.5f}]"
def ci4(c): return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def call_of(r, bar=0.005):
    m = r["MRR"]; holm_ok = all(m.get(f"holm_p_{k}", m[k]["p_one_sided_le0"]) < 0.05 for k in ("query", "ts"))
    g, l = m["excluded_gain_ge_bar"], m["excluded_loss_ge_bar"]
    if m["present_pos"] and holm_ok:
        return "present (+); gain ≥ 0.005 excluded" if g else "present (+)"
    if m["present_neg"]:
        return "present (−); loss ≥ 0.005 excluded" if l else "present (−)"
    ex = "bounded within ±0.005" if (g and l) else "gain ≥ 0.005 excluded" if g else "loss ≥ 0.005 excluded" if l else ""
    return "not established" + (f"; {ex}" if ex else "")


def arm_table(arms, keys=None):
    keys = keys or list(arms)
    L = ["| system | MRR | H@1 | H@3 | H@10 | per-seed MRR |", "|---|---|---|---|---|---|"]
    for k in keys:
        if k not in arms: continue
        v = arms[k]; ps = ", ".join(f"{x:.5f}" for x in v["per_seed_MRR"]) if v["n_seeds"] > 1 else "-"
        L.append(f"| {k} | {v['MRR']:.5f} | {v['H1']:.4f} | {v['H3']:.4f} | {v['H10']:.4f} | {ps} |")
    return "\n".join(L)


def cmp_table(rows):
    L = ["| comparison | role | ΔMRR | query 95% CI | timestamp 95% CI | U_ts | ΔH@1 [ts CI] | ΔH@3 [ts CI] | ΔH@10 [ts CI] | MDE80_ts | call |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["MRR"]
        hs = " | ".join(f"{r[h]['delta']:+.4f} {ci4(r[h]['ts']['CI95'])}" for h in ("H1", "H3", "H10"))
        call = call_of(r)
        hp = ""
        fp = lambda p: "<0.0005" if p == 0 else f"{p:.3g}"
        if "holm_p_ts" in m: hp = f" (Holm p query/ts {fp(m['holm_p_query'])}/{fp(m['holm_p_ts'])})"
        L.append(f"| {r['comparison']} | {r['role']} | {f5(m['delta'])} | {ci(m['query']['CI95'])} | {ci(m['ts']['CI95'])} | {m['ts']['U']:+.5f} | {hs} | {m['MDE80_ts']:.5f} | {call}{hp} |")
    return "\n".join(L)


def csv_rows(item, dataset, base, rows):
    out = []
    for r in rows:
        for met in ("MRR", "H1", "H3", "H10"):
            m = r[met]
            out.append(dict(item=item, dataset=dataset, base=base, comparison=r["comparison"], role=r["role"], metric=met,
                            delta=round(m["delta"], 6), query_lo=round(m["query"]["CI95"][0], 6), query_hi=round(m["query"]["CI95"][1], 6),
                            ts_lo=round(m["ts"]["CI95"][0], 6), ts_hi=round(m["ts"]["CI95"][1], 6), L_query=round(m["query"]["L"], 6),
                            L_ts=round(m["ts"]["L"], 6), U_ts=round(m["ts"]["U"], 6), MDE80_ts=round(m["MDE80_ts"], 6),
                            present_pos=m["present_pos"], present_neg=m["present_neg"], gain_ge_bar_excluded=m["excluded_gain_ge_bar"],
                            loss_ge_bar_excluded=m["excluded_loss_ge_bar"], call=(call_of(r) if met == "MRR" else "")))
    return out


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
