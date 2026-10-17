import os
# tie diagnostics on the TEST dumps: share of queries whose gold is in the list with a non-floor score that is tied with >= 1 other listed entity
import numpy as np, json, glob
B = os.path.expandvars("$WORK_DIR/tracks/v78_external/dumps")
out = {}
for D in sorted(glob.glob(f"{B}/*/*/test")):
    g = np.load(f"{D}/gold.npy"); gs = np.load(f"{D}/gold_score.npy"); fl = np.load(f"{D}/floor.npy"); rr = np.load(f"{D}/rr_official.npy")
    ti = np.load(f"{D}/top_ids.npy", mmap_mode="r"); ts = np.load(f"{D}/top_scores.npy", mmap_mode="r")
    Q = len(g); tied = 0; gold_scored = 0; tied_rr = []; at_floor = int((gs <= fl).sum())
    for c0 in range(0, Q, 20000):
        a = np.asarray(ti[c0:c0+20000]); s = np.asarray(ts[c0:c0+20000]); gg = g[c0:c0+20000]; gsc = gs[c0:c0+20000].astype(s.dtype)
        sel = (a == gg[:, None]).any(1) & (gsc > fl[c0:c0+20000].astype(s.dtype))
        gold_scored += int(sel.sum())
        nt = (s == gsc[:, None]).sum(1)
        t = sel & (nt > 1); tied += int(t.sum()); tied_rr.append(rr[c0:c0+20000][t])
    tr = np.concatenate(tied_rr)
    out[D.replace(B + "/", "")] = dict(rows=Q, gold_scored_above_floor=gold_scored, gold_tied_with_other_listed=tied, frac_tied=tied / Q,
                                       mean_official_rr_of_tied=float(tr.mean()) if len(tr) else None, gold_at_or_below_floor=at_floor)
print(json.dumps(out, indent=1))
