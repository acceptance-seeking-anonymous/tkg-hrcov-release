"""
retriever track (idea 6.1) / rt_common.py -- shared pieces for stages A-D.

Imports the shared harness READ-ONLY (oih_common, bridge_features); never writes to it.
"""
import os, sys, json, time
import numpy as np

HARNESS = os.path.expandvars("$WORK_DIR")
TRACK = f"{HARNESS}/tracks/retriever"
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from patterns import EventIndex, instantiate, decode, lag_bucket, L_BUCKETS   # noqa: E402

KREF = {"ICEWS18": 50, "GDELT": 200}          # base window used for gold-free features (harness convention)
M1, M2 = 64, 32                               # pattern instantiation caps
DELTA0 = 1.0                                  # bump used inside the predicted residual utility
B_GRID = [10, 30, 100, 300, 1000]
D_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
BMAX = max(B_GRID)
BETAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [0, 1, 2]
TE_A = 10.0                                   # target-encoding smoothing strength
DYAD_WINDOW = 65                              # BridgeMem: <=8 most recent dyad events within 65 snapshots
DYAD_MAXEV = 8
DYAD_KAPPA = 100.0                            # BridgeMem support-adaptive backoff tau=100


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


# ----------------------------------------------------------------------------- data
class DS:
    """Dataset bundle: harness Splits, dumps (train/valid/test views), event index."""

    def __init__(self, name, base="tirgn", load_valid=True):
        from oih_common import Splits, BaseDump, DATASETS
        t0 = time.time()
        self.name = name; self.base = base
        self.cfg = DATASETS[name]
        self.S = Splits(name)
        self.N = self.S.n_ent; self.R = self.S.R; self.step = self.cfg["step_unit"]
        self.K = KREF[name]
        self.D = BaseDump(self.cfg["dumps"][base], self.S)
        vkey = f"{base}_valid"
        self.DV = BaseDump(self.cfg["dumps"][vkey], self.S) if (load_valid and vkey in self.cfg["dumps"]) else self.D
        allq = np.concatenate([self.S.q["train"], self.S.q["valid"], self.S.q["test"]])
        self.EI = EventIndex(allq, self.N, self.R, self.step)
        tr = self.S.q["train"]
        self.freq = self.S.freq
        self.logfreq = np.log1p(self.freq).astype(np.float32)
        self.sfreq = np.bincount(np.concatenate([tr[:, 0], tr[:, 2]]), minlength=self.N)
        self.rfreq = np.bincount(np.concatenate([tr[:, 1], tr[:, 1] + self.R]), minlength=2 * self.R)
        self.load_s = round(time.time() - t0, 1)

    def dump_for(self, split):
        return self.DV if split == "valid" else self.D

    def query(self, D, i):
        s, r, o, ts = (int(x) for x in D.sro[i])
        return s, r, o, ts, ts // self.step, D.topk_ids[i, :self.K], D.topk_scores[i, :self.K], float(D.gold_score[i])


# ----------------------------------------------------------------------------- features
QNUM = ["q_entropy", "q_top1_lp", "q_margin12", "q_tailmass", "q_subj_logfreq", "q_rel_logfreq",
        "q_log_nhop1", "q_log_npat", "q_log_union"]
PNUM = ["hop2", "log_ncand", "log_npath", "log_ninterm", "log_mass", "frac_intop", "log1p_nintop",
        "max_lp", "best_pos", "mean_lf", "max_lf", "min_lf", "eq1", "eq2", "eq1inv", "lb1", "lb2"]
TEF = ["te_full", "te_full_n", "te_rr", "te_rr_n", "te_r2", "te_r2_n", "te_p", "te_p_n", "te_hier"]


def te_keys(r, pid, hop, rel1, rel2, R):
    R2p = 2 * R + 1
    hb = (hop - 1).astype(np.int64)
    k_full = np.int64(r) * np.int64(20_000_000) + pid                      # pid < (2R*L)^2 + 2R*L < 2e7
    k_rr = ((np.int64(r) * 2 + hb) * R2p + rel1) * R2p + (rel2 + 1)
    k_r2 = (np.int64(r) * 2 + hb) * R2p + np.where(hop == 2, rel2, rel1)
    k_p = pid
    return k_full, k_rr, k_r2, k_p


class TETables:
    """Target-encoding tables (hits, n) at 4 levels, built on the STATS part of train."""

    def __init__(self, path=None, arrs=None):
        if path is not None:
            z = np.load(path)
            arrs = {k: z[k] for k in z.files}
        self.a = arrs
        self.prior = arrs["prior"]                 # [2] global hit rate per hop

    def lookup(self, lvl, keys):
        K = self.a[f"{lvl}_k"]; H = self.a[f"{lvl}_h"]; Nn = self.a[f"{lvl}_n"]
        pos = np.searchsorted(K, keys); pos = np.clip(pos, 0, len(K) - 1)
        ok = K[pos] == keys
        return np.where(ok, H[pos], 0.0), np.where(ok, Nn[pos], 0.0)

    def features(self, r, pid, hop, rel1, rel2, R):
        kf, krr, kr2, kp = te_keys(r, pid, hop, rel1, rel2, R)
        pri = self.prior[(hop - 1).astype(np.int64)]
        a = TE_A
        out = []
        hs = {}
        for lvl, k in (("full", kf), ("rr", krr), ("r2", kr2), ("p", kp)):
            h, n = self.lookup(lvl, k); hs[lvl] = (h, n)
            out.append((h + a * pri) / (n + a)); out.append(np.log1p(n))
        # hierarchical back-off: r2 -> rr -> full
        c_r2 = (hs["r2"][0] + a * pri) / (hs["r2"][1] + a)
        c_rr = (hs["rr"][0] + a * c_r2) / (hs["rr"][1] + a)
        c_full = (hs["full"][0] + a * c_rr) / (hs["full"][1] + a)
        out.append(c_full)
        return np.stack(out, 1).astype(np.float32)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class FeatureBuilder:
    def __init__(self, ds, te=None):
        self.ds = ds; self.te = te
        self.N = ds.N; self.R = ds.R; self.K = ds.K

    def base_view(self, tki, tks):
        """gold-free base distribution summary (top-K exact + tail-uniform)."""
        N, K = self.N, len(tki)
        sc = tks.astype(np.float64)
        p = np.exp(sc)
        tail = max(1.0 - p.sum(), 1e-12)
        lp_tail = np.log(tail / max(N - K, 1))
        lp = np.full(N, lp_tail); lp[tki] = sc
        pos = np.full(N, K, np.int64); pos[tki] = np.arange(K)
        pw = np.exp(sc - sc.max()); pw /= pw.sum()
        ent = float(-(pw * np.log(pw + 1e-12)).sum())
        return lp, pos, dict(ent=ent, top1=float(sc[0]), m12=float(sc[0] - sc[1]), tail=float(np.log(tail)))

    def build(self, s, r, t_snap, tki, tks, gold=None, gold_lp=None):
        ds = self.ds
        d = instantiate(ds.EI, s, t_snap, M1, M2)
        if d is None:
            return None
        lp, pos, qv = self.base_view(tki, tks)
        pid = d["pid"]; P = len(pid); pp = d["pc_p"].astype(np.int64); cc = d["pc_c"].astype(np.int64)
        lpc = lp[cc]
        prob = np.exp(lpc)
        ncand = np.bincount(pp, minlength=P).astype(np.float64)
        mass = np.bincount(pp, weights=prob, minlength=P)
        intop = (pos[cc] < self.K)
        nin = np.bincount(pp, weights=intop, minlength=P)
        maxlp = np.full(P, -1e9); np.maximum.at(maxlp, pp, lpc)
        bestpos = np.full(P, self.K, np.int64); np.minimum.at(bestpos, pp, pos[cc])
        lf = ds.logfreq[cc].astype(np.float64)
        meanlf = np.bincount(pp, weights=lf, minlength=P) / np.maximum(ncand, 1)
        maxlf = np.full(P, -1.0); np.maximum.at(maxlf, pp, lf)
        minlf = np.full(P, 1e9); np.minimum.at(minlf, pp, lf)
        hop, rel1, lb1, rel2, lb2 = decode(pid, self.R)
        rinv = r - self.R if r >= self.R else r + self.R
        n_union = len(np.unique(cc))
        Xp = np.stack([hop == 2, np.log1p(ncand), np.log1p(d["npath"]), np.log1p(d["ninterm"]),
                       np.log(np.maximum(mass, 1e-12)), nin / np.maximum(ncand, 1), np.log1p(nin),
                       maxlp, bestpos / self.K, meanlf, maxlf, minlf,
                       rel1 == r, rel2 == r, (rel1 == rinv) | (rel2 == rinv), lb1, np.maximum(lb2, 0)], 1).astype(np.float32)
        qn = np.array([qv["ent"], qv["top1"], qv["m12"], qv["tail"], np.log1p(ds.sfreq[s]), np.log1p(ds.rfreq[r]),
                       np.log1p(d["n_hop1"]), np.log1p(P), np.log1p(n_union)], np.float32)
        cat = np.stack([rel1 + 1, lb1 + 1, rel2 + 1, lb2 + 1], 1).astype(np.int64)   # 0 = none
        Xte = self.te.features(r, pid, hop, rel1, rel2, self.R) if self.te is not None else None
        out = dict(d=d, P=P, pid=pid, hop=hop, rel1=rel1, rel2=rel2, Xp=Xp, qn=qn, cat=cat, Xte=Xte,
                   mass=mass, ncand=ncand, lp=lp, pos=pos, pp=pp, cc=cc)
        if gold is not None:
            hitpair = cc == gold
            out["hit"] = np.bincount(pp[hitpair], minlength=P) > 0
            # harness-consistent utility target at bump DELTA0 (full partition; exact gold prob)
            if gold_lp is not None:
                # harness UtilityEstimator.u_gold_entities convention: exact top-K and gold probs,
                # tail-uniform over the remaining N-K(-1) entities
                p_top = np.exp(tks.astype(np.float64)); intop_g = bool(np.any(tki == gold))
                tail_h = min(max(1.0 - p_top.sum() - (0.0 if intop_g else np.exp(gold_lp)), 0.0), 1.0)
                pt_h = tail_h / max(self.N - self.K - (0 if intop_g else 1), 1)
                ph = np.where(intop, prob, pt_h)
                if not intop_g:
                    ph = np.where(hitpair, np.exp(gold_lp), ph)
                m_exact = np.bincount(pp, weights=ph, minlength=P)
            else:
                m_exact = mass
            out["u_target"] = DELTA0 * out["hit"] - np.log1p(m_exact * (np.exp(DELTA0) - 1.0))
        return out


def pattern_utility(p_hit, mass, delta0=DELTA0):
    """predicted residual utility of bumping C_p by delta0: E[u] = d0*P(hit) - log(1+(e^d0-1) m_p)."""
    return delta0 * p_hit - np.log1p(mass * (np.exp(delta0) - 1.0))


def candidate_order(fb_out, util, rng_tiebreak=None, bmax=BMAX):
    """Rank candidates by the best utility of any pattern that contains them (tie-break: summed
    positive utility, then base log-prob).  Returns ordered candidate ids (<= bmax)."""
    pp = fb_out["pp"]; cc = fb_out["cc"]
    u_pair = util[pp]
    N = fb_out["lp"].shape[0]
    best = np.full(N, -np.inf); np.maximum.at(best, cc, u_pair)
    supp = np.bincount(cc, weights=np.maximum(u_pair, 0.0), minlength=N)
    cand = np.unique(cc)
    o = np.lexsort((-fb_out["lp"][cand], -supp[cand], -best[cand]))
    return cand[o][:bmax]


# ----------------------------------------------------------------------------- numpy MLP retriever
def np_forward(W, qcat_r, qn, cat, Xp, Xte):
    """Deterministic (z = mu) forward of the retriever exported to numpy. Returns logits [P]."""
    relu = lambda x: np.maximum(x, 0.0)
    eq = W["E_q"][qcat_r]
    qin = np.concatenate([eq, (qn - W["qmu"]) / W["qsd"]])
    h = relu(qin @ W["q1w"] + W["q1b"])
    mu = h @ W["qmuw"] + W["qmub"]
    pin = np.concatenate([W["E_rel"][cat[:, 0]], W["E_lb"][cat[:, 1]], W["E_rel"][cat[:, 2]], W["E_lb"][cat[:, 3]],
                          (np.concatenate([Xp, Xte], 1) - W["pmu"]) / W["psd"]], 1)
    hp = relu(pin @ W["p1w"] + W["p1b"]); hp = relu(hp @ W["p2w"] + W["p2b"])
    zz = np.broadcast_to(mu, (len(hp), len(mu)))
    inter = zz * (hp @ W["iw"])
    x = np.concatenate([zz, hp, inter], 1)
    x = relu(x @ W["h1w"] + W["h1b"])
    return (x @ W["h2w"] + W["h2b"])[:, 0] + W["logit_corr"]
