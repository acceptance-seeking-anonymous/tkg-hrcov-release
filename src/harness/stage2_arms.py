"""
harness / stage2_arms.py -- train/evaluate the GraphBridge stage-2 ranker arms (G10-G12
recipe: MLP NF->64->64->1, BPR, 40 epochs, Adam 1e-3 wd 1e-5, bs 4096, 64 sampled negatives per
reached train gold) from the g12 bridge cache, over several seeds, and write the G12 regression gate.

  python3 stage2_arms.py --dataset ICEWS18 --arms arm1,arm4,arm5 --seeds 0,1,2,3,4
Outputs (job cwd): regression_gate_<DS>.json, and <HARNESS>/results/arm5_<DS>_seed<k>.pt
(state_dict + feature mu/sd), reusable as the backfill ranker.

Metric definitions identical to stage2_ranker.run_arm: realized_hit@k = #(gold reached & rank<=k)/N_test,
realized_mrr = sum(1/rank over reached)/N_test, conditional_hit = over reached, retrieval_coverage =
reached/N_test; gate expanded_reach = reached/#built, median_cset over built test queries.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
from bridge_cache import BridgeCache
from oih_common import HARNESS, jdump

G12_REF = {
    "ICEWS18": dict(gate=dict(expanded_reach=0.7012, median_cset=6907),
                    arm1=dict(H50=0.1214, H200=0.2694, MRR=0.0163), arm4=dict(H50=0.1280, H200=0.2814, MRR=0.0177),
                    arm5=dict(H50=0.1252, H200=0.2896, MRR=0.0182)),
    "GDELT": dict(gate=dict(expanded_reach=0.9073, median_cset=7053),
                  arm1=dict(H50=0.0572, H200=0.1683, MRR=0.0097), arm4=dict(H50=0.0574, H200=0.1714, MRR=0.0087),
                  arm5=dict(H50=0.0670, H200=0.1716, MRR=0.0100)),
}
KS = [1, 10, 50, 200]


def make_net(nf):
    return torch.nn.Sequential(torch.nn.Linear(nf, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                               torch.nn.ReLU(), torch.nn.Linear(64, 1))


def train_arm(bc, tr_idx, arm, seed, n_neg=64, epochs=40, dev="cpu", Q=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Xpos, Xneg = [], []
    for j in tr_idx:
        if not (bc.built[j] and bc.reached[j]):
            continue
        X = Q[j] if Q is not None else bc.arm_features(j, arm)
        gj = int(bc.gold_pos[j]); n = X.shape[0]
        if n <= 1:
            continue
        negj = np.delete(np.arange(n), gj)
        take = rng.choice(negj, size=min(n_neg, len(negj)), replace=False)
        Xpos.append(np.repeat(X[gj:gj + 1], len(take), 0)); Xneg.append(X[take])
    Xpos = np.concatenate(Xpos).astype(np.float32); Xneg = np.concatenate(Xneg).astype(np.float32)
    allx = np.concatenate([Xpos, Xneg]); mu = allx.mean(0); sd = allx.std(0) + 1e-6
    Pp = torch.tensor((Xpos - mu) / sd, device=dev); Pn = torch.tensor((Xneg - mu) / sd, device=dev)
    net = make_net(Xpos.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(Pp); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = -torch.nn.functional.logsigmoid(net(Pp[idx]).squeeze(1) - net(Pn[idx]).squeeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd, int(n)


def score_with(net, mu, sd, X):
    with torch.no_grad():
        return net(torch.tensor((X - mu) / sd)).squeeze(1).numpy()


def eval_arm(bc, te_idx, net, mu, sd, arm, Q=None):
    N = len(te_idx); real = {k: 0 for k in KS}; mrr = 0.0; ncov = 0; ranks = np.zeros(N, np.int32)
    for n_, j in enumerate(te_idx):
        if not (bc.built[j] and bc.reached[j]):
            continue
        X = Q[j] if Q is not None else bc.arm_features(j, arm)
        s = score_with(net, mu, sd, X); order = np.argsort(-s)
        gr = int(np.where(order == int(bc.gold_pos[j]))[0][0]) + 1
        ncov += 1; mrr += 1.0 / gr; ranks[n_] = gr
        for k in KS:
            real[k] += int(gr <= k)
    return dict(retrieval_coverage=round(ncov / N, 4), realized_mrr=round(mrr / N, 4),
                realized_hit={k: round(real[k] / N, 4) for k in KS},
                conditional_hit={k: round(real[k] / max(1, ncov), 4) for k in KS}), ranks


def mcnemar_exact(a, b):
    from math import comb
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b)); n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", default="arm1,arm4,arm5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--save_arm5", default="1")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
    t0 = time.time()
    bc = BridgeCache(a.dataset, "g12")
    tr = np.where(bc.qsplit == 0)[0]; te = np.where(bc.qsplit == 2)[0]
    built_te = te[np.asarray(bc.built)[te]]
    sizes = np.diff(np.asarray(bc.offsets))
    gate = dict(expanded_reach=round(float(np.asarray(bc.reached)[built_te].mean()), 4),
                median_cset=int(np.median(sizes[built_te])), n_test=int(len(te)), n_test_built=int(len(built_te)),
                n_train=int(len(tr)), n_train_reached=int(np.sum(np.asarray(bc.reached)[tr])))
    ref = G12_REF[a.dataset]
    gate["ref"] = ref["gate"]
    gate["reach_match"] = abs(gate["expanded_reach"] - ref["gate"]["expanded_reach"]) <= 0.002
    gate["medC_match"] = abs(gate["median_cset"] - ref["gate"]["median_cset"]) <= max(5, 0.01 * ref["gate"]["median_cset"])
    print(json.dumps(gate), flush=True)
    res = dict(dataset=a.dataset, gate=gate, arms={}, cache=bc.dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    hits = {}
    for arm in a.arms.split(","):
        Q = {j: bc.arm_features(j, arm) for j in np.concatenate([tr, te]) if bc.built[j]}
        per = []
        for sd_ in seeds:
            net, mu, sd, npairs = train_arm(bc, tr, arm, sd_, Q=Q)
            m, ranks = eval_arm(bc, te, net, mu, sd, arm, Q=Q)
            m["seed"] = sd_; m["n_train_pairs"] = npairs; per.append(m)
            hits[(arm, sd_)] = ranks
            if arm == "arm5" and a.save_arm5 == "1":
                os.makedirs(f"{HARNESS}/results", exist_ok=True)
                torch.save(dict(state_dict=net.state_dict(), mu=mu, sd=sd, nf=int(len(mu)), arm=arm, seed=sd_,
                                dataset=a.dataset, cache=bc.dir), f"{HARNESS}/results/arm5_{a.dataset}_seed{sd_}.pt")
            print(f"[{arm} s{sd_}] H50={m['realized_hit'][50]} H200={m['realized_hit'][200]} MRR={m['realized_mrr']} ({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for key, getter in (("H50", lambda m: m["realized_hit"][50]), ("H200", lambda m: m["realized_hit"][200]),
                            ("H10", lambda m: m["realized_hit"][10]), ("H1", lambda m: m["realized_hit"][1]),
                            ("MRR", lambda m: m["realized_mrr"])):
            v = np.array([getter(m) for m in per])
            agg[key] = dict(mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                            min=round(float(v.min()), 4), max=round(float(v.max()), 4))
        cmp = {}
        if arm in ref:
            for key in ("H50", "H200", "MRR"):
                r_ = ref[arm][key]; mu_ = agg[key]["mean"]; s_ = agg[key]["sd"]
                tol = max(2 * s_, 0.005 if key != "MRR" else 0.001)
                cmp[key] = dict(ref=r_, mean=mu_, sd=s_, diff=round(mu_ - r_, 4), tol=round(tol, 4),
                                within=bool(abs(mu_ - r_) <= tol),
                                ref_in_seed_range=bool(agg[key]["min"] - 0.003 <= r_ <= agg[key]["max"] + 0.003))
        res["arms"][arm] = dict(per_seed=per, aggregate=agg, vs_G12=cmp)
    if ("arm4", seeds[0]) in hits and ("arm5", seeds[0]) in hits:
        pt = {}
        for k in (50, 200):
            pv = []
            for s_ in seeds:
                a4 = (hits[("arm4", s_)] > 0) & (hits[("arm4", s_)] <= k); a5 = (hits[("arm5", s_)] > 0) & (hits[("arm5", s_)] <= k)
                pv.append(dict(seed=s_, delta=round(float(a5.mean() - a4.mean()), 4), p_mcnemar=round(mcnemar_exact(a4, a5), 4)))
            pt[f"H{k}"] = pv
        res["paired_arm4_to_arm5"] = pt
    ok5 = res["arms"].get("arm5", {}).get("vs_G12", {})
    res["verdict"] = dict(
        gate_reach_medC_pass=bool(gate["reach_match"] and gate["medC_match"]),
        arm5_H200_pass=bool(ok5.get("H200", {}).get("within", False)),
        arm5_H50_pass=bool(ok5.get("H50", {}).get("within", False)))
    res["verdict"]["PASS"] = all(res["verdict"].values())
    res["elapsed_s"] = round(time.time() - t0, 1)
    np.savez(f"regression_gate_ranks_{a.dataset}.npz", te_rows=np.asarray(bc.qrow)[te],
             **{f"{k[0]}_s{k[1]}": v for k, v in hits.items()})
    jdump(res, f"regression_gate_{a.dataset}.json")
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    main()
