# Fusion and reader follow-ups: co-participation and eventuality-transition edges, a reader without document links, and dated Wikidata edges

**Track:**  Task 2 (temporal knowledge fusion) and Task 3 (contextual reasoning with a time-masked reader).
**Verdict file:** `fusion_followups_verdict.json`. Detailed statistics: `s12_verdict_raw.json` (steps 1–2) and `s3_verdict.json` (step 3).
**Pre-registered rules** (each saved before any number of its step was read): step 1, step 2, step 3. The step-3 rule carries a provenance correction: the raw entity-date fetch had already started when the rule was written. The criteria are unchanged.
**Hosts:**
* Steps 1–2: every list, feature and model number comes from `serverB`. Lists and features ran on CPU with 8 workers. The main re-rankers ran on CPU with 8 threads, because the 4090 was occupied by another user's job at every training boundary. The post-hoc RE-GCN arms ran on the 4090 once it was idle.
* Step 3: Wikidata dates were fetched from the local sandbox via query.wikidata.org. Lists, features and re-rankers ran on `serverA` on CPU with 8 workers/threads. GPU 0 was busy at every training boundary, and GPU 1 was never used.
* Aggregation, bootstrap and figures: local laptop.
**Protocol (all steps):** time-aware filtered ranking, both directions, full test split, exact ranks, 3 seeds (seed-averaged per-query RR), valid-only selection. Statistics use the harness Poisson(1) paired bootstrap (B = 2,000 for rule clauses, 1,000 for other strata; one-sided p). The head-damage guardrail is the pilots' convention.

## 0. Verdicts

| step | question | pre-registered answer |
|---|---|---|
| 1a | Same-article **co-participation** as a fusion edge type in HRCOV (MidEast-TE) | **Useful**: +0.0063 (RE-GCN) / +0.0051 (LogCL) MRR over HRCOV, PASS on both. It works as an edge **feature**; the candidate block itself adds ≤ +0.0007. |
| 1b | **Eventuality succession** edges P_ECCKG(r_q \| r_prev) | **Not useful**: +0.0028 (pass) / +0.0017 (fail, < 0.002). |
| 1b′ | Succession from **training statistics** (labelled substitute) | **Useful**: +0.0030 / +0.0024, PASS on both. Post-hoc: 25–70% of this is plain pair activity; the transition weighting itself adds +0.0010 / +0.0018 overall and about +0.003 on dyadic-only golds. |
| 2 | Does the time-masked **text reader alone** carry a gain once document links are removed? | **Yes**: +0.0035 / +0.0025 overall, +0.0036 / +0.0029 on exact-recurrence-in-pool, beats shuffled text, PASS on both. It keeps about 80% of the linked-reader gain and stays below co-participation. |
| 3 | Does the smallpedia **2-hop static gain** survive when static edges are dated (valid before t only)? | **Yes, 3 of 3 bases**: +0.0033 / +0.0031 / +0.0019 over HRCOV (survival fraction 1.09 / 0.89 / 0.37). Statement qualifiers are essentially absent (1.05%), so the dating uses entity lifetimes, which date 91.4% of edges. Full static fusion loses 23–30% of its gain to dated-out facts. |

!Fusion edges and reader

*Figure 1.* Δ MRR with 95% paired-bootstrap CIs (3 seeds, MidEast-TE full test, n = 89,896 queries); blue = RE-GCN base, orange = LogCL base. (a) Fusion-edge arms against HRCOV (= history re-ranker over top-50 + pattern-prior candidates). (b) The same arms on dyadic-only golds (pair history only in another relation, 17.1% of queries). (c) Story-reader arms against HR (top-50 re-ranker). Filled circles = pre-registered arms; open diamonds = diagnostics and post-hoc controls ("pair-event count only" and both pilot-4 reference arms in c). Dotted line = the pre-registered +0.002.


## 1. Step 1: co-participation and eventuality-transition fusion edges ( Task 2), MidEast-TE

**Question.** Do (a) same-article co-participation edges and (b) eventuality-style relation-succession edges help when they are fused into HRCOV, both as extra candidates and as features?

**Setup.** We used the pilot-4 frozen bases (RE-GCN, 16 epochs; LogCL, 12 epochs) and the full test split (89,896 queries, both directions, time-aware filtered, exact ranks from the full log-prob rows). Every selection used the valid split only; each arm has 3 seeds. Each query gets a W = 300 candidate layout: [base top-50 | pattern prior 50 | COOC 50 | SUCC_TR 50 | SUCC_E 50 | degree-matched random 50]. Every edge block is the first 50 entities of its own list that are not already in top-50 ∪ pattern. An arm is a column subset plus a feature subset, so all arms share one feature tensor.
* **HRCOV (reference)** = history re-ranker over the base top-50 plus 50 pattern-prior candidates (6.1 patterns, exact base masses). Features are 8 base, 11 history, 5 pattern and a block flag. This is the pilots' HRCOV_PAT arm. The stage-2 bridge block is left out because it contributed about 0 on polecat and smallpedia (deviation).
* **COOC edges (a).** c is linked to s if both take part in events extracted from the same article, using articles dated ≤ t−1 over the full history. Candidates are ranked by co-article count decayed with τ = 30 d. There are 8 features: log counts for all time, for articles without a direct s–c event, for the last 30 d and the last 7 d, the decayed count, recency, a within-pool z-score and an any-flag.
* **SUCC edges (b).** For s's events in the last 365 d (≤ 4,096), score(c) = Σ_e exp(−lag_e/30)·P(r_q | r_e). There are 4 features: log score, maximum and recency-weighted mean lift over s–c events, and the event count. Two priors are used:
  * **SUCC_E (eventuality commonsense, ECCKG).** Each CAMEO relation name is anchored to its 10 nearest ECCKG eventualities (bge-m3, mean top-1 cosine 0.757). Precedence uses Before / Causes / CausesDesire / HasPrerequisite⁻¹ paths, 1-hop plus 0.5 × 2-hop (18,402 eventualities, 800,260 edges). No TKG data are used. 40.1% of the 234 × 234 relation pairs have ECCKG support, and they cover 60.8% of the training transition mass. The Spearman correlation with the training lift is only 0.14.
  * **SUCC_TR (labelled substitute).** The same construction with P_tr(r_q | r_prev) estimated from consecutive same-pair train events (3.42 M transitions).
* **Rule** (`fusion_followups_step1_decision_rule.json`): an edge type is useful iff HRCOV+edge − HRCOV ≥ +0.002 MRR with one-sided p < 0.05 on both bases, and the head stratum is not damaged (gold train frequency ≥ 500; on MidEast-TE this is 82.8% of test queries, so the guardrail is close to an overall check). The dyadic-only stratum is reported separately.

**Coverage.** Gold in the test pool: top-50 84.9% / 85.0% (RE-GCN / LogCL); the pattern block adds 3.5%. The COOC block adds a further 2.9% / 2.8%, and the succession blocks add 2.3% / 2.3% (training statistics) and 2.2% / 2.2% (ECCKG). The degree-matched block adds 0.5%. The share of golds inside the arm's columns rises from 88.4% (HRCOV) to 91.3% (HRCOV_COOC) on RE-GCN.


| arm | RE-GCN MRR | H@1 | H@3 | H@10 | LogCL MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|---|---|
| frozen base | 0.4482 | 0.3260 | 0.5129 | 0.6849 | 0.4569 | 0.3345 | 0.5238 | 0.6927 |
| HRCOV (reference) | 0.4863 | 0.3615 | 0.5591 | 0.7225 | 0.4875 | 0.3633 | 0.5594 | 0.7223 |
| HRCOV + co-participation edges | 0.4926 | 0.3686 | 0.5649 | 0.7260 | 0.4926 | 0.3685 | 0.5653 | 0.7262 |
| HRCOV + succession edges (ECCKG) | 0.4891 | 0.3642 | 0.5616 | 0.7242 | 0.4892 | 0.3648 | 0.5612 | 0.7243 |
| HRCOV + succession edges (train statistics, substitute) | 0.4894 | 0.3644 | 0.5618 | 0.7249 | 0.4899 | 0.3654 | 0.5619 | 0.7251 |
| diag: co-participation features, no block | 0.4918 | 0.3681 | 0.5643 | 0.7257 | 0.4925 | 0.3687 | 0.5653 | 0.7259 |
| diag: degree-matched random block + co-participation features | 0.4925 | 0.3686 | 0.5648 | 0.7260 | 0.4921 | 0.3682 | 0.5646 | 0.7261 |
| diag: all three edge types | 0.4930 | 0.3688 | 0.5655 | 0.7266 | 0.4933 | 0.3694 | 0.5660 | 0.7270 |
| post-hoc: succession block, 365-d pair-event count only | 0.4884 | 0.3636 | 0.5612 | 0.7245 | 0.4881 | 0.3637 | 0.5603 | 0.7239 |
| post-hoc: train-succession features, no block | 0.4889 | 0.3642 | 0.5617 | 0.7247 | 0.4898 | 0.3654 | 0.5623 | 0.7250 |
| post-hoc: ECCKG-succession features, no block | 0.4889 | 0.3642 | 0.5617 | 0.7243 | 0.4893 | 0.3652 | 0.5612 | 0.7238 |

| edge type | base | Δ MRR vs HRCOV [95% CI] | p | head Δ [CI] | DAMAGE | dyadic-only Δ [CI] | verdict |
|---|---|---|---|---|---|---|---|
| co-participation (a) | RE-GCN | +0.0063 [+0.0056, +0.0069] | < 5e-4 | +0.0064 [+0.0056, +0.0072] | no | +0.0054 [+0.0040, +0.0069] | **PASS** |
| co-participation (a) | LogCL | +0.0051 [+0.0045, +0.0058] | < 5e-4 | +0.0048 [+0.0039, +0.0056] | no | +0.0069 [+0.0054, +0.0083] | **PASS** |
| co-participation (a) | both |  |  |  |  |  | **useful** |
| succession, ECCKG (b) | RE-GCN | +0.0028 [+0.0023, +0.0033] | < 5e-4 | +0.0025 [+0.0019, +0.0031] | no | +0.0029 [+0.0019, +0.0040] | **PASS** |
| succession, ECCKG (b) | LogCL | +0.0017 [+0.0012, +0.0022] | < 5e-4 | +0.0016 [+0.0010, +0.0021] | no | +0.0013 [+0.0003, +0.0024] | fail |
| succession, ECCKG (b) | both |  |  |  |  |  | **not useful** |
| succession, train statistics (b, substitute) | RE-GCN | +0.0030 [+0.0025, +0.0036] | < 5e-4 | +0.0029 [+0.0023, +0.0035] | no | +0.0033 [+0.0023, +0.0044] | **PASS** |
| succession, train statistics (b, substitute) | LogCL | +0.0024 [+0.0018, +0.0029] | < 5e-4 | +0.0024 [+0.0018, +0.0030] | no | +0.0035 [+0.0024, +0.0047] | **PASS** |
| succession, train statistics (b, substitute) | both |  |  |  |  |  | **useful** |

**Result (pre-registered decision).**
* **Co-participation edges (a): useful (PASS on both bases).** HRCOV_COOC − HRCOV = +0.0063 [+0.0056, +0.0069] on RE-GCN and +0.0051 [+0.0045, +0.0058] on LogCL (p < 5e-4 each), with no head damage. The dyadic-only stratum gains +0.0054 / +0.0069.
* **Succession edges from ECCKG (b): not useful.** The arm passes on RE-GCN (+0.0028 [+0.0023, +0.0033]) but fails the +0.002 bar on LogCL (+0.0017 [+0.0012, +0.0022], p < 5e-4). Dyadic-only: +0.0029 / +0.0013.
* **Succession edges from training statistics (labelled substitute): useful (PASS on both).** +0.0030 [+0.0025, +0.0036] / +0.0024 [+0.0018, +0.0029]; dyadic-only +0.0033 / +0.0035. The pre-stated expectation that this substitute would be absorbed by the pattern prior and the count reader `ev_dyad_lr` was **wrong**.

**What the gains consist of (diagnostic arms in the rule file; post-hoc arms marked).**
* **Co-participation acts as an edge feature, not as a candidate generator.** The COOC features alone, on HRCOV's own candidates, give +0.0055 / +0.0050. A random degree-matched block that carries the same features gives +0.0062 / +0.0046. The COOC block over that random block adds +0.0001 (n.s.) / +0.0005. The block does bring golds into the pool (gold inside the arm's columns 88.4% → 91.3% on RE-GCN), but those queries move only from MRR 0.008 to 0.025. Its contribution is confined to R_out (+0.0037 / +0.0050 over the feature-only arm; 4% of queries). The signal lies in "who co-appears with s in articles", which re-orders partners that are already in the pool. This is the same signal pilot 4 called COOC (7-day window); here it covers the full history and sits on top of HRCOV.
* **Succession: mostly features, and partly plain pair activity (post-hoc).** The same succession block with only the 365-day pair-event count gives +0.0021 / +0.0006, and about 0 on dyadic-only. The training-statistics succession weighting adds +0.0010 / +0.0018 on top, and +0.0030 / +0.0035 on dyadic-only. The ECCKG weighting adds +0.0007 / +0.0011 on top, and +0.0025 / +0.0013 on dyadic-only. The succession features without the block give +0.0026 / +0.0022; the block on top of them adds +0.0004 / +0.0002 overall (+0.0047 / +0.0040 on R_out). The relation-transition part therefore helps mainly where the design-implications note predicted, on dyadic-only golds. Its effect there is +0.003 MRR with training statistics, and about half of that, base-dependent, with ECCKG.
* **ECCKG versus training statistics.** SUCC_E − SUCC_TR = −0.0003 (n.s.) / −0.0007 (E worse). This is the first ECCKG signal in the project that is distinguishable from 0 over a matched no-prior control (+0.0007 / +0.0011, post-hoc). However, it does not reach the pre-registered bar on LogCL and never beats training statistics overall. On RE-GCN dyadic-only golds, the ECCKG features beat the training-statistics features (SUCCFEAT_TR − SUCCFEAT_E = −0.0017 [−0.0026, −0.0008]); on LogCL the opposite holds (+0.0015).
* **Combination.** All three edge types together give +0.0067 / +0.0058 over HRCOV (diagnostic). Succession on top of co-participation adds +0.0004 / +0.0007 overall and +0.0024 / +0.0004 on dyadic-only.
* **Noise floor.** Pilot 4 measured about ±0.001 MRR from re-training with uninformative extra columns. All rule-level deltas here exceed it, but the post-hoc sub-attributions of ≤ 0.001 are at that floor.


*Attribution (Δ MRR [95% CI]; R_out = exact recurrence but gold outside the base top-50, 4% of queries):*

| comparison | RE-GCN all | RE-GCN dyadic-only | RE-GCN R_out | LogCL all | LogCL dyadic-only | LogCL R_out |
|---|---|---|---|---|---|---|
| co-participation features, no block | +0.0055 [+0.0049, +0.0062] | +0.0048 [+0.0034, +0.0061] | +0.0033 [+0.0022, +0.0044] | +0.0050 [+0.0043, +0.0057] | +0.0052 [+0.0039, +0.0064] | +0.0051 [+0.0037, +0.0068] |
| co-participation block on top of its features | +0.0007 [+0.0004, +0.0010] | +0.0006 [-0.0001, +0.0014] | +0.0037 [+0.0027, +0.0047] | +0.0001 [-0.0002, +0.0005] | +0.0017 [+0.0009, +0.0026] | +0.0050 [+0.0040, +0.0061] |
| random degree-matched block + co-participation features | +0.0062 [+0.0055, +0.0068] | +0.0056 [+0.0042, +0.0069] | +0.0010 [-0.0002, +0.0021] | +0.0046 [+0.0040, +0.0053] | +0.0049 [+0.0036, +0.0062] | +0.0033 [+0.0020, +0.0046] |
| co-participation block vs random block (same features) | +0.0001 [-0.0002, +0.0004] | -0.0002 [-0.0010, +0.0004] | +0.0060 [+0.0050, +0.0069] | +0.0005 [+0.0002, +0.0008] | +0.0021 [+0.0013, +0.0028] | +0.0069 [+0.0059, +0.0081] |
| succession block + 365-d pair-event count only (post-hoc) | +0.0021 [+0.0016, +0.0026] | +0.0003 [-0.0005, +0.0012] | +0.0066 [+0.0054, +0.0078] | +0.0006 [+0.0002, +0.0011] | +0.0001 [-0.0008, +0.0008] | +0.0035 [+0.0025, +0.0045] |
| train-succession weighting on top (post-hoc) | +0.0010 [+0.0005, +0.0014] | +0.0030 [+0.0020, +0.0042] | -0.0004 [-0.0013, +0.0006] | +0.0018 [+0.0013, +0.0023] | +0.0035 [+0.0023, +0.0046] | +0.0020 [+0.0007, +0.0035] |
| ECCKG-succession weighting on top (post-hoc) | +0.0007 [+0.0003, +0.0012] | +0.0025 [+0.0017, +0.0035] | +0.0003 [-0.0007, +0.0014] | +0.0011 [+0.0006, +0.0016] | +0.0013 [+0.0002, +0.0024] | +0.0026 [+0.0014, +0.0040] |
| train-succession features, no block (post-hoc) | +0.0026 [+0.0021, +0.0031] | +0.0004 [-0.0005, +0.0015] | +0.0015 [+0.0007, +0.0024] | +0.0022 [+0.0017, +0.0027] | +0.0026 [+0.0015, +0.0038] | +0.0015 [+0.0004, +0.0025] |
| train-succession block on top of its features (post-hoc) | +0.0004 [+0.0001, +0.0008] | +0.0029 [+0.0021, +0.0036] | +0.0047 [+0.0039, +0.0055] | +0.0002 [-0.0002, +0.0005] | +0.0009 [+0.0002, +0.0016] | +0.0040 [+0.0031, +0.0051] |
| ECCKG vs train statistics | -0.0003 [-0.0006, +0.0001] | -0.0004 [-0.0013, +0.0005] | +0.0007 [-0.0002, +0.0015] | -0.0007 [-0.0011, -0.0003] | -0.0022 [-0.0030, -0.0013] | +0.0006 [-0.0003, +0.0016] |
| succession edges on top of co-participation | +0.0004 [+0.0001, +0.0008] | +0.0024 [+0.0015, +0.0033] | +0.0007 [-0.0001, +0.0015] | +0.0007 [+0.0003, +0.0011] | +0.0004 [-0.0007, +0.0014] | +0.0002 [-0.0007, +0.0010] |

*Per-stratum Δ vs HRCOV:*

| stratum | share | RE-GCN HRCOV MRR | Δ COOC | Δ SUCC_E | Δ SUCC_TR | LogCL HRCOV MRR | Δ COOC | Δ SUCC_E | Δ SUCC_TR |
|---|---|---|---|---|---|---|---|---|---|
| R_in | 0.716 | 0.6127 | +0.0068 [+0.0059, +0.0077] | +0.0028 [+0.0021, +0.0035] | +0.0030 [+0.0023, +0.0037] | 0.6122 | +0.0048 [+0.0039, +0.0058] | +0.0016 [+0.0009, +0.0022] | +0.0021 [+0.0014, +0.0028] |
| R_out | 0.044 | 0.0395 | +0.0070 [+0.0056, +0.0083] | +0.0069 [+0.0058, +0.0080] | +0.0062 [+0.0052, +0.0073] | 0.0321 | +0.0102 [+0.0084, +0.0122] | +0.0061 [+0.0048, +0.0074] | +0.0055 [+0.0040, +0.0068] |
| D_only | 0.171 | 0.2565 | +0.0054 [+0.0040, +0.0069] | +0.0029 [+0.0019, +0.0040] | +0.0033 [+0.0023, +0.0044] | 0.2540 | +0.0069 [+0.0054, +0.0083] | +0.0013 [+0.0003, +0.0024] | +0.0035 [+0.0024, +0.0047] |
| NOHIST | 0.069 | 0.0308 | +0.0020 [+0.0014, +0.0027] | -0.0003 [-0.0005, +0.0000] | +0.0009 [+0.0006, +0.0012] | 0.0317 | +0.0007 [+0.0000, +0.0013] | +0.0010 [+0.0006, +0.0014] | +0.0012 [+0.0008, +0.0016] |
| head | 0.828 | 0.5678 | +0.0064 [+0.0056, +0.0072] | +0.0025 [+0.0019, +0.0031] | +0.0029 [+0.0023, +0.0035] | 0.5697 | +0.0048 [+0.0039, +0.0056] | +0.0016 [+0.0010, +0.0021] | +0.0024 [+0.0018, +0.0030] |
| non_head | 0.172 | 0.0952 | +0.0055 [+0.0046, +0.0065] | +0.0041 [+0.0033, +0.0048] | +0.0039 [+0.0031, +0.0046] | 0.0933 | +0.0069 [+0.0060, +0.0079] | +0.0023 [+0.0016, +0.0030] | +0.0023 [+0.0016, +0.0031] |
| gold_in_cooc_block | 0.029 | 0.0079 | +0.0173 [+0.0161, +0.0187] | +0.0122 [+0.0111, +0.0135] | +0.0127 [+0.0116, +0.0139] | 0.0080 | +0.0157 [+0.0143, +0.0171] | +0.0097 [+0.0087, +0.0109] | +0.0108 [+0.0098, +0.0120] |
| gold_in_sutr_block | 0.023 | 0.0081 | +0.0160 [+0.0146, +0.0176] | +0.0158 [+0.0144, +0.0173] | +0.0173 [+0.0159, +0.0187] | 0.0081 | +0.0141 [+0.0127, +0.0159] | +0.0129 [+0.0115, +0.0145] | +0.0149 [+0.0135, +0.0162] |
| gold_in_sue_block | 0.022 | 0.0081 | +0.0163 [+0.0148, +0.0181] | +0.0168 [+0.0154, +0.0184] | +0.0169 [+0.0155, +0.0183] | 0.0081 | +0.0143 [+0.0127, +0.0161] | +0.0137 [+0.0124, +0.0154] | +0.0146 [+0.0132, +0.0161] |

## 2. Step 2: story reader without document links (Task 3), MidEast-TE

**Question.** Pilot 4 found its reading gain mostly carried by same-article co-participation, which needs the event→article links. Does the time-masked text reader still carry a gain when articles are not linked to events at all?

**Setup.** Article ids are not linked to events here. The corpus is treated as dated texts only. The context of (s, r, ?, t) is up to 16 of the most recent text-bearing articles dated [t−7, t−1] whose **text lexically mentions s** (pilot-4 mention dictionary; no event link is used). Co-participation is therefore not computable. The shuffled control uses the same number of s-mentioning articles dated ≤ t−8. The reader is the same 10 bge-m3 features as pilot-4 CTX (cached vectors) over the same HR top-50 pools. HR, HR_CTX (linked) and HR_COOC were re-trained in the same run. The re-trained HR reproduces pilot 4: HR_COOC − HR = +0.0052 / +0.0035, against +0.0050 / +0.0037 in pilot 4.

**Leakage audit (test, n = 89,896).** Windows with an article dated ≥ t: 0. Windows with an article of the gold event: 0 (both asserted). 82.3% of queries have a no-link context (linked: 86.9%), with a mean of 11.5 articles when present. Full text makes up 25.3% of these context articles; the rest are headlines. Overlap with the pilot-4 linked context is 55.1% of the no-link articles. However, s is an extracted event participant in 91.9% of the no-link articles: mention-based retrieval largely re-finds the articles the extractor used, because the corpus contains only articles that produced events. The gold is named in 48.1% of windows with context (shuffled: 41.3%).

**Rule** (`fusion_followups_step2_decision_rule.json`, the pilot-4 rule): on both bases, (a) HR_CTXNL − HR ≥ +0.002 overall with p < 0.05; (b) the same on R_in (exact recurrence, gold in the top-50); (c) HR_CTXNL > HR_SHUFNL with p < 0.05; (d) no head damage.


| arm | RE-GCN MRR | H@1 | H@3 | H@10 | LogCL MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|---|---|---|
| frozen base | 0.4482 | 0.3260 | 0.5129 | 0.6849 | 0.4569 | 0.3345 | 0.5238 | 0.6927 |
| HR (history re-ranker) | 0.4816 | 0.3580 | 0.5538 | 0.7157 | 0.4836 | 0.3606 | 0.5554 | 0.7164 |
| HR + reader, no document links | 0.4851 | 0.3619 | 0.5570 | 0.7182 | 0.4861 | 0.3631 | 0.5579 | 0.7184 |
| HR + reader, shuffled stale text (no links) | 0.4820 | 0.3589 | 0.5534 | 0.7151 | 0.4831 | 0.3599 | 0.5547 | 0.7163 |
| diag: HR + reader, linked context (pilot-4 retrieval) | 0.4859 | 0.3633 | 0.5576 | 0.7180 | 0.4866 | 0.3640 | 0.5580 | 0.7186 |
| diag: HR + co-participation (pilot 4) | 0.4868 | 0.3649 | 0.5581 | 0.7175 | 0.4871 | 0.3648 | 0.5587 | 0.7178 |
| diag: HR + no-link reader + co-participation | 0.4875 | 0.3650 | 0.5590 | 0.7183 | 0.4880 | 0.3657 | 0.5596 | 0.7189 |

| base | (a) Δ vs HR, all | (b) Δ vs HR, R_in | (c) Δ vs shuffled | (d) head Δ | verdict |
|---|---|---|---|---|---|
| RE-GCN | +0.0035 [+0.0028, +0.0042] | +0.0036 [+0.0026, +0.0045] | +0.0032 [+0.0024, +0.0038] | +0.0046 [+0.0038, +0.0054] | **PASS** |
| LogCL | +0.0025 [+0.0019, +0.0031] | +0.0029 [+0.0021, +0.0038] | +0.0030 [+0.0024, +0.0036] | +0.0035 [+0.0028, +0.0042] | **PASS** |

| comparison | RE-GCN all | RE-GCN R_in | RE-GCN dyadic-only | RE-GCN no history | LogCL all | LogCL R_in | LogCL dyadic-only | LogCL no history |
|---|---|---|---|---|---|---|---|---|
| linked-context reader vs HR | +0.0043 [+0.0037, +0.0050] | +0.0049 [+0.0040, +0.0058] | +0.0050 [+0.0037, +0.0062] | -0.0004 [-0.0009, +0.0001] | +0.0030 [+0.0024, +0.0036] | +0.0033 [+0.0025, +0.0041] | +0.0036 [+0.0023, +0.0048] | -0.0001 [-0.0005, +0.0004] |
| no-link vs linked reader | -0.0008 [-0.0014, -0.0003] | -0.0013 [-0.0020, -0.0006] | -0.0001 [-0.0013, +0.0012] | +0.0019 [+0.0013, +0.0027] | -0.0005 [-0.0010, +0.0000] | -0.0004 [-0.0010, +0.0002] | -0.0018 [-0.0029, -0.0006] | +0.0017 [+0.0010, +0.0025] |
| co-participation vs HR | +0.0052 [+0.0045, +0.0058] | +0.0061 [+0.0052, +0.0069] | +0.0047 [+0.0036, +0.0059] | +0.0005 [+0.0002, +0.0007] | +0.0035 [+0.0029, +0.0042] | +0.0039 [+0.0030, +0.0048] | +0.0043 [+0.0032, +0.0053] | -0.0007 [-0.0009, -0.0005] |
| no-link reader vs co-participation | -0.0017 [-0.0024, -0.0010] | -0.0025 [-0.0034, -0.0016] | +0.0002 [-0.0011, +0.0016] | +0.0010 [+0.0003, +0.0018] | -0.0010 [-0.0016, -0.0003] | -0.0010 [-0.0019, -0.0001] | -0.0025 [-0.0038, -0.0013] | +0.0023 [+0.0017, +0.0030] |
| no-link reader on top of co-participation | +0.0007 [+0.0001, +0.0012] | -0.0001 [-0.0008, +0.0006] | +0.0037 [+0.0025, +0.0049] | +0.0012 [+0.0004, +0.0021] | +0.0009 [+0.0004, +0.0014] | +0.0007 [+0.0000, +0.0014] | +0.0015 [+0.0004, +0.0027] | +0.0023 [+0.0016, +0.0031] |

**Result (pre-registered decision): the reader carries a gain without document links. PASS on both bases.**
* HR_CTXNL − HR = +0.0035 [+0.0028, +0.0042] (RE-GCN) / +0.0025 [+0.0019, +0.0031] (LogCL). On R_in: +0.0036 / +0.0029. Against the shuffled stale text: +0.0032 / +0.0030 (all p < 5e-4). The shuffled no-link context itself gives +0.0003 (n.s.) / −0.0005, and there is no head damage.
* **Relative to the linked reader and to co-participation (diagnostics).** The no-link reader keeps most of the linked reader's gain: HR_CTXNL − HR_CTX = −0.0008 [−0.0014, −0.0003] / −0.0005 [−0.0010, +0.0000], about 80% of +0.0043 / +0.0030. It stays below co-participation (HR_CTXNL − HR_COOC = −0.0017 / −0.0010). On top of co-participation it adds +0.0007 / +0.0009 overall, +0.0037 / +0.0015 on dyadic-only and +0.0012 / +0.0023 on no-history queries.
* **Where the no-link reader differs.** On no-history queries the mention-retrieved context beats the linked context: HR_CTXNL − HR_CTX = +0.0019 / +0.0017, CI excludes 0. Text that mentions s but did not produce an s-event carries information that the event links never point to. Queries without a no-link context (17.7%) are unchanged (+0.0003 / −0.0000).
* **Caveat that bounds the claim.** Removing the article ids removes the explicit links, not the corpus selection. Every article in this corpus produced at least one extracted event, and s is an event participant in 91.9% of the retrieved articles. The test therefore shows that text alone (lexical retrieval + dense reading) recovers about 80% of the linked-reader gain. It does not show the same on an independent news stream whose articles were never used for extraction. That would need a corpus such as the GDELT-2025 slice with its own crawl, or ACLED notes.


## 3. Step 3: dated Wikidata static edges for tkgl-smallpedia (Task 2 leakage control)

**Question.** Pilot 2 found that the static-KG fusion gain survives the removal of direct s–o static edges (HRCOV_ST_NODIRECT). That left one leakage path open: the 2-hop paths run through a time-less 2024 Wikidata snapshot. Does the 2-hop gain survive when a query at year t may only use static facts that were valid before t?

**Feasibility (measured before the rule; local sandbox → query.wikidata.org, sequential, ≥ 1 s pause).**
* **Static edges.** 978,022 edges over 309,794 nodes, 753 properties. Row order equals the TGB CSV, and a full id→QID map was rebuilt and checked against pilot 2's stored QIDs.
* **Statement qualifiers.** These are the route the task names first, and they are essentially absent. In a uniform sample of 3,000 static edges, 2,845 statements still exist in current Wikidata, and only 1.05% carry P580 / P582 / P585 (0.81% have P580). Only 1 of the 3,000 has a start year ≥ 1998. This is expected from how TGB 2.0 builds smallpedia: time-qualified statements became the temporal edges and the unqualified ones the static KG. Dating by statement qualifiers would change about 0.03% of edges for the valid/test years, so the full statement-qualifier fetch was not run (declared).
* **Dated dumps.** Not attempted: historical Wikidata JSON dumps are ~100 GB each, and none ships with TGB.
* **Entity-lifetime lower bound (the dating channel used).** valid_from(h, p, x) = the later of the earliest start-type date (P569 birth, P571 inception, P580 start, P1619 opening, P729 service entry) of h and of x. We fetched these dates for all 309,794 nodes (775 SPARQL batches of 400; about 100 min including 429/502 back-offs). 86.8% of temporal entities and 52.4% of static-only nodes have such a date, so **91.4% of edges get a lower bound** (both endpoints dated: 43.8%). The bound falls in the valid/test period (≥ 1998) for 4.70% of edges and in the test period (≥ 2008) for 1.82%. An undated edge is kept (assumed old).
* **Effect on the leakage cases identified in pilot 2.** Of the 6,762 direct-edge test queries, only 3.1% lose every direct edge under dating. In the outcome-recording pairs, municipal mergers (`P131` / `P3842` vs `P1366` replaced-by / `P1365` replaces) are partly caught because the merged entity has an inception. Draft-by-team pairs (`P54` vs `P647`) are never caught: the player and the team both predate the draft.
* **Gold reach of the static candidate set (test).** 21.3% → 20.3% with direct edges, 19.4% → 18.5% without.

**What was re-run (host serverA, CPU).** HRCOV_ST (mode `dated`) and HRCOV_ST_NODIRECT (mode `dated_nd`) were re-run with every static hop (S, SS, ST, TS) restricted to edges with valid_from < t. The static degree features were computed at t. The static path-type precision table was refitted under the same dating on its 100,000 early-train queries. The pilot-2 bridge rankers, pattern tables and C_cov were reused (C_cov asserted identical on all valid/test rows), and the 3 seeds and valid-only selection were unchanged. The comparators are the pilot-2 HRCOV, HRCOV_ST and HRCOV_ST_NODIRECT ranks on the same 163,172 test queries; the base ranks are asserted identical across all arms. GPU 0 was never idle at a training boundary and GPU 1 was never used, so all re-ranker training ran on CPU (8 threads).

**Rule** (`fusion_followups_step3_decision_rule.json`). The 2-hop gain survives dating iff, on ≥ 2 of 3 bases, (a) HRCOV_ST_NODIRECT_DATED − HRCOV ≥ +0.001 with p < 0.05, (b) there is no head damage, and (c) the no-pair-history Δ is > 0 with p < 0.05.


| arm | RE-GCN MRR | H@1 | H@10 | LogCL MRR | H@1 | H@10 | TiRGN MRR | H@1 | H@10 |
|---|---|---|---|---|---|---|---|---|---|
| frozen base | 0.4218 | 0.3625 | 0.5345 | 0.4664 | 0.4059 | 0.5786 | 0.6181 | 0.5523 | 0.7239 |
| HRCOV (pilot 2) | 0.6595 | 0.6081 | 0.7367 | 0.6590 | 0.6072 | 0.7371 | 0.6601 | 0.6089 | 0.7353 |
| HRCOV_ST, undated (pilot 2) | 0.6672 | 0.6135 | 0.7493 | 0.6668 | 0.6130 | 0.7491 | 0.6682 | 0.6146 | 0.7480 |
| HRCOV_ST, dated static edges | 0.6649 | 0.6105 | 0.7473 | 0.6646 | 0.6102 | 0.7480 | 0.6663 | 0.6128 | 0.7464 |
| HRCOV_ST_NODIRECT, undated (pilot 2) | 0.6625 | 0.6081 | 0.7460 | 0.6625 | 0.6079 | 0.7461 | 0.6653 | 0.6120 | 0.7448 |
| HRCOV_ST_NODIRECT, dated static edges | 0.6628 | 0.6091 | 0.7448 | 0.6621 | 0.6080 | 0.7452 | 0.6620 | 0.6075 | 0.7432 |

| base | (a) Δ NODIRECT_DATED vs HRCOV | p | (b) head Δ | (c) no-pair Δ | dated − undated (NODIRECT) | survival fraction | verdict |
|---|---|---|---|---|---|---|---|
| RE-GCN | +0.0033 [+0.0028, +0.0038] | < 5e-4 | -0.0007 [-0.0024, +0.0011] | +0.0143 [+0.0132, +0.0153] | +0.0003 [-0.0001, +0.0006] | 1.09 | **PASS** |
| LogCL | +0.0031 [+0.0026, +0.0036] | < 5e-4 | +0.0005 [-0.0012, +0.0022] | +0.0143 [+0.0133, +0.0154] | -0.0004 [-0.0007, -0.0001] | 0.89 | **PASS** |
| TiRGN | +0.0019 [+0.0014, +0.0024] | < 5e-4 | -0.0014 [-0.0032, +0.0006] | +0.0149 [+0.0139, +0.0160] | -0.0032 [-0.0036, -0.0029] | 0.37 | **PASS** |

| base | Δ ST_DATED vs HRCOV | dated − undated (ST) | survival fraction (ST) | dated − undated on direct-edge queries whose edges are dated out | dated − undated on outcome-recording pairs |
|---|---|---|---|---|---|
| RE-GCN | +0.0054 [+0.0048, +0.0059] | -0.0023 [-0.0027, -0.0020] | 0.70 | -0.6655 [-0.7702, -0.5574] | -0.1492 [-0.2080, -0.0942] |
| LogCL | +0.0056 [+0.0052, +0.0061] | -0.0022 [-0.0025, -0.0018] | 0.72 | -0.6596 [-0.7706, -0.5516] | -0.1548 [-0.2133, -0.0956] |
| TiRGN | +0.0062 [+0.0057, +0.0067] | -0.0018 [-0.0022, -0.0015] | 0.77 | -0.6520 [-0.7619, -0.5443] | -0.1644 [-0.2210, -0.1077] |

**Result (pre-registered decision): the 2-hop static gain survives dating on 3 of 3 bases.** HRCOV_ST_NODIRECT_DATED − HRCOV is +0.0033 [+0.0028, +0.0038] on RE-GCN, +0.0031 [+0.0026, +0.0036] on LogCL and +0.0019 [+0.0014, +0.0024] on TiRGN (each p < 5e-4). No head damage occurs, and the no-pair-history stratum gains +0.0143 / +0.0143 / +0.0149.

* **How much of the 2-hop gain survives.** The survival fraction is Δ(dated) / Δ(undated) against HRCOV: 1.09 on RE-GCN, 0.89 on LogCL and 0.37 on TiRGN. Dated − undated within NODIRECT is +0.0003 (n.s.) / −0.0004 / −0.0032. The TiRGN figure is partly a training artefact: seed 0 of the dated TiRGN run early-stopped at epoch 4 (test MRR 0.6588, against 0.6636 / 0.6637 for seeds 1–2). Even on seeds 1–2 the dated arm sits about 0.002 below the undated one (0.6644–0.6661). This is a post-hoc observation and is not used in the decision. Pilot 2 had already noted that the no-direct arms vary more across seeds than HRCOV_ST.
* **Full static fusion (with direct edges).** Dating removes 23–30% of the gain: HRCOV_ST − HRCOV falls from +0.0077 / +0.0078 / +0.0080 to +0.0054 / +0.0056 / +0.0062, and dated − undated is −0.0023 / −0.0022 / −0.0018. Most of this loss is the leakage pilot 2 identified. On the 0.13% of test queries whose every direct s–o edge is dated after t, the undated arm's lift over HRCOV (≈ +0.66 MRR) disappears completely (dated − undated −0.67 / −0.66 / −0.65). On the 326 outcome-recording queries (merger, present-day territory and draft pairs), the lift over HRCOV shrinks from +0.64 / +0.62 / +0.71 to +0.49 / +0.47 / +0.55. The draft pairs stay, because the entity lifetimes do not date them.
* **What the dating does not reach.** 8.6% of edges are undated, and entity lifetimes cannot date a fact whose two endpoints both existed long before it became true. Examples are a player who is a member of the team that later drafted him, or a person whose residence is a present-day city. The surviving +0.002–0.003 is therefore an upper bound on the forecasting-attributable 2-hop gain, tighter than pilot 2's +0.0030–0.0051 but still not leak-proof. Only statement-level history (Wikidata revision timestamps) would close it, and that was not attempted.
* **First-seen entities.** These lose the most under dating (dated − undated −0.005 to −0.012). Static facts about brand-new entities are disproportionately "future" facts.


!Dated static edges

*Figure 2.* tkgl-smallpedia, full test (n = 163,172), 3 seeds, 95% paired-bootstrap CIs; blue = RE-GCN, orange = LogCL, green = TiRGN. (a) Gain over HRCOV of static fusion with and without direct s–o edges, undated (pilot 2, open diamonds) and dated (this step, filled). Dotted line = the pre-registered +0.001. (b, c) Dated − undated MRR by stratum, for full static fusion (b) and the 2-hop-only arm (c). The x-axis is symmetric-log (linear within ±0.01). Strata: direct s–o static edge (4.1% of queries), outcome-recording pairs (0.2%), direct edge dated out (0.13%).

## 4. Mapping to the  tasks

* **Task 2 (fusion).** This project's two most useful fusion signals so far are both *entity-level* sources that the TKG lacks: static Wikidata attributes (pilot 2, now dating-checked) and document co-participation (here). Neither is an eventuality-commonsense source. ECCKG relation succession is the first ECCKG signal distinguishable from zero against a matched activity control (+0.0007 / +0.0011 overall, +0.0025 / +0.0013 on dyadic-only), but it misses the bar on one base and does not beat training-data succession statistics. The defensible Task-2 statement is: "eventuality commonsense adds a small, base-dependent relation-shift signal on dyadic-only golds; entity-level co-participation and static attributes carry most of the fusion gain, as edge features rather than as candidate generators."
* **Task 3 (reader).** The reader is not merely a proxy for the document links. With the links removed, lexical retrieval plus dense reading still passes the pilot-4 rule and keeps about 80% of the linked reader's gain. This supports framing the LLM / encoder as a *reader of retrieved text* in a TKG without document provenance. The remaining confound is corpus selection: all articles are extraction sources. An independent news stream is the next test.
* **Evaluation hygiene.** Static-KG fusion on TGB smallpedia should be reported with dated static edges. The undated +0.008 includes about 0.002 of dated-out future facts, and even the dated 2-hop +0.002–0.003 is an upper bound.

## 5. Deviations and caveats

1. HRCOV on MidEast-TE is the pattern-prior variant (the pilots' HRCOV_PAT); the stage-2 bridge block was omitted.
2. The MidEast-TE head guardrail (gold train frequency ≥ 500) covers 82.8% of test queries and is close to an overall check.
3. All step-1/2 main re-rankers ran on CPU (the 4090 was busy); CPU and GPU runs of the same recipe are not bitwise identical.
4. The step-1 post-hoc arms (succession count-only, succession features without block, ECCKG features without block) were added after the step-1 test numbers had been read. They are attribution only and not part of any decision.
5. The ECCKG prior anchors CAMEO relation *names* to ECCKG eventualities by embedding similarity (top-10, mean top-1 cosine 0.757). Its hub eventualities make "Forgive" / "Use unconventional violence" frequent successors (`ecckg_succession_summary.json`). The anchoring was fixed before any test number; no alternative anchoring was tried.
6. Step 2 removes article ids but not the corpus selection: every article produced extracted events.
7. Step 3: the statement-qualifier fetch was not run beyond the 3,000-edge sample (estimated 0.03% of edges affected). The dating channel is an entity-lifetime lower bound, and the 8.6% of edges with no bound are kept. The hub cap uses the time-less static degree; the degree features use the dated degree. The official py-tgb evaluator was not re-run for the dated arms (pilot 2 showed harness ranks equal it to 1e-6).
8. Step 3 training: TiRGN dated-NODIRECT seed 0 stopped at epoch 4, and the 3-seed survival fraction 0.37 is sensitive to it. The rule outcome (3/3) does not depend on it.
9. Re-training with uninformative extra columns moves MRR by about ±0.001 (pilot 4). Sub-attributions of that size are not interpretable.

## 6. Reproduction

| step | host | command (track dir `tracks/fusion_followups/`) | time |
|---|---|---|---|
| prep (pattern TE tables, train succession) | serverB CPU | `ff_s1_prep.py --workers 8` | 87 s |
| ECCKG succession prior | serverB CPU | `ff_ecckg.py --ecckg ecckg_full.csv --device cpu` | 138 s |
| W = 300 fusion layout + 45 features, per base | serverB CPU | `ff_s1_build.py --base <b> --workers 8`; identity check `ff_s1_check.py` (base-rank identity 1.000; HR features equal pilot 4) | ~100 s / base |
| no-link contexts + reader features | serverB CPU | `ff_s2_ctx.py --device cpu --threads 8` | 73 s |
| re-rankers | serverB CPU (GPU if idle) | `ff_train.py --base <b> --layout s1\|s2 [--arms …] [--tag _posthoc]` | s1 ~25 min / base, s2 ~5 min / base (CPU) |
| Wikidata entity dates | local | `ff_fetch_dates.py ff_idq.npy ent_dates.jsonl 400` → `ff_build_lb.py` | 100 min |
| dated static re-run | serverA CPU | `sp_chain_dated.sh 8 regcn,logcl,tirgn dated,dated_nd` (modified `sp_common/sp_static/sp_lists`, `sp_ste_dated.py`) | 7.5 h (host load 17–54) |
| aggregation / figures / tables | local | `ff_agg_s12.py`, `ff_agg_s3.py`, `make_tables.py`, `fig2.py` | < 1 min |

Runs: serverB (prep + smoke, steps 1–2, post-hoc); serverA (step 3). Data files (not released): static_lb.npz (per-edge validity lower bound), wikidata_entity_dates.jsonl.gz (raw date rows), static_lb_summary.json.
