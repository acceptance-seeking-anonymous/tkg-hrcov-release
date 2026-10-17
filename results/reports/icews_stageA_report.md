# Pilot 3 of 4 — history re-ranker on TGB 2.0 `tkgl-icews`: STAGE A (pilot of the pilot)

**Question.** Does the history re-ranker (HR) or its history-aware pool widening (HRCOV, pilot 1) beat both a frozen base and the
full-history recurrency rule RecB-lite on `tkgl-icews`, a dataset where RecB-lite (0.303) already exceeds the best published model (0.269)?
How much of the weakness is the relation-code artefact (117 post-2020 relation types that are zero-stripped copies of older CAMEO codes)?

**Stage A design (investigator redesign).** Bases trained on a contiguous recent window (last 20% of train days, 2,266,194 facts); every
feature, candidate list and RecB-lite reads the FULL history before t; test = a uniform 10% sample of test queries (465,138 queries, both
directions, rng 31415), identical for every arm; re-ranker training / selection on a uniform 200,000-query valid sample.

**Answer (pre-registered GO / NO-GO, `stageA_decision_rule.json`, registered
2026-10-12T14:54:56Z before any base, list or model existed): GO.** On both bases the valid-chosen arm is HRCOV with B = 150
pattern-prior candidates. HRCOV − RecB-lite: RE-GCN +0.0717 [+0.0709, +0.0724], LogCL +0.0711 [+0.0703, +0.0718]; HRCOV − window-trained
base: RE-GCN +0.1103 [+0.1094, +0.1112], LogCL +0.0758 [+0.0751, +0.0766] (one-sided p < 5e-4 for all four; threshold ΔMRR ≥ 0.005). The head
guardrail passes (head-stratum deltas positive against both comparators). py-tgb's official Evaluator reproduces every harness MRR within
4e-7.

**How to read these numbers.** Stage A numbers are **not leaderboard-comparable**: the bases are window-trained and the test is a 10%
sample. **Base-relative gains are inflated by window training** (the bases saw 20% of train facts; the re-ranker features see all of
history). **The RecB-lite comparison is the meaningful one**: both RecB-lite and the re-ranker read the full history, and the margin is
+0.071 to +0.072 MRR on both bases.

**Relation-code artefact (secondary, pre-registered).** Recomputing HRCOV's features on relation-normalised ids (base unchanged) adds
+0.0095 / +0.0093 MRR overall. The gain sits entirely on the 43.8% of test facts whose relation is unseen in train (+0.0215 / +0.0211);
facts with a seen relation change by ≤ 0.0002. After normalisation, unseen-relation facts are still 0.085 (RE-GCN) / 0.088 (LogCL) MRR below seen-relation facts for
HRCOV. Most of the gap is therefore not the code artefact. A post-hoc control (not pre-registered) runs RecB-lite on the normalised data:
0.3146, +0.0129 over RecB-lite. HRCOV RELNORM still beats it by +0.0682 [+0.0675, +0.0690] (RE-GCN) and
+0.0675 [+0.0667, +0.0682] (LogCL). The bases themselves were not retrained on normalised relations in Stage A, so the
artefact's cost to the base models is not measured here.

Verdict: `icews_stageA_verdict.json`; mapping: `relnorm_mapping.csv`;
pipeline: `pilot_pipeline_README_v3.md` (§8).

---

## 1. Setup

**Data and protocol.** TGB 2.0 `tkgl-icews`: 87,856 entities, 391 relations (+391 inverse), 10,224 daily timestamps; train / valid / test
10,861,600 / 2,326,157 / 2,325,689 facts (TGB 70/15/15 by timestamp quantile; the test period runs from 2018-05 to 2022-12). Queries in both directions (valid 4,652,314,
test 4,651,378). Time-aware filtered, 1-vs-all over all entities, ground-truth online history before t.

**TGB identity checks** (`pi_verify.py`, job, py-tgb 2.3.0). The entity map is the identity, harness relation ids equal the
TGB edge types, the inverse type is r + 391 (TGB `num_rels` = 782), and TGB rows are in CSV order. TGB conflict sets equal the harness time-aware filter
for all 3,587,665 valid and 3,599,019 test (ts, s, r') keys. The harness exact filtered rank is therefore the official statistic.

**Relation normalisation** (`pi_relnorm.py`). 117 relation types first appear on or after 2020-01-01. Each is
a CAMEO code with its leading zero stripped (e.g. `10` vs `010`, `40` vs `040`), and each has a unique zero-padded original. All 117 merges are
accepted, on four kinds of evidence:
* No merged pair shares a usage day.
* No original code is used on or after 2020-01-01 (last uses 2018-10-04 to 2019-12-31; new codes first used 2020-01-01 to 2021-03-01).
* Among the 266 codes active in 2018–2019, the original has the most similar (subject, object) usage profile for 71 of 117 codes,
  covering 98.1% of the merged test facts (99.1% are within the 3 most similar).
* The median volume ratio, facts per month in 2020–21 over 2018–19, is 0.88.

After the merge, 43.82% of test facts move from a relation never seen in train to a seen one (0% unseen afterwards). Valid contains no new code. The
normalised data directory is `$P/stage/tkgl-icews-relnorm/` (`pi_relnorm_data.py`). This is a
string-matching and usage-profile inference; it has not been confirmed against the ICEWS source. (The rule file's "117 zero-stripped codes" was
taken from the scouting note at registration; the merge script, run afterwards, confirmed exactly 117 accepted merges.)

**Bases** (`pi_base.py`). Harness native-torch RE-GCN and LogCL (h 200, 2 layers, history 10, seed 42).
* Gradient steps are taken only on train days 5,749–7,186 (1,438 days, 2,266,194 facts).
* RE-GCN's 10-snapshot history reaches back before the window.
* LogCL's global copy-history is pre-rolled over all earlier train facts.
* The epoch grid {1,2,3,4,6,8,10,12} (extension 14–20) is selected on FULL-valid filtered MRR.

| base | full-valid MRR by epoch | selected |
|---|---|---|
| RE-GCN | 1 ep 0.2718, 2 ep 0.2893, 3 ep 0.2966, 4 ep 0.2998, 6 ep 0.3031, 8 ep 0.3049, 10 ep 0.3049, 12 ep 0.3053, 14 ep 0.3043 | epoch 12 (interior after the pre-registered extension to 14) |
| LogCL | 1 ep 0.3014, 2 ep 0.3139, 3 ep 0.3190, 4 ep 0.3208, 6 ep 0.3220, 8 ep 0.3222, 10 ep 0.3221, 12 ep 0.3214 | epoch 8 (interior) |

Both window bases exceed RecB-lite on valid (RE-GCN 0.3053, LogCL 0.3222 vs RecB-lite 0.3032, full valid). On the test sample they fall
below it (§2). Most of the drop sits on facts with post-2020 relation codes, whose relation embeddings the bases never trained.

**Systems.** All on frozen bases; every choice on valid. Valid-sample days below the 70% quantile are used for fitting (fit rows capped at 120,000 queries), the rest for selection.
Test is scored once.

| system | candidates | features | notes |
|---|---|---|---|
| `HR` | base top-50 | 8 base + 11 online evidence (residual-track Hist / DyadReader) | MLP 64-64, listwise CE, s = α log p + β f calibrated on valid-select, 3 seeds |
| `HRCOV_B` | top-50 ++ first B pattern-prior candidates outside the top-50, B ∈ {50, 100, 150} | + 6 pattern coverage features | pilot-1 pool widening (6.1 non-learned prior, base mass from the dump top-200); stage-2 bridge block not used |
| `HRCOV150_RN` | as the chosen arm, candidates from the normalised pattern lists | features on relation-normalised ids | same recipe; base and evaluation ids unchanged |
| RecB-lite | full row | — | full history, λ = 0.99 selected on full valid; harness profiler ranks; also scored by the official Evaluator |
| RecB official | full row | — | TGB repo predictor (commit 3b83e5cf), default mode λ 0.1, α 0.99, window 100, same sample |

Valid selection (3-seed-mean valid-select MRR):

| base | HR | HRCOV50 | HRCOV100 | HRCOV150 | B* | A* |
|---|---|---|---|---|---|---|
| RE-GCN | 0.3542 | 0.3743 | 0.3744 | 0.3748 | 150 | HRCOV150 |
| LogCL | 0.3566 | 0.3738 | 0.3744 | 0.3749 | 150 | HRCOV150 |

On both bases, B* = 150 is the top of the pre-registered grid, and the margins over B = 50 / 100 are ≤ 0.0011.

## 2. Test-sample results (Table 1; n = 465,138; harness rows are 3-seed means for re-rankers)

| base | system | evaluator | seeds | MRR | H@1 | H@3 | H@10 | valid-select MRR | host |
|---|---|---|---|---|---|---|---|---|---|
| - | RecB-lite (full history, lambda=0.99, valid-selected) | harness | - | 0.3017 | 0.2173 | 0.3429 | 0.4664 | — | serverA CPU |
| - | RecB-lite (full history) | official py-tgb | - | 0.3017 | — | — | 0.4664 | — | serverA CPU |
| - | RecB official TGB predictor, default mode (lambda 0.1, alpha 0.99), window 100 | official py-tgb | - | 0.2034 | — | — | 0.3203 | — | serverA CPU |
| RE-GCN | regcn (window-trained frozen base, epoch 12) | harness | - | 0.2630 | 0.1698 | 0.2962 | 0.4494 | — | serverA GPU |
| RE-GCN | regcn (window-trained frozen base) | official py-tgb | - | 0.2630 | — | — | 0.4494 | — | serverA GPU+CPU |
| RE-GCN | HR | harness | 3 | 0.3501 | 0.2551 | 0.3973 | 0.5359 | 0.3542 | serverA GPU |
| RE-GCN | HRCOV50 | harness | 3 | 0.3729 | 0.2670 | 0.4236 | 0.5828 | 0.3743 | serverA GPU |
| RE-GCN | HRCOV100 | harness | 3 | 0.3726 | 0.2660 | 0.4234 | 0.5837 | 0.3744 | serverA GPU |
| RE-GCN | HRCOV150 — **A* (valid-chosen)** | harness | 3 | 0.3733 | 0.2664 | 0.4239 | 0.5847 | 0.3748 | serverA GPU |
| RE-GCN | HRCOV150_RN — **RELNORM variant of A*** | harness | 3 | 0.3828 | 0.2747 | 0.4339 | 0.5959 | 0.3748 | serverA GPU |
| RE-GCN | HR_s0 | official py-tgb (seed 0) | - | 0.3502 | — | — | 0.5360 | — | serverA GPU+CPU |
| RE-GCN | HRCOV150_s0 | official py-tgb (seed 0) | - | 0.3739 | — | — | 0.5846 | — | serverA GPU+CPU |
| RE-GCN | HRCOV150_RN_s0 | official py-tgb (seed 0) | - | 0.3829 | — | — | 0.5957 | — | serverA GPU+CPU |
| LogCL | logcl (window-trained frozen base, epoch 8) | harness | - | 0.2969 | 0.2018 | 0.3358 | 0.4843 | — | serverA GPU |
| LogCL | logcl (window-trained frozen base) | official py-tgb | - | 0.2969 | — | — | 0.4843 | — | serverA GPU+CPU |
| LogCL | HR | harness | 3 | 0.3557 | 0.2583 | 0.4038 | 0.5460 | 0.3566 | serverA GPU |
| LogCL | HRCOV50 | harness | 3 | 0.3721 | 0.2655 | 0.4228 | 0.5837 | 0.3738 | serverA GPU |
| LogCL | HRCOV100 | harness | 3 | 0.3725 | 0.2655 | 0.4228 | 0.5844 | 0.3744 | serverA GPU |
| LogCL | HRCOV150 — **A* (valid-chosen)** | harness | 3 | 0.3727 | 0.2655 | 0.4228 | 0.5846 | 0.3749 | serverA GPU |
| LogCL | HRCOV150_RN — **RELNORM variant of A*** | harness | 3 | 0.3821 | 0.2740 | 0.4330 | 0.5947 | 0.3749 | serverA GPU |
| LogCL | HR_s0 | official py-tgb (seed 0) | - | 0.3552 | — | — | 0.5451 | — | serverA GPU+CPU |
| LogCL | HRCOV150_s0 | official py-tgb (seed 0) | - | 0.3733 | — | — | 0.5844 | — | serverA GPU+CPU |
| LogCL | HRCOV150_RN_s0 | official py-tgb (seed 0) | - | 0.3819 | — | — | 0.5945 | — | serverA GPU+CPU |
| - | RecB-lite RELNORM (full history, relation-normalised data; POST-HOC) | harness | - | 0.3146 | 0.2229 | 0.3552 | 0.4949 | — | serverA CPU |

Published TGB 2.0 rows for this dataset are RE-GCN 0.182, CEN 0.187, TGB RecB 0.211 (train mode) and best published (EdgeReMIND) 0.269.
Those rows were computed on the full test with full-train models and history length 3. They are listed for orientation only and are **not
comparable** with the Stage A rows above.

!Test-sample MRR and pre-registered contrasts

*Figure 1.* (a, b) Test-sample MRR per system for each window-trained base. RecB-lite and the official TGB RecB are base-independent and
repeated in both panels. (c) ΔMRR of the valid-chosen arm HRCOV (B = 150) against RecB-lite and against the window-trained base, with
95% paired-bootstrap CIs (B = 2,000; the CIs are narrower than the markers). The dashed line is the pre-registered GO threshold, 0.005.
n = 465,138 test queries; re-rankers are 3-seed means.

## 3. Decision (Table 2)

| base | A* (B*) | (a) ΔMRR A* − base [95% CI] | p | (b) ΔMRR A* − RecB-lite [95% CI] | p | (c) head Δ vs base / vs RecB-lite | DAMAGE | official = harness (max abs diff) | PASS |
|---|---|---|---|---|---|---|---|---|---|
| RE-GCN | HRCOV150 (150) | +0.1103 [+0.1094, +0.1112] | 0.0000 | +0.0717 [+0.0709, +0.0724] | 0.0000 | +0.1015 [+0.1005, +0.1025] / +0.0804 [+0.0795, +0.0813] | no | yes (3.6e-07) | **yes** |
| LogCL | HRCOV150 (150) | +0.0758 [+0.0751, +0.0766] | 0.0000 | +0.0711 [+0.0703, +0.0718] | 0.0000 | +0.0621 [+0.0613, +0.0629] / +0.0808 [+0.0800, +0.0818] | no | yes (3.6e-07) | **yes** |

`GO = true` (2 of 2 bases pass; both required). For seed 0, the official Evaluator matches the harness MRR within 3.6e-7 for the bases,
RecB-lite and all three systems. Per-query ranks are identical for 99.93% / 99.91% of base queries; the remaining queries differ only through
ties. A 20,000-query subsample scored through `query_batch` agrees to the float32 precision of 1/MRR.

## 4. Where the gain lands (Table 3; Δ with 95% CI, B = 1,000)

| stratum | share | base | RecB-lite | HR | HRCOV150 | HRCOV150 RELNORM | Δ HRCOV − RecB-lite [95% CI] | Δ HRCOV − base | Δ RELNORM − HRCOV |
|---|---|---|---|---|---|---|---|---|---|
| **RE-GCN** | | | | | | | | | |
| all | 1.000 | 0.2630 | 0.3017 | 0.3501 | 0.3733 | 0.3828 | +0.0717 [+0.0709, +0.0724] | +0.1103 | +0.0095 |
| exact recurrence (s,r',o) before t | 0.629 | 0.3606 | 0.4759 | 0.4940 | 0.5240 | 0.5283 | +0.0481 [+0.0472, +0.0491] | +0.1635 | +0.0042 |
| non-recurrent | 0.371 | 0.0977 | 0.0062 | 0.1062 | 0.1178 | 0.1361 | +0.1116 [+0.1104, +0.1128] | +0.0201 | +0.0183 |
| pair history (any (s,o) event before t) | 0.892 | 0.2926 | 0.3376 | 0.3905 | 0.4153 | 0.4257 | +0.0777 [+0.0768, +0.0785] | +0.1227 | +0.0105 |
| no pair history | 0.108 | 0.0197 | 0.0059 | 0.0184 | 0.0281 | 0.0292 | +0.0223 [+0.0214, +0.0231] | +0.0085 | +0.0011 |
| pool-absent (gold outside base top-50) | 0.350 | 0.0053 | 0.0892 | 0.0053 | 0.0944 | 0.1002 | +0.0052 [+0.0043, +0.0059] | +0.0891 | +0.0059 |
| gold in base top-50 | 0.650 | 0.4016 | 0.4159 | 0.5355 | 0.5233 | 0.5347 | +0.1074 [+0.1063, +0.1084] | +0.1217 | +0.0114 |
| relation unseen in train (post-2020 zero-stripped code) | 0.438 | 0.1765 | 0.2380 | 0.2905 | 0.3138 | 0.3353 | +0.0759 [+0.0747, +0.0769] | +0.1373 | +0.0215 |
| relation seen in train | 0.562 | 0.3306 | 0.3514 | 0.3967 | 0.4198 | 0.4199 | +0.0684 [+0.0674, +0.0694] | +0.0892 | +0.0001 |
| cold gold (train freq < 50) | 0.110 | 0.0194 | 0.1616 | 0.0616 | 0.2100 | 0.2084 | +0.0484 [+0.0464, +0.0503] | +0.1906 | -0.0016 |
| mid gold | 0.113 | 0.0785 | 0.1371 | 0.1421 | 0.1716 | 0.1732 | +0.0344 [+0.0327, +0.0360] | +0.0930 | +0.0016 |
| head gold (train freq >= 500) | 0.777 | 0.3242 | 0.3453 | 0.4211 | 0.4257 | 0.4379 | +0.0804 [+0.0795, +0.0813] | +0.1015 | +0.0122 |
| first-seen entity | 0.002 | 0.0133 | 0.0176 | 0.0182 | 0.0168 | 0.0166 | -0.0008 [-0.0076, +0.0053] | +0.0035 | -0.0003 |
| hist=no (no train+valid co-occurrence) | 0.188 | 0.0552 | 0.1232 | 0.1098 | 0.1762 | 0.1817 | +0.0530 [+0.0516, +0.0543] | +0.1210 | +0.0056 |
| **LogCL** | | | | | | | | | |
| all | 1.000 | 0.2969 | 0.3017 | 0.3557 | 0.3727 | 0.3821 | +0.0711 [+0.0703, +0.0718] | +0.0758 | +0.0093 |
| exact recurrence (s,r',o) before t | 0.629 | 0.4202 | 0.4759 | 0.5045 | 0.5233 | 0.5284 | +0.0474 [+0.0464, +0.0482] | +0.1031 | +0.0051 |
| non-recurrent | 0.371 | 0.0879 | 0.0062 | 0.1034 | 0.1174 | 0.1340 | +0.1112 [+0.1100, +0.1124] | +0.0295 | +0.0166 |
| pair history (any (s,o) event before t) | 0.892 | 0.3306 | 0.3376 | 0.3966 | 0.4146 | 0.4250 | +0.0770 [+0.0762, +0.0778] | +0.0840 | +0.0104 |
| no pair history | 0.108 | 0.0199 | 0.0059 | 0.0188 | 0.0279 | 0.0285 | +0.0220 [+0.0212, +0.0228] | +0.0080 | +0.0006 |
| pool-absent (gold outside base top-50) | 0.332 | 0.0052 | 0.0667 | 0.0052 | 0.0730 | 0.0793 | +0.0063 [+0.0055, +0.0071] | +0.0679 | +0.0063 |
| gold in base top-50 | 0.668 | 0.4417 | 0.4182 | 0.5296 | 0.5214 | 0.5323 | +0.1032 [+0.1020, +0.1041] | +0.0798 | +0.0109 |
| relation unseen in train (post-2020 zero-stripped code) | 0.438 | 0.2263 | 0.2380 | 0.2966 | 0.3117 | 0.3328 | +0.0737 [+0.0726, +0.0748] | +0.0854 | +0.0211 |
| relation seen in train | 0.562 | 0.3520 | 0.3514 | 0.4019 | 0.4203 | 0.4205 | +0.0690 [+0.0679, +0.0700] | +0.0683 | +0.0002 |
| cold gold (train freq < 50) | 0.110 | 0.0288 | 0.1616 | 0.0763 | 0.2015 | 0.2016 | +0.0399 [+0.0379, +0.0421] | +0.1727 | +0.0001 |
| mid gold | 0.113 | 0.0951 | 0.1371 | 0.1474 | 0.1711 | 0.1721 | +0.0340 [+0.0324, +0.0357] | +0.0760 | +0.0010 |
| head gold (train freq >= 500) | 0.777 | 0.3641 | 0.3453 | 0.4254 | 0.4262 | 0.4380 | +0.0808 [+0.0800, +0.0818] | +0.0621 | +0.0119 |
| first-seen entity | 0.002 | 0.0148 | 0.0176 | 0.0175 | 0.0166 | 0.0161 | -0.0010 [-0.0084, +0.0062] | +0.0018 | -0.0005 |
| hist=no (no train+valid co-occurrence) | 0.188 | 0.0668 | 0.1232 | 0.1184 | 0.1710 | 0.1787 | +0.0479 [+0.0464, +0.0493] | +0.1042 | +0.0077 |

!Where the gain lands

*Figure 2.* ΔMRR of HRCOV (B = 150) against RecB-lite (orange) and against the window-trained base (grey) per stratum, with 95% CIs.
The test-sample share of each stratum is shown in brackets. The first-seen-entity stratum (0.2%) is omitted; its Δ vs RecB-lite is
−0.001 with a CI spanning 0.

* The margin over RecB-lite is largest where recurrence alone cannot answer: non-recurrent facts (+0.111 / +0.111) and golds inside the
  base top-50 (+0.107 / +0.103). It is +0.048 / +0.047 on exactly recurrent facts.
* Pool widening helps against the base, not against the recurrency rule. On golds outside the base top-50 (35% / 33%), HRCOV lifts MRR
  from 0.005 to 0.094 / 0.073, but RecB-lite already reaches 0.089 / 0.067 there, so the margin over it is only +0.005 / +0.006.
* No-pair-history facts (10.8%) gain +0.022 over RecB-lite and +0.008 / +0.008 over the base. First-seen facts (0.2%) gain ≈ 0.
* Ceilings (gold placed at rank 1 when it is in the set): top-50 0.652 / 0.670; C_cov at B = 150 0.795 / 0.800. HRCOV realises 0.373 / 0.373,
  i.e. 47% of the candidate-set ceiling.

## 5. The relation-code artefact (Table 4; secondary, pre-registered; † = post-hoc)

| base | Δ RELNORM − orig, all [95% CI] | relation unseen in train (43.8%) | relation seen in train | MRR unseen-rel: base / RecB-lite / RecB-lite RN† / HR / HRCOV / HRCOV RN | MRR seen-rel: base / RecB-lite / HRCOV / HRCOV RN | official seed 0 HRCOV → RN |
|---|---|---|---|---|---|---|
| RE-GCN | +0.0095 [+0.0091, +0.0098] | +0.0215 [+0.0208, +0.0223] | +0.0001 [-0.0000, +0.0002] | 0.1765 / 0.2380 / 0.2674 / 0.2905 / 0.3138 / 0.3353 | 0.3306 / 0.3514 / 0.4198 / 0.4199 | 0.3739 → 0.3829 |
| LogCL | +0.0093 [+0.0090, +0.0097] | +0.0211 [+0.0204, +0.0218] | +0.0002 [+0.0001, +0.0003] | 0.2263 / 0.2380 / 0.2674 / 0.2966 / 0.3117 / 0.3328 | 0.3520 / 0.3514 / 0.4203 / 0.4205 | 0.3733 → 0.3819 |

!Relation-code artefact

*Figure 3.* Test-sample MRR on facts whose relation was seen in train (light) and unseen in train (hatched; 43.8% of the sample, all post-2020
zero-stripped codes). Shown for the window-trained base, RecB-lite, HR, HRCOV and HRCOV with relation-normalised features.

* **Bases.** The bases are hit hardest. MRR on unseen-relation facts is 0.154 (RE-GCN) / 0.126 (LogCL) below that on seen-relation facts:
  their embeddings for ids 274–390 are untrained, and their global history keyed on those ids only starts in 2020.
* **RecB-lite.** The recurrency rule is hit too, because its history is keyed on the relation id. Unseen-relation MRR is 0.238 vs 0.351 on
  seen relations. On normalised data (post-hoc) the unseen-relation MRR rises to 0.267, and overall MRR to 0.3146.
* **Re-rankers.** HRCOV is hit least, because its dyadic and entity-level evidence does not depend on the relation id. Normalised relation
  features add +0.021 on the unseen-relation facts and ≈ 0 elsewhere.
* **Residual gap.** After normalisation, a gap of 0.085 / 0.088 MRR remains between unseen- and seen-relation facts for HRCOV. These facts are
  the 2020–2022 part of the test period, so the remainder is attributable to period, not to relation ids, in this design.
* **Answer to question (b) in Stage A.** For history-based systems the artefact costs ≈ 0.02–0.03 MRR on the affected 44% of facts
  (≈ 0.009–0.013 overall). Its cost to the published graph encoders needs a base retrained on normalised relations (Stage B item). The
  window-trained RE-GCN already scores 0.263 on the sample against the published 0.182. This suggests the published weakness relative to
  RecB also reflects configuration (history 3, full-train schedule), but the protocols differ and the comparison is not controlled.

## 6. Negative and null results kept in view

1. Both window-trained bases are **below** RecB-lite on the test sample (RE-GCN -0.0386 [-0.0397, -0.0375],
   LogCL -0.0048 [-0.0056, -0.0038]), although they are above it on valid. The base-vs-RecB ordering flips across the 2020
   code change.
2. Inside the base top-50, HRCOV is below HR (RE-GCN 0.5233 vs 0.5355, LogCL 0.5214 vs 0.5296): the 150 extra candidates dilute in-pool
   re-ranking, as on polecat. The net HRCOV − HR is +0.0232 / +0.0170 because the out-of-pool gain is larger.
3. Over RecB-lite, pool widening adds only +0.005 / +0.006 on out-of-pool golds; the out-of-pool recurrence mass is already reachable by the rule.
4. First-seen entities (0.2%) gain nothing (−0.0008 / −0.0010 vs RecB-lite, CI includes 0). No-pair-history facts gain +0.022 over RecB-lite but
   only +0.008 over the bases.
5. RELNORM: ≈ 0 on facts with seen relations, as expected. On the RE-GCN cold-gold stratum it is −0.0016 [−0.0027, −0.0006].
6. B* = 150 is the top edge of the block-size grid on both bases. The valid differences are ≤ 0.0011, so a larger block may add a little.
7. The official TGB RecB in default mode (window 100) scores 0.2034, below the TGB train-mode row (0.211, full test) and far below
   full-history RecB-lite (0.3017). History truncation, not the rule, explains most of the published RecB-vs-RecB-lite gap.

## 7. Deviations from the task specification

1. The study follows the Stage A redesign in place of the original full-data pilot. Bases are trained on the last 20% of train days
   (days 5,749–7,186; 2,266,194 of 10,861,600 facts). The test is a uniform 10% query sample (465,138 of 4,651,378). Re-rankers are trained
   and selected on a uniform 200,000-query valid sample (of 4,652,314). Every test number is provisional for the full-data question; Stage B
   was not started.
2. HRCOV drops pilot-1's stage-2 bridge block and its 5 bridge features (25 features in total). The block size B ∈ {50, 100, 150} is chosen on
   valid (the "history block cap").
3. The official py-tgb Evaluator scored seed-0 systems only, using a batched path: full score rows with the TGB conflict set masked to −inf.
   It was cross-checked per query through `query_batch` on 20,000 queries. 3-seed means use the harness evaluator, which is verified identical.
4. The official TGB RecB uses the repository's unmodified predictor called per relation on the sampled queries (fork pool, not the script's
   ray driver). Default mode only; train mode was not run.
5. RE-GCN was not retrained on relation-normalised data (optional item; no relnorm base row).
6. The RecB-lite RELNORM control is post-hoc, added after the GO decision was read; it is labelled as such and does not enter the decision.
7. LogCL's global copy-history is pre-rolled over pre-window train facts. This is a window-training design choice, stated in the rule file.
8. Pattern target-encoding keys use multiplier 4e7 instead of the retriever track's 2e7 (fixes key collisions for R = 391).
9. Strata come from the full online history (Hist). Pattern-reach and 2-hop-reach strata were not computed.
10. The head guardrail was applied against both comparators, which is stricter than the single-comparator form of earlier pilots.

## 8. Implications for Stage B (if requested) and the project

* Stage A supports running Stage B. The margin over RecB-lite (+0.07) is 14× the GO threshold, it holds on both bases, and it survives
  relation normalisation of both sides (+0.067 / +0.068, post-hoc). Stage B (full-train bases, full test) will remove the base-handicap
  inflation and give leaderboard-comparable rows.
* Stage B design points: open the block-size grid upward (B* at the edge); add a RE-GCN base retrained on `tkgl-icews-relnorm` to measure
  the artefact on the base itself; keep RecB-lite (and RecB-lite RN) as the decisive comparators.
* For Task 3, the useful structure on icews is again entity-pair and entity-level history read beyond the base's top-50. The
  relation-id-free evidence is what makes the re-ranker robust to the 2020 relation-code change.

## 9. Hosts and jobs

All base training, dumps, lists, features, re-rankers, RecB scoring and official evaluation ran on `serverA` (2× RTX PRO 6000
Blackwell Max-Q, docker `tkg-gpu:latest`, torch 2.8.0+cu128, py-tgb 2.3.0); GPU 0 for RE-GCN, GPU 1 for LogCL. Aggregation, bootstrap and
figures ran on the local laptop (CPU). Jobs:
* smoke: 61c7c7c7
* bases: RE-GCN 3a0a75e9, LogCL 04d84518
* relnorm + prep + verify: ff0a2c30
* RecB-lite official + TGB RecB: 5e88174c (the first attempt, 5d4db400, failed on a docker option order)
* chains: RE-GCN 43d0cb97, LogCL eb65c060
* relnorm data + post-hoc RecB-lite RN: 3e1a0314

## 10. Artifacts

* `icews_stageA_verdict.json`: all numbers, decision, strata, ceilings, RELNORM, post-hoc block, base grids
* `stageA_decision_rule.json`: pre-registered rule
* `relnorm_mapping.csv`: 117 merges with per-merge evidence
* `pilot_pipeline_README_v3.md`: paths, commands, run times, pitfalls, Stage B reuse (§8)
* Figures: `fig1_stageA_mrr.png`, `fig2_strata.png`, `fig3_relation_codes.png`
* Code: `pi_common.py`, `pi_relnorm.py`, `pi_relnorm_data.py`, `pi_verify.py`, `pi_prep.py`, `pi_base.py`, `pi_lists.py`, `pi_replay.py`, `pi_features.py`, `pi_train.py`, `pi_recblite_official.py`, `pi_recb_official.py`, `pi_chain.sh`, `pi_agg.py`, `pi_figs.py`
