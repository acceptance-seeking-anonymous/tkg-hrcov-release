# Addendum to "SOTA Re-Survey: TKG Extrapolation 2024-01 → 2026-07" — novelty refresh, 2026-07-01 → 2026-10-11

**Prepared 2026-10-11 for the  temporal-knowledge-discovery project (Task 2 fusion / Task 3 extrapolation).** Parent document: `sota_sweep_tkg_2024-2026.md` (2026-07-12), §6 novelty verdicts and §7 bibliography. This addendum re-checks ideas **6.1**, **6.2** and **6.5** against literature posted or published 2026-07-01 → 2026-10-11, adds SOTA rows that exceed the parent tables, and reports the status of the parent's watch items.

**Marks.** [V] = I read the abstract and/or full text myself (sub-tag *full* = numbers/method read from the PDF; *abs* = abstract only). [S] = search snippet, title or TLDR only. Every entry carries an arXiv id or DOI that was returned by an API or PDF in this session.

**Window note.** Four papers dated *before* the window (HisRES ICDE'25, CHE-TKG May 2026, CFARL Apr 2026, CAPRE ICAART'26) were missed by the July sweep and surfaced here through citation chasing. They are labelled "pre-window, missed by sweep" wherever they appear.

---

## 0. Summary

1. **6.1 (IB-trained pattern retriever over abstracted paths): OPEN → NARROWED.** No TKG paper trains a retriever with an IB/compression objective or reuses one retriever across several frozen bases. However, **BridgeMem** (arXiv:2609.29268, 2026-09-24) now retrieves entity-anonymised transition evidence (relation, direction, lag tokens) and feeds it as a residual into a frozen forecaster, and **GRIP** (arXiv:2608.16776) applies a capacity bottleneck so that retrieved evidence carries only the information the query lacks (text RAG). Two of the idea's three ingredients now have published TKG or near-TKG instances; the IB objective and cross-base reuse do not.
2. **6.2 (residual / Δ-log-likelihood objective against a frozen base): OPEN → NARROWED.** **BridgeMem** adds a learned likelihood-ratio residual to the log-scores of a *frozen* CENET and gates it by the base's own uncertainty. "Residual correction on a frozen TKG base" can no longer be claimed as new. BridgeMem's residual is, however, **trained base-agnostically** (self-supervised next-relation cross-entropy on dyads; the base enters only through a validation-selected λ). The *objective* of optimising conditional predictive information given the base's prediction is still unclaimed in TKG. Outside TKG, the same idea appears in frozen-forecaster retrieval for time series (FreshCast arXiv:2610.07834; PRR arXiv:2608.23221) and in competence-gated pooling for event forecasting (arXiv:2609.12101).
3. **6.5 (utility-gated evidence augmentation for cold coverage in extrapolation, per-stratum reporting): remains OPEN.** The nearest new work, ICPE-STKG (KBS'26, [S]), completes missing prior facts with a reverse-causal RL agent on *artificially sparsified* ICEWS/WIKI variants; it has no utility gate relative to a frozen base and no stratum accounting. The other near neighbours are CFARL (gated augmented actions for few-shot unseen entities, completion) and H-MoE (query-aware gating with a cold-start expert, no augmentation). BridgeMem leaves candidates *without* dyadic history unchanged by construction. It is therefore complementary to cold-coverage work and gives a clean control.
4. **SOTA (taf, both directions, RE-GCN split).** The July tables are superseded. **CHE-TKG** (arXiv:2605.04652, preprint) reports 51.91 / 38.77 / 60.37 / 27.38 MRR on ICEWS14 / ICEWS18 / ICEWS05-15 / GDELT [V full], and **HisRES** (ICDE'25, arXiv:2405.10621) reports 50.48 / 37.69 / 59.07 / 26.58 [V full]. Both exceed SRM-LLM, LogCL and CSI on ≥3 datasets and were missed in July. **FreqDiff** (EMNLP'26 Main, arXiv:2608.20804) ties SRM-LLM on ICEWS14 (50.85 vs 50.84) and is below the July best elsewhere. BridgeMem (CENET protocol) and EST (ACL'26) report much higher numbers that are **not protocol-comparable** (§3).
5. **Watch items.** I found no TKG-extrapolation follow-up from the Booster, TransFIR, GFM-Retriever (only DualG-MRAG, multimodal RAG), SIBR or RECIPE-TKG groups. SRM-LLM's authors published CRI (ACL'26 Findings). The strikingness paper is now IJCAI'26, and its group also published FETA (TASLP'26). MTRM is now in TKDE'26 (closed access, numbers not retrieved). ICPE-STKG is now in KBS vol. 347 (abstract only, via snippet).

---

## 1. Verdict table

| Idea | July 2026 verdict | **Oct 2026 verdict** | Nearest neighbours (id, mark) | What our experiment must establish to stay novel |
|---|---|---|---|---|
| **6.1** IB-trained pattern RETRIEVER over abstracted temporal paths; residual utility + compression; decoupled, reusable across frozen bases | OPEN | **NARROWED** | BridgeMem arXiv:2609.29268 [V full]; GRIP arXiv:2608.16776 [V full-partial]; SIBR 10.1016/j.eswa.2024.126226 (July anchor); GFM-Retriever arXiv:2603.07179 (July anchor) and follow-up DualG-MRAG arXiv:2607.28580 [V abs]; CRI 10.18653/v1/2026.findings-acl.1471 [V full-partial]; LANTERN 10.18653/v1/2026.findings-acl.559 [V full-partial]; AECR arXiv:2608.05205 [V abs]; FITTER arXiv:2608.10668 [V abs]; SNAM arXiv:2610.04869 [V full-partial] | Train **one** retriever over entity-anonymised **multi-hop** patterns with an IB objective. Show that, without retraining, it improves **≥2 different frozen bases** (e.g., TiRGN + CENET or HisRES) on the full test sets. At **matched retrieved-set size**, it must beat (i) its own β=0 (no-compression) ablation, (ii) **random relation admission** (the G9 control that our IB edge selector failed), and (iii) a BridgeMem-style 1-hop dyadic residual. The gain must also appear on the **no-dyadic-history stratum**, where BridgeMem is zero by construction. |
| **6.2** Residual / Δ-log-likelihood TRAINING OBJECTIVE against a frozen base (TiRGN) | OPEN (crowded surroundings) | **NARROWED** (architecture taken; objective open) | BridgeMem arXiv:2609.29268 [V full]; SNAM arXiv:2610.04869 [V full-partial]; FreshCast arXiv:2610.07834 [V abs]; PRR arXiv:2608.23221 [V abs]; Competence-gated pooling arXiv:2609.12101 [V abs]; generator-in-the-loop reranker arXiv:2609.08188 [V abs]; ENOUGH arXiv:2609.08180 [V abs]; EV-COT / SPARK / CoH (July anchors) | At identical module, evidence, parameter count and λ/gate-selection protocol, show that training **through the combined score with the frozen base's logits inside the loss** (maximise log p(gold \| base ⊕ evidence) − log p(gold \| base)) beats three alternatives: (a) the same module trained **base-agnostically** (BridgeMem's next-relation CE); (b) **post-hoc validation-calibrated mixing** (SNAM/FreshCast-style β); (c) our own **utility-as-feature** ranker (G10/G11). The gain should concentrate where the base is wrong or uncertain (high-strikingness and cold strata), with no head-stratum damage, on ≥2 bases. |
| **6.5** Learned, conditional-utility-GATED evidence augmentation for cold coverage in EXTRAPOLATION, per-stratum coverage-up-without-head-damage | OPEN | **OPEN** (minor narrowing) | ICPE-STKG 10.1016/j.knosys.2026.116268 [S]; CFARL 10.1109/TETCI.2026.3683653 [V abs, pre-window]; H-MoE 10.3390/technologies14100633 [V abs]; CAPRE 10.5220/0014349100004052 [V abs, pre-window]; ContraTGT 10.24963/ijcai.2026/303 [V abs]; GAttNHP arXiv:2607.14733 [V abs]; BridgeMem arXiv:2609.29268 [V full] (no-history candidates left unchanged); Booster arXiv:2501.00252 (July anchor; no extrapolation follow-up found); TransFIR arXiv:2604.10164 (July anchor; no follow-up found) | Report **full test, all strata** on ICEWS14/18/05-15/GDELT (taf, both, RE-GCN split); this is the number GraphBridge still lacks. A **learned** conditional-utility gate on augmented evidence (train-time synthetic quadruples and/or inference-time bridge evidence) must raise cold-stratum metrics (pool-absent gold, no-dyadic-history, first-seen entity) over the frozen base, with a **pre-registered head-damage bound**. At matched augmentation budget, it must beat four alternatives: ungated completion (ICPE-STKG/Booster-style), **random admission** (G9 lesson), base-uncertainty gating g(q)=4u(1−u) (BridgeMem), and the utility-feature ranker (G10–G12). |

---

## 2. Per-idea analysis

### 2.1 Idea 6.1 — IB-trained pattern retriever over abstracted temporal paths

**What appeared since July.**
- **BridgeMem** (arXiv:2609.29268, 2026-09-24) [V full]. For each candidate *c* of query (a, r, ?, t), it retrieves the ≤8 most recent events on the ordered pair (a, c) within 65 snapshots. Each event is encoded as relation, direction and log-binned lag tokens. A count reader conditions on (relation, direction, recency) "rather than entity identities", so transitions are reused across dyads; this is an abstracted pattern space. The output is a likelihood-ratio residual added to a frozen CENET's log-scores. A support-adaptive mixture (ρ = τ/(n+τ), τ = 100) backs off from counts to a neural reader on sparse dyads.
  - **What it does not do:** it has no IB or compression objective; it is trained only with next-relation cross-entropy. Retrieval is limited to **1-hop direct dyads**, so candidates with no prior event with *a* get zero residual. It uses one backbone, and the authors list transfer to other backbones as future work.
- **GRIP** (arXiv:2608.16776) [V full-partial]. Text RAG. Retrieved evidence passes through a severe stochastic bottleneck while the decoder keeps full access to the query, which forces the evidence channel to encode the *residual* information not available from the query. It reports a 30× reduction in a query–latent MI diagnostic. This is the closest published "IB-for-residual-evidence" objective, but it is neither graph nor temporal.
- **GFM-Retriever follow-up**: DualG-MRAG (arXiv:2607.28580, ACM MM'26; same group: Q. Sun, H. Yuan, J. Li) [V abs]. It uses a GNN retriever for multimodal RAG with no IB term and no TKG setting. No TKG or IB follow-up of GFM-Retriever was found (S2 citers: 4, none TKG).
- **Abstraction-level rule/pattern work:**
  - CRI (ACL'26 Findings; SRM-LLM authors F. Zhang, J. Cheng) [V full-partial]: an LLM generator abstracts seed rules and a fact-grounded critic filters them, with no trained retriever.
  - AECR (arXiv:2608.05205) [V abs]: abstract event causal rules are retrieved and injected; its gains concentrate on rare and unseen events. It is evaluated on CGEP, not ICEWS/GDELT. It is also relevant to **Task 2** (eventuality-level abstraction).
  - FITTER (arXiv:2608.10668, ISWC'26) and GRATE (arXiv:2607.10197) [V abs]: vocabulary-agnostic structural encoders for cross-TKG transfer. They are encoders, not retrievers.
- **Training-free evidence selection.** LANTERN (ACL'26 Findings) [V full-partial] selects evidence for an LLM by combining a long-window strength score and a short-window novelty score with Pareto-greedy selection. It is heuristic, and is evaluated under a 2-hop candidate protocol.
- **Cross-base reuse of a plug-in.** SNAM (arXiv:2610.04869) [V full-partial] shows that a <20-parameter source-activity term, fitted once, transfers to four *frozen* temporal link predictors with only β selected on validation. This shows that reviewers will expect real multi-base evidence for any "reusable" claim. It is evaluated on dynamic-graph link prediction (AP), not TKG.

**What GraphBridge already did.** G9 found that an IB edge selector does **not** beat random relation admission at matched set size. Taken together with G3 (four embedding-space coverage attempts failed) and G5/G8 (a relation-typed 2-hop bridge reaches cold golds and relation expansion lifts reach to ~0.70), the IB component is the weakest part of 6.1 on our own evidence. The abstraction and multi-hop components are where we have positive signal.

**Verdict: NARROWED.** The unclaimed combination is: an IB/compression-trained retriever + abstracted **multi-hop** patterns + demonstrated reuse across ≥2 frozen TKG bases.

**Required claim (falsifiable).** See the verdict table. **Kill criterion:** if IB ≈ random admission at matched size again (as in G9), drop "IB" from the contribution. The remaining "abstracted multi-hop retriever as a frozen-base plug-in" is then adjacent to BridgeMem + CRI/AECR and should be treated as CROWDED.

### 2.2 Idea 6.2 — residual / Δ-log-likelihood objective against a frozen base

**What appeared since July.**
- **BridgeMem** [V full] is the decisive new neighbour. It uses S(q,c) = log p₀(c|q) + λ·g(q)·A(q,c) over a frozen CENET, with g(q) = 4u_q(1−u_q) taken from CENET's own history-oracle probability. λ and the gate are selected on validation under the constraint that none of MRR or Hits@{1,3,10} decreases relative to the frozen backbone. This is a regression gate, analogous to the one in our harness.
  - **Training objective:** the residual is trained with a self-supervised next-relation-on-dyad cross-entropy. The paper calls this objective self-supervised and notes that it never requires scoring the full entity vocabulary. The base's prediction is therefore **absent from the training loss**; the conditional-utility objective of 6.2 is not what BridgeMem optimises.
  - **Protocol caveat:** see §3.
- **SNAM** [V full-partial] adds β·log λ_u to frozen backbone logits, with β chosen on validation. This is post-hoc calibration, not a conditional objective.
- **FreshCast** (arXiv:2610.07834) [V abs] keeps a time-series forecaster frozen. It names alignment between the memory-induced correction and the forecaster's *residual errors* as a key determinant of retrieval utility, and calibrates the memory weight in closed form on validation. It is conceptually the nearest "conditional utility vs frozen base" statement, but outside TKG.
- **PRR** (arXiv:2608.23221) [V abs] (time series) learns retrieval relevance from realised futures and connects to five frozen backbones with validation-calibrated trust. It reports that predictive relevance and forecast utility are "empirically distinct objectives", which is useful motivation for 6.2.
- **Competence-gated pooling** (arXiv:2609.12101) [V abs] defines a model's value as its *marginal* value beyond an external forecast (Brier loss) and gates pooling weights per domain. It addresses LLM question forecasting, not TKG.
- **Generator-in-the-loop alignment** (arXiv:2609.08188) and **ENOUGH** (arXiv:2609.08180) [V abs] train rerankers/controllers from a frozen generator's answer-level utility. These are text RAG, not temporal.

**What GraphBridge already did.** G10/G11 used conditional utility, including the reusability-gated g(c) = 1 − 1/rolecount(c), as a ranker **feature**, not as the training objective. The 6.2 objective therefore remains untested on our side too.

**Verdict: NARROWED.** "A residual on a frozen TKG base, gated by base uncertainty" is now prior art (BridgeMem, 2026-09-24). Novelty rests entirely on the **training objective**: conditional predictive information given the base, optimised through the combined score.

**Required claim.** See the verdict table. Two practical implications for the harness:
1. Add a **BridgeMem-style control**: the same residual architecture trained with next-relation CE, plus a λ/gate selected on validation under the same no-metric-decrease rule. This is the minimal ablation that separates "objective" from "architecture".
2. Run on **≥2 bases**. BridgeMem used only CENET, so a multi-base result is a differentiator on its own.

### 2.3 Idea 6.5 — utility-gated evidence augmentation for cold coverage in extrapolation

**What appeared since July.**
- **ICPE-STKG** (KBS vol. 347, art. 116268; online 2026-05-23, issue dated 2026-07-01) [S: ScienceDirect snippet; full text closed]. A multi-hop RL agent uses reverse causality to recover missing prior facts; a wavelet-based temporal network then reasons over the completed history. It is evaluated on eight sparse variants of ICEWS14, ICEWS05-15 and WIKI. This is "complete history before reasoning" for sparse TKGs, the closest augmentation-for-extrapolation paper found. **Missing:** no utility gate relative to a frozen base, no per-stratum head-damage accounting, and no standard full-test splits (from the snippet; unverified).
- **CFARL** (TETCI 2026, 10.1109/TETCI.2026.3683653; pre-window, missed by sweep) [V abs]. It augments the first-hop action space of unseen entities via collaborative filtering. The policy "adaptively reduces the selection probability of augmented actions", which is a learned gate on augmentation. It addresses few-shot out-of-graph TKG *completion* (inductive), not extrapolation on standard splits.
- **H-MoE** (Technologies 14(10):633, 2026-10-02) [V abs]. A query-aware gate routes among a TGNN, a temporal transformer and a PLM "semantic adapter" for sparse/cold-start queries; the largest gains are on queries with limited historical evidence. It is a routing gate, not augmentation, and is evaluated on three ICEWS sets only.
- **CAPRE** (ICAART 2026; pre-window) [V abs]: counterfactual *negative* augmentation for path-based extrapolation, not cold coverage. **ContraTGT** (IJCAI'26) [V abs]: learnable augmentation for contrastive pre-training on temporal (non-KG) link prediction. **GAttNHP** (arXiv:2607.14733) [V abs]: largest gains on long-tail event chains, with no augmentation.
- **BridgeMem** [V full] explicitly leaves candidates without dyadic history unchanged. It cannot address pool-absent or no-history golds, which is the regime where GraphBridge G5–G8 found reach.
- **Booster** (arXiv:2501.00252): no extrapolation extension found. OpenAlex lists 0 citers. S2 lists 2 citers (few-shot KGC mixup, 10.1016/j.neucom.2026.134303; multimodal KGC, 10.14778/3819518.3819526), neither TKG extrapolation. The first and last authors' 2026-07+ output contains no TKG paper (OpenAlex author disambiguation is noisy; checked on arXiv as well).
- **TransFIR** (arXiv:2604.10164): 1 S2 citer, unrelated (graph coarsening); no author follow-up found.

**Verdict: OPEN.** The specific combination is still unclaimed: extrapolation on standard splits + learned conditional-utility gate vs a frozen base + per-stratum coverage-up / head-damage reporting. Cite ICPE-STKG, CFARL and H-MoE as the closest prior work.

**Required claim.** See the verdict table. The full-test number across all strata is the main gap carried over from GraphBridge (G6–G7 report realised H@50 on the base-zero cohort only: 0.113 ICEWS18 / 0.068 GDELT). The claim must show that this reach converts into full-test MRR/H@k without head damage.

---

## 3. SOTA rows (protocol: time-aware filtered, both directions, RE-GCN split) — MRR / H@1 / H@10, ×100

Protocol checks used: (a) the paper states time-aware filtering; (b) inverse-query or both-direction averaging; (c) the train/valid/test counts in the paper's own stats table (ICEWS14 74,845/8,514/7,371; ICEWS18 373,018/45,995/49,545; ICEWS05-15 368,868/46,302/46,159; GDELT 1,734,399/238,765/305,241); (d) the TiRGN row in the paper reproduces ≈44.0 / 33.7 / 50.0 / 21.7.

### 3.1 Rows that exceed the July tables (comparable)

| Method | Venue / date | ICEWS14 | ICEWS18 | ICEWS05-15 | GDELT | Protocol (a)(b)(c)(d) | Mark |
|---|---|---|---|---|---|---|---|
| **CHE-TKG** | arXiv:2605.04652, preprint 2026-05-06 (pre-window, missed by sweep); no code link found | **51.91** / 41.37 / 72.17 | **38.77** / 27.58 / 60.95 | **60.37** / 50.04 / 79.44 | **27.38** / 17.73 / 47.02 | taf ✓; both (MRR = ½(1/k_o + 1/k_s)) ✓; split ✓ ("ICEWS14s" label but 74,845/8,514/7,371); TiRGN row 44.75 / 33.66 / 50.04 / 21.67 ✓ | [V full] |
| **HisRES** | ICDE 2025, arXiv:2405.10621 (pre-window, missed by sweep) | 50.48 / 39.57 / 71.09 | **37.69** / 26.46 / 59.70 | **59.07** / 48.62 / 78.48 | **26.58** / 16.90 / 46.31 | taf ✓; raw + inverse query sets ✓; split ✓; TiRGN row ✓ | [V full] |
| FreqDiff | EMNLP 2026 Main, arXiv:2608.20804 (2026-08-21) | 50.85 / 39.82 / 71.48 | 36.85 / 26.85 / 61.47 | 54.71 / 45.02 / 74.45 | 25.04 / 17.64 / 41.83 | taf ✓; inverse augmentation ✓; split ✓; TiRGN row 44.04 / 33.66 / 50.04 / 21.67 ✓. Ties SRM-LLM on ICEWS14 (+0.01), below the July best elsewhere; omits LogCL / HisRES / SRM-LLM from its baselines | [V full] |

**Updated best-verified rows (taf/both/RE-GCN):** ICEWS14 **CHE-TKG 51.91** (prev. SRM-LLM 50.84); ICEWS18 **CHE-TKG 38.77** (prev. SRM-LLM 37.23); ICEWS05-15 **CHE-TKG 60.37** (prev. SRM-LLM 57.97); GDELT **CHE-TKG 27.38** (prev. CSI 25.6). CHE-TKG is an un-refereed preprint without public code. **The best peer-reviewed rows are HisRES (ICEWS18 37.69, ICEWS05-15 59.07, GDELT 26.58) and SRM-LLM (ICEWS14 50.84).**

### 3.2 Rows that are reported higher but are NOT protocol-comparable

| Method | Venue | Reported MRR (ICEWS14 / 18 / 05-15 / GDELT) | Why not comparable | Mark |
|---|---|---|---|---|
| BridgeMem (frozen CENET + residual) | arXiv:2609.29268 | 58.96 / 54.17 / 71.30 / 65.40 | Uses the released CENET filtering protocol, which removes "non-gold entities recorded as another true answer to the same directed query in the evaluated data". Its CENET backbone scores 56.83 on ICEWS14 versus **39.02** for CENET under taf in the HisRES and FreqDiff tables. Its reruns of TPAR / CSI / INFER / EST (33.69 / 33.06 / 30.54 / 25.04 on ICEWS14) are far below their published taf numbers. **The +Δ over its own frozen CENET (e.g., +2.13 ICEWS14) is internally valid; the absolute rows are not.** | [V full] |
| EST (Entity State Tuning) | ACL 2026 long (per BridgeMem ref. [5]); arXiv:2602.12389 | 51.3 / 46.6 / 62.1 / 41.2 | "Standard filtered" is stated without specifying time-awareness. GDELT 41.2 is far above every taf row (≤27.4) and the recurrency ceiling. H@1 is unusually high relative to H@10 (ICEWS14 45.5 vs 62.9). BridgeMem's shared-protocol rerun gives 25.04 on ICEWS14. Possible state write-back leakage. Treat as protocol-incompatible until reproduced. | [V full] |
| CRI (GPT-3.5 generator) | ACL 2026 Findings, 10.18653/v1/2026.findings-acl.1471 | 49.8 / 38.8 / 53.1 / — | taf ✓, inverse relations ✓, but splits not confirmed; its baseline block copies the AnRe / LLM-DA (ICL-lineage) numbers. The 70B row (60.6 / 56.4 / 65.9) suggests LLM exposure to pre-cutoff ICEWS events. ICEWS18 38.8 nominally ties CHE-TKG. | [V full-partial] |
| LANTERN | ACL 2026 Findings | 48.0 / 36.5 / 51.5 / 25.5 | LLM ranking over a **2-hop candidate set** (AnRe protocol), not full-vocabulary ranking | [V full-partial] |
| TTGFN | IJCAI 2026, 10.24963/ijcai.2026/727 | H@1 relative gains only (+20.85% / +24.22% / +31.14%) | Uses extra source-text context, which the benchmark does not provide; numbers not read | [V abs] |
| BSCA | arXiv:2609.14279 | GDELT 52.1 | Interpolation (TKG completion), not extrapolation | [V abs] |
| MTRM | TKDE 2026, 10.1109/TKDE.2026.3684753 | "up to +5.89% MRR" | Closed access; table not retrieved | [V abs] |
| FETA | TASLP 2026, 10.1109/TASLPRO.2026.3700268 | "new SOTA" on ICEWS14/18/05-15/GDELT | Closed access; table not retrieved; strikingness group | [V abs] |
| EdgeReMIND | LoG 2026, arXiv:2609.17916 | best test MRR on 6/8 TGB 2.0 sets | TGB 2.0 (tkgl-*) benchmark, not RE-GCN splits | [V abs] |

**Implication for the harness.** Use HisRES or LogCL alongside TiRGN as a second frozen base where code is available; LogCL code is at github.com/WeiChen3690/LogCL (link printed in the CHE-TKG PDF; HisRES code availability not checked). Never compare against BridgeMem's or EST's absolute numbers. For 6.2, replicate BridgeMem's *mechanism* as a control under our taf protocol.

---

## 4. Watch-item status (as requested in the July sweep)

| Watch item | Status 2026-10-11 | Evidence |
|---|---|---|
| Booster (arXiv:2501.00252) authors → augmentation for extrapolation | **No extension found** | OpenAlex citers 0; S2 citers 2 (non-TKG); arXiv `au:"Jiasheng Zhang"` window → 1 unrelated paper (HyperClaim, arXiv:2607.28375); OpenAlex author works for J. Zhang / J. Shao since 07-01 → none TKG |
| ICPE-STKG (KBS'26) | Published, KBS vol. 347, art. 116268 | Abstract via ScienceDirect snippet [S]; full text closed (Unpaywall / S2 / Crossref TDM all negative). See §2.3 |
| GFM-Retriever (arXiv:2603.07179) follow-ups | One same-group follow-up, **not IB, not TKG**: DualG-MRAG (arXiv:2607.28580, ACM MM'26); plus a TKDE survey (10.1109/TKDE.2026.3717198) | S2 citers 4; arXiv author queries |
| MTRM (TKDE'26) | Published online 2026-04-16 (closed); abstract unchanged | Retrieval of semantic trend clusters + contrastive alignment; no IB, no frozen base, no per-query utility |
| TransFIR follow-ups | None found | S2 citers 1 (non-TKG); OpenAlex author works since 07-01: none TKG |
| SIBR follow-ups | None from the authors (Kai Chen, Aiping Li: no TKG works since 07-01); 17 S2 citers, none IB-for-TKG in window | — |
| SRM-LLM | Same-author follow-up **CRI** (ACL'26 Findings; F. Zhang, J. Cheng): LLM rule generator + fact-grounded critic | §2.1, §3.2 |
| Strikingness-aware evaluation (arXiv:2605.13153) | **Published at IJCAI 2026** (10.24963/ijcai.2026/432); same group: **FETA** (TASLP'26) rule-based frequency/tendency model | [V abs] |
| RECIPE-TKG | No follow-ups (OpenAlex 1 citer; author works since 07-01: none TKG) | — |
| New full-test SOTA rows | CHE-TKG, HisRES (missed in July), FreqDiff tie | §3 |

---

## 5. Other in-window TKG papers screened (context; not near 6.1/6.2/6.5)

- **Evaluation / benchmarks:**
  - TKG forecasting under distribution shifts, synthetic generator (arXiv:2607.09232, ECML-PKDD'26 WS) [V abs]: memory baselines are competitive when recurrence dominates; latent-community shifts are hardest.
  - Music-festival TKG resource (arXiv:2609.24467, AKBC'26) [V abs].
  - Unified TKG representation-learning benchmark (10.3778/j.issn.1673-9418.2510055) [V abs].
- **Continual / long-horizon:** HiTS-CL (arXiv:2609.36559) [V abs], a backbone-agnostic continual-learning framework that argues the fixed-prefix protocol is misaligned with extrapolation. This is relevant to how the harness treats test-time history. MA-DAR (arXiv:2607.21949) [V abs]; Aggregative Hawkes Autoformer (10.1109/TBDATA.2026.3702410) [V abs].
- **Encoders / representation:** GAttNHP (arXiv:2607.14733); Cog-RMH (10.1109/TKDE.2026.3689542, pre-window); H²-Evo (10.1109/TKDE.2026.3694589); TWTNet (10.1145/3828652); TRGH (10.1038/s41598-026-69859-8); TCD diffusion (10.1016/j.jnlssr.2026.100355); AdaTKG (arXiv:2605.07121, pre-window); CVGSAE full extrapolation with unseen entities/relations/timestamps (10.1109/TSMC.2026.3663166, pre-window); FFCT few-shot fuzzy TKG (10.1109/TFUZZ.2026.3702333) [all V abs].
- **Rules / LLM hybrids:** TKG-LPEHD (10.1016/j.datak.2026.102623) [V abs]: LLM-driven dynamic weight fusion of rule- and graph-based reasoning, a weak 6.2 neighbour (fusion, not residual objective). LLM-RDO (10.1109/JIOT.2026.3719675) [V abs]. RAPTOR reachability-aware RL pre-training (arXiv:2607.14886) [V abs]. Structure-aware generative TKGR (10.1007/s11280-026-01428-5) [S: TLDR].
- **Selection / pruning:** multi-granularity knowledge refinement that prunes historical TKGs at edge and node level (10.1007/s11704-026-51857-8) [S: TLDR], a weak 6.1 neighbour (heuristic selection inside the predictor). Title-only [S]: KBS "Learn from historical subgraph…" (10.1016/j.knosys.2026.117132); KBS noise-aware query-guided learning (10.1016/j.knosys.2026.116707); EAAI sparse-TKG feature learning (10.1016/j.engappai.2026.114563); ASOC zero-shot TKGC LLM-guided (10.1016/j.asoc.2026.116469).
- **Task 2 relevance (eventuality fusion):** AECR abstract event causal rules (arXiv:2608.05205) [V abs]. Concrete cause–effect pairs are abstracted to relation-level causal rules; retrieved rules are injected through gated fusion; gains are largest on rare and unseen events. This is the closest in-window analogue to our ECCKG fusion line and worth citing in the Task 2 chapter.

---

## 6. Search log

**Window:** 2026-07-01 → 2026-10-11 (arXiv `submittedDate`, OpenAlex `from_publication_date`, S2 `publicationDateOrYear`). Citation chases kept citers dated ≥ 2026-07-01, or undated 2026.

| Source | Calls | Succeeded | Records returned | Notes |
|---|---|---|---|---|
| arXiv export API — topical queries | 48 | 48 | 306 | phrase and AND queries on TKG / IB / retrieval / augmentation / cold / frozen / residual / ICEWS / GDELT |
| arXiv export API — concept queries (frozen, residual, utility, augmentation, link forecasting, pattern) | 15 | 15 | 57 | surfaced FreshCast, PRR, ENOUGH, generator-in-the-loop |
| arXiv export API — author queries (Booster, GFM-Retriever, TransFIR, strikingness, RECIPE-TKG, SRM-LLM, SIBR/CSI, Gastinger, HisRES groups) | 13 | 13 | 10 | |
| OpenAlex `search=` (API key) | 28 | 28 | 2,766 | 243 TKG-relevant after keyword filter |
| OpenAlex `title_and_abstract.search` (cursor-paged) | 6 | 6 | 392 | complete counts, e.g., "temporal knowledge graph" = 111 works in window |
| OpenAlex `cites:` (watch items + anchors) | 22 seeds | 21 (RECIPE-TKG arXiv DOI not indexed; EACL DOI used) | 11 | OpenAlex citation coverage of arXiv-only seeds is near zero |
| OpenAlex author works since 07-01 | 10 seeds × first/last author | 10 | — | name disambiguation noisy (Booster's "Jiasheng Zhang") |
| Semantic Scholar `/citations` | 23 seeds | 23 | 789 citers total; 74 in window | primary citation-chase source |
| Semantic Scholar `/paper/search` | 17 distinct queries (29 calls with retries) | **9 of 17 queries ultimately succeeded** | 813 | **20 of 29 calls returned HTTP 429.** On the first pass, 12 of 17 distinct queries failed. On retry, 4 of those succeeded (100 hits each): "TKG retrieval augmented", "TKG pattern retrieval", "residual learning frozen base model knowledge graph", "event forecasting knowledge graph large language model". **8 queries never succeeded:** TKG extrapolation; data augmentation; sparse entities; emerging/inductive; long-tail; rules + LLM; IB retriever KG; temporal link prediction augmentation. These topics were covered by the arXiv/OpenAlex equivalents. Per-call status is in `search_log_api.csv`. The configured S2 key is daemon-only and not exportable to analysis cells, so calls ran in the unauthenticated pool. |
| Semantic Scholar `/paper/batch` | 1 | 1 | 24 ids | abstracts/TLDRs for closed-access Elsevier/IEEE items (most returned no abstract) |
| Crossref / Unpaywall via full-text tool | 6 DOIs | 2 full texts (ACL Anthology) | — | ICPE-STKG, CAPRE, MTRM closed; arXiv DOI for BridgeMem not in Crossref (PDF fetched from arxiv.org instead) |
| arxiv.org PDFs | 9 | 9 | — | BridgeMem, SNAM, GRIP, FreqDiff, competence-gated pooling, CHE-TKG, EST, AECR, HisRES |
| Web search | 9 | 9 | — | EMNLP/NeurIPS/CIKM/ICLR-2027 lists; Booster; ICPE-STKG; MTRM; residual-plug-in phrasing. CIKM'26 (7–11 Nov 2026) accepted list not yet public; **no public ICLR 2027 TKG submissions found**; no NeurIPS 2026 TKG-extrapolation paper surfaced. |

**Totals:** 2,351 unique titles harvested (exact-title dedupe across sources); ≈450 candidate titles scanned by hand (243 OpenAlex TKG-filtered, ≈120 arXiv, 74 in-window S2 citers, plus S2 keyword hits); ≈60 abstracts read; **9 full texts read in part or whole** (BridgeMem in full; method/table sections of SNAM, GRIP, FreqDiff, CHE-TKG, EST, HisRES, CRI, LANTERN). The AECR and competence-gated-pooling PDFs were downloaded but only their abstracts were used. Saturation check: the 12 retried S2 keyword queries and the 15 arXiv concept queries returned no TKG-relevant paper beyond those listed.

---

## 7. Bibliography of this addendum (new entries; July entries are in the parent §7)

Near neighbours:
1. BridgeMem — Li, Chen, Zhuo, Tang, Xu. "BridgeMem: Causal Dyadic Transition Residuals for Temporal Knowledge Graph Forecasting." arXiv:2609.29268 (2026-09-24). [V full]
2. GRIP — Teng. "Grounded Reasoning via Information-Restricted Premises." arXiv:2608.16776. [V full-partial]
3. SNAM — Zhang et al. "Your Temporal Link Predictor Is Blind to Who Is Active." arXiv:2610.04869. [V full-partial]
4. FreshCast — "Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters." arXiv:2610.07834. [V abs]
5. PRR — "Which Histories Matter for Time Series Forecasting? Learning Predictive Relevance with Future Supervision." arXiv:2608.23221. [V abs]
6. Competence-gated pooling — Tiwari, Bandaru, Ji. arXiv:2609.12101. [V abs]
7. Generator-in-the-loop alignment for multimodal RAG — arXiv:2609.08188. [V abs]
8. ENOUGH (minimal sufficient profiles) — arXiv:2609.08180. [V abs]
9. CRI — Ning, Zhang, Cheng, Peng, Wang. ACL 2026 Findings. DOI 10.18653/v1/2026.findings-acl.1471. [V full-partial]
10. LANTERN — Jin et al. ACL 2026 Findings. DOI 10.18653/v1/2026.findings-acl.559. [V full-partial]
11. AECR — Zheng, Chen, Wang. "Abstract Event Causal Rules: Induction and Application." arXiv:2608.05205. [V abs]
12. FITTER — Pan et al. ISWC 2026. arXiv:2608.10668. [V abs]
13. GRATE — Pan et al. ICML'26 GFM Workshop. arXiv:2607.10197. [V abs]
14. DualG-MRAG — Tao, Sun, Yuan, Zhang, Li. ACM MM 2026. arXiv:2607.28580. [V abs]
15. ICPE-STKG — Peng, Zhang, Dong, Yin. KBS 347 (2026) 116268. DOI 10.1016/j.knosys.2026.116268. [S]
16. CFARL — Li et al. IEEE TETCI 2026. DOI 10.1109/TETCI.2026.3683653. [V abs]
17. H-MoE — Chen et al. Technologies 14(10) 633 (2026). DOI 10.3390/technologies14100633. [V abs]
18. CAPRE — ICAART 2026. DOI 10.5220/0014349100004052. [V abs]
19. ContraTGT — Jin et al. IJCAI 2026. DOI 10.24963/ijcai.2026/303. [V abs]
20. GAttNHP — Tian et al. arXiv:2607.14733. [V abs]

SOTA / protocol:

21. CHE-TKG — Lei et al. arXiv:2605.04652. [V full]
22. HisRES — Zhang et al. ICDE 2025. arXiv:2405.10621. [V full]
23. FreqDiff — Gan et al. EMNLP 2026 Main. arXiv:2608.20804. [V full]
24. EST — Li et al. ACL 2026. arXiv:2602.12389. [V full]
25. TTGFN — Tian, Tao, Zhang. IJCAI 2026. DOI 10.24963/ijcai.2026/727. [V abs]
26. BSCA — Geng, Luo. arXiv:2609.14279. [V abs]
27. MTRM — Pang et al. TKDE 2026. DOI 10.1109/TKDE.2026.3684753. [V abs]
28. FETA — Huang et al. IEEE TASLP 2026. DOI 10.1109/TASLPRO.2026.3700268. [V abs]
29. Strikingness-aware evaluation — Huang, Zhang, Wei. IJCAI 2026. DOI 10.24963/ijcai.2026/432 (arXiv:2605.13153). [V abs]
30. EdgeReMIND — Pollard. LoG 2026. arXiv:2609.17916. [V abs]

Context (§5):

31. arXiv:2607.09232 [V abs]
32. arXiv:2609.24467 [V abs]
33. DOI 10.3778/j.issn.1673-9418.2510055 [V abs]
34. arXiv:2609.36559 [V abs]
35. arXiv:2607.21949 [V abs]
36. DOI 10.1109/TBDATA.2026.3702410 [V abs]
37. DOI 10.1109/TKDE.2026.3689542 [V abs]
38. DOI 10.1109/TKDE.2026.3694589 [V abs]
39. DOI 10.1145/3828652 [V abs]
40. DOI 10.1038/s41598-026-69859-8 [V abs]
41. DOI 10.1016/j.jnlssr.2026.100355 [V abs]
42. arXiv:2605.07121 [V abs]
43. DOI 10.1109/TSMC.2026.3663166 [V abs]
44. DOI 10.1109/TFUZZ.2026.3702333 [V abs]
45. DOI 10.1016/j.datak.2026.102623 [V abs]
46. DOI 10.1109/JIOT.2026.3719675 [V abs]
47. arXiv:2607.14886 [V abs]
48. DOI 10.1007/s11280-026-01428-5 [S]
49. DOI 10.1007/s11704-026-51857-8 [S]
50. DOI 10.1016/j.knosys.2026.117132 [S]
51. DOI 10.1016/j.knosys.2026.116707 [S]
52. DOI 10.1016/j.engappai.2026.114563 [S]
53. DOI 10.1016/j.asoc.2026.116469 [S]
54. Booster citers: DOI 10.1016/j.neucom.2026.134303 [S], DOI 10.14778/3819518.3819526 [S]
55. GFM survey: DOI 10.1109/TKDE.2026.3717198 [S]
