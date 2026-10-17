# Forward holdout for Manuscript A (GDELT25X): Stage 2 report, the single test pass

**Outcome: PRACTICAL_ADVANCE.** Applied mechanically under `fh_decision_rule.json`, registered 2026-10-14T10:28:13Z, sha256 `07b9c2ca…2702`.

* **Validity checks.** V1–V5 all pass.
  * All 87 frozen files matched their sha256 before scoring.
  * Every frozen re-ranker reproduced its valid-select ranks on 100% of rows, with maximum score deviation 0.
  * Test full rows agree with the dump on 100% of rows.
  * All 364,718 test queries (61 days) were scored.
  * Identity check = 1.0.
* **Evaluator.** Harness exact time-aware filtered rank, both directions. Standalone RecB uses its own average-tie formula.

## Chronology (UTC)

| time | event |
|---|---|
| 2026-10-14T10:28:13Z | decision rule written; file saved 10:28:19Z |
| 10:33:49Z – 11:34:31Z | Stage 1 (job): base, controls, re-rankers fitted on valid only; 87-file sha256 manifest 11:34:28Z |
| ≈14:00Z | **release**: approval of Stage 2 by the investigators, relayed  (exact minute not stated) |
| 14:02:38Z | Stage-2 job submitted with the reported command, wrapped in a GPU-1-only wait (below) |
| 14:02:41Z | job start; GPU 1 had 38,150 MiB free (> 35 GB), so the frozen script was launched |
| 14:02:44Z | hash verification: 87 checked, 0 problems |
| 14:02:45Z | frozen GPU check selects physical GPU 1 (GPU 0 held by G8); cap 30 GB |
| 14:03:20Z | TiRGN test rows replayed (35 s GPU) |
| 14:08:10Z | every arm scored once |
| 14:10:00Z | fh_verdict.json written; job end (rc 0); harvested 14:10:11Z |

## Disclosures

1. **Earlier exposure of this window.** Before registration, the status pilot had already scored the GDELT25X test window with its own systems: RE-GCN and LogCL bases, its HR re-rankers, RecB and Stage-0 statistics. While auditing what existed, I read one of those numbers, the pilot's RE-GCN base test MRR. No TiRGN, HRCOV_PAT, BLEND or POOL number existed for this window before this pass. The untouched-test claim applies to this comparison only.
2. **Order changed by the investigators' decision.** The rule's stated order was to run Stage 2 after the status pilot's own Stage-2 test. This test ran **before** that, by the investigators' decision. No status-pilot Stage-2 result existed, so none could have informed any choice.
3. **GPU-1-only guard.** At release the coordinator restricted GPU use to GPU 1. The job command therefore wrapped the frozen script in a wait for GPU 1 > 35 GB free. The frozen script's own check then selected GPU 1. No frozen file was modified, and the hash check passed.
4. **Rule-level deviations from G2, declared at registration:**
   * Single base, TiRGN, trained here with the harness's entity-only objective.
   * The decision uses the timestamp-block bootstrap; the query-level bootstrap is reported alongside.
   * The full valid window was used for fitting and selection.
   * One base seed (42).

## Every arm (test, 364,718 queries)

| arm | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| TiRGN base (GDELT25X-trained) | 0.3979 | 0.2779 | 0.4416 | 0.6492 |
| RecB default (clean) | 0.3918 | 0.2760 | 0.4349 | 0.6299 |
| RecB trained (clean) | 0.4011 | 0.2853 | 0.4442 | 0.6411 |
| TiRGN(+)RecB BLEND [def MIX 0.3] | 0.4156 | 0.2937 | 0.4632 | 0.6688 |
| RecB-POOL [K=100, def MIX 0.3] | 0.4156 | 0.2937 | 0.4632 | 0.6688 |
| HR (W=50, 19 feat., 3 seeds) (seeds 0.4295, 0.4293, 0.4297; sd 0.00018) | 0.4295 | 0.3083 | 0.4779 | 0.6837 |
| HRCOV_PAT (W=100, 25 feat., 3 seeds) (seeds 0.4304, 0.4307, 0.4303; sd 0.00022) | 0.4305 | 0.3090 | 0.4788 | 0.6846 |

## Decision

**C1: HRCOV_PAT vs each recurrency control.** The margin is ≥ 0.005 and p < 0.05 against both, so C1 passes. The better control is BLEND, so U is reported against BLEND.

| contrast | Δ MRR | block CI95 (61 day-blocks) | one-sided p | U (one-sided 95%) | query-level CI95 |
|---|---|---|---|---|---|
| HRCOV_PAT − BLEND | +0.0149 | [+0.0140, +0.0157] | < 0.0005 (0 of 2,000 draws ≤ 0) | +0.0156 | [+0.0144, +0.0154] |
| HRCOV_PAT − POOL | +0.0149 | [+0.0140, +0.0157] | < 0.0005 (0 of 2,000 draws ≤ 0) | +0.0156 | [+0.0144, +0.0154] |

Paired Hits deltas against BLEND (block CI):

* H1: +0.0153 [+0.0143, +0.0163]
* H3: +0.0157 [+0.0143, +0.0171]
* H10: +0.0158 [+0.0145, +0.0172]

The G7 multinomial block bootstrap gives the same conclusion: HRCOV_PAT − BLEND CI [+0.0140, +0.0157].

**C3 (head guardrail).** The head stratum covers 92.4% of test queries (n = 337,065). HRCOV_PAT − BLEND = +0.0157, block CI [+0.0147, +0.0166], so there is no DAMAGE. **C2** reduces to the BLEND part of C1.

**Secondary contrasts (reported; not in the decision).**

| contrast | Δ MRR | block CI95 | U | query-level CI95 |
|---|---|---|---|---|
| HRCOV_PAT − HR (pool widening) | +0.0010 | [+0.0006, +0.0013] | +0.0013 | [+0.0008, +0.0012] |
| HR − BLEND | +0.0139 | [+0.0131, +0.0147] | +0.0146 | [+0.0134, +0.0144] |
| HRCOV_PAT − RecB trained (clean) | +0.0294 | [+0.0282, +0.0306] | +0.0304 | [+0.0287, +0.0300] |
| BLEND − TiRGN base | +0.0177 | [+0.0168, +0.0186] | +0.0184 | [+0.0172, +0.0182] |
| HRCOV_PAT − TiRGN base | +0.0326 | [+0.0313, +0.0338] | +0.0335 | [+0.0320, +0.0332] |

## Strata and stability (pre-registered reporting; B = 1000)

Contrast throughout: HRCOV_PAT − BLEND.

* mid (50-499) (6.2%): +0.0058 [+0.0042, +0.0075].
* low-frequency (< 50) (1.4%): +0.0031 [+0.0001, +0.0063].
* no-pair-history (1.8%): -0.0080 [-0.0093, -0.0067].
* exact recurrence (s,r,o) before t (83.7%): +0.0121 [+0.0112, +0.0132].
* gold outside base top-50 (12.1%): +0.0005 [+0.0002, +0.0008].
* Weekly blocks (9): all positive, +0.0125 to +0.0169. Leave-one-week-out: +0.0146 to +0.0151.

**Boundary.** On the 1.8% of queries with no pair history, HRCOV_PAT is *below* the blend. Pool widening adds only +0.0010 (U 0.0013). The advance therefore comes from learned re-ranking of a history-rich pool; candidate expansion contributes little on this slice, where the gold is in the TiRGN top-50 for 87.9% of test queries.

**Context, not a test.** On the inspected G2 GDELT benchmark the same contrast was +0.0099 against TiRGN(+)RecB, using the authors' checkpoint. The forward-holdout value is +0.0149.

Files:
* `fh_verdict.json`: machine-readable verdict.
* `fh_stage2_table.csv`: every arm and contrast.
