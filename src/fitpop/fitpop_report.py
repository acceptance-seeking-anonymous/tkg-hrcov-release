"""fitpop_report.py -- writes $FITPOP_OUT/fitpop_audit.json and $FITPOP_OUT/fitpop_audit.md from FP (fitpop_agg.py)."""
import os
OUT = os.environ.get("FITPOP_OUT", "fitpop_out"); os.makedirs(OUT, exist_ok=True)
import json, hashlib, datetime
ARMS = ['SA', '+C', '+S', '+T', 'HRcov']
def ci(m): return dict(delta=m['delta'], query_L=m['query']['L'], query_U=m['query']['U'], ts_L=m['ts']['L'], ts_U=m['ts']['U'])
def fmt(m): return f"{m['delta']:+.4f} | [{m['query']['L']:+.4f}, {m['query']['U']:+.4f}] | [{m['ts']['L']:+.4f}, {m['ts']['U']:+.4f}]"
out = dict(title='Fit-population audit of the v7.8 ladder and HRcov, with a common-intersection sensitivity refit (review item 2)',
           label='audit + sensitivity analysis (post-verdict; does not replace the registered v7.8 verdict F4_CONDITIONAL)',
           created_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
           rule=dict(rule_id='v78_framing_rule', sha256='8e05c67b7ce8f3a475ac391e813463b9d435f8add5af13e5c4d03f78483a5f2d', amendment_sha256='8fffffcaad75b066fe0fb35127e101c5bbe11f65de7ff4654068f5170962d84e'),
           registered_results_version_id='redacted-id',
           definitions=dict(
               common_start_fit_population='validation fit part: queries with timestamp id below the 70% cut of the validation timestamps (S1: all 162,066 validation queries; S2/S3: the 200,000-query HRcov validation sample)',
               retained='fit queries whose gold entity is among the arm pool candidates (v78_train.py: gold_pos >= 0; hcpat_train.py for HRcov: gold_pos < W = 100 after masking the bridge columns); cap 120,000 (rng 0) never binds',
               pools={'SA': 'SA50: the first 100 candidates of the strict history order', '+C, +S, +T': 'PC_<base>: base top-50 followed by 50 history candidates (same candidates for the three rungs)',
                      'HRcov': 'stored HRcov pool: base top-50 followed by <= 50 pattern-prior candidates'},
               refit='each arm retrained on the intersection of the five arms\' retained fit queries; everything else as registered: seeds 0-2, Adam 1e-3, bs 512, <= 30 epochs, patience 5 on the selection-part MRR (all selection queries, unchanged), feature standardisation recomputed on the arm\'s fit rows, calibrated fusion for +S, +T and HRcov, official py-tgb 2.3.0 scoring (ties averaged) of the final promoted lists',
               hrcov_refit='HRcov refit in the v7.8 trainer as arm HRCOV_RE (stored HRcov pool and 25 HRcov features, the hcpat_train.py recipe); its own-population re-run reproduces the stored HRcov to check the code path',
               bootstraps='paired, query Poisson and timestamp block, B = 2000, shared draws (seed 0); L, U = 5th / 95th percentiles (one-sided 95% bounds); BEATS / MATCHES / SMALL with bar 0.005 as registered'),
           configs={})
for c, r in FP['configs'].items():
    k = FP['counts'][c]; S = r['comparisons']; sy = r['systems']
    out['configs'][c] = dict(dataset=r['dataset'], base=r['base'], n_test=r['n_test'],
        common_start_fit_population=k['common_start_fit_population'], cut_ts=k['cut_ts'],
        per_arm={a: dict(pool=k['per_arm'][a]['pool'], gold_in_pool_fit_queries_used=k['per_arm'][a]['used'], frac_of_start=k['per_arm'][a]['frac_of_start'], capped=k['per_arm'][a]['capped']) for a in ARMS},
        identities_identical=k['identical_identities'], pairwise_overlap=k['pairwise'],
        intersection=dict(n=k['intersection_all_arms'], frac_of_start=k['intersection_frac_of_start']),
        hrcov_stored_training_n_fit_queries=k['hrcov_stored_training_n_fit_queries'], hrcov_count_matches_stored=k['hrcov_count_matches_stored'], row_id_note=k.get('row_id_note'),
        refit_fit_queries=r['fit_queries_refit'], own_fit_queries_recomputed_in_trainer=r['fit_queries_own_recomputed_in_trainer'],
        test_MRR={'registered': {a: sy[a]['MRR'] for a in ARMS}, 'common_intersection': {a: sy[a + '_int']['MRR'] for a in ARMS}, 'HRcov_own_population_rerun': sy['HRcov_own_rerun']['MRR']},
        test_H={'registered': {a: {h: sy[a][h] for h in ('H1', 'H3', 'H10')} for a in ARMS}, 'common_intersection': {a: {h: sy[a + '_int'][h] for h in ('H1', 'H3', 'H10')} for a in ARMS}},
        per_seed_MRR_common_intersection={a: sy[a + '_int']['per_seed_MRR'] for a in ARMS},
        harness_minus_official_common_intersection={a: sy[a + '_int']['harness_minus_official_per_seed'] for a in ARMS},
        select_MRR_common_intersection=r['select_MRR_int'],
        comparisons={lab: ci(S[lab]['MRR']) for lab in S},
        comparisons_H10={lab: ci(S[lab]['H10']) for lab in S if lab.startswith(('int:', 'reg:'))},
        hrcov_minus_sa_call={'registered': r['calls']['reg: HRcov - SA'], 'common_intersection': r['calls']['int: HRcov - SA']},
        base_identity_mean_abs_rr=r['base_identity_mean_abs_rr'])
cc = out['configs']
out['summary'] = dict(
    identities_identical_any={c: cc[c]['identities_identical'] for c in cc},
    hrcov_minus_sa_call=dict((c, cc[c]['hrcov_minus_sa_call']) for c in cc),
    ladder_registered={c: {k: cc[c]['comparisons'][f'reg: {k}']['delta'] for k in ('+C - SA', '+S - +C', '+T - +S', 'HRcov - +T (+P)')} for c in cc},
    ladder_common_intersection={c: {k: cc[c]['comparisons'][f'int: {k}']['delta'] for k in ('+C - SA', '+S - +C', '+T - +S', 'HRcov - +T (+P)')} for c in cc},
    hrcov_rerun_minus_stored={c: cc[c]['comparisons']['rerun - stored: HRcov (own population)']['delta'] for c in cc})
body = json.dumps(out, indent=1, sort_keys=True, default=str); out['sha256'] = hashlib.sha256(body.encode()).hexdigest()
json.dump(out, open(os.path.join(OUT, 'fitpop_audit.json'), 'w'), indent=1, default=str)
L = ['# Fit-population audit of the v7.8 ladder and HRcov (review item 2)', '',
     '**Label: audit plus sensitivity analysis, run after the v7.8 verdict. The common-intersection refit does not replace the registered verdict (F4_CONDITIONAL, results version `ff0dcaea`).**', '',
     '## 1. Fit populations', '',
     'Every arm starts from the same validation fit part (timestamp id below the 70% cut) and keeps only the fit queries whose gold is in its own candidate pool (v7.8 trainer and HRcov trainer alike; the 120,000 cap never binds). +C, +S and +T share one pool, so their fit queries are identical; SA (history pool) and HRcov (pattern pool) keep different sets.', '',
     '| config | common start | SA | +C = +S = +T | HRcov | intersection of all five | Jaccard SA vs +C | Jaccard SA vs HRcov | Jaccard +C vs HRcov |', '|---|---|---|---|---|---|---|---|---|']
for c in cc:
    g = cc[c]; p = g['per_arm']; o = g['pairwise_overlap']
    L.append(f"| {c} ({g['dataset']}, {g['base']}) | {g['common_start_fit_population']:,} | {p['SA']['gold_in_pool_fit_queries_used']:,} ({p['SA']['frac_of_start']:.1%}) | {p['+C']['gold_in_pool_fit_queries_used']:,} ({p['+C']['frac_of_start']:.1%}) | {p['HRcov']['gold_in_pool_fit_queries_used']:,} ({p['HRcov']['frac_of_start']:.1%}) | {g['intersection']['n']:,} ({g['intersection']['frac_of_start']:.1%}) | {o['SA vs +C']['jaccard']:.4f} | {o['SA vs HRcov']['jaccard']:.4f} | {o['+C vs HRcov']['jaccard']:.4f} |")
L += ['', '| config | only SA (not +C) | only +C (not SA) | only +C (not HRcov) | only HRcov (not +C) | HRcov count = stored HRcov training count |', '|---|---|---|---|---|---|']
for c in cc:
    o = cc[c]['pairwise_overlap']
    L.append(f"| {c} | {o['SA vs +C']['only_first']:,} | {o['SA vs +C']['only_second']:,} | {o['+C vs HRcov']['only_first']:,} | {o['+C vs HRcov']['only_second']:,} | {cc[c]['hrcov_stored_training_n_fit_queries']:,} ({'yes' if cc[c]['hrcov_count_matches_stored'] else 'NO'}) |")
L += ['', 'The retained identities are therefore not identical across arms: the registered ladder step +C − SA changes the pool and the fit population together (SA → +C adds base candidates; queries whose gold is only in the base top-50 enter the fit set, and some history-only golds leave it), and HRcov − +T (+P) also changes both. +S − +C and +T − +S use identical fit queries.', '',
      '## 2. Common-intersection sensitivity refit', '',
      'All five arms retrained on the intersection of their fit sets (same seeds, selection rows, early stopping, fusion and official scoring as registered). HRcov is refitted in the v7.8 trainer (arm HRCOV_RE: stored HRcov pool and features, the HRcov recipe).', '',
      '| config | system | registered MRR (official evaluator) | common-intersection refit MRR (official evaluator) | refit − registered |', '|---|---|---|---|---|']
for c in cc:
    g = cc[c]
    for a in ARMS:
        L.append(f"| {c} | {a} | {g['test_MRR']['registered'][a]:.4f} | {g['test_MRR']['common_intersection'][a]:.4f} | {g['test_MRR']['common_intersection'][a] - g['test_MRR']['registered'][a]:+.4f} |")
L += ['', '### Ladder and HRcov − SA (MRR; L, U = 5th / 95th bootstrap percentiles)', '', '| config | step | registered Δ | query [L, U] | ts-block [L, U] | common-intersection Δ | query [L, U] | ts-block [L, U] |', '|---|---|---|---|---|---|---|---|']
for c in cc:
    S = FP['configs'][c]['comparisons']
    for k in ('+C - SA', '+S - +C', '+T - +S', 'HRcov - +T (+P)', 'HRcov - SA'):
        L.append(f"| {c} | {k} | {fmt(S[f'reg: {k}']['MRR'])} | {fmt(S[f'int: {k}']['MRR'])} |")
L += ['', '| config | HRcov − SA call, registered | HRcov − SA call, common intersection |', '|---|---|---|']
for c in cc:
    L.append(f"| {c} | {cc[c]['hrcov_minus_sa_call']['registered']} | {cc[c]['hrcov_minus_sa_call']['common_intersection']} |")
L += ['', '## 3. Checks', '']
for c in cc:
    S = FP['configs'][c]['comparisons']; g = cc[c]
    L.append(f"- {c}: HRcov re-run in the v7.8 trainer on its own population (n = {FP['configs'][c]['n_fit_own_rerun_HRcov']:,}) vs stored HRcov: Δ MRR {fmt(S['rerun - stored: HRcov (own population)']['MRR'])}; refit fit-query counts {json.dumps(g['refit_fit_queries'])}; base reciprocal ranks equal to the registered base (mean |Δrr| {g['base_identity_mean_abs_rr']:.1e}).")
open(os.path.join(OUT, 'fitpop_audit.md'), 'w').write('\n'.join(L) + '\n')
# ---------------------------------------------------------------- diagnostics (size-matched control, test strata)
if 'DIAG' in globals():
    for c in cc:
        d = DIAG[c]; S = d['comparisons']
        cc[c]['diagnostic_size_matched_harness'] = dict(
            note='training-only refit of each arm on a random subsample (rng 1) of its OWN fit population with the size of the common intersection; harness exact ranks of v78_train.py (not the official evaluator); registered references re-read in the same harness scale (HRcov: own-population re-run)',
            harness_MRR=d['harness_MRR'], comparisons={lab: ci(S[lab]['MRR']) for lab in S}, hrcov_minus_sa_call=d['calls'])
        cc[c]['diagnostic_test_strata_official'] = dict(note='test queries split by where the gold is (SA pool = history top-100; HRcov pool = base top-50 + pattern candidates; +C/+S/+T pool differs from the HRcov pool by <= 2% of fit queries); official per-query RR, seed mean', strata=STR[c])
    out['summary']['size_matched_minus_registered_harness'] = {c: {a: DIAG[c]['comparisons'][f'sm - reg: {a}']['MRR']['delta'] for a in ARMS} for c in cc}
    out['summary']['hrcov_minus_sa_size_matched_harness'] = {c: dict(delta=DIAG[c]['comparisons']['sm: HRcov - SA']['MRR']['delta'], call=DIAG[c]['calls']['sm']) for c in cc}
    out.pop('sha256', None); body = json.dumps(out, indent=1, sort_keys=True, default=str); out['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(out, open(os.path.join(OUT, 'fitpop_audit.json'), 'w'), indent=1, default=str)
    L = open(os.path.join(OUT, 'fitpop_audit.md')).read().rstrip('\n').split('\n')
    L += ['', '## 4. Diagnostic: size-matched random subsamples (harness ranks)', '',
          'To separate the size of the fit set from its composition, each arm was also retrained on a random subsample of its own fit population with exactly the size of the common intersection (training only; harness exact ranks from the trainer, compared with the registered arms in the same harness scale). This table is not comparable in level with the official-evaluator table of section 2: in S3 the trainer harness ranks of the base-tail arms (+T, HRcov) are about 0.027 MRR higher than official.', '',
          '| config | system | registered (trainer harness ranks) | size-matched refit (trainer harness ranks) | size-matched − registered | common-intersection refit (trainer harness ranks) |', '|---|---|---|---|---|---|']
    for c in cc:
        d = DIAG[c]; S = d['comparisons']
        for a in ARMS:
            L.append(f"| {c} | {a} | {d['harness_MRR'][a + '_reg']:.4f} | {d['harness_MRR'][a + '_sm']:.4f} | {fmt(S[f'sm - reg: {a}']['MRR']).split(' | ')[0]} | {d['harness_MRR'][a + '_int']:.4f} |")
    L += ['', '| config | HRcov − SA (harness): registered | size-matched | common intersection |', '|---|---|---|---|']
    for c in cc:
        S = DIAG[c]['comparisons']; k = DIAG[c]['calls']
        L.append(f"| {c} | {S['reg: HRcov - SA']['MRR']['delta']:+.4f} ({k['reg']}) | {S['sm: HRcov - SA']['MRR']['delta']:+.4f} ({k['sm']}) | {S['int: HRcov - SA']['MRR']['delta']:+.4f} ({k['int']}) |")
    L += ['', 'In S3 the trainer\'s harness rank of the base-tail arms (+T, HRcov) is about 0.027 MRR above the official rank because CognTKE\'s tail has many tied scores; S1 and S2 agree with the official evaluator to within 0.001. The size-matched comparison stays within one scale and is unaffected.', '',
          '## 5. Diagnostic: where the intersection losses occur (official ranks, test queries)', '',
          '| config | test stratum (gold location) | share | SA reg → int | +C reg → int | HRcov reg → int | contribution to HRcov int − reg |', '|---|---|---|---|---|---|---|']
    for c in cc:
        for sn, v in STR[c].items():
            m = v['MRR']; L.append(f"| {c} | {sn} | {v['share']:.4f} | {m['SA']['registered']:.4f} → {m['SA']['intersection']:.4f} | {m['+C']['registered']:.4f} → {m['+C']['intersection']:.4f} | {m['HRcov']['registered']:.4f} → {m['HRcov']['intersection']:.4f} | {v['contribution_to_int_minus_reg']['HRcov']:+.4f} |")
    L += ['', '## 6. Reading', '',
          '1. Fit populations are not identical across the ladder. All arms start from the same validation fit part, but each keeps the queries whose gold is in its own pool: SA keeps 75.1% (S1) and 58.7% (S2, S3), the base-candidate pool of +C/+S/+T keeps 77.8%, 62.4% and 63.7%, and HRcov keeps 77.5%, 61.9% and 63.4%. +C, +S and +T use identical fit queries, so the steps +S − +C and +T − +S compare like with like; the steps +C − SA and HRcov − +T change the pool and the fit population together.',
          '2. Fewer fit queries alone does not explain it: random subsamples of each arm\'s own population, of the intersection\'s size, move every arm by at most 0.0006 MRR and HRcov − SA by at most 0.0007 (harness: S1 +0.0040 → +0.0047, S2 +0.0071 → +0.0073, S3 +0.0559 → +0.0559; calls unchanged).',
          '3. The common-intersection refit removes a specific kind of fit query: those whose gold is reachable only through the base (or pattern) candidates. Without them, the arms that use base candidates lose most of their gain on test queries whose gold is base-only (S2: +C 0.141 → 0.042, HRcov 0.153 → 0.067; S3: +C 0.372 → 0.041, HRcov 0.407 → 0.357), while SA is unchanged; in S2 HRcov also loses on queries whose gold both pools reach (−0.0035 of its −0.0084). Under this restriction HRcov − SA becomes MATCHES in S2 (−0.0013, SA slightly ahead) and in S1 (+0.0023) and stays BEATS in S3 (+0.0238), where base-score features recover most base-only golds even without such examples (+S on base-only golds: 0.409 → 0.355 with CognTKE, against 0.155 → 0.060 with TiRGN in S2).',
          '4. Consequence for the attribution: the base-candidate effect in the ladder (+C − SA) cannot be separated from the training-population change, because the population change is part of the mechanism: admitting base candidates is what admits training queries with base-only golds. The ladder should be described as "base candidates together with the training queries they make usable". The end-to-end comparisons (HRcov vs SA, HRcov vs the external comparator) remain valid as comparisons of complete systems, each trained on the queries its own pool makes usable. The intersection is not a neutral control: by construction it removes the examples that only base-candidate arms can learn from. When every arm is restricted to fit queries whose gold SA can also reach, HRcov no longer beats SA on polecat–TiRGN (S2), while the CognTKE advantage (S3) survives.',
          '', '## Deviations', '',
          '1. HRcov was refitted in the v7.8 trainer (arm HRCOV_RE: stored HRcov pool and features, the hcpat_train.py recipe) instead of hcpat_train.py, which has no fit-row option; its own-population re-run reproduces the stored HRcov (official MRR within 0.0001 in all three configurations).',
          '2. Feature standardisation (mean and SD) is recomputed on each refit\'s own fit rows, as in the registered trainer; the intersection therefore also changes the standardisation slightly.',
          '3. The size-matched control and its comparisons use the trainer\'s harness ranks, not the official evaluator (to keep GPU use small); in S3 the harness level of base-tail arms is about 0.027 above official (tied CognTKE tail), which affects levels but not within-scale differences.',
          '4. The test strata use the HRcov pool for all base-candidate arms; the +C/+S/+T pool differs from it for at most 2% of fit queries (Jaccard 0.980-0.997).',
          '5. GPU 0 only (never GPU 1), one process at a time, with a free-memory guard before each stage; GPU 0 was otherwise idle for most of the run (a second process of about 0.7 GB appeared briefly).']
    open(os.path.join(OUT, 'fitpop_audit.md'), 'w').write('\n'.join(L) + '\n')
