"""Builds v78_adapted_check.json / .md from R_AD, OUT_AD, OS_AD, SEL, SELA, reg (registered v78_results v3), chk."""
import json, hashlib, datetime, os
CFG = {'S1': 'tkgl-smallpedia, TiRGN', 'S2': 'tkgl-polecat, TiRGN', 'S3': 'tkgl-polecat, CognTKE'}
def ci(m): return dict(delta=m['delta'], query=[m['query']['L'], m['query']['U']], ts=[m['ts']['L'], m['ts']['U']])
def fmt(m): return f"{m['delta']:+.4f} | [{m['query']['L']:+.4f}, {m['query']['U']:+.4f}] | [{m['ts']['L']:+.4f}, {m['ts']['U']:+.4f}]"
def sysm(r, k): m = r['systems'][k]; return dict(MRR=m['MRR'], H1=m['H1'], H3=m['H3'], H10=m['H10'])
J = dict(
    title='v7.8 post-verdict check: external_comparator re-applied with the label-adapted CountTRuCoLa run (EXT_adapted)',
    label='registered-optional comparator run (amendment v78_rule_amendment_1, field EXT.EXT_adapted), executed and evaluated AFTER the v7.8 framing verdict was read; the registered verdict stays as recorded',
    created_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    rule=dict(rule_id='v78_framing_rule', sha256='8e05c67b7ce8f3a475ac391e813463b9d435f8add5af13e5c4d03f78483a5f2d',
              amendment='v78_rule_amendment_1', amendment_sha256='8fffffcaad75b066fe0fb35127e101c5bbe11f65de7ff4654068f5170962d84e'),
    protocol_note=dict(file='v78_external_protocol_note_2_ext_adapted.json', version_id='redacted-id',
                       sha256='ce9ef0ebc845dafd70fbf5ecee0911a7357295b4408032c7dcc423b1d48e7d6b', created_utc='2026-10-16T08:20:08Z'),
    registered_verdict=dict(label=reg['outcome']['label'], external_calls=reg['outcome']['external_calls'], standalone_calls=reg['outcome']['standalone_calls'],
                            results_version_id='redacted-id'),
    method=dict(
        EXT_adapted='CountTRuCoLa released code/config with the validation facts below the harness 70% timestamp cut moved into its learning data (sibling run; dumps tracks/v78_external/dumps/counttrucola_adapted/<dataset>/{val,test}); EXT_adapted alone = its scores, floor = minimum listed score (amendment)',
        EXT_adapted_blend='a * z(frozen-base log-prob) + (1 - a) * z(EXT_adapted score / tau), z over the top-1000 list, same 52-configuration grid as EXT_blend (a in {0.02,0.05,0.1,0.2,...,0.9,0.95,0.98} x tau in {0.5,1,2,4}), selected on the validation selection part only',
        selection='external_comparator re-applied per the rule: per method the variant with the highest selection-part MRR (harness, ties averaged), then the method with the highest selected MRR; CountTRuCoLa variants {alone, blend, adapted, adapted_blend}, EdgeReMIND variants {alone, blend}',
        selection_part=dict(S1='56,460 validation rows with ts id >= 105', S2='60,764 rows of the 200,000-row validation sample with ts id >= 1423', S3='same rows as S2'),
        test='official py-tgb 2.3.0 Evaluator (ties averaged) on all test queries; HRcov and SA seed-mean (seeds 0-2); paired bootstraps B = 2000 (query Poisson and timestamp block, shared draws, seed 0), one-sided 95% bounds; BEATS / MATCHES / SMALL as registered (bar 0.005)',
        validation_rows_inside_learning_data='NaN in the adapted validation dumps; never requested (asserted: no requested row has in_learning_data = True, all requested scores finite)'),
    deviation_disclosed=('tkgl-polecat: CountTRuCoLa learned on ALL 372,024 validation facts with ts id < 1423 (not only the 139,236 fit rows of the 200,000-row sample), '
                         'because its learning treats facts absent from its train index as negatives; this gives EXT_adapted more fit-window label access than the HRcov '
                         'ranker (capped at 120,000 fit queries). Declared in protocol note 2 before any adapted number. tkgl-smallpedia: both readings coincide.'),
    consistency_checks=dict(
        adapted_alone_selection_rr_vs_sibling_official_max_abs={c: SELA[c]['alone_vs_sibling_official_max_abs_rr']['counttrucola_adapted'] for c in SELA},
        adapted_alone_selection_MRR={c: SELA[c]['counttrucola_adapted']['alone_select_MRR'] for c in SELA},
        sibling_reported_selection_MRR_official={'tkgl-smallpedia': 0.6385, 'tkgl-polecat': 0.2649},
        adapted_alone_test_MRR={c: R_AD[c]['systems']['EXT_adapted_counttrucola']['MRR'] for c in R_AD},
        sibling_reported_test_MRR_official={'tkgl-smallpedia': 0.6373, 'tkgl-polecat': 0.2559},
        base_identity_adapted_npz_vs_v78_base_mean_abs_rr={c: R_AD[c]['adapted_base_identity'] for c in R_AD},
        registered_comparisons_reproduced_exactly={c: f"{chk[c]['n_same']} of {chk[c]['n_same'] + chk[c]['n_diff']} pairs, all systems' MRR identical (max |diff| {chk[c]['sys_max']})" for c in chk},
        harness_minus_official_EXT_adapted_alone={c: R_AD[c]['systems']['EXT_adapted_counttrucola']['harness_minus_official_per_seed'][0] for c in R_AD},
        note_harness_gap='EXT_adapted alone has the same large harness-vs-official gap on tkgl-smallpedia as EXT_alone (0.132 vs 0.133; tie blocks among listed candidates); all decisions use the official tie-averaged ranks'),
    configs={})
for c in ('S1', 'S2', 'S3'):
    r = R_AD[c]; rr = reg['configs'][c]; es = r['external_selection']; X = r['decision']['external']['X']; Xr = rr['decision']['external']['X']
    J['configs'][c] = dict(
        config=CFG[c], n_test_queries=r['n_queries'],
        selection_candidates={m: d['candidates'] for m, d in es['per_method'].items()},
        selected_grid_point_adapted_blend=dict(a=SELA[c]['counttrucola_adapted']['a'], tau=SELA[c]['counttrucola_adapted']['tau']),
        selected_grid_point_registered_blend=dict(a=SEL[c]['counttrucola']['a'], tau=SEL[c]['counttrucola']['tau']),
        external_comparator_registered=Xr, external_comparator_reapplied=X,
        selection_margin_reapplied_minus_registered=es['per_method']['counttrucola']['select_MRR'] - rr['external_selection']['per_method']['counttrucola']['select_MRR'],
        test={k: sysm(r, k) for k in ('HRcov', 'SA', 'EXT_alone_counttrucola', 'EXT_blend_counttrucola', 'EXT_adapted_counttrucola', 'EXT_adapted_blend_counttrucola')},
        comparisons={lab: ci(r['comparisons'][lab]['MRR']) for lab in (f'HRcov - {X}', f'SA - {X}', f'HRcov - {Xr}', f'SA - {Xr}', 'HRcov - EXT_adapted_counttrucola') if lab in r['comparisons']},
        external_call_registered=rr['decision']['external']['call'], external_call_reapplied=r['decision']['external']['call'],
        standalone_beats_external_registered=rr['decision']['external']['standalone_beats_external'],
        standalone_beats_external_reapplied=r['decision']['external']['standalone_beats_external'],
        standalone_call=r['decision']['standalone']['call'])
J['outcome_reapplied'] = OUT_AD; J['outcome_sensitivity_reapplied'] = OS_AD
J['changes'] = dict(external_comparator_changed={c: J['configs'][c]['external_comparator_registered'] != J['configs'][c]['external_comparator_reapplied'] for c in J['configs']},
                    external_call_changed={c: J['configs'][c]['external_call_registered'] != J['configs'][c]['external_call_reapplied'] for c in J['configs']},
                    standalone_beats_external_changed={c: J['configs'][c]['standalone_beats_external_registered'] != J['configs'][c]['standalone_beats_external_reapplied'] for c in J['configs']},
                    outcome_label_changed=OUT_AD['label'] != reg['outcome']['label'], outcome_sensitivity_label_changed=OS_AD['label'] != reg['outcome_sensitivity']['label'])
body = json.dumps({k: v for k, v in J.items()}, indent=1, sort_keys=True, default=str)
J['sha256'] = hashlib.sha256(body.encode()).hexdigest()
json.dump(J, open(os.path.join(os.environ.get('V78_OUT', '.'), 'v78_adapted_check.json'), 'w'), indent=1, default=str)
# ---------------- markdown
L = ['# v7.8 post-verdict check: external comparator with the label-adapted CountTRuCoLa run', '',
     '**Label: registered-optional comparator run (amendment `v78_rule_amendment_1`, field `EXT.EXT_adapted`), executed and evaluated after the v7.8 framing verdict was read. The registered verdict (`F4_CONDITIONAL`, results version `b037d61d`) stays as recorded; this check is reported next to it.**', '',
     f"Protocol decisions for the adapted run were saved before any adapted number: `v78_external_protocol_note_2_ext_adapted.json` (version `03fb6eb1`, sha256 `ce9ef0ebc845dafd…`, 2026-10-16T08:20:08Z). Rule `v78_framing_rule` (sha256 `8e05c67b7ce8f3a4…`), amendment 1 (sha256 `8fffffcaad75b066…`).", '',
     '## Result', '',
     f"Re-applying the rule's `external_comparator` selection over {{EXT_alone, EXT_blend, EXT_adapted, EXT_adapted_blend}} (CountTRuCoLa) and {{EXT_alone, EXT_blend}} (EdgeReMIND) selects **EXT_adapted_blend (CountTRuCoLa)** in all three configurations, by selection-part margins of +{J['configs']['S1']['selection_margin_reapplied_minus_registered']:.5f} (S1), +{J['configs']['S2']['selection_margin_reapplied_minus_registered']:.5f} (S2) and +{J['configs']['S3']['selection_margin_reapplied_minus_registered']:.5f} (S3) MRR over the registered EXT_blend. **No call changes and the outcome label does not change**: external calls stay S1 MATCHES, S2 BEATS, S3 BEATS; SA does not beat the external comparator in S1 or S3 and beats it in S2, as before; outcome `{OUT_AD['label']}` (sensitivity with SA_GEN: `{OS_AD['label']}`), the same as registered.", '',
     '| config | external comparator (registered → re-applied) | HRcov − X: Δ MRR | query bootstrap [L, U] | ts-block bootstrap [L, U] | call (registered → re-applied) | SA beats X |', '|---|---|---|---|---|---|---|']
for c in ('S1', 'S2', 'S3'):
    g = J['configs'][c]; r = R_AD[c]; X = g['external_comparator_reapplied']
    L.append(f"| {c} ({g['config']}) | {g['external_comparator_registered']} → {X} | {fmt(r['comparisons'][f'HRcov - {X}']['MRR'])} | {g['external_call_registered']} → {g['external_call_reapplied']} | {g['standalone_beats_external_registered']} → {g['standalone_beats_external_reapplied']} |")
L += ['', 'L, U = 5th / 95th bootstrap percentiles (one-sided 95% bounds), B = 2000. The registered HRcov − EXT_blend rows are reproduced exactly by the same aggregation (all 30 registered comparison pairs per configuration identical, shared bootstrap draws).', '',
      '## Selection (validation selection part, harness ranks with ties averaged)', '',
      '| config | EdgeReMIND alone | EdgeReMIND blend | CTR alone | CTR blend (registered X) | CTR adapted alone | CTR adapted blend | adapted-blend grid point (a, τ) | registered blend (a, τ) |', '|---|---|---|---|---|---|---|---|---|']
for c in ('S1', 'S2', 'S3'):
    g = J['configs'][c]; e = g['selection_candidates']['edgeremind']; t = g['selection_candidates']['counttrucola']
    L.append(f"| {c} | {e['alone']:.5f} | {e['blend']:.5f} | {t['alone']:.5f} | {t['blend']:.5f} | {t['adapted']:.5f} | **{t['adapted_blend']:.5f}** | ({g['selected_grid_point_adapted_blend']['a']}, {g['selected_grid_point_adapted_blend']['tau']}) | ({g['selected_grid_point_registered_blend']['a']}, {g['selected_grid_point_registered_blend']['tau']}) |")
L += ['', 'CTR = CountTRuCoLa. Selection rows: S1 56,460; S2/S3 60,764 (same rows as the registered EXT selection; asserted).', '',
      '## Test (official evaluator, all test queries)', '',
      '| config | system | MRR | H@1 | H@3 | H@10 |', '|---|---|---|---|---|---|']
for c in ('S1', 'S2', 'S3'):
    for k in ('HRcov', 'SA', 'EXT_alone_counttrucola', 'EXT_blend_counttrucola', 'EXT_adapted_counttrucola', 'EXT_adapted_blend_counttrucola'):
        m = J['configs'][c]['test'][k]; L.append(f"| {c} | {k} | {m['MRR']:.4f} | {m['H1']:.4f} | {m['H3']:.4f} | {m['H10']:.4f} |")
L += ['', '## Further comparisons (MRR; L, U = 5th / 95th percentiles, i.e. one-sided 95% bounds)', '', '| config | pair | Δ | query [L, U] | ts-block [L, U] |', '|---|---|---|---|---|']
for c in ('S1', 'S2', 'S3'):
    r = R_AD[c]
    for lab in J['configs'][c]['comparisons']:
        L.append(f"| {c} | {lab} | {fmt(r['comparisons'][lab]['MRR'])} |")
cc = J['consistency_checks']
L += ['', '## Checks', '',
      f"- EXT_adapted alone on the selection rows equals the sibling's official per-query reciprocal ranks (max |Δrr| {max(cc['adapted_alone_selection_rr_vs_sibling_official_max_abs'].values()):.1e}); selection MRR {cc['adapted_alone_selection_MRR']['S1']:.4f} (smallpedia) and {cc['adapted_alone_selection_MRR']['S2']:.4f} (polecat) vs the sibling's 0.6385 and 0.2649.",
      f"- EXT_adapted alone test MRR {cc['adapted_alone_test_MRR']['S1']:.4f} (smallpedia) and {cc['adapted_alone_test_MRR']['S2']:.4f} (polecat), equal to the sibling's official 0.6373 and 0.2559.",
      f"- Frozen-base reciprocal ranks inside the adapted scoring equal the v7.8 base (mean |Δrr| {max(cc['base_identity_adapted_npz_vs_v78_base_mean_abs_rr'].values()):.1e}).",
      '- Validation rows inside the adapted learning data (NaN in the dumps) were never requested (asserted per run).',
      f"- Harness vs official: EXT_adapted alone shows the same tie-block gap as EXT_alone on tkgl-smallpedia (harness − official {cc['harness_minus_official_EXT_adapted_alone']['S1']:+.4f}); decisions use official ranks.", '',
      '## Caveats', '',
      '1. Timing: the adapted run, its blend and this re-selection were made after the v7.8 verdict was read. The amendment lists EXT_adapted as optional, so the selection is the registered procedure, but the decision to run it was taken post-verdict; the registered verdict is unchanged and remains the reference.',
      '2. The selection margins are small (at most ' + f"{max(J['configs'][c]['selection_margin_reapplied_minus_registered'] for c in J['configs']):.5f}" + ' MRR) and the test MRR differences EXT_adapted_blend − EXT_blend are ' + ', '.join(f"{J['configs'][c]['test']['EXT_adapted_blend_counttrucola']['MRR'] - J['configs'][c]['test']['EXT_blend_counttrucola']['MRR']:+.5f} ({c})" for c in J['configs']) + '; the label adaptation changes the comparator only marginally, and HRcov − X moves by the same amounts.',
      '3. tkgl-polecat (S2, S3): CountTRuCoLa learned on all 372,024 validation facts below the cut, more fit-window label access than the HRcov ranker (120,000 fit queries); declared in protocol note 2. The direction of this deviation favours the comparator.',
      '4. EdgeReMIND was not adapted (out of scope of protocol note 2).', '',
      f"File: `v78_adapted_check.json` (sha256 of its canonical body `{J['sha256'][:16]}…`). Code: `v78_ext.py --tag _adapted`, `v78_adapted_chain.sh`, `v78_adapted_agg.py`, `v78_adapted_report.py`."]
open(os.path.join(os.environ.get('V78_OUT', '.'), 'v78_adapted_check.md'), 'w').write('\n'.join(L) + '\n')
