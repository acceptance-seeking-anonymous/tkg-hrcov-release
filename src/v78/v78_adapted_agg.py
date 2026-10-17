"""v78 post-verdict check (registered-optional, disclosed): external_comparator selection re-applied over
{EXT_alone, EXT_blend} x {EdgeReMIND, CountTRuCoLa} plus {EXT_adapted, EXT_adapted_blend} (CountTRuCoLa, label-adapted),
selection by selection-part MRR (validation, ts >= 70% cut); HRcov - external with query Poisson + ts-block bootstraps.
Usage: exec(open(os.path.join(os.environ['SRC'], 'v78', 'v78_adapted_agg.py')).read())  -> R_AD, OUT_AD
"""
import json, os, numpy as np
exec(open(os.path.join(os.environ['SRC'], 'v78', 'v78_agg.py')).read())
RD = os.path.join(os.environ.get("RESULTS_DIR", "."), "v78")   # per-query files of the v7.8 runs (results/LARGE_FILES.tsv), flat
JP = RD; DL = RD
J1 = RD; XJ = RD; XA = RD
CONF = {'S1': dict(ds='smallpedia', base='tirgn', hrcov=f'{RD}/off_smallpedia_tirgn_hcpat.npz', v78_hist=f'{J1}/official_test_smallpedia_v78_hist.npz', v78_base=f'{J1}/official_test_smallpedia_v78_tirgn.npz', train=f'{J1}/train_v78_smallpedia.json'),
        'S2': dict(ds='polecat', base='tirgn', hrcov=f'{RD}/off_polecat_tirgn_hcpat.npz', v78_hist=f'{DL}/official_test_polecat_v78_hist.npz', v78_base=f'{DL}/official_test_polecat_v78_tirgn.npz', train=f'{JP}/train_v78_polecat.json'),
        'S3': dict(ds='polecat', base='cogn', hrcov=f'{RD}/official_test_polecat_cogn.npz', v78_hist=f'{DL}/official_test_polecat_v78_hist.npz', v78_base=f'{DL}/official_test_polecat_v78_cogn.npz', train=f'{JP}/train_v78_polecat.json')}
SEL, SELA, R_AD = {}, {}, {}
for c, cf in CONF.items():
    ds, b = cf['ds'], cf['base']
    p = f'{XJ}/official_test_{ds}_v78_ext_{b}.npz'; pa = f'{XA}/official_test_{ds}_v78_ext_adapted_{b}.npz'
    for q in (p, pa, cf['hrcov'], cf['v78_hist'], cf['v78_base'], cf['train']): assert os.path.exists(q), (c, q)
    sel = json.load(open(f'{XJ}/sel_ext_{ds}_{b}.json')); sela = json.load(open(f'{XA}/sel_ext_adapted_{ds}_{b}.json'))
    assert sel['n_select'] == sela['n_select'] and sel['cut'] == sela['cut'], (c, 'selection part differs')
    SEL[c], SELA[c] = sel, sela
    ext = {f'EXT_{v}_{m}': (p, f'EXT_{v}_{m}') for m in ('edgeremind', 'counttrucola') for v in ('alone', 'blend')}
    ext.update({'EXT_adapted_counttrucola': (pa, 'EXT_alone_counttrucola_adapted'), 'EXT_adapted_blend_counttrucola': (pa, 'EXT_blend_counttrucola_adapted')})
    sa = sela['counttrucola_adapted']
    cf.update(ext=ext, sa_erm=(f'{XJ}/official_test_{ds}_v78_erm.npz', f'{XJ}/train_v78_erm_{ds}.json'),
              ext_select={'edgeremind': {'alone': sel['edgeremind']['alone_select_MRR'], 'blend': sel['edgeremind']['blend_select_MRR']},
                          'counttrucola': {'alone': sel['counttrucola']['alone_select_MRR'], 'blend': sel['counttrucola']['blend_select_MRR'],
                                           'adapted': sa['alone_select_MRR'], 'adapted_blend': sa['blend_select_MRR']}})
    cf.pop('external_comparator', None); cf.pop('sa_erm_select', None)
    R_AD[c] = run_config(c, cf, ext=cf['ext'])
    Za = np.load(pa); Zb = np.load(cf['v78_base'])
    R_AD[c]['adapted_base_identity'] = float(np.abs(np.asarray(Za['rr_base'], np.float64) - np.asarray(Zb['rr_base_s0'] if 'rr_base_s0' in Zb.files else Zb['rr_base'], np.float64)).mean())
OUT_AD = outcome(R_AD)
