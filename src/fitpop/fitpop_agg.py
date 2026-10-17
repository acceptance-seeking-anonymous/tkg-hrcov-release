"""fitpop_agg.py -- aggregate the fit-population audit and the common-intersection sensitivity refit (sensitivity analysis;
does not replace the registered v7.8 verdict).  Official per-query RR (py-tgb, ties averaged), seed-mean of seeds 0-2,
paired bootstraps as registered (query Poisson + ts-block, B = 2000, shared draws, seed 0; v7_common.summ_multi).
Usage: source configs/env.sh; exec(open(os.path.join(os.environ['SRC'], 'fitpop', 'fitpop_agg.py')).read())  -> FP (dict)
"""
import os, sys, json, numpy as np
sys.dont_write_bytecode = True
exec(open(os.path.join(os.environ['SRC'], 'v78', 'v78_agg.py')).read())
RD = os.path.join(os.environ.get("RESULTS_DIR", "."), "v78")   # per-query files of the v7.8 runs (results/LARGE_FILES.tsv), flat
XJ = RD; XA = RD
JP = RD; DL = RD; J1 = RD
CONFP = {'S1': dict(ds='smallpedia', b='tirgn', hrcov=f'{RD}/off_smallpedia_tirgn_hcpat.npz', hist=f'{J1}/official_test_smallpedia_v78_hist.npz', base=f'{J1}/official_test_smallpedia_v78_tirgn.npz'),
         'S2': dict(ds='polecat', b='tirgn', hrcov=f'{RD}/off_polecat_tirgn_hcpat.npz', hist=f'{DL}/official_test_polecat_v78_hist.npz', base=f'{DL}/official_test_polecat_v78_tirgn.npz'),
         'S3': dict(ds='polecat', b='cogn', hrcov=f'{RD}/official_test_polecat_cogn.npz', hist=f'{DL}/official_test_polecat_v78_hist.npz', base=f'{DL}/official_test_polecat_v78_cogn.npz')}
REG = json.load(open(os.path.join(os.path.dirname(os.environ['SRC']), 'results', 'v78', 'v78_results.json')))   # registered v7.8 results
COUNTS = json.load(open(f'{XA}/fitpop_counts.json'))
FP = dict(counts=COUNTS, configs={})
for c, cf in CONFP.items():
    ds, b = cf['ds'], cf['b']
    Zr = {k: v for p in (cf['hist'], cf['base']) for Z in [np.load(p)] for k, v in ((k, Z[k]) for k in Z.files)}
    Hs = np.load(cf['hrcov']); n = len(Zr['rows']); ts = np.load(cf['hist'])['ts'].astype(np.int64)
    Zi = {}
    for p in (f'{XJ}/official_test_{ds}_fitpop_{c}_hist.npz', f'{XJ}/official_test_{ds}_fitpop_{c}_{b}.npz'):
        Z = np.load(p); assert np.array_equal(np.asarray(Z['rows']), np.arange(n)), (c, p); Zi.update({k: Z[k] for k in Z.files})
    if 'ts' in Zi: assert np.array_equal(Zi['ts'].astype(np.int64), ts), (c, 'ts')
    A = {'SA': arm(Zr, 'SA_W100'), '+C': arm(Zr, f'LAD_C_{b}'), '+S': arm(Zr, f'LAD_S_{b}'), '+T': arm(Zr, f'LAD_T_{b}'),
         'HRcov': arm({k: Hs[k] for k in Hs.files}, 'HRCOV_PAT'),
         'SA_int': arm(Zi, 'SA_W100'), '+C_int': arm(Zi, f'LAD_C_{b}'), '+S_int': arm(Zi, f'LAD_S_{b}'), '+T_int': arm(Zi, f'LAD_T_{b}'),
         'HRcov_int': arm(Zi, f'HRCOV_RE_{b}'), 'HRcov_own_rerun': arm(Zi, f'HRCOVOWN_{b}')}
    bi = arm(Zi, 'base'); bz = arm(Zr, 'base'); base_ident = float(np.abs(bi['rr'] - bz['rr']).mean())
    pairs = [('int: +C - SA', '+C_int', 'SA_int'), ('int: +S - +C', '+S_int', '+C_int'), ('int: +T - +S', '+T_int', '+S_int'),
             ('int: HRcov - +T (+P)', 'HRcov_int', '+T_int'), ('int: HRcov - SA', 'HRcov_int', 'SA_int'),
             ('reg: +C - SA', '+C', 'SA'), ('reg: +S - +C', '+S', '+C'), ('reg: +T - +S', '+T', '+S'), ('reg: HRcov - +T (+P)', 'HRcov', '+T'), ('reg: HRcov - SA', 'HRcov', 'SA'),
             ('int - reg: SA', 'SA_int', 'SA'), ('int - reg: +C', '+C_int', '+C'), ('int - reg: +S', '+S_int', '+S'), ('int - reg: +T', '+T_int', '+T'), ('int - reg: HRcov', 'HRcov_int', 'HRcov'),
             ('rerun - stored: HRcov (own population)', 'HRcov_own_rerun', 'HRcov')]
    S = compare(A, pairs, ts)
    trj = json.load(open(f'{XJ}/train_int_{c}.json')); tro = json.load(open(f'{XJ}/train_own_{c}.json'))
    fitn = {k: trj['arms'][a]['n_fit_queries'] for k, a in [('SA', 'SA_W100'), ('+C', f'LAD_C_{b}'), ('+S', f'LAD_S_{b}'), ('+T', f'LAD_T_{b}'), ('HRcov', f'HRCOV_RE_{b}')]}
    own = {k: trj['arms'][a]['n_fit_own'] for k, a in [('SA', 'SA_W100'), ('+C', f'LAD_C_{b}'), ('+S', f'LAD_S_{b}'), ('+T', f'LAD_T_{b}'), ('HRcov', f'HRCOV_RE_{b}')]}
    FP['configs'][c] = dict(dataset=COUNTS[c]['dataset'], base=b, n_test=int(n), base_identity_mean_abs_rr=base_ident,
        fit_queries_refit=fitn, fit_queries_own_recomputed_in_trainer=own, n_fit_own_rerun_HRcov=tro['arms'][f'HRCOV_RE_{b}']['n_fit_queries'],
        systems={k: mets(v) for k, v in A.items()}, comparisons=S,
        calls={'int: HRcov - SA': criterion(S['int: HRcov - SA']), 'reg: HRcov - SA': criterion(S['reg: HRcov - SA'])},
        select_MRR_int={k: float(np.mean([v['select']['MRR'] for v in trj['arms'][a]['seeds'].values()])) for k, a in [('SA', 'SA_W100'), ('+C', f'LAD_C_{b}'), ('+S', f'LAD_S_{b}'), ('+T', f'LAD_T_{b}'), ('HRcov', f'HRCOV_RE_{b}')]})
