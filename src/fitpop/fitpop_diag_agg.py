"""fitpop_diag_agg.py -- post-verdict fit-population audit, diagnostics: size-matched random-subsample control and
test strata by gold-pool membership (trainer harness ranks from fitpop_diag.py / fitpop_diag_chain.sh).
Run after fitpop_agg.py in the same namespace (needs CONFP, FP, compare, criterion, arm, XJ, np); defines DIAG and STR,
which the diagnostics section of fitpop_report.py reads.  The two blocks below are the two analysis cells as run,
in order (the second recomputes the comparisons and keeps gold_frac from the first)."""
# ---- block 1: gold-pool fractions and first comparison set
XD = RD   # harness_ranks_<c>.npz and test_goldflags_<c>.npz (results/LARGE_FILES.tsv)
DIAG={}
for c,cf in CONFP.items():
    b=cf['b']; Hn=np.load(f'{XD}/harness_ranks_{c}.npz'); ts=np.load(cf['hist'])['ts'].astype(np.int64)
    def harm(prefix, name):
        ks=[f'{prefix}__{name}_s{s}' for s in (0,1,2)]
        rr=np.mean([1.0/np.asarray(Hn[k],np.float64) for k in ks],0)
        H={h: np.mean([(np.asarray(Hn[k])<=h).astype(np.float64) for k in ks],0) for h in (1,3,10)}
        return dict(rr=rr,H=H,per_seed=[float(np.mean(1.0/np.asarray(Hn[k],np.float64))) for k in ks],n_seeds=3,harness_per_seed=[])
    Hs=np.load(cf['hrcov'])
    hreg=dict(rr=np.mean([1.0/np.asarray(Hs[f'hrank_HRCOV_PAT_s{s}'],np.float64) for s in (0,1,2)],0), H={h: np.mean([(np.asarray(Hs[f'hrank_HRCOV_PAT_s{s}'])<=h).astype(np.float64) for s in (0,1,2)],0) for h in (1,3,10)}, per_seed=[], n_seeds=3, harness_per_seed=[])
    A={'SA_reg':harm('reg','SA_W100'),'+C_reg':harm('reg',f'LAD_C_{b}'),'+S_reg':harm('reg',f'LAD_S_{b}'),'+T_reg':harm('reg',f'LAD_T_{b}'),'HRcov_reg':hreg,
       'SA_int':harm('int','SA_W100'),'+C_int':harm('int',f'LAD_C_{b}'),'+S_int':harm('int',f'LAD_S_{b}'),'+T_int':harm('int',f'LAD_T_{b}'),'HRcov_int':harm('int',f'HRCOV_RE_{b}'),
       'SA_sm':harm('smsa','SA_W100'),'+C_sm':harm('smpc',f'LAD_C_{b}'),'+S_sm':harm('smpc',f'LAD_S_{b}'),'+T_sm':harm('smpc',f'LAD_T_{b}'),'HRcov_sm':harm('smpt',f'HRCOV_RE_{b}'),
       'HRcov_own':harm('own',f'HRCOV_RE_{b}')}
    pairs=[(f'{x}: HRcov - SA',f'HRcov_{x}',f'SA_{x}') for x in ('reg','int','sm')]+[(f'{x}: +C - SA',f'+C_{x}',f'SA_{x}') for x in ('reg','int','sm')]+[(f'{x}: +S - +C',f'+S_{x}',f'+C_{x}') for x in ('reg','int','sm')]+\
          [(f'sm - reg: {a}',f'{a}_sm',f'{a}_reg') for a in ('SA','+C','+S','+T','HRcov')]+[(f'int - reg: {a}',f'{a}_int',f'{a}_reg') for a in ('SA','+C','+S','+T','HRcov')]+[('own rerun - stored (harness): HRcov','HRcov_own','HRcov_reg')]
    S=compare(A,pairs,ts)
    # strata on official rr
    G=np.load(f'{XD}/test_goldflags_{c}.npz'); gs=G['test_gold_in_sa']; gp=G['test_gold_in_pt']; gc=G['test_gold_in_pc']
    sy=FP['configs'][c]['systems']
    # need official per-query rr arrays: rebuild from agg A (not stored); recompute quickly
    DIAG[c]=dict(harness={k:float(v['rr'].mean()) for k,v in A.items()}, cmp=S, gold_frac=dict(in_SA=float(gs.mean()), in_HRcov_pool=float(gp.mean()), in_PC=float(gc.mean()),
             HRcov_pool_not_SA=float((gp&~gs).mean()), SA_not_HRcov_pool=float((gs&~gp).mean()), neither=float((~gs&~gp).mean())))
    print(c, {k:round(v,4) for k,v in DIAG[c]['harness'].items()})
    for lab in S:
        m=S[lab]['MRR']; print(f"   {lab:36s} {m['delta']:+.4f} q[{m['query']['L']:+.4f},{m['query']['U']:+.4f}] ts[{m['ts']['L']:+.4f},{m['ts']['U']:+.4f}]")
    print('   gold', {k:round(v,4) for k,v in DIAG[c]['gold_frac'].items()})

# ---- block 2: final comparisons, calls and test strata
STR={}
for c,cf in CONFP.items():
    b=cf['b']; ds=cf['ds']; ts=np.load(cf['hist'])['ts'].astype(np.int64)
    Hn=np.load(f'{XD}/harness_ranks_{c}.npz')
    def harm(prefix, name):
        ks=[f'{prefix}__{name}_s{s}' for s in (0,1,2)]
        return dict(rr=np.mean([1.0/np.asarray(Hn[k],np.float64) for k in ks],0), H={h: np.mean([(np.asarray(Hn[k])<=h).astype(np.float64) for k in ks],0) for h in (1,3,10)},
                    per_seed=[float(np.mean(1.0/np.asarray(Hn[k],np.float64))) for k in ks], n_seeds=3, harness_per_seed=[])
    A={}
    for tag,pre in (('reg',None),('int','int'),('sm',None)):
        pass
    A.update({'SA_reg':harm('reg','SA_W100'),'+C_reg':harm('reg',f'LAD_C_{b}'),'+S_reg':harm('reg',f'LAD_S_{b}'),'+T_reg':harm('reg',f'LAD_T_{b}'),'HRcov_reg':harm('own',f'HRCOV_RE_{b}'),
              'SA_int':harm('int','SA_W100'),'+C_int':harm('int',f'LAD_C_{b}'),'+S_int':harm('int',f'LAD_S_{b}'),'+T_int':harm('int',f'LAD_T_{b}'),'HRcov_int':harm('int',f'HRCOV_RE_{b}'),
              'SA_sm':harm('smsa','SA_W100'),'+C_sm':harm('smpc',f'LAD_C_{b}'),'+S_sm':harm('smpc',f'LAD_S_{b}'),'+T_sm':harm('smpc',f'LAD_T_{b}'),'HRcov_sm':harm('smpt',f'HRCOV_RE_{b}')})
    pairs=[(f'{x}: {k}',f'{p}_{x}',f'{q}_{x}') for x in ('reg','int','sm') for k,p,q in (('+C - SA','+C','SA'),('+S - +C','+S','+C'),('+T - +S','+T','+S'),('HRcov - +T (+P)','HRcov','+T'),('HRcov - SA','HRcov','SA'))]+\
          [(f'sm - reg: {a}',f'{a}_sm',f'{a}_reg') for a in ('SA','+C','+S','+T','HRcov')]+[(f'int - reg: {a}',f'{a}_int',f'{a}_reg') for a in ('SA','+C','+S','+T','HRcov')]
    S=compare(A,pairs,ts)
    DIAG[c]=dict(harness_MRR={k:float(v['rr'].mean()) for k,v in A.items()}, harness_per_seed={k:v['per_seed'] for k,v in A.items()}, comparisons=S, gold_frac=DIAG[c]['gold_frac'],
                 calls={x: criterion(S[f'{x}: HRcov - SA']) for x in ('reg','int','sm')})
    # strata on official rr (registered vs intersection arms)
    Zr = {k: v for p in (cf['hist'], cf['base']) for Z in [np.load(p)] for k, v in ((k, Z[k]) for k in Z.files)}
    Hs=np.load(cf['hrcov']); Zi={}
    for p in (f'{XJ}/official_test_{ds}_fitpop_{c}_hist.npz', f'{XJ}/official_test_{ds}_fitpop_{c}_{b}.npz'):
        Z=np.load(p); Zi.update({k:Z[k] for k in Z.files})
    O={'SA':(arm(Zr,'SA_W100'),arm(Zi,'SA_W100')),'+C':(arm(Zr,f'LAD_C_{b}'),arm(Zi,f'LAD_C_{b}')),'+S':(arm(Zr,f'LAD_S_{b}'),arm(Zi,f'LAD_S_{b}')),
       '+T':(arm(Zr,f'LAD_T_{b}'),arm(Zi,f'LAD_T_{b}')),'HRcov':(arm({k:Hs[k] for k in Hs.files},'HRCOV_PAT'),arm(Zi,f'HRCOV_RE_{b}'))}
    G=np.load(f'{XD}/test_goldflags_{c}.npz'); gs=G['test_gold_in_sa']; gp=G['test_gold_in_pt']
    strata={'gold in SA and HRcov pools':gs&gp,'gold in HRcov pool only (base / pattern candidates)':gp&~gs,'gold in SA pool only':gs&~gp,'gold in neither pool':~gs&~gp}
    st={}
    for sn,msk in strata.items():
        st[sn]=dict(share=float(msk.mean()), MRR={a:{'registered':float(O[a][0]['rr'][msk].mean()),'intersection':float(O[a][1]['rr'][msk].mean())} for a in O},
                    contribution_to_int_minus_reg={a: float(msk.mean()*(O[a][1]['rr'][msk].mean()-O[a][0]['rr'][msk].mean())) for a in O})
    STR[c]=st
    print(c, 'harness', {k:round(v,4) for k,v in DIAG[c]['harness_MRR'].items()}, DIAG[c]['calls'])
    for lab in [f'{x}: HRcov - SA' for x in ('reg','int','sm')]+[f'sm - reg: {a}' for a in ('SA','+C','+S','+T','HRcov')]:
        m=S[lab]['MRR']; print(f"   {lab:26s} {m['delta']:+.4f} q[{m['query']['L']:+.4f},{m['query']['U']:+.4f}] ts[{m['ts']['L']:+.4f},{m['ts']['U']:+.4f}]")
    for sn,v in st.items(): print('   ',sn[:30], round(v['share'],4), {a:(round(v['MRR'][a]['registered'],4),round(v['MRR'][a]['intersection'],4),round(v['contribution_to_int_minus_reg'][a],4)) for a in ('SA','+C','HRcov')})
