import os
import numpy as np, json
P=os.path.expandvars("$WORK_DIR/tracks/pc_tirgn")
D=np.load(os.path.expandvars("$WORK_DIR/tracks/hardening/tirgn_polecat/tirgn_scores_POLECAT_traintest.npz"), mmap_mode="r")
sp=D["split"]; gf=D["gold_rank_filt"]; gs=D["gold_score"]; b=D["split_bounds"]
out={}
for name,code,lo,hi in (("valid",1,b[0],b[1]),("test",2,b[1],b[2])):
    L=np.load(f"{P}/work/lookup_tirgn/{name}.npz"); rows=L["rows"]
    dr=gf[lo:hi][rows].astype(np.int64); lr=L["rank_filt"].astype(np.int64)
    dg=gs[lo:hi][rows]; lg=L["gold_lp"]
    d=np.abs(dr-lr); rel=d/np.maximum(dr,1)
    out[name]=dict(n=int(len(rows)), agree=float((d==0).mean()), le1=float((d<=1).mean()), le2=float((d<=2).mean()), rel_le_1pct=float((rel<=0.01).mean()),
                   max_abs=int(d.max()), q999_abs=float(np.quantile(d,0.999)), gold_lp_maxabs=float(np.abs(dg-lg).max()), gold_lp_q99=float(np.quantile(np.abs(dg-lg),0.99)),
                   MRR_dump=float(np.mean(1/dr)), MRR_lookup=float(np.mean(1/lr)), top1_disagree=float(((dr==1)!=(lr==1)).mean()),
                   disagree_rank_median=float(np.median(dr[d>0])) if (d>0).any() else None)
print(json.dumps(out,indent=1))
