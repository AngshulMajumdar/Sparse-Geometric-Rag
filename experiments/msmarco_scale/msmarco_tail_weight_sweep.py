from __future__ import annotations
import sys,json
import numpy as np
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); c=np.load(OUT/'coordination_features.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; SM=z['sem']; RAW=c['rawlex']; LF=c['lenfac']; QC=c['qcount']; QT=c['qterms'].astype(int); qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
WTS=[-1.,-.5,-.25,0.,.1,.25,.5,.75,1.,1.25,1.5,2.,3.,4.]; WLS=[2.,3.,4.,5.,6.,8.]; WSS=[0.,.1,.2,.3,.5,.75]
def Z(x):return m.zscore(x)
def top100(sc):
 n=len(sc); k=min(100,n); ii=np.argpartition(sc,-k)[-k:] if n>k else np.arange(n); return ii[np.argsort(sc[ii])[::-1]]
def eval_cfg(wt,wl,ws):
 run={}
 for i,qid in enumerate(qids):
  k=valid[i]; ratio=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum(.9+.1*ratio,1e-6); cov=QC[i,:k]/max(1,QT[i]); ladj=lx*np.power(np.maximum(cov,1e-6),.25); sc=wt*Z(T[i,:k])+wl*Z(ladj)+ws*Z(SM[i,:k]); oo=top100(sc); run[qid]=[int(x) for x in docs[i,oo]]
 return m.eval_run(run,qrels)
# stage1 vary tail at current lexical/sem
rows=[]
for wt in WTS:
 met=eval_cfg(wt,4.,.3); r={'wt':wt,'wl':4.,'ws':.3,**met}; rows.append(r); print('T',r,flush=True)
bestwt=max(rows,key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']))['wt']; print('BESTWT',bestwt,flush=True)
# stage2 tune ratios around selected tail; avoid huge cartesian by using chosen wt only
for wl in WLS:
 for ws in WSS:
  met=eval_cfg(bestwt,wl,ws); r={'wt':bestwt,'wl':wl,'ws':ws,**met}; rows.append(r); print('W',r,flush=True)
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); out={'protocol':'fixed eta=1 P=2000 TRAIN validation; final b=.1 coordination alpha=.25; tune final geometry/lexical/semantic trust only','best':rows[0],'top20':rows[:20],'stage1_best_wt':bestwt}; json.dump(out,open(OUT/'tail_weight_sweep.json','w'),indent=2); print('BEST',rows[0])
