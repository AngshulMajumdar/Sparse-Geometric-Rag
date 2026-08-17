from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); c=np.load(OUT/'coordination_features.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; S=z['sem']; RAW=c['rawlex']; LF=c['lenfac']; QC=c['qcount']; QT=c['qterms'].astype(int); qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
def top100(sc):
 n=len(sc); kk=min(100,n); ii=np.argpartition(sc,-kk)[-kk:] if n>kk else np.arange(n); return ii[np.argsort(sc[ii])[::-1]]
def evalx(name,fn):
 run={}
 for i,qid in enumerate(qids):
  k=valid[i]; sc=fn(i,k); oo=top100(sc); run[qid]=[int(x) for x in docs[i,oo]]
 r={'name':name,**m.eval_run(run,qrels)}; print(name,r['nDCG@10'],r['MRR@10'],r['R@100'],flush=True); return r
def Z(x):return m.zscore(x)
def base(i,k,alpha=.25):
 ratio=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum(.9+.1*ratio,1e-6); cov=QC[i,:k]/max(1,QT[i]); ladj=lx*np.power(np.maximum(cov,1e-6),alpha); return Z(T[i,:k])+4*Z(ladj)+.3*Z(S[i,:k]),cov
rows=[]
rows.append(evalx('struct_base',lambda i,k:base(i,k)[0]))
for thr in [.4,.5,.6,.67,.75,.8,.9,1.0]:
 for wb in [.05,.1,.2,.3,.5,.75,1.0]:
  rows.append(evalx(f'bonus_t{thr}_w{wb}',lambda i,k,thr=thr,wb=wb: base(i,k)[0]+wb*(base(i,k)[1]>=thr).astype(np.float32)))
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); out={'protocol':'fixed direct eta=1 P=2000 validation pool; structural base b=.1 alpha=.25 wl=4 ws=.3 plus coverage threshold bonus','best':rows[0],'top20':rows[:20]}; json.dump(out,open(OUT/'coverage_bonus_sweep.json','w'),indent=2); print('BEST',rows[0])
