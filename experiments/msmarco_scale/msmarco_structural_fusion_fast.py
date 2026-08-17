from __future__ import annotations
import sys,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True)
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; L=z['lex']; S=z['sem']; nq=len(qids); P=docs.shape[1]
qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
# masks/grades in pool
REL=np.zeros((nq,P),np.float32); POS=np.zeros((nq,P),bool); NPOS=np.ones(nq,np.int32); IDCG=np.ones(nq,np.float64)
for i,qid in enumerate(qids):
 k=valid[i]; mp={int(d):j for j,d in enumerate(docs[i,:k].tolist())}
 grades=[]
 for d,r in qrels[qid].items():
  rr=float(r); grades.append(rr)
  j=mp.get(int(d));
  if j is not None: REL[i,j]=rr; POS[i,j]=rr>0
 NPOS[i]=max(1,sum(float(r)>0 for r in qrels[qid].values()))
 ideal=sorted(grades,reverse=True)[:10]; IDCG[i]=sum((2**r-1)/math.log2(j+2) for j,r in enumerate(ideal)) or 1.0
# transforms
ZT=np.zeros_like(T); ZL=np.zeros_like(L); ZS=np.zeros_like(S); PT=np.zeros_like(T); PL=np.zeros_like(L); PS=np.zeros_like(S); RRT=np.zeros_like(T); RRL=np.zeros_like(L); RRS=np.zeros_like(S)
for i,k in enumerate(valid):
 if not k: continue
 for X,Z,PCT,RR in [(T,ZT,PT,RRT),(L,ZL,PL,RRL),(S,ZS,PS,RRS)]:
  x=X[i,:k].astype(np.float64); sd=x.std(); Z[i,:k]=0 if sd<1e-8 else ((x-x.mean())/(sd+1e-8)).astype(np.float32)
  r=rankdata(-x,method='average').astype(np.float32); RR[i,:k]=r; PCT[i,:k]=(k-r)/max(1,k-1)
# invalid trailing slots: force score low in evaluator by mask
VALID=np.arange(P)[None,:] < valid[:,None]
disc10=(1/np.log2(np.arange(2,12))).astype(np.float64)

def evaluate(name, score):
 sc=np.asarray(score,np.float32).copy(); sc[~VALID]=-np.inf
 # top 100 sorted
 ix=np.argpartition(sc,-100,axis=1)[:,-100:]
 vals=np.take_along_axis(sc,ix,axis=1); ord=np.argsort(vals,axis=1)[:,::-1]; top100=np.take_along_axis(ix,ord,axis=1)
 rel100=np.take_along_axis(REL,top100,axis=1); pos100=rel100>0; rel10=rel100[:,:10]; pos10=pos100[:,:10]
 dcg=((2.0**rel10-1.0)*disc10).sum(axis=1); nd=(dcg/IDCG).mean()
 any10=pos10.any(axis=1); first=np.argmax(pos10,axis=1); rr=np.where(any10,1.0/(first+1),0.0).mean()
 h10=pos10.sum(axis=1); h100=pos100.sum(axis=1)
 return {'name':name,'nDCG@10':float(nd),'MRR@10':float(rr),'P@10':float((h10/10).mean()),'R@10':float((h10/NPOS).mean()),'R@100':float((h100/NPOS).mean()),'Hit@10':float(any10.mean()),'Hit@100':float(pos100.any(axis=1).mean()),'n_queries':nq}
rows=[]
def add(name,score):
 r=evaluate(name,score); rows.append(r); print(name,round(r['nDCG@10'],6),round(r['MRR@10'],6),round(r['R@100'],6),flush=True)

add('baseline_z_1_4_.1',ZT+4*ZL+.1*ZS)
# robust clip: focused grid
for c in [0.75,1.,1.5,2.,3.,4.]:
 for wl in [3.,4.,5.,6.,8.]:
  for ws in [0.,.1,.25]: add(f'clip_c{c}_wl{wl}_ws{ws}',np.clip(ZT,-c,c)+wl*np.clip(ZL,-c,c)+ws*np.clip(ZS,-c,c))
# tanh
for ss in [.75,1.,1.5,2.,3.]:
 for wl in [3.,4.,5.,6.,8.]:
  for ws in [0.,.1,.25]: add(f'tanh_s{ss}_wl{wl}_ws{ws}',np.tanh(ZT/ss)+wl*np.tanh(ZL/ss)+ws*np.tanh(ZS/ss))
# percentile
for wt in [.25,.5,1.,2.]:
 for wl in [1.,2.,4.,8.,12.]:
  for ws in [0.,.1,.25,.5]: add(f'percent_wt{wt}_wl{wl}_ws{ws}',wt*PT+wl*PL+ws*PS)
# RRF focused
for K in [10.,30.,60.,100.,200.]:
 for wt in [.5,1.,2.]:
  for wl in [1.,2.,4.,8.]:
   for ws in [0.,.1,.25,.5]: add(f'rrf_K{int(K)}_wt{wt}_wl{wl}_ws{ws}',wt/(K+RRT)+wl/(K+RRL)+ws/(K+RRS))
# interactions on z: lexical-tail agreement and absolute gaps
for inter in [-1.,-.5,-.25,.25,.5,1.]:
 for wl in [3.,4.,5.]:
  add(f'zprod_i{inter}_wl{wl}',ZT+wl*ZL+.1*ZS+inter*(ZT*ZL))
# min/consensus bonus using percentiles
for c in [.25,.5,1.,2.]:
 for wl in [2.,4.,6.]: add(f'consensus_min_c{c}_wl{wl}',PT+wl*PL+.1*PS+c*np.minimum(PT,PL))
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'fixed eta=1 P=2000 validation pools; structural fusion only, no index/candidate changes','baseline':next(x for x in rows if x['name']=='baseline_z_1_4_.1'),'best':rows[0],'top30':rows[:30],'n_formulas':len(rows)}
json.dump(out,open(OUT/'structural_fusion_fast.json','w'),indent=2)
print('=== BEST ==='); print(json.dumps(rows[0],indent=2)); print('TOP10'); [print(r) for r in rows[:10]]
