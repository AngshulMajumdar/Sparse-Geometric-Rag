from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True)
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid']; docs=z['docs']; tail=z['tail']; lex=z['lex']; sem=z['sem']
qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)

# Per-query transformed arrays cached in RAM. 1000 x 2000 float32 each.
NQ=len(qids); P=docs.shape[1]
ZT=np.zeros((NQ,P),np.float32); ZL=np.zeros_like(ZT); ZS=np.zeros_like(ZT)
PT=np.zeros_like(ZT); PL=np.zeros_like(ZT); PS=np.zeros_like(ZT)
RRT=np.zeros_like(ZT); RRL=np.zeros_like(ZT); RRS=np.zeros_like(ZT)
for i in range(NQ):
    k=int(valid[i])
    if not k: continue
    for raw,outz,outp,outr in [(tail[i,:k],ZT[i,:k],PT[i,:k],RRT[i,:k]),(lex[i,:k],ZL[i,:k],PL[i,:k],RRL[i,:k]),(sem[i,:k],ZS[i,:k],PS[i,:k],RRS[i,:k])]:
        outz[:] = m.zscore(raw)
        # average rank under ties, best score rank=1
        r=rankdata(-np.asarray(raw,dtype=np.float64), method='average')
        # percentile quality: best near 1, worst near 0
        outp[:] = (k-r)/(max(1,k-1))
        outr[:] = r
    if (i+1)%200==0: print('transform',i+1,flush=True)

def top100(sc):
    n=len(sc); kk=min(100,n)
    if n<=kk: return np.argsort(sc)[::-1]
    ii=np.argpartition(sc,-kk)[-kk:]
    return ii[np.argsort(sc[ii])[::-1]]

def eval_formula(name, fn):
    run={}
    for i,qid in enumerate(qids):
        k=int(valid[i]);
        if not k: run[qid]=[]; continue
        sc=fn(i,k)
        oo=top100(sc)
        run[qid]=[int(x) for x in docs[i,oo]]
    met=m.eval_run(run,qrels); return {'name':name,**met}

rows=[]
# exact baseline
rows.append(eval_formula('baseline_z_1_4_.1', lambda i,k: ZT[i,:k]+4*ZL[i,:k]+.1*ZS[i,:k]))

# 1) clipped-z robust fusion. Preserve lexical dominance but sweep saturation.
for c in [0.5,1.0,1.5,2.0,3.0,4.0,6.0]:
  for wl in [2.0,3.0,4.0,5.0,6.0,8.0]:
    for ws in [0.0,0.05,0.1,0.25]:
      rows.append(eval_formula(f'clip_c{c}_wl{wl}_ws{ws}', lambda i,k,c=c,wl=wl,ws=ws: np.clip(ZT[i,:k],-c,c)+wl*np.clip(ZL[i,:k],-c,c)+ws*np.clip(ZS[i,:k],-c,c)))

# 2) tanh saturation. scale controls saturation speed.
for s in [0.5,1.0,2.0,4.0]:
  for wl in [2.0,3.0,4.0,5.0,6.0,8.0]:
    for ws in [0.0,0.1,0.25]:
      rows.append(eval_formula(f'tanh_s{s}_wl{wl}_ws{ws}', lambda i,k,s=s,wl=wl,ws=ws: np.tanh(ZT[i,:k]/s)+wl*np.tanh(ZL[i,:k]/s)+ws*np.tanh(ZS[i,:k]/s)))

# 3) percentile/Borda fusion (tie-aware)
for wt in [0.25,0.5,1.0,2.0]:
  for wl in [1.0,2.0,4.0,8.0]:
    for ws in [0.0,0.1,0.25,0.5,1.0]:
      rows.append(eval_formula(f'percent_wt{wt}_wl{wl}_ws{ws}', lambda i,k,wt=wt,wl=wl,ws=ws: wt*PT[i,:k]+wl*PL[i,:k]+ws*PS[i,:k]))

# 4) Reciprocal rank fusion, tie-aware ranks.
for K in [10.,20.,50.,100.,200.,500.]:
  for wt in [0.5,1.0,2.0]:
    for wl in [1.0,2.0,4.0,8.0]:
      for ws in [0.0,0.1,0.25,0.5,1.0]:
        rows.append(eval_formula(f'rrf_K{int(K)}_wt{wt}_wl{wl}_ws{ws}', lambda i,k,K=K,wt=wt,wl=wl,ws=ws: wt/(K+RRT[i,:k])+wl/(K+RRL[i,:k])+ws/(K+RRS[i,:k])))

# 5) Agreement/product-like: weighted log percentile with epsilon, promotes candidates jointly strong.
for eps in [0.01,0.05,0.1,0.2]:
  for wt in [0.25,0.5,1.0]:
    for wl in [1.0,2.0,4.0]:
      rows.append(eval_formula(f'logrank_eps{eps}_wt{wt}_wl{wl}', lambda i,k,eps=eps,wt=wt,wl=wl: wt*np.log(eps+PT[i,:k])+wl*np.log(eps+PL[i,:k])+.1*np.log(eps+PS[i,:k])))

rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']), reverse=True)
out={'protocol':'fixed eta=1 P=2000 pools; deterministic 1000 TRAIN validation; no index/candidate changes; structural decision-rule sweep only','baseline':next(r for r in rows if r['name']=='baseline_z_1_4_.1'),'best':rows[0],'top25':rows[:25],'n_formulas':len(rows)}
json.dump(out,open(OUT/'structural_fusion_sweep.json','w'),indent=2)
print('BASE',out['baseline'])
print('BEST',out['best'])
print('TOP10')
for r in rows[:10]: print(r)
print('N',len(rows))
