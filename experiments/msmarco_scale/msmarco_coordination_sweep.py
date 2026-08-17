from __future__ import annotations
import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True)
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); c=np.load(OUT/'coordination_features.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; L=z['lex']; S=z['sem']; QC=c['qcount']; RAW=c['rawlex']; LF=c['lenfac']; QT=c['qterms'].astype(int)
qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)

def top100(sc):
 n=len(sc); kk=min(100,n)
 if n<=kk:return np.argsort(sc)[::-1]
 ii=np.argpartition(sc,-kk)[-kk:]; return ii[np.argsort(sc[ii])[::-1]]

def eval_model(name, scorer):
 run={}
 for i,qid in enumerate(qids):
  k=valid[i]
  if k<=0: run[qid]=[]; continue
  sc=scorer(i,k); oo=top100(sc); run[qid]=[int(x) for x in docs[i,oo]]
 met=m.eval_run(run,qrels); row={'name':name,**met}; print(name,round(met['nDCG@10'],6),round(met['MRR@10'],6),round(met['R@100'],6),flush=True); return row

def Z(x): return m.zscore(x)
rows=[]
rows.append(eval_model('baseline',lambda i,k:Z(T[i,:k])+4*Z(L[i,:k])+.1*Z(S[i,:k])))
# recover doc length ratio from current b=.2 denominator LF=.8+.2*r
# stage 1: length correction b only, current fusion weights
for bb in [0.,.05,.1,.15,.2,.3,.4,.5,.75,1.]:
 def f(i,k,bb=bb):
  r=(LF[i,:k]-.8)/.2; den=(1-bb)+bb*r; lx=RAW[i,:k]/np.maximum(den,1e-6); return Z(T[i,:k])+4*Z(lx)+.1*Z(S[i,:k])
 rows.append(eval_model(f'length_b{bb}',f))
# choose best length b by nDCG
lenrows=[r for r in rows if r['name'].startswith('length_b')]; bestb=float(max(lenrows,key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']))['name'].split('length_b')[1]); print('BEST_B',bestb,flush=True)
# stage 2: weights at best b
for wl in [2.,3.,4.,5.,6.,8.,10.,12.]:
 for ws in [0.,.05,.1,.2,.3]:
  def f(i,k,bb=bestb,wl=wl,ws=ws):
   r=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum((1-bb)+bb*r,1e-6); return Z(T[i,:k])+wl*Z(lx)+ws*Z(S[i,:k])
  rows.append(eval_model(f'bestb_wl{wl}_ws{ws}',f))
weightrows=[r for r in rows if r['name'].startswith('bestb_')]; bestw=max(weightrows,key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100'])); import re
mt=re.search(r'wl([0-9.]+)_ws([0-9.]+)',bestw['name']); bestwl=float(mt.group(1)); bestws=float(mt.group(2)); print('BEST_W',bestwl,bestws,flush=True)
# stage 3: add coordination count z-score
for wc in [-2.,-1.,-.5,-.25,0.,.125,.25,.5,1.,2.,4.]:
 def f(i,k,wc=wc,bb=bestb,wl=bestwl,ws=bestws):
  r=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum((1-bb)+bb*r,1e-6); return Z(T[i,:k])+wl*Z(lx)+ws*Z(S[i,:k])+wc*Z(QC[i,:k])
 rows.append(eval_model(f'coord_wc{wc}',f))
# stage 4: coordination-adjusted lexical score lx*(coverage)^alpha; keep weights and also tune lex weight modestly
for alpha in [.125,.25,.5,1.,1.5,2.]:
 for wl in [bestwl*.75,bestwl,bestwl*1.25]:
  def f(i,k,alpha=alpha,wl=wl,bb=bestb,ws=bestws):
   r=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum((1-bb)+bb*r,1e-6); cov=QC[i,:k]/max(1,QT[i]); ladj=lx*np.power(np.maximum(cov,1e-6),alpha); return Z(T[i,:k])+wl*Z(ladj)+ws*Z(S[i,:k])
  rows.append(eval_model(f'coordlex_a{alpha}_wl{wl}',f))
# stage 5: exact all-query-terms and high-coverage bonuses (z of binary masks)
for thr in [.5,.67,.75,.8,1.0]:
 for wb in [.125,.25,.5,1.,2.]:
  def f(i,k,thr=thr,wb=wb,bb=bestb,wl=bestwl,ws=bestws):
   r=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum((1-bb)+bb*r,1e-6); cov=QC[i,:k]/max(1,QT[i]); bonus=(cov>=thr).astype(np.float32); return Z(T[i,:k])+wl*Z(lx)+ws*Z(S[i,:k])+wb*bonus
  rows.append(eval_model(f'covbonus_thr{thr}_wb{wb}',f))
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'same fixed eta=1 P=2000 TRAIN validation pools; only existing-index lexical/coordination structure; no candidate changes','baseline':next(r for r in rows if r['name']=='baseline'),'best':rows[0],'top30':rows[:30],'best_length_b':bestb,'best_weight_base':bestw,'n_models':len(rows)}
json.dump(out,open(OUT/'coordination_sweep.json','w'),indent=2); print('===BEST==='); print(json.dumps(rows[0],indent=2)); print('TOP10'); [print(r) for r in rows[:10]]
