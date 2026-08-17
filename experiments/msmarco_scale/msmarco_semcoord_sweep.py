from __future__ import annotations
import sys,json
import numpy as np
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); c=np.load(OUT/'coordination_features.npz',allow_pickle=False); s=np.load(OUT/'semantic_coord_features.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; RAW=c['rawlex']; LF=c['lenfac']; LQC=c['qcount']; LQT=c['qterms'].astype(int); SEM=s['sem_sum']; SQC=s['sem_qcount']; SQT=s['qterms'].astype(int); SMX=s['sem_maxterm']; qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
def Z(x):return m.zscore(x)
def top100(sc):
 n=len(sc); k=min(100,n); ii=np.argpartition(sc,-k)[-k:] if n>k else np.arange(n); return ii[np.argsort(sc[ii])[::-1]]
def lexical(i,k):
 ratio=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum(.9+.1*ratio,1e-6); cov=LQC[i,:k]/max(1,LQT[i]); return lx*np.power(np.maximum(cov,1e-6),.25)
def evalx(name,fn):
 run={}
 for i,qid in enumerate(qids):
  k=valid[i]; oo=top100(fn(i,k)); run[qid]=[int(x) for x in docs[i,oo]]
 r={'name':name,**m.eval_run(run,qrels)}; print(name,round(r['nDCG@10'],6),round(r['MRR@10'],6),round(r['R@100'],6),flush=True); return r
rows=[]; rows.append(evalx('base_sem_raw_.3',lambda i,k:Z(T[i,:k])+4*Z(lexical(i,k))+.3*Z(SEM[i,:k])))
# semantic coverage-adjusted sum
for a in [.125,.25,.5,1.,1.5,2.]:
 for ws in [.1,.2,.3,.5,.75,1.,1.5]:
  rows.append(evalx(f'semcoord_a{a}_w{ws}',lambda i,k,a=a,ws=ws: Z(T[i,:k])+4*Z(lexical(i,k))+ws*Z(SEM[i,:k]*np.power(np.maximum(SQC[i,:k]/max(1,SQT[i]),1e-6),a))))
# separate semantic coverage bonus
for ws in [.1,.2,.3,.5]:
 for wc in [-1.,-.5,-.25,.1,.25,.5,1.,2.]:
  rows.append(evalx(f'semraw_w{ws}_cov{wc}',lambda i,k,ws=ws,wc=wc: Z(T[i,:k])+4*Z(lexical(i,k))+ws*Z(SEM[i,:k])+wc*(SQC[i,:k]/max(1,SQT[i]))))
# max-per-query-term semantic evidence as extra confidence
for wm in [-.5,-.25,.1,.25,.5,1.]: rows.append(evalx(f'semmax_{wm}',lambda i,k,wm=wm: Z(T[i,:k])+4*Z(lexical(i,k))+.3*Z(SEM[i,:k])+wm*Z(SMX[i,:k])))
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); out={'protocol':'fixed eta=1 P=2000 TRAIN validation; final lexical structure b=.1 alpha=.25 wl4; semantic support decomposed by original query-term coverage using current graph only','baseline':next(r for r in rows if r['name']=='base_sem_raw_.3'),'best':rows[0],'top25':rows[:25],'n_models':len(rows)}; json.dump(out,open(OUT/'semantic_coord_sweep.json','w'),indent=2); print('BEST',rows[0]); print('TOP10'); [print(r) for r in rows[:10]]
