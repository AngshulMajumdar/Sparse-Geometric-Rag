from __future__ import annotations
import sys,json
import numpy as np
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); c=np.load(OUT/'coordination_features.npz',allow_pickle=False); g=np.load(OUT/'branch_features.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(int); docs=z['docs']; T=z['tail']; SM=z['sem']; RAW=c['rawlex']; LF=c['lenfac']; QC=c['qcount']; QT=c['qterms'].astype(int)
G=g['geom']; C=g['cons']; BC=g['branch_count']; GA=g['geom_abs']; GM=g['geom_max']; CM=g['cons_max']; PC=g['pos_count']; NC=g['neg_count']; qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
def Z(x):return m.zscore(x)
def top100(sc):
 n=len(sc); k=min(100,n); ii=np.argpartition(sc,-k)[-k:] if n>k else np.arange(n); return ii[np.argsort(sc[ii])[::-1]]
def common(i,k):
 ratio=(LF[i,:k]-.8)/.2; lx=RAW[i,:k]/np.maximum(.9+.1*ratio,1e-6); cov=QC[i,:k]/max(1,QT[i]); ladj=lx*np.power(np.maximum(cov,1e-6),.25); return 4*Z(ladj)+.3*Z(SM[i,:k])
def evalx(name,fn):
 run={}
 for i,qid in enumerate(qids):
  k=valid[i]; sc=fn(i,k); oo=top100(sc); run[qid]=[int(x) for x in docs[i,oo]]
 r={'name':name,**m.eval_run(run,qrels)}; print(name,round(r['nDCG@10'],6),round(r['MRR@10'],6),round(r['R@100'],6),flush=True); return r
rows=[]; rows.append(evalx('struct_base',lambda i,k:Z(T[i,:k])+common(i,k)))
# separate geometry and consensus
for wc in [-2.,-1.,-.5,-.25,0.,.0625,.125,.25,.5,1.,2.,4.]: rows.append(evalx(f'split_wc{wc}',lambda i,k,wc=wc: Z(G[i,:k])+wc*Z(C[i,:k])+common(i,k)))
# add extra consensus to current tail
for w in [-2.,-1.,-.5,-.25,.25,.5,1.,2.]: rows.append(evalx(f'base_plus_cons{w}',lambda i,k,w=w: Z(T[i,:k])+common(i,k)+w*Z(C[i,:k])))
# branch-count/diversity one-at-a-time
for nm,X in [('bc',BC),('gabs',GA),('gmax',GM),('cmax',CM)]:
 for w in [-1.,-.5,-.25,.25,.5,1.]: rows.append(evalx(f'base_{nm}_{w}',lambda i,k,w=w,X=X: Z(T[i,:k])+common(i,k)+w*Z(X[i,:k])))
# coherence and positive fraction, bounded structural signals
for w in [-2.,-1.,-.5,-.25,.25,.5,1.,2.]:
 rows.append(evalx(f'coherence_{w}',lambda i,k,w=w: Z(T[i,:k])+common(i,k)+w*(G[i,:k]/np.maximum(GA[i,:k],1e-6))))
 rows.append(evalx(f'posfrac_{w}',lambda i,k,w=w: Z(T[i,:k])+common(i,k)+w*(PC[i,:k]/np.maximum(PC[i,:k]+NC[i,:k],1.0))))
# concentration penalty/bonus: max branch / total absolute evidence
for w in [-1.,-.5,-.25,.25,.5,1.]: rows.append(evalx(f'concentration_{w}',lambda i,k,w=w: Z(T[i,:k])+common(i,k)+w*(GM[i,:k]/np.maximum(GA[i,:k],1e-6))))
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); out={'protocol':'fixed eta=1 P=2000 TRAIN validation; branch-level structure from current index only; structural lexical base b=.1 alpha=.25 wl4 ws=.3','baseline':next(r for r in rows if r['name']=='struct_base'),'best':rows[0],'top25':rows[:25],'n_models':len(rows)}; json.dump(out,open(OUT/'branch_fusion_sweep.json','w'),indent=2); print('BEST',rows[0]); print('TOP10'); [print(r) for r in rows[:10]]
