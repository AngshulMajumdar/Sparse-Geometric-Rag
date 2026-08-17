from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_early_lex_validation_fast as e
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=e.idx; P=2000; set_num_threads(5)
BS=[0.,.05,.1,.15,.2,.3,.4]; ETAS=[.125,.25,.5,.75,1.,1.5,2.,3.,4.]
def topk(score,k):
 n=len(score); k=min(k,n); ii=np.argpartition(score,-k)[-k:] if n>k else np.arange(n); return ii
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr
rng=np.random.default_rng(20260815); qids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; texts=m.load_query_texts(qids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
hit={(b,eta):0 for b in BS for eta in ETAS}; den=route=0; times=[]
_=e.prepare_all(texts[qids[0]])
for qi,qid in enumerate(qids):
 t=time.perf_counter(); p=e.prepare_all(texts[qid]); times.append((time.perf_counter()-t)*1000)
 rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
 if p is None: continue
 ud=p['ud']; relidx=[]
 for d in rels:
  k=np.searchsorted(ud,d); ok=k<len(ud) and int(ud[k])==d; route+=int(ok)
  if ok: relidx.append(int(k))
 if not relidx: continue
 ratio=np.asarray(idx.dl[ud],np.float32)/np.float32(idx.avgdl); raw=p['lex']*((1-.2)+.2*ratio); zt=m.zscore(p['tail'])
 for bb in BS:
  lb=raw/np.maximum((1-bb)+bb*ratio,1e-6); zl=m.zscore(lb)
  for eta in ETAS:
   sel=topk(zt+np.float32(eta)*zl,P); ss=set(map(int,sel.tolist())); hit[(bb,eta)]+=sum(r in ss for r in relidx)
 if (qi+1)%100==0: print('q',qi+1,'median',float(np.median(times)),'route',route/den,flush=True)
rows=[{'pre_b':bb,'eta':eta,'pool_relevant_recall':hit[(bb,eta)]/den} for bb in BS for eta in ETAS]
rows.sort(key=lambda r:r['pool_relevant_recall'],reverse=True); out={'protocol':'deterministic 1000 TRAIN validation; same routed candidates/tail, only preselection lexical length penalty b and eta varied; objective top-P relevant survival','route_relevant_recall':route/den,'rows':rows,'best':rows[0],'baseline':next(r for r in rows if r['pre_b']==.2 and r['eta']==1.),'median_prepare_ms':float(np.median(times))}; json.dump(out,open(WORK/'preselection_length_eta_sweep.json','w'),indent=2); print('BEST',rows[0]); print('BASE',out['baseline']); print('TOP15'); [print(r) for r in rows[:15]]
