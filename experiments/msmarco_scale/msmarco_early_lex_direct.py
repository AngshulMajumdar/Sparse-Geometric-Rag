from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
LLEX=np.float32(4.0); LSEM=np.float32(0.1)
ETAS=[0.0,0.125,0.25,0.5,1.0,2.0,4.0]
set_num_threads(5)

def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if n>k:
  ii=np.argpartition(score,-k)[-k:]
  return ii[np.argsort(score[ii])[::-1]]
 return np.argsort(score)[::-1]

def prepare_all(text):
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
 spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans:return None
 docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False)
 mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False)
 rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]
  ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)])
 base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
 ud,inv=np.unique(docs,return_inverse=True)
 tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 lex,sem=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
 return ud,tail,lex,sem

def rank(ud,tail,lex,sem,sel):
 fin=m.zscore(tail[sel])+LLEX*m.zscore(lex[sel])+LSEM*m.zscore(sem[sel]); oo=np.argsort(fin)[::-1][:100]; return [int(x) for x in ud[sel[oo]]]

tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
_=prepare_all(texts[ids[0]])
runs={e:{} for e in ETAS}; pool={e:0 for e in ETAS}; den=route=0; times=[]; strategy_times=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); p=prepare_all(texts[qid]); times.append((time.perf_counter()-t)*1000)
 if p is None:
  for e in ETAS:runs[e][qid]=[]
  continue
 ud,tail,lex,sem=p; rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels); relset=set()
 for d in rels:
  kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; route+=int(ok)
  if ok:relset.add(int(kk))
 zt=m.zscore(tail); zl=m.zscore(lex)
 st=time.perf_counter()
 for e in ETAS:
  sel=topk_desc(zt+np.float32(e)*zl,P); pool[e]+=sum(int(i) in relset for i in sel); runs[e][qid]=rank(ud,tail,lex,sem,sel)
 strategy_times.append((time.perf_counter()-st)*1000)
 if (z+1)%100==0:print('q',z+1,'prepare',float(np.median(times)),'strategies',float(np.median(strategy_times)),flush=True)
rows=[]
for e in ETAS:
 met=m.eval_run(runs[e],qrels); row={'eta':e,'pool_relevant_recall':pool[e]/den,**met}; rows.append(row); print('DIRECT',row,flush=True)
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'rows':rows,'best':rows[0],'route_relevant_recall':route/den,'timing':{'median_prepare_alllexsem_ms':float(np.median(times)),'p95_prepare_ms':float(np.percentile(times,95)),'median_all_eta_strategy_ms':float(np.median(strategy_times))}}
json.dump(out,open(WORK/'early_lex_direct_validation.json','w'),indent=2); print('BEST',rows[0]); print('TIMING',out['timing'])
