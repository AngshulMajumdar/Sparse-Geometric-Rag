from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
LLEX=np.float32(4.0); LSEM=np.float32(0.1)
ETAS=[0.0,0.0625,0.125,0.25,0.5,1.0,2.0,4.0,8.0]
LEX_QUOTAS=[0,100,250,500,750,1000,1250,1500,1750,2000]
set_num_threads(5)

def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if k<=0:return np.empty(0,np.int64)
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
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 # One CSR scan per routed doc supplies both validation features.
 lex,sem=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
 return {'ud':ud,'tail':tail,'lex':lex,'sem':sem,'candidate_docs':len(ud),'candidate_memberships':len(docs)}

def final_rank(p,sel,k=100):
 docs=p['ud'][sel]; fin=m.zscore(p['tail'][sel])+LLEX*m.zscore(p['lex'][sel])+LSEM*m.zscore(p['sem'][sel]); oo=np.argsort(fin)[::-1][:k]; return [int(x) for x in docs[oo]]

def quota_from_orders(tail_order,lex_order,n,lq):
 k=min(P,n); gq=max(0,k-min(lq,k)); out=np.empty(k,np.int64); z=0; chosen=np.zeros(n,np.uint8)
 if gq:
  g=tail_order[:gq]; out[:gq]=g; chosen[g]=1; z=gq
  if z==k: return out
 for ii in lex_order:
  if chosen[ii]==0:
   chosen[ii]=1; out[z]=ii; z+=1
   if z==k:break
 return out[:z]

tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
p=prepare_all(texts[ids[0]]); _=final_rank(p,topk_desc(p['tail'],P)); del p
runsD={e:{} for e in ETAS}; runsQ={q:{} for q in LEX_QUOTAS}; poolD={e:0 for e in ETAS}; poolQ={q:0 for q in LEX_QUOTAS}; den=routehit=0; times=[]; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); p=prepare_all(texts[qid]); times.append((time.perf_counter()-t)*1000)
 rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
 if p is None:
  for e in ETAS:runsD[e][qid]=[]
  for qv in LEX_QUOTAS:runsQ[qv][qid]=[]
  continue
 cands.append(p['candidate_docs']); ud=p['ud']; relset=set()
 for d in rels:
  kk=np.searchsorted(ud,d); ok=(kk<len(ud) and int(ud[kk])==d); routehit+=int(ok)
  if ok: relset.add(int(kk))
 zt=m.zscore(p['tail']); zl=m.zscore(p['lex'])
 for e in ETAS:
  sel=topk_desc(zt+np.float32(e)*zl,P); poolD[e]+=sum(int(i) in relset for i in sel); runsD[e][qid]=final_rank(p,sel)
 tail_order=topk_desc(p['tail'],P); lex_order=np.argsort(p['lex'])[::-1]
 for qv in LEX_QUOTAS:
  sel=quota_from_orders(tail_order,lex_order,len(ud),qv); poolQ[qv]+=sum(int(i) in relset for i in sel); runsQ[qv][qid]=final_rank(p,sel)
 if (z+1)%100==0:print('q',z+1,'median_prepare',float(np.median(times)),'route',routehit/den,flush=True)
rowsD=[]
for e in ETAS:
 met=m.eval_run(runsD[e],qrels); row={'eta':e,'pool_relevant_recall':poolD[e]/den,**met}; rowsD.append(row); print('DIRECT',row,flush=True)
rowsQ=[]
for qv in LEX_QUOTAS:
 met=m.eval_run(runsQ[qv],qrels); row={'lex_quota':qv,'pool_relevant_recall':poolQ[qv]/den,**met}; rowsQ.append(row); print('QUOTA',row,flush=True)
rowsD.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); rowsQ.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'same deterministic 1000 TRAIN validation; gamma=.25 lambdaM=.125 S=16 P=2000 h=0; final lambda_lex=4 lambda_sem=.1 locked; lexical support injected before P selection','direct_rows':rowsD,'quota_rows':rowsQ,'best_direct':rowsD[0],'best_quota':rowsQ[0],'route_relevant_recall':routehit/den,'timing_validation_amortized':{'median_prepare_ms':float(np.median(times)),'p95_prepare_ms':float(np.percentile(times,95)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(out,open(WORK/'early_lex_validation.json','w'),indent=2); print('BEST_DIRECT',rowsD[0],flush=True); print('BEST_QUOTA',rowsQ[0],flush=True); print('TIMING',out['timing_validation_amortized'],flush=True)
