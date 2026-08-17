from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
ETA=np.float32(1.0); QUOTA=500; LLEX=np.float32(4.0); LSEM=np.float32(0.1)
set_num_threads(5)

def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if k<=0:return np.empty(0,np.int64)
 if n>k:
  ii=np.argpartition(score,-k)[-k:]
  return ii[np.argsort(score[ii])[::-1]]
 return np.argsort(score)[::-1]

def quota_select(tail,lex,lq=500):
 n=len(tail); k=min(P,n); gq=k-min(lq,k); gt=topk_desc(tail,gq)
 if gq==k:return gt
 lex_order=topk_desc(lex,min(n,2*P)); chosen=np.zeros(n,np.uint8); chosen[gt]=1; out=np.empty(k,np.int64); out[:gq]=gt; z=gq
 for ii in lex_order:
  if chosen[ii]==0:
   chosen[ii]=1; out[z]=ii; z+=1
   if z==k:return out
 # fallback
 for ii in np.argsort(lex)[::-1]:
  if chosen[ii]==0:
   out[z]=ii; z+=1
   if z==k:return out
 return out[:z]

def prepare_geometry_lex(text):
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
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32)
 lex,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl)
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 return {'ud':ud,'tail':tail,'lex':lex,'semvec':semvec,'candidate_memberships':len(docs)}

def rank_selected(p,sel):
 docs=p['ud'][sel]; ts=p['tail'][sel]; lx=p['lex'][sel]; zero=np.zeros(M,np.float32)
 _,sem=m.score_support_pool(docs,idx.sup_ip,idx.sup_ids,zero,p['semvec'],idx.dl,idx.avgdl)
 fin=m.zscore(ts)+LLEX*m.zscore(lx)+LSEM*m.zscore(sem); oo=np.argsort(fin)[::-1][:100]
 return [int(x) for x in docs[oo]]

def select_direct(p):return topk_desc(m.zscore(p['tail'])+ETA*m.zscore(p['lex']),P)

df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True)
p=prepare_geometry_lex(texts[ids[0]]); sd=select_direct(p); _=rank_selected(p,sd); del p
runD={}; runQ={}; timesD=[]; routehit=poolD=poolQ=den=0; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); p=prepare_geometry_lex(texts[qid])
 if p is None: runD[qid]=[]; runQ[qid]=[]; continue
 sd=select_direct(p); rd=rank_selected(p,sd); timesD.append((time.perf_counter()-t)*1000); runD[qid]=rd
 sq=quota_select(p['tail'],p['lex'],QUOTA); runQ[qid]=rank_selected(p,sq)
 ud=p['ud']; cands.append(len(ud)); rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels); setD=set(map(int,sd)); setQ=set(map(int,sq))
 for d in rels:
  kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; routehit+=int(ok)
  if ok: poolD+=int(int(kk) in setD); poolQ+=int(int(kk) in setQ)
 if (z+1)%500==0:print('dev',z+1,'median_direct_ms',float(np.median(timesD)),'route',routehit/max(1,den),'poolD',poolD/max(1,den),'poolQ',poolQ/max(1,den),flush=True)
metD=m.eval_run(runD,qrels); metQ=m.eval_run(runQ,qrels)
out={'protocol':'shortlist strategy locked on deterministic TRAIN validation; DEV untouched','selected':{'strategy':'direct early lexical fusion','eta':1.0,'P':2000,'gamma_tail':0.25,'lambda_M':0.125,'lambda_lex_final':4.0,'lambda_sem_final':0.1,'h':0},'secondary_validation_fixed_comparator':{'strategy':'quota rescue','lex_quota':500},'direct_dev_metrics':metD,'quota500_dev_metrics':metQ,'route_relevant_recall':routehit/den,'direct_pool_relevant_recall':poolD/den,'quota_pool_relevant_recall':poolQ/den,'direct_timing':{'median_ms':float(np.median(timesD)),'p95_ms':float(np.percentile(timesD,95)),'mean_ms':float(np.mean(timesD)),'qps':1000/float(np.mean(timesD)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(out,open(WORK/'early_lex_dev_results.json','w'),indent=2); print('DIRECT_DEV',metD,flush=True); print('QUOTA_DEV',metQ,flush=True); print('SUMMARY',out,flush=True)
