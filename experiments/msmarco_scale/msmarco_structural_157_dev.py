from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
ETA=np.float32(1.0); FINAL_B=np.float32(0.1); ALPHA=np.float32(0.25); WLEX=np.float32(5.0); WSEM=np.float32(0.75)
set_num_threads(5)

def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if n<=k:return np.argsort(score)[::-1]
 ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

@njit(parallel=True,cache=False)
def selected_final_features(dd,ip,ids,lexvec,semvec,dl,avgdl):
 n=len(dd); lx=np.zeros(n,np.float32); sm=np.zeros(n,np.float32); qc=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.0; s=0.0; c=0.0
  for k in range(a,bb):
   t=int(ids[k]); v=lexvec[t]
   if v>0: raw+=v; c+=1.0
   s+=semvec[t]
  ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
  lx[z]=raw/(den if den>0 else 1.0); sm[z]=s; qc[z]=c
 return lx,sm,qc

def prepare(text):
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
 spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans:return None
 docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False); mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False); rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False); sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]; ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)]); base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
 ud,inv=np.unique(docs,return_inverse=True); tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32); oldlex,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl)
 sel=topk_desc(m.zscore(tail)+ETA*m.zscore(oldlex),P); dd=ud[sel]; ts=tail[sel]
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 lx,sm,qc=selected_final_features(dd,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
 cov=qc/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA)
 fin=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sm); oo=np.argsort(fin)[::-1][:100]
 return [int(x) for x in dd[oo]],ud,sel

_=selected_final_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),np.zeros(M,np.float32),idx.dl,idx.avgdl)
df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True)
_=prepare(texts[ids[0]])
run={}; times=[]; routehit=poolhit=den=0; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); out=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
 if out is None: run[qid]=[]; continue
 rank,ud,sel=out; run[qid]=rank; cands.append(len(ud)); rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels); pooldocs=set(map(int,ud[sel].tolist()))
 for d in rels:
  kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; routehit+=int(ok); poolhit+=int(ok and d in pooldocs)
 if (z+1)%500==0: print('dev',z+1,'median',float(np.median(times)),'route',routehit/max(1,den),'pool',poolhit/max(1,den),flush=True)
met=m.eval_run(run,qrels); result={'protocol':'structural final rule locked entirely on deterministic 1000 TRAIN validation; untouched full DEV','params':{'preselection_eta':1.0,'P':2000,'gamma_tail':0.25,'lambda_M':0.125,'final_length_b':0.1,'coordination_alpha':0.25,'lambda_lex':4.0,'lambda_sem':0.3,'h':0,'S':16},'dev_metrics':met,'route_relevant_recall':routehit/den,'pool_relevant_recall':poolhit/den,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(result,open(WORK/'structural_157_dev_results.json','w'),indent=2); print('RESULT',json.dumps(result,indent=2),flush=True)
