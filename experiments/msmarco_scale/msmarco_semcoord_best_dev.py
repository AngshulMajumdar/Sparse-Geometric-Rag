from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
ETA=np.float32(1.0); FINAL_B=np.float32(0.1); LEX_ALPHA=np.float32(0.25); WLEX=np.float32(4.0); SEM_ALPHA=np.float32(0.5); WSEM=np.float32(1.0); KSEM=m.SEMK
set_num_threads(5)
def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if n<=k:return np.argsort(score)[::-1]
 ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]
@njit(parallel=True,cache=False)
def selected_lex_features(dd,ip,ids,lexvec,dl,avgdl):
 n=len(dd); lx=np.zeros(n,np.float32); qc=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.0; c=0.0
  for k in range(a,bb):
   t=int(ids[k]); v=lexvec[t]
   if v>0: raw+=v; c+=1.0
  ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
  lx[z]=raw/(den if den>0 else 1.0); qc[z]=c
 return lx,qc
@njit(cache=False)
def contains(ids,a,bb,t):
 lo=np.int64(a); hi=np.int64(bb)
 while lo<hi:
  md=(lo+hi)//2; x=int(ids[md])
  if x<t: lo=md+1
  else: hi=md
 return lo<bb and int(ids[lo])==t
@njit(parallel=True,cache=False)
def semcoord(dd,ip,ids,nbr,nw):
 n=len(dd); ssum=np.zeros(n,np.float32); qcount=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); total=0.0; cnt=0.0
  for qi in range(nbr.shape[0]):
   local=0.0
   for r in range(nbr.shape[1]):
    t=int(nbr[qi,r])
    if t<0: continue
    if contains(ids,a,bb,t): local += float(nw[qi,r])
   if local>0: cnt+=1.0
   total += local
  ssum[z]=total; qcount[z]=cnt
 return ssum,qcount
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
 lx,qc=selected_lex_features(dd,idx.sup_ip,idx.sup_ids,lexvec,idx.dl,idx.avgdl); lcov=qc/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(lcov,1e-6),LEX_ALPHA)
 nbr=np.full((len(q.indices),KSEM),-1,np.int32); nw=np.zeros((len(q.indices),KSEM),np.float32)
 for u,(t,amp) in enumerate(zip(q.indices,q.data)):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:KSEM]; sv=idx.A.data[a:bb][:KSEM]; l=len(nb); nbr[u,:l]=nb; nw[u,:l]=np.float32(amp)*sv*idx.idf[nb]
 ssum,sqc=semcoord(dd,idx.sup_ip,idx.sup_ids,nbr,nw); scov=sqc/max(1,len(q.indices)); sadj=ssum*np.power(np.maximum(scov,1e-6),SEM_ALPHA)
 fin=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sadj); oo=np.argsort(fin)[::-1][:100]
 return [int(x) for x in dd[oo]],ud,sel
_=selected_lex_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),idx.dl,idx.avgdl); _=semcoord(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.full((1,KSEM),-1,np.int32),np.zeros((1,KSEM),np.float32))
df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True); _=prepare(texts[ids[0]])
run={}; times=[]; routehit=poolhit=den=0; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); out=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
 if out is None: run[qid]=[]; continue
 rank,ud,sel=out; run[qid]=rank; cands.append(len(ud)); rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels); pooldocs=set(map(int,ud[sel].tolist()))
 for d in rels:
  kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; routehit+=int(ok); poolhit+=int(ok and d in pooldocs)
 if (z+1)%500==0: print('dev',z+1,'median',float(np.median(times)),'route',routehit/max(1,den),'pool',poolhit/max(1,den),flush=True)
met=m.eval_run(run,qrels); result={'protocol':'semantic-coordination rule locked on deterministic 1000 TRAIN validation; untouched full DEV','params':{'preselection_eta':1.0,'preselection_length_b':0.2,'P':2000,'gamma_tail':0.25,'lambda_M':0.125,'final_length_b':0.1,'lex_coord_alpha':0.25,'lambda_lex':4.0,'sem_coord_alpha':0.5,'lambda_sem_coord':1.0,'h':0,'S':16},'dev_metrics':met,'route_relevant_recall':routehit/den,'pool_relevant_recall':poolhit/den,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))}}
out=WORK/'semcoord_best_dev_results.json'; json.dump(result,open(out,'w'),indent=2); print('RESULT',json.dumps(result,indent=2),flush=True); print('SAVED',out,flush=True)
