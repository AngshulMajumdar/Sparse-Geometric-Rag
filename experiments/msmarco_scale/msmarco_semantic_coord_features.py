from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
import msmarco_best_tail_core as b
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; idx=b.idx; set_num_threads(5); KSEM=m.SEMK
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(np.int32); docs=z['docs']; texts=m.load_query_texts(qids)
@njit(cache=False)
def contains(ids,a,bb,t):
 lo=np.int64(a); hi=np.int64(bb)
 while lo<hi:
  md=(lo+hi)//2; x=int(ids[md])
  if x<t: lo=md+1
  else: hi=md
 return lo<bb and int(ids[lo])==t
@njit(parallel=True,cache=False)
def semcoord(dd,ip,ids,qterms,qamps,nbr,nw):
 n=len(dd); ssum=np.zeros(n,np.float32); qcount=np.zeros(n,np.float32); smax=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); total=0.; cnt=0.; mx=0.
  for qi in range(len(qterms)):
   local=0.
   for r in range(nbr.shape[1]):
    t=int(nbr[qi,r])
    if t<0: continue
    if contains(ids,a,bb,t): local += float(nw[qi,r])
   if local>0: cnt+=1.; mx=max(mx,local)
   total += local
  ssum[z]=total; qcount[z]=cnt; smax[z]=mx
 return ssum,qcount,smax
shape=docs.shape; SS=np.zeros(shape,np.float32); QC=np.zeros(shape,np.float32); MX=np.zeros(shape,np.float32); QT=np.zeros(len(qids),np.int16); times=[]; errs=[]
# warmup dummy
_=semcoord(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.array([0],np.int32),np.array([1],np.float32),np.full((1,KSEM),-1,np.int32),np.zeros((1,KSEM),np.float32))
for i,qid in enumerate(qids):
 k=int(valid[i]);
 if not k:continue
 q=idx.query_vec(texts[qid]); qt=q.indices.astype(np.int32); qa=q.data.astype(np.float32); QT[i]=len(qt); nbr=np.full((len(qt),KSEM),-1,np.int32); nw=np.zeros((len(qt),KSEM),np.float32)
 for u,(t,amp) in enumerate(zip(qt,qa)):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:KSEM]; sv=idx.A.data[a:bb][:KSEM]; l=len(nb); nbr[u,:l]=nb; nw[u,:l]=np.float32(amp)*sv*idx.idf[nb]
 t0=time.perf_counter(); ss,qc,mx=semcoord(docs[i,:k],idx.sup_ip,idx.sup_ids,qt,qa,nbr,nw); times.append((time.perf_counter()-t0)*1000); SS[i,:k]=ss; QC[i,:k]=qc; MX[i,:k]=mx; errs.append(float(np.max(np.abs(ss-z['sem'][i,:k]))))
 if (i+1)%200==0:print(i+1,'median',float(np.median(times)),'max_sem_err',max(errs),flush=True)
np.savez_compressed(OUT/'semantic_coord_features.npz',qids=np.asarray(qids),valid=valid,sem_sum=SS,sem_qcount=QC,sem_maxterm=MX,qterms=QT)
meta={'protocol':'fixed eta=1 P=2000 validation pools; semantic support decomposed by original query term using existing A graph and binary support only','median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'max_sem_reconstruction_error':max(errs)}; json.dump(meta,open(OUT/'semantic_coord_meta.json','w'),indent=2); print('DONE',meta,flush=True)
