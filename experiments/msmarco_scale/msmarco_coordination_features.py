from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True); set_num_threads(5)
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(np.int32); docs=z['docs']; idx=m.FullIndex(); M=m.M
texts=m.load_query_texts(qids)
@njit(parallel=True,cache=False)
def extras(dd,ip,ids,lexvec,dl,avgdl):
 n=len(dd); cnt=np.zeros(n,np.float32); raw=np.zeros(n,np.float32); lf=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); b=int(ip[d+1]); c=0.0; r=0.0
  for k in range(a,b):
   t=int(ids[k]); v=lexvec[t]
   if v>0: c+=1.; r+=v
  cnt[z]=c; raw[z]=r; lf[z]=(1.0-m.LENGTH_B)+m.LENGTH_B*(float(dl[d])/avgdl)
 return cnt,raw,lf
_=extras(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),idx.dl,idx.avgdl)
COUNT=np.zeros_like(z['tail'],np.float32); RAW=np.zeros_like(COUNT); LF=np.zeros_like(COUNT); QTERMS=np.zeros(len(qids),np.int16); times=[]
for i,qid in enumerate(qids):
 k=int(valid[i]);
 if not k:continue
 q=idx.query_vec(texts[qid]); QTERMS[i]=len(q.indices); lv=np.zeros(M,np.float32); lv[q.indices]=idx.idf[q.indices]
 t=time.perf_counter(); c,r,l=extras(docs[i,:k],idx.sup_ip,idx.sup_ids,lv,idx.dl,idx.avgdl); times.append((time.perf_counter()-t)*1000); COUNT[i,:k]=c; RAW[i,:k]=r; LF[i,:k]=l
 if (i+1)%200==0: print(i+1,float(np.median(times)),flush=True)
np.savez_compressed(OUT/'coordination_features.npz',qids=np.asarray(qids),valid=valid,qcount=COUNT,rawlex=RAW,lenfac=LF,qterms=QTERMS)
print('DONE',float(np.median(times)),float(np.percentile(times,95)),flush=True)
