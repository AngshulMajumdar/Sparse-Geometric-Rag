from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
import msmarco_best_tail_core as b
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True); idx=b.idx; set_num_threads(5); M=m.M; S=m.S
z=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(np.int32); docs=z['docs']; texts=m.load_query_texts(qids)
@njit(cache=False)
def find_doc(pd,a,bb,d):
 lo=np.int64(a); hi=np.int64(bb)
 while lo<hi:
  md=(lo+hi)//2; x=int(pd[md])
  if x<d: lo=md+1
  else: hi=md
 if lo<bb and int(pd[lo])==d:return lo
 return -1
@njit(parallel=True,cache=False)
def pool_features(dd,rterms,rd,offs,pd,pm,pr,ps,qd,ct,cv,rp,ri,rv):
 n=len(dd); geom=np.zeros(n,np.float32); cons=np.zeros(n,np.float32); bc=np.zeros(n,np.float32); gabs=np.zeros(n,np.float32); gmax=np.zeros(n,np.float32); cmax=np.zeros(n,np.float32); pos=np.zeros(n,np.float32); neg=np.zeros(n,np.float32)
 for zz in prange(n):
  d=int(dd[zz]); gs=0.; cs=0.; cnt=0.; ab=0.; mx=-1e30; cm=0.; pp=0.; nn=0.
  for jj in range(len(rterms)):
   j=int(rterms[jj]); a=int(offs[j]); bb=int(offs[j+1]); p=find_doc(pd,a,bb,d)
   if p<0: continue
   cnt+=1.; c=float(pm[p])*float(rd[j]); local=0.; sig=0.; bits=int(ps[p])
   for r in range(S):
    t=int(pr[p,r])
    if t==65535: continue
    qv=float(qd[t]); cen=float(m.lookup_center(ct[j],cv[j],t)); rel=float(m.lookup_rel(rp,ri,rv,j,t)); sgn=1. if ((bits>>r)&1) else -1.
    local += rel*(qv-cen)*sgn; sig += qv*qv
   g=c*local*(sig**0.25 if sig>0 else 0.)
   gs+=g; cs+=c; ab+=abs(g); mx=max(mx,g); cm=max(cm,c); pp+=1. if g>0 else 0.; nn+=1. if g<0 else 0.
  geom[zz]=gs; cons[zz]=cs; bc[zz]=cnt; gabs[zz]=ab; gmax[zz]=0 if mx<-1e20 else mx; cmax[zz]=cm; pos[zz]=pp; neg[zz]=nn
 return geom,cons,bc,gabs,gmax,cmax,pos,neg
# warmup
q=idx.query_vec(texts[qids[0]]); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rt,rd=idx.route(q); _=pool_features(docs[0,:1],rt,rd,idx.offs,idx.pd,idx.pm,idx.pr,idx.ps,qd,idx.ct,idx.cv,idx.rp,idx.ri,idx.rv)
shape=docs.shape; names=['geom','cons','branch_count','geom_abs','geom_max','cons_max','pos_count','neg_count']; arr={n:np.zeros(shape,np.float32) for n in names}; times=[]; errs=[]
for i,qid in enumerate(qids):
 k=int(valid[i]);
 if not k:continue
 q=idx.query_vec(texts[qid]); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rt,rd=idx.route(q); t=time.perf_counter(); vals=pool_features(docs[i,:k],rt,rd,idx.offs,idx.pd,idx.pm,idx.pr,idx.ps,qd,idx.ct,idx.cv,idx.rp,idx.ri,idx.rv); times.append((time.perf_counter()-t)*1000)
 for nm,v in zip(names,vals): arr[nm][i,:k]=v
 # reconstructed tail consistency
 recon=vals[0]+.125*vals[1]; errs.append(float(np.max(np.abs(recon-z['tail'][i,:k]))))
 if (i+1)%200==0:print(i+1,'median_ms',float(np.median(times)),'max_tail_err',max(errs),flush=True)
np.savez_compressed(OUT/'branch_features.npz',qids=np.asarray(qids),valid=valid,**arr)
meta={'protocol':'fixed eta=1 P=2000 validation pools; branch-level features recovered from current sorted branch postings only','features':names,'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'max_tail_reconstruction_error':max(errs)}; json.dump(meta,open(OUT/'branch_feature_meta.json','w'),indent=2); print('DONE',meta,flush=True)
