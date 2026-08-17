from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
OUT=WORK/'structural_fusion'; OUT.mkdir(exist_ok=True)
set_num_threads(5)

@njit(parallel=True,cache=False)
def support_features(cand_docs,ip,ids,lexvec,semvec,dl,avgdl):
 n=len(cand_docs); lex=np.zeros(n,np.float32); rawlex=np.zeros(n,np.float32); sem=np.zeros(n,np.float32); qcount=np.zeros(n,np.uint8); lenfac=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(cand_docs[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.0; sm=0.0; cnt=0
  for k in range(a,bb):
   t=int(ids[k]); v=lexvec[t]
   if v>0:
    raw += v; cnt += 1
   sm += semvec[t]
  denom=(1.0-m.LENGTH_B)+m.LENGTH_B*(float(dl[d])/avgdl)
  rawlex[z]=raw; lex[z]=raw/(denom if denom>0 else 1.0); sem[z]=sm; qcount[z]=cnt; lenfac[z]=denom
 return lex,rawlex,sem,qcount,lenfac

def topk(score,k):
 n=len(score); k=min(k,n)
 if n<=k:return np.argsort(score)[::-1]
 ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

# exact validation IDs
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr
rng=np.random.default_rng(20260815); qids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]
texts=m.load_query_texts(qids)
# allocate fixed P arrays
shape=(len(qids),P)
docsO=np.zeros(shape,np.uint32); valid=np.zeros(len(qids),np.int32)
features={k:np.zeros(shape,dtype) for k,dtype in [
 ('geom',np.float32),('cons',np.float32),('tail',np.float32),('lex',np.float32),('rawlex',np.float32),('sem',np.float32),('qcount',np.float32),('lenfac',np.float32),('branch_count',np.float32),('max_cons',np.float32),('max_geom',np.float32)]}
qterms=np.zeros(len(qids),np.int16)
prep=[]
# warmup
_=support_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),np.zeros(M,np.float32),idx.dl,idx.avgdl)
for qi,qid in enumerate(qids):
 t0=time.perf_counter(); text=texts[qid]
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; qterms[qi]=len(q.indices)
 rterms,rd=idx.route(q)
 spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans: continue
 dmem=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False)
 mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False)
 rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]
  ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)])
 base,sig,consmem=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
 ud,inv=np.unique(dmem,return_inverse=True)
 geom=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)
 cons=np.bincount(inv,weights=consmem,minlength=len(ud)).astype(np.float32)
 tail=geom+b.LAM*cons
 bc=np.bincount(inv,minlength=len(ud)).astype(np.float32)
 # max per-document membership-level contributions, preserved as structural evidence
 mxcon=np.full(len(ud),-np.inf,np.float32); np.maximum.at(mxcon,inv,consmem); mxcon[~np.isfinite(mxcon)]=0
 memgeom=base*np.power(sig,b.GAMMA,dtype=np.float32); mxg=np.full(len(ud),-np.inf,np.float32); np.maximum.at(mxg,inv,memgeom); mxg[~np.isfinite(mxg)]=0
 # support features for all routed docs, needed for eta=1 selection
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]
 semvec=np.zeros(M,np.float32)
 for tt,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[tt],idx.A.indptr[tt+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 lx,rlx,sm,qc,lf=support_features(ud,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
 sel=topk(m.zscore(tail)+m.zscore(lx),P) # locked eta=1
 k=len(sel); valid[qi]=k; docsO[qi,:k]=ud[sel]
 for name,arr in [('geom',geom),('cons',cons),('tail',tail),('lex',lx),('rawlex',rlx),('sem',sm),('qcount',qc.astype(np.float32)),('lenfac',lf),('branch_count',bc),('max_cons',mxcon),('max_geom',mxg)]: features[name][qi,:k]=arr[sel]
 prep.append((time.perf_counter()-t0)*1000)
 if (qi+1)%100==0: print('q',qi+1,'median',float(np.median(prep)),flush=True)

save={'qids':np.asarray(qids),'valid':valid,'docs':docsO,'qterms':qterms}|features
np.savez_compressed(OUT/'fixed_eta1_structural_features.npz',**save)
meta={'protocol':'same deterministic 1000 TRAIN validation and locked eta=1 P=2000 pools; features use only current index','features':list(features),'median_prepare_ms':float(np.median(prep)),'p95_prepare_ms':float(np.percentile(prep,95))}
json.dump(meta,open(OUT/'structural_feature_meta.json','w'),indent=2); print('DONE',meta,flush=True)
