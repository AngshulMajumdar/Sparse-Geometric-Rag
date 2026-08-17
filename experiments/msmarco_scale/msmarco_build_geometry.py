from __future__ import annotations
import gzip,json,pickle,time,re,gc,os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import njit, prange, set_num_threads

ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; GEOM=WORK/'geometry_1m'; GEOM.mkdir(parents=True,exist_ok=True)
N_CAL=1_000_000; M=50_000; F=4; B=64; S=16; L=12; TAU=20.; BETA=-.2; EPS=1e-6
GRAPH_TAU=10.; ASSOC_K=64; ROUTE_K=32
SENT=np.uint16(65535)
set_num_threads(5)

def load_vocab():
 with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as g: z=pickle.load(g)
 terms=z['terms'].tolist(); idf=np.asarray(z['idf'],np.float32)
 return terms,idf,{t:i for i,t in enumerate(terms)}

def shard_path(i):
 hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz')); assert len(hits)==1,(i,hits); return hits[0]

def tfidf_shard(p,vocab,idf):
 texts=[]
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:
   o=json.loads(line); texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip())
 cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32)
 X=cv.transform(texts).tocsr().astype(np.float32)
 X.data *= idf[X.indices]
 normalize(X,norm='l2',axis=1,copy=False)
 X.sort_indices()
 return X

@njit(parallel=True,cache=False)
def topk_memberships(indptr,indices,data,F):
 N=indptr.size-1
 branches=np.full((N,F),np.uint16(65535),np.uint16)
 mem=np.zeros((N,F),np.float32)
 topL=np.full((N,12),np.uint16(65535),np.uint16)
 for d in prange(N):
  a=indptr[d]; b=indptr[d+1]
  # top 12 descending insertion
  vals=np.zeros(12,np.float32); tids=np.full(12,np.uint16(65535),np.uint16)
  for p in range(a,b):
   v=data[p]; t=np.uint16(indices[p])
   # locate insertion descending
   pos=12
   for r in range(12):
    if v>vals[r]: pos=r; break
   if pos<12:
    for r in range(11,pos,-1): vals[r]=vals[r-1]; tids[r]=tids[r-1]
    vals[pos]=v; tids[pos]=t
  den=0.0
  for s in range(F): den += vals[s]
  if den>0:
   for s in range(F):
    branches[d,s]=tids[s]; mem[d,s]=vals[s]/den
  for s in range(12): topL[d,s]=tids[s]
 return branches,mem,topL

@njit(cache=False)
def lookup_center(ct,cv,t):
 lo=0; hi=ct.size
 while lo<hi:
  mid=(lo+hi)//2; x=ct[mid]
  if x==65535 or x>=t: hi=mid
  else: lo=mid+1
 if lo<ct.size and ct[lo]==t: return cv[lo]
 return 0.0

@njit(parallel=True,cache=False)
def residual_codes(indptr,indices,data,branches,center_terms,center_values,S):
 N=indptr.size-1; F=branches.shape[1]
 rt=np.full((N,F,S),np.uint16(65535),np.uint16)
 rs=np.zeros((N,F,S),np.int8)
 for d in prange(N):
  a=indptr[d]; b=indptr[d+1]
  for sl in range(F):
   j=int(branches[d,sl])
   if j==65535: continue
   best=np.zeros(S,np.float32); bt=np.full(S,np.uint16(65535),np.uint16); bs=np.zeros(S,np.int8)
   for p in range(a,b):
    t=np.uint16(indices[p]); r=data[p]-lookup_center(center_terms[j],center_values[j],t); ar=abs(r)
    # replace current minimum
    mi=0; mv=best[0]
    for q in range(1,S):
     if best[q]<mv: mi=q; mv=best[q]
    if ar>mv:
     best[mi]=ar; bt[mi]=t; bs[mi]=1 if r>=0 else -1
   # sort retained residuals descending by magnitude for determinism
   for x in range(S):
    mx=x
    for y in range(x+1,S):
     if best[y]>best[mx]: mx=y
    if mx!=x:
     tv=best[x]; best[x]=best[mx]; best[mx]=tv
     tt=bt[x]; bt[x]=bt[mx]; bt[mx]=tt
     ss=bs[x]; bs[x]=bs[mx]; bs[mx]=ss
   for q in range(S): rt[d,sl,q]=bt[q]; rs[d,sl,q]=bs[q]
 return rt,rs


def prune_rows(mat,k):
 rows=[]; cols=[]; vals=[]; mat=mat.tocsr()
 for r in range(mat.shape[0]):
  a,b=mat.indptr[r],mat.indptr[r+1]; idx=mat.indices[a:b]; dat=mat.data[a:b]
  if len(dat)==0: continue
  kk=min(k,len(dat)); pick=np.argpartition(dat,-kk)[-kk:]; pick=pick[np.argsort(dat[pick])[::-1]]
  rows.extend([r]*kk); cols.extend(idx[pick].tolist()); vals.extend(dat[pick].astype(np.float32).tolist())
 return sparse.csr_matrix((np.asarray(vals,np.float32),(np.asarray(rows,np.int32),np.asarray(cols,np.int32))),shape=mat.shape)

if __name__=='__main__':
 t_all=time.time(); terms,idf,vocab=load_vocab(); np.save(GEOM/'idf.npy',idf); 
 with gzip.open(GEOM/'terms.pkl.gz','wb',compresslevel=1) as g: pickle.dump(terms,g,protocol=5)
 print('GEOMETRY calibration on first 1,000,000 passages; lexical basis from all 8.84M',flush=True)
 # X calibration
 xcache=GEOM/'cal_X.npz'
 if xcache.exists():
  X=sparse.load_npz(xcache).tocsr(); print(' X checkpoint loaded',X.shape,X.nnz,flush=True)
 else:
  xs=[]
  for sid in range(4):
   t=time.time(); Xs=tfidf_shard(shard_path(sid),vocab,idf); xs.append(Xs); print(' tfidf shard',sid,Xs.shape,Xs.nnz,'sec',time.time()-t,flush=True)
  X=sparse.vstack(xs,format='csr'); del xs; gc.collect(); print(' X',X.shape,X.nnz,flush=True); sparse.save_npz(xcache,X,compressed=False); print(' X checkpoint saved',flush=True)
 # branches/topL
 if (GEOM/'cal_branches.npy').exists() and (GEOM/'cal_memberships.npy').exists() and (GEOM/'cal_topL.npy').exists():
  branches=np.load(GEOM/'cal_branches.npy'); mem=np.load(GEOM/'cal_memberships.npy'); topL=np.load(GEOM/'cal_topL.npy'); print(' membership checkpoint loaded',flush=True)
 else:
  t=time.time(); branches,mem,topL=topk_memberships(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),F); print(' memberships sec',time.time()-t,flush=True)
  np.save(GEOM/'cal_branches.npy',branches); np.save(GEOM/'cal_memberships.npy',mem); np.save(GEOM/'cal_topL.npy',topL); print(' membership checkpoint saved',flush=True)
 # centers exact using sparse algebra
 if (GEOM/'center_terms.npy').exists() and (GEOM/'center_values.npy').exists():
  center_terms=np.load(GEOM/'center_terms.npy'); center_values=np.load(GEOM/'center_values.npy'); print(' centers checkpoint loaded',flush=True)
 else:
  t=time.time(); wr=np.repeat(np.arange(N_CAL,dtype=np.int32),F); wc=branches.ravel().astype(np.int32); wd=mem.ravel(); valid=wc!=65535
  W=sparse.csr_matrix((wd[valid],(wr[valid],wc[valid])),shape=(N_CAL,M),dtype=np.float32); del wr,wc,wd,valid
  mass=np.asarray(W.sum(axis=0)).ravel().astype(np.float32)
  center_terms=np.full((M,B),SENT,np.uint16); center_values=np.zeros((M,B),np.float32)
  block=512
  for start in range(0,M,block):
   end=min(M,start+block); C=(W[:,start:end].T@X).tocsr()
   for local in range(end-start):
    j=start+local
    if mass[j]<=0: continue
    a,b=C.indptr[local],C.indptr[local+1]; idx=C.indices[a:b]; dat=C.data[a:b]/mass[j]
    if len(dat)==0: continue
    kk=min(B,len(dat)); pick=np.argpartition(dat,-kk)[-kk:]; ii=idx[pick]; vv=dat[pick]; oo=np.argsort(ii); ii=ii[oo]; vv=vv[oo]
    center_terms[j,:kk]=ii.astype(np.uint16); center_values[j,:kk]=vv.astype(np.float32)
   if start%4096==0: print(' centers',end,'/',M,flush=True)
  np.save(GEOM/'center_terms.npy',center_terms); np.save(GEOM/'center_values.npy',center_values); del W,mass,C; gc.collect(); print(' centers sec',time.time()-t,flush=True)
 # residuals
 if (GEOM/'cal_res_terms.npy').exists() and (GEOM/'cal_res_signs.npy').exists():
  rt=np.load(GEOM/'cal_res_terms.npy',mmap_mode='r'); rs=np.load(GEOM/'cal_res_signs.npy',mmap_mode='r'); print(' residual checkpoint loaded',flush=True)
 else:
  t=time.time(); rt,rs=residual_codes(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),branches,center_terms,center_values,S); print(' residual sec',time.time()-t,flush=True); np.save(GEOM/'cal_res_terms.npy',rt); np.save(GEOM/'cal_res_signs.npy',rs); print(' residual checkpoint saved',flush=True)
 # reliability global
 total_memberships=int(np.sum(branches!=SENT)); gcnt=np.zeros(M,np.float64); gsum=np.zeros(M,np.float64)
 for d0 in range(0,N_CAL,50_000):
  tt=rt[d0:d0+50_000].ravel(); zz=rs[d0:d0+50_000].ravel().astype(np.float64); ok=tt!=SENT
  gcnt += np.bincount(tt[ok].astype(np.int64),minlength=M)
  gsum += np.bincount(tt[ok].astype(np.int64),weights=zz[ok],minlength=M)
 ge2=gcnt/max(1,total_memberships); ge1=gsum/max(1,total_memberships); gvar=np.maximum(ge2-ge1*ge1,0.)
 # branch order calibration
 flat=branches.ravel(); valid=np.flatnonzero(flat!=SENT); order=valid[np.argsort(flat[valid],kind='stable')]; sorted_br=flat[order].astype(np.int64); counts=np.bincount(sorted_br,minlength=M); offs=np.zeros(M+1,np.int64); np.cumsum(counts,out=offs[1:])
 f_rt=rt.reshape(N_CAL*F,S); f_rs=rs.reshape(N_CAL*F,S)
 # Memory-bounded reliability CSR. Upper bound is one entry per residual occurrence.
 max_rel=N_CAL*F*S
 rel_i=np.memmap(GEOM/'rel_indices.u16',dtype=np.uint16,mode='w+',shape=(max_rel,))
 rel_v=np.memmap(GEOM/'rel_data.f32',dtype=np.float32,mode='w+',shape=(max_rel,))
 rel_p=np.zeros(M+1,np.uint64); pos_rel=0
 t=time.time()
 for j in range(M):
  a,b=offs[j],offs[j+1]; mpos=order[a:b]; nj=len(mpos)
  if nj>0:
   tj=f_rt[mpos].ravel(); sj=f_rs[mpos].ravel().astype(np.float64); ok=tj!=SENT
   if np.any(ok):
    u,inv=np.unique(tj[ok],return_inverse=True); cnt=np.bincount(inv).astype(np.float64); sm=np.bincount(inv,weights=sj[ok]).astype(np.float64)
    e2=cnt/nj; e1=sm/nj; lv=np.maximum(e2-e1*e1,0.); shr=(cnt/(cnt+TAU))*lv+(TAU/(cnt+TAU))*gvar[u.astype(np.int64)]; w=np.power(shr+EPS,BETA)
    if len(w) and np.isfinite(w).all() and w.mean()>0: w=w/w.mean()
    nrel=len(u); rel_i[pos_rel:pos_rel+nrel]=u.astype(np.uint16); rel_v[pos_rel:pos_rel+nrel]=w.astype(np.float32); pos_rel+=nrel
  rel_p[j+1]=pos_rel
  if j%5000==0 and j: print(' reliability branch',j,'pairs',pos_rel,flush=True)
 rel_i.flush(); rel_v.flush(); np.save(GEOM/'rel_indptr.npy',rel_p); np.save(GEOM/'global_sign_var.npy',gvar.astype(np.float32));
 with open(GEOM/'rel_meta.json','w') as f: json.dump({'nnz':int(pos_rel),'max_entries':int(max_rel)},f)
 print(' reliability nnz',pos_rel,'sec',time.time()-t,flush=True)
 del rt,rs,order,flat,valid,sorted_br,counts,offs,f_rt,f_rs,rel_i,rel_v,rel_p; gc.collect()
 # graph exact from calibration topL: n_i and unordered pair counts
 t=time.time(); flatL=topL.ravel(); good=flatL!=SENT; ni=np.bincount(flatL[good].astype(np.int64),minlength=M).astype(np.float64); del flatL,good
 maxpairs=N_CAL*66; pairkeys=np.full(maxpairs,np.uint32(0xffffffff),np.uint32); pos=0
 # vectorized per pair position across documents: only 66 loops, each handles 1m rows
 for a in range(L):
  ia=topL[:,a]
  for b in range(a+1,L):
   ib=topL[:,b]; ok=(ia!=SENT)&(ib!=SENT); n=int(ok.sum())
   x=ia[ok].astype(np.uint32); y=ib[ok].astype(np.uint32); lo=np.minimum(x,y); hi=np.maximum(x,y); pairkeys[pos:pos+n]=(lo<<16)|hi; pos+=n
 print(' pair occurrences',pos,'sorting...',flush=True); keys=pairkeys[:pos]; keys.sort(); del pairkeys,topL; gc.collect()
 # run-length encode sorted keys without np.unique's large extra sort
 change=np.empty(len(keys),dtype=bool); change[0]=True; change[1:]=keys[1:]!=keys[:-1]; starts=np.flatnonzero(change); ukeys=keys[starts].copy(); cnt=np.diff(np.append(starts,len(keys))).astype(np.float32); del keys,change,starts; gc.collect(); print(' unique pairs',len(ukeys),flush=True)
 ii=(ukeys>>16).astype(np.int32); jj=(ukeys & np.uint32(65535)).astype(np.int32); nij=cnt.astype(np.float64); ppmi=np.log((nij*float(N_CAL)+1e-12)/(ni[ii]*ni[jj]+1e-12)); ppmi=np.maximum(ppmi,0.); score=(nij/(nij+GRAPH_TAU))*ppmi; mask=score>0; ii=ii[mask]; jj=jj[mask]; sv=score[mask].astype(np.float32); del ukeys,cnt,nij,ppmi,score,mask; gc.collect(); print(' positive pair edges',len(sv),flush=True)
 rows=np.concatenate([ii,jj]); cols=np.concatenate([jj,ii]); vals=np.concatenate([sv,sv]); del ii,jj,sv; Afull=sparse.csr_matrix((vals,(rows,cols)),shape=(M,M)); del rows,cols,vals; gc.collect(); A=prune_rows(Afull,ASSOC_K); del Afull; gc.collect(); sparse.save_npz(GEOM/'assoc_ppmi.npz',A,compressed=True); print(' A nnz',A.nnz,flush=True)
 An=normalize(A,norm='l2',axis=1,copy=True); gr=[]; gc2=[]; gv=[]; bs=256
 for start in range(0,M,bs):
  end=min(M,start+bs); sim=(An[start:end]@An.T).tocsr()
  for local in range(end-start):
   i=start+local; a,b=sim.indptr[local],sim.indptr[local+1]; js=sim.indices[a:b]; vv2=sim.data[a:b]; mk=(js!=i)&(vv2>0); js=js[mk]; vv2=vv2[mk]
   if len(vv2)==0: continue
   kk=min(ROUTE_K,len(vv2)); pk=np.argpartition(vv2,-kk)[-kk:]; pk=pk[np.argsort(vv2[pk])[::-1]]; gr.extend([i]*kk); gc2.extend(js[pk].tolist()); gv.extend(vv2[pk].astype(np.float32).tolist())
  if start%4096==0: print(' G',end,'/',M,flush=True)
 G=sparse.csr_matrix((np.asarray(gv,np.float32),(np.asarray(gr,np.int32),np.asarray(gc2,np.int32))),shape=(M,M)); sparse.save_npz(GEOM/'context_similarity.npz',G,compressed=True); print(' G nnz',G.nnz,'graph sec',time.time()-t,flush=True)
 # Save metadata
 with open(GEOM/'meta.json','w') as f: json.dump({'calibration_docs':N_CAL,'full_corpus_docs':8_841_823,'F':F,'B':B,'S':S,'L':L,'tau':TAU,'beta':BETA,'graph_tau':GRAPH_TAU,'assoc_k':ASSOC_K,'route_k':ROUTE_K,'build_seconds':time.time()-t_all},f,indent=2)
 print('GEOMETRY DONE total sec',time.time()-t_all,flush=True)
