from __future__ import annotations
import gzip,json,pickle,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import njit,prange,set_num_threads
ROOT=Path('/mnt/data'); W=ROOT/'msmarco_scale_work'; G=W/'geometry_uniform1m_s32'; OLD=W/'full_index'; NEW=W/'full_index_uniform1m_s32'; NEW.mkdir(exist_ok=True)
N=8_841_823; M=50_000; F=4; S=32; SENT=np.uint16(65535)

def shard_path(i):
 hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz')); assert len(hits)==1; return hits[0]
def load_vocab():
 with gzip.open(W/'final_vocab_50k.pkl.gz','rb') as g:z=pickle.load(g)
 terms=z['terms'].tolist(); idf=np.asarray(z['idf'],np.float32); return idf,{t:i for i,t in enumerate(terms)}
@njit(cache=False)
def lookup(ct,cv,t):
 lo=0; hi=ct.size
 while lo<hi:
  mid=(lo+hi)//2; x=ct[mid]
  if x==65535 or x>=t: hi=mid
  else: lo=mid+1
 if lo<ct.size and ct[lo]==t:return cv[lo]
 return 0.0
@njit(parallel=True,cache=False)
def kernel(indptr,indices,data,ct,cv):
 n=indptr.size-1; br=np.full((n,F),SENT,np.uint16); rt=np.full((n,F,S),SENT,np.uint16); sb=np.zeros((n,F),np.uint32)
 for d in prange(n):
  a=indptr[d]; b=indptr[d+1]; tv=np.zeros(F,np.float32); tt=np.full(F,SENT,np.uint16)
  for p in range(a,b):
   v=data[p]; t=np.uint16(indices[p]); pos=F
   for r in range(F):
    if v>tv[r]:pos=r;break
   if pos<F:
    for r in range(F-1,pos,-1):tv[r]=tv[r-1];tt[r]=tt[r-1]
    tv[pos]=v;tt[pos]=t
  den=0.0
  for s in range(F):den+=tv[s]
  if den<=0:continue
  for s in range(F):br[d,s]=tt[s]
  for sl in range(F):
   j=int(tt[sl]); best=np.zeros(S,np.float32); bt=np.full(S,SENT,np.uint16); bp=np.zeros(S,np.uint8)
   for p in range(a,b):
    t=np.uint16(indices[p]); rr=data[p]-lookup(ct[j],cv[j],t); ar=abs(rr); mi=0; mv=best[0]
    for q in range(1,S):
     if best[q]<mv:mi=q;mv=best[q]
    if ar>mv:best[mi]=ar;bt[mi]=t;bp[mi]=1 if rr>=0 else 0
   for x in range(S):
    mx=x
    for y in range(x+1,S):
     if best[y]>best[mx]:mx=y
    if mx!=x:
     zz=best[x];best[x]=best[mx];best[mx]=zz; zt=bt[x];bt[x]=bt[mx];bt[mx]=zt; zp=bp[x];bp[x]=bp[mx];bp[mx]=zp
   bits=np.uint32(0)
   for q in range(S):
    rt[d,sl,q]=bt[q]
    if bt[q]!=SENT and bp[q]:bits|=np.uint32(1)<<np.uint32(q)
   sb[d,sl]=bits
 return br,rt,sb

def work(sid):
 set_num_threads(5); t=time.time(); idf,vocab=load_vocab(); ct=np.load(G/'center_terms.npy',mmap_mode='r'); cvv=np.load(G/'center_values.npy',mmap_mode='r'); oldbr=np.memmap(OLD/'branches.u16',np.uint16,'r',shape=(N,F)); rtg=np.memmap(NEW/'res_terms.u16',np.uint16,'r+',shape=(N,F,S)); sbg=np.memmap(NEW/'signbits.u32',np.uint32,'r+',shape=(N,F))
 texts=[]
 with gzip.open(shard_path(sid),'rt',encoding='utf-8') as f:
  for line in f:
   o=json.loads(line);texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip())
 cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32);X=cv.transform(texts).tocsr().astype(np.float32);X.data*=idf[X.indices];normalize(X,norm='l2',axis=1,copy=False);X.sort_indices(); br,rt,sb=kernel(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),ct,cvv)
 n=len(texts);off=sid*250_000;sl=slice(off,off+n);mism=int(np.sum(br!=np.asarray(oldbr[sl])));assert mism==0,(sid,mism);rtg[sl]=rt;sbg[sl]=sb;rtg.flush();sbg.flush();sec=time.time()-t;json.dump({'offset':off,'n':n,'nnz':int(X.nnz),'seconds':sec,'branch_mismatches':mism},open(NEW/f'shard_{sid:04d}.json','w'));return sid,n,sec
if __name__=='__main__':
 if not (NEW/'res_terms.u16').exists():a=np.memmap(NEW/'res_terms.u16',np.uint16,'w+',shape=(N,F,S));a[:]=SENT;a.flush();del a
 if not (NEW/'signbits.u32').exists():a=np.memmap(NEW/'signbits.u32',np.uint32,'w+',shape=(N,F));a[:]=0;a.flush();del a
 done={int(p.stem.split('_')[1]) for p in NEW.glob('shard_*.json')};missing=[i for i in range(36) if i not in done];print('missing',missing,flush=True);t=time.time()
 with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as ex:
  fs=[ex.submit(work,i) for i in missing]
  for k,fu in enumerate(as_completed(fs),1):sid,n,sec=fu.result();print(f'[{k:02d}/{len(missing):02d}] shard {sid:04d} n={n:,} sec={sec:.1f}',flush=True)
 print('S32 ENCODE DONE',time.time()-t,flush=True)
