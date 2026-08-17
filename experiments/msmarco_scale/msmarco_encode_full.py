from __future__ import annotations
import gzip,json,pickle,time,re,gc,os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import njit, prange, set_num_threads

ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; GEOM=WORK/'geometry_1m'; IDX=WORK/'full_index'; IDX.mkdir(parents=True,exist_ok=True)
N=8_841_823; M=50_000; F=4; S=16; SENT=np.uint16(65535)
set_num_threads(5)
TOKEN_RE=re.compile(r'(?u)\b\w\w+\b')

def shard_path(i):
 hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz')); assert len(hits)==1,(i,hits); return hits[0]

def load_vocab():
 with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as g: z=pickle.load(g)
 terms=z['terms'].tolist(); idf=np.asarray(z['idf'],np.float32); return terms,idf,{t:i for i,t in enumerate(terms)}

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
def encode_kernel(indptr,indices,data,center_terms,center_values):
 n=indptr.size-1
 branches=np.full((n,F),SENT,np.uint16); mem=np.zeros((n,F),np.float32)
 rt=np.full((n,F,S),SENT,np.uint16); signbits=np.zeros((n,F),np.uint16)
 for d in prange(n):
  a=indptr[d]; b=indptr[d+1]
  # top4 document coordinates
  tv=np.zeros(F,np.float32); tt=np.full(F,SENT,np.uint16)
  for p in range(a,b):
   v=data[p]; t=np.uint16(indices[p]); pos=F
   for r in range(F):
    if v>tv[r]: pos=r; break
   if pos<F:
    for r in range(F-1,pos,-1): tv[r]=tv[r-1]; tt[r]=tt[r-1]
    tv[pos]=v; tt[pos]=t
  den=0.0
  for s in range(F): den+=tv[s]
  if den<=0: continue
  for s in range(F): branches[d,s]=tt[s]; mem[d,s]=tv[s]/den
  # residual codes per fuzzy branch
  for sl in range(F):
   j=int(tt[sl])
   if j==65535: continue
   best=np.zeros(S,np.float32); bt=np.full(S,SENT,np.uint16); bp=np.zeros(S,np.uint8)
   for p in range(a,b):
    t=np.uint16(indices[p]); r=data[p]-lookup_center(center_terms[j],center_values[j],t); ar=abs(r)
    mi=0; mv=best[0]
    for q in range(1,S):
     if best[q]<mv: mi=q; mv=best[q]
    if ar>mv:
     best[mi]=ar; bt[mi]=t; bp[mi]=1 if r>=0 else 0
   # sort descending to stabilize
   for x in range(S):
    mx=x
    for y in range(x+1,S):
     if best[y]>best[mx]: mx=y
    if mx!=x:
     z=best[x]; best[x]=best[mx]; best[mx]=z
     zt=bt[x]; bt[x]=bt[mx]; bt[mx]=zt
     zp=bp[x]; bp[x]=bp[mx]; bp[mx]=zp
   bits=np.uint16(0)
   for q in range(S):
    rt[d,sl,q]=bt[q]
    if bt[q]!=SENT and bp[q]: bits |= np.uint16(1<<q)
   signbits[d,sl]=bits
 return branches,mem,rt,signbits

if __name__=='__main__':
 terms,idf,vocab=load_vocab(); center_terms=np.load(GEOM/'center_terms.npy',mmap_mode='r'); center_values=np.load(GEOM/'center_values.npy',mmap_mode='r')
 # Global disk-backed arrays
 branches=np.memmap(IDX/'branches.u16',dtype=np.uint16,mode='w+',shape=(N,F)); branches[:]=SENT
 memberships=np.memmap(IDX/'memberships.f32',dtype=np.float32,mode='w+',shape=(N,F)); memberships[:]=0
 res_terms=np.memmap(IDX/'res_terms.u16',dtype=np.uint16,mode='w+',shape=(N,F,S)); res_terms[:]=SENT
 signbits=np.memmap(IDX/'signbits.u16',dtype=np.uint16,mode='w+',shape=(N,F)); signbits[:]=0
 doc_lengths=np.memmap(IDX/'doc_lengths.u16',dtype=np.uint16,mode='w+',shape=(N,)); doc_lengths[:]=0
 cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32)
 total_len=0; t_all=time.time(); offset=0
 for sid in range(36):
  t=time.time(); texts=[]; lens=[]
  with gzip.open(shard_path(sid),'rt',encoding='utf-8') as f:
   for line in f:
    o=json.loads(line); tx=((o.get('title') or '')+' '+(o.get('text') or '')).strip(); texts.append(tx); lens.append(min(65535,len(TOKEN_RE.findall(tx.lower()))))
  n=len(texts); X=cv.transform(texts).tocsr().astype(np.float32); X.data *= idf[X.indices]; normalize(X,norm='l2',axis=1,copy=False); X.sort_indices()
  br,mm,rt,sb=encode_kernel(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),center_terms,center_values)
  sl=slice(offset,offset+n); branches[sl]=br; memberships[sl]=mm; res_terms[sl]=rt; signbits[sl]=sb; doc_lengths[sl]=np.asarray(lens,np.uint16); total_len += int(np.sum(lens,dtype=np.int64))
  # whole-document binary support, one pair of files per corpus shard
  X.indices.astype(np.uint16).tofile(IDX/f'support_{sid:04d}.u16')
  X.indptr.astype(np.uint32).tofile(IDX/f'support_indptr_{sid:04d}.u32')
  with open(IDX/f'shard_{sid:04d}.json','w') as f: json.dump({'offset':offset,'n':n,'nnz':int(X.nnz),'seconds':time.time()-t},f)
  offset+=n; branches.flush(); memberships.flush(); res_terms.flush(); signbits.flush(); doc_lengths.flush()
  print(f'[{sid+1:02d}/36] n={n:,} nnz={X.nnz:,} offset={offset:,} sec={time.time()-t:.1f}',flush=True)
  del texts,lens,X,br,mm,rt,sb; gc.collect()
 assert offset==N,(offset,N)
 avg=total_len/N
 # Build branch postings by a global stable sort over 35.4M uint16 branch IDs.
 print('building branch postings...',flush=True); t=time.time(); flat=np.memmap(IDX/'branches.u16',dtype=np.uint16,mode='r',shape=(N*F,)); order=np.argsort(flat,kind='stable'); sorted_br=flat[order]; nvalid=int(np.searchsorted(sorted_br,SENT,side='left'))
 bo=np.memmap(IDX/'branch_order.u32',dtype=np.uint32,mode='w+',shape=(nvalid,)); bo[:]=order[:nvalid].astype(np.uint32); bo.flush(); counts=np.bincount(sorted_br[:nvalid].astype(np.int64),minlength=M); offsets=np.zeros(M+1,np.uint64); np.cumsum(counts,dtype=np.uint64,out=offsets[1:]); np.save(IDX/'branch_offsets.npy',offsets); print('postings valid memberships',nvalid,'sec',time.time()-t,flush=True)
 with open(IDX/'meta.json','w') as f: json.dump({'N':N,'M':M,'F':F,'S':S,'avg_doc_length':avg,'build_seconds':time.time()-t_all,'geometry':'geometry_1m_fullcorpus_vocab'},f,indent=2)
 print('FULL INDEX DONE avgdl',avg,'total sec',time.time()-t_all,flush=True)
