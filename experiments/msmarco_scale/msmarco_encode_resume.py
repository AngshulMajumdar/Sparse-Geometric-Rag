from __future__ import annotations
import sys, gzip, json, pickle, time, re, gc, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_encode_full as base

ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; GEOM=WORK/'geometry_1m'; IDX=WORK/'full_index'
N=8_841_823; M=50_000; F=4; S=16
TOKEN_RE=re.compile(r'(?u)\b\w\w+\b')

def shard_path(i):
 hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz')); assert len(hits)==1,(i,hits); return hits[0]

def load_vocab():
 with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as g:z=pickle.load(g)
 terms=z['terms'].tolist(); idf=np.asarray(z['idf'],np.float32); return idf,{t:i for i,t in enumerate(terms)}

def work(sid):
 set_num_threads(1)
 t=time.time(); idf,vocab=load_vocab(); center_terms=np.load(GEOM/'center_terms.npy',mmap_mode='r'); center_values=np.load(GEOM/'center_values.npy',mmap_mode='r')
 branches=np.memmap(IDX/'branches.u16',np.uint16,'r+',shape=(N,F)); memberships=np.memmap(IDX/'memberships.f32',np.float32,'r+',shape=(N,F)); rtg=np.memmap(IDX/'res_terms.u16',np.uint16,'r+',shape=(N,F,S)); sbg=np.memmap(IDX/'signbits.u16',np.uint16,'r+',shape=(N,F)); dlg=np.memmap(IDX/'doc_lengths.u16',np.uint16,'r+',shape=(N,))
 texts=[]; lens=[]
 with gzip.open(shard_path(sid),'rt',encoding='utf-8') as f:
  for line in f:
   o=json.loads(line); tx=((o.get('title') or '')+' '+(o.get('text') or '')).strip(); texts.append(tx); lens.append(min(65535,len(TOKEN_RE.findall(tx.lower()))))
 n=len(texts); cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32); X=cv.transform(texts).tocsr().astype(np.float32); X.data*=idf[X.indices]; normalize(X,norm='l2',axis=1,copy=False); X.sort_indices()
 br,mm,rt,sb=base.encode_kernel(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),center_terms,center_values)
 offset=sid*250_000; sl=slice(offset,offset+n); branches[sl]=br; memberships[sl]=mm; rtg[sl]=rt; sbg[sl]=sb; dlg[sl]=np.asarray(lens,np.uint16)
 X.indices.astype(np.uint16).tofile(IDX/f'support_{sid:04d}.u16'); X.indptr.astype(np.uint32).tofile(IDX/f'support_indptr_{sid:04d}.u32')
 branches.flush(); memberships.flush(); rtg.flush(); sbg.flush(); dlg.flush()
 secs=time.time()-t
 with open(IDX/f'shard_{sid:04d}.json','w') as f:json.dump({'offset':offset,'n':n,'nnz':int(X.nnz),'seconds':secs},f)
 return sid,n,int(X.nnz),secs

if __name__=='__main__':
 missing=[i for i in range(36) if not (IDX/f'shard_{i:04d}.json').exists()]
 print('missing',missing,flush=True); t0=time.time()
 with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as ex:
  fs={ex.submit(work,i):i for i in missing}; done=0
  for f in as_completed(fs):
   sid,n,nnz,sec=f.result(); done+=1; print(f'[{done:02d}/{len(missing):02d}] shard {sid:04d} n={n:,} nnz={nnz:,} sec={sec:.1f}',flush=True)
 print('encoding complete sec',time.time()-t0,flush=True)
 # validate all shards and calculate avg dl
 metas=[]
 for i in range(36):
  with open(IDX/f'shard_{i:04d}.json') as f:metas.append(json.load(f))
 assert sum(x['n'] for x in metas)==N
 dl=np.memmap(IDX/'doc_lengths.u16',np.uint16,'r',shape=(N,)); avg=float(np.mean(dl,dtype=np.float64));
 # postings
 print('building postings',flush=True); t=time.time(); flat=np.memmap(IDX/'branches.u16',np.uint16,'r',shape=(N*F,)); order=np.argsort(flat,kind='stable'); sorted_br=flat[order]; nvalid=int(np.searchsorted(sorted_br,np.uint16(65535),side='left')); bo=np.memmap(IDX/'branch_order.u32',np.uint32,'w+',shape=(nvalid,)); bo[:]=order[:nvalid].astype(np.uint32); bo.flush(); counts=np.bincount(sorted_br[:nvalid].astype(np.int64),minlength=M); offs=np.zeros(M+1,np.uint64); np.cumsum(counts,dtype=np.uint64,out=offs[1:]); np.save(IDX/'branch_offsets.npy',offs); print('postings',nvalid,'sec',time.time()-t,flush=True)
 with open(IDX/'meta.json','w') as f:json.dump({'N':N,'M':M,'F':F,'S':S,'avg_doc_length':avg,'build_seconds_resume':time.time()-t0,'geometry':'geometry_1m_fullcorpus_vocab'},f,indent=2)
 print('FULL INDEX DONE avgdl',avg,'total sec',time.time()-t0,flush=True)
