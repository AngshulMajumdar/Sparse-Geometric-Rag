from __future__ import annotations
import sys,gzip,json,pickle,time,gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_encode_full as base
ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; GEOM=WORK/'geometry_uniform1m'; OLD=WORK/'full_index'; NEW=WORK/'full_index_uniform1m'; NEW.mkdir(parents=True,exist_ok=True)
N=8_841_823; M=50_000; F=4; S=16

def shard_path(i):
    hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz')); assert len(hits)==1,(i,hits); return hits[0]

def load_vocab():
    with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as g:z=pickle.load(g)
    terms=z['terms'].tolist(); idf=np.asarray(z['idf'],np.float32); return idf,{t:i for i,t in enumerate(terms)}

def work(sid):
    set_num_threads(1); t=time.time(); idf,vocab=load_vocab(); ct=np.load(GEOM/'center_terms.npy',mmap_mode='r'); cvv=np.load(GEOM/'center_values.npy',mmap_mode='r')
    oldbr=np.memmap(OLD/'branches.u16',np.uint16,'r',shape=(N,F)); rtg=np.memmap(NEW/'res_terms.u16',np.uint16,'r+',shape=(N,F,S)); sbg=np.memmap(NEW/'signbits.u16',np.uint16,'r+',shape=(N,F))
    texts=[]
    with gzip.open(shard_path(sid),'rt',encoding='utf-8') as f:
        for line in f:
            o=json.loads(line); texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip())
    n=len(texts); cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32); X=cv.transform(texts).tocsr().astype(np.float32); X.data*=idf[X.indices]; normalize(X,norm='l2',axis=1,copy=False); X.sort_indices()
    br,mm,rt,sb=base.encode_kernel(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),ct,cvv)
    offset=sid*250_000; sl=slice(offset,offset+n); mism=int(np.sum(br!=np.asarray(oldbr[sl])))
    if mism: raise RuntimeError(f'branch mismatch shard {sid}: {mism}')
    rtg[sl]=rt; sbg[sl]=sb; rtg.flush(); sbg.flush(); secs=time.time()-t
    with open(NEW/f'shard_{sid:04d}.json','w') as f:json.dump({'offset':offset,'n':n,'nnz':int(X.nnz),'seconds':secs,'branch_mismatches':mism},f)
    return sid,n,int(X.nnz),secs

if __name__=='__main__':
    # initialize once if missing
    if not (NEW/'res_terms.u16').exists():
        a=np.memmap(NEW/'res_terms.u16',np.uint16,'w+',shape=(N,F,S)); a[:]=np.uint16(65535); a.flush(); del a
    if not (NEW/'signbits.u16').exists():
        a=np.memmap(NEW/'signbits.u16',np.uint16,'w+',shape=(N,F)); a[:]=0; a.flush(); del a
    done={int(p.stem.split('_')[1]) for p in NEW.glob('shard_*.json')}; missing=[i for i in range(36) if i not in done]
    print('missing',missing,flush=True); t0=time.time()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as ex:
        fs={ex.submit(work,i):i for i in missing}; k=0
        for fu in as_completed(fs):
            sid,n,nnz,sec=fu.result(); k+=1; print(f'[{k:02d}/{len(missing):02d}] shard {sid:04d} n={n:,} nnz={nnz:,} sec={sec:.1f}',flush=True)
    print('UNIFORM RESIDUAL ENCODE DONE sec',time.time()-t0,flush=True)
