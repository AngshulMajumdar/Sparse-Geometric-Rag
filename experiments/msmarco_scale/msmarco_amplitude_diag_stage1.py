from __future__ import annotations
import sys,time,json,gzip,pickle
from pathlib import Path
import numpy as np,pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_early_lex_validation_fast as e
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; idx=e.idx; P=2000; M=m.M
set_num_threads(5)
OUT=WORK/'amplitude_diag'; OUT.mkdir(exist_ok=True)

def topk_desc(score,k):
    n=len(score); k=min(k,n)
    if n<=k: return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]
    return ii[np.argsort(score[ii])[::-1]]

# exact same deterministic validation query IDs
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id'])
uq=np.unique(tr['query-id'].to_numpy()); del tr
rng=np.random.default_rng(20260815)
ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]
texts=m.load_query_texts(ids)
qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)

# Warm up
p=e.prepare_all(texts[ids[0]]); del p

# Freeze current direct eta=1 pools. Store the existing final-score components too.
docs2k=np.empty((len(ids),P),np.uint32)
tail2k=np.empty((len(ids),P),np.float32)
lex2k=np.empty((len(ids),P),np.float32)
sem2k=np.empty((len(ids),P),np.float32)
valid=np.zeros(len(ids),np.int32)
route_relhit=pool_relhit=den=0
start=time.time(); prep=[]
for qi,qid in enumerate(ids):
    t=time.perf_counter(); p=e.prepare_all(texts[qid]); prep.append((time.perf_counter()-t)*1000)
    if p is None: continue
    sel=topk_desc(m.zscore(p['tail']) + m.zscore(p['lex']), P)  # eta=1
    k=len(sel); valid[qi]=k
    docs2k[qi,:k]=p['ud'][sel]; tail2k[qi,:k]=p['tail'][sel]; lex2k[qi,:k]=p['lex'][sel]; sem2k[qi,:k]=p['sem'][sel]
    if k<P:
        docs2k[qi,k:]=np.uint32(0); tail2k[qi,k:]=0; lex2k[qi,k:]=0; sem2k[qi,k:]=0
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    ud=p['ud']; poolset=set(map(int,p['ud'][sel].tolist()))
    for d in rels:
        kk=np.searchsorted(ud,d); route_relhit += int(kk<len(ud) and int(ud[kk])==d); pool_relhit += int(d in poolset)
    if (qi+1)%100==0: print('pool',qi+1,'median_ms',float(np.median(prep)),'route',route_relhit/den,'pool',pool_relhit/den,flush=True)

np.savez_compressed(OUT/'fixed_eta1_pools.npz', qids=np.asarray(ids), valid=valid, docs=docs2k, tail=tail2k, lex=lex2k, sem=sem2k)
union=np.unique(np.concatenate([docs2k[i,:valid[i]] for i in range(len(ids))]))
np.save(OUT/'union_docs.npy',union)
meta={'n_queries':len(ids),'P':P,'union_docs':int(len(union)),'route_relevant_recall':route_relhit/den,'pool_relevant_recall':pool_relhit/den,'median_prepare_ms':float(np.median(prep)),'seconds':time.time()-start}
json.dump(meta,open(OUT/'stage1_meta.json','w'),indent=2)
print('DONE',meta,flush=True)
