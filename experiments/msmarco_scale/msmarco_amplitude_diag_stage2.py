from __future__ import annotations
import sys,time,json,gzip
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'amplitude_diag'; IDX=m.IDX
idx=m.FullIndex()
union=np.load(OUT/'union_docs.npy',mmap_mode='r')
# exact support counts from existing binary support index
sup_ip=idx.sup_ip; sup_ids=idx.sup_ids
counts=(np.asarray(sup_ip[union.astype(np.int64)+1],dtype=np.uint64)-np.asarray(sup_ip[union.astype(np.int64)],dtype=np.uint64))
uip=np.empty(len(union)+1,np.uint64); uip[0]=0; np.cumsum(counts,out=uip[1:]); nnz=int(uip[-1])
np.save(OUT/'exact_tfidf_indptr.npy',uip)
data=np.memmap(OUT/'exact_tfidf_data.f32',np.float32,'w+',shape=(nnz,))
cv=idx.cvq; idf=idx.idf
# Locate all 36 uploaded shards
shards=[]
for sid in range(36):
    a=ROOT/f'corpus_{sid:04d}.jsonl.gz'; b=ROOT/f'corpus_{sid:04d}.jsonl(1).gz'
    p=a if a.exists() else b
    if not p.exists(): raise FileNotFoundError((a,b))
    shards.append(p)

BATCH=5000; texts=[]; rows=[]; start=time.time(); found=0; verified=0

def flush():
    global texts,rows,found,verified
    if not rows:return
    X=cv.transform(texts).tocsr().astype(np.float32)
    X.data*=idf[X.indices]; normalize(X,norm='l2',axis=1,copy=False)
    for bi,ur in enumerate(rows):
        doc=int(union[ur]); ga=int(sup_ip[doc]); gb=int(sup_ip[doc+1]); expected=np.asarray(sup_ids[ga:gb],dtype=np.int32)
        a=int(X.indptr[bi]); b=int(X.indptr[bi+1]); got=X.indices[a:b]
        if len(got)!=len(expected) or not np.array_equal(got,expected):
            raise RuntimeError(f'support mismatch doc={doc} expected={len(expected)} got={len(got)}')
        ua=int(uip[ur]); ub=int(uip[ur+1]); data[ua:ub]=X.data[a:b]
        verified+=1
    found+=len(rows); texts=[]; rows=[]

p_union=0
for sid,path in enumerate(shards):
    lo=sid*250000; hi=min((sid+1)*250000,m.N)
    # target union row range for this shard
    r0=int(np.searchsorted(union,lo)); r1=int(np.searchsorted(union,hi))
    if r0==r1:
        continue
    targets=np.asarray(union[r0:r1],dtype=np.uint32); ti=0
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for local,line in enumerate(f):
            d=lo+local
            if ti>=len(targets): break
            td=int(targets[ti])
            if d<td: continue
            if d!=td: raise RuntimeError((sid,d,td))
            o=json.loads(line); oid=int(o['_id'])
            if oid!=d: raise RuntimeError(f'id mismatch line {d} json {oid}')
            texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip()); rows.append(r0+ti); ti+=1
            if len(rows)>=BATCH: flush()
    if ti!=len(targets): raise RuntimeError(f'shard {sid} found {ti}/{len(targets)}')
    flush(); data.flush()
    print('shard',sid,'selected',len(targets),'total_found',found,'elapsed',time.time()-start,flush=True)
flush(); data.flush()
meta={'union_docs':int(len(union)),'nnz':nnz,'avg_nnz':float(nnz/len(union)),'verified_docs':verified,'seconds':time.time()-start}
json.dump(meta,open(OUT/'stage2_meta.json','w'),indent=2)
print('DONE',meta,flush=True)
