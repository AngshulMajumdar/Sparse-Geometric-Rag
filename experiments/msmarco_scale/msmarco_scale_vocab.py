from __future__ import annotations
import gzip, json, pickle, time, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT=Path('/mnt/data')
OUT=ROOT/'msmarco_scale_work'
LOC=OUT/'local_vocab'
LOC.mkdir(parents=True,exist_ok=True)
LOCAL_K=150_000
WORKERS=3
pat=re.compile(r'corpus_(\d{4})')

def shard_paths():
    out={}
    for p in ROOT.glob('corpus_*.jsonl*.gz'):
        m=pat.search(p.name)
        if m: out[int(m.group(1))]=p
    return [out[i] for i in sorted(out)]

def process_one(arg):
    sid,p=arg
    dst=LOC/f'vocab_{sid:04d}.pkl.gz'
    if dst.exists(): return sid, 'cached', dst.stat().st_size
    t=time.time(); texts=[]
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            o=json.loads(line)
            texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip())
    cv=CountVectorizer(max_features=LOCAL_K,min_df=1,lowercase=True,
                       token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32)
    X=cv.fit_transform(texts)
    terms=cv.get_feature_names_out()
    tf=np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    with gzip.open(dst,'wb',compresslevel=1) as g:
        pickle.dump((terms,tf),g,protocol=5)
    return sid, time.time()-t, dst.stat().st_size

if __name__=='__main__':
    paths=shard_paths()
    assert len(paths)==36, len(paths)
    print('PASS1 local heavy hitters: shards=',len(paths),'workers=',WORKERS,flush=True)
    t0=time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(process_one,(i,p)):i for i,p in enumerate(paths)}
        done=0
        for fut in as_completed(futs):
            sid,secs,sz=fut.result(); done+=1
            print(f'[{done:02d}/36] shard {sid:04d}: {secs}  file={sz/2**20:.1f} MiB',flush=True)
    print('local pass seconds',time.time()-t0,flush=True)

    print('MERGE local candidates',flush=True)
    c=Counter()
    for sid in range(36):
        with gzip.open(LOC/f'vocab_{sid:04d}.pkl.gz','rb') as g:
            terms,tf=pickle.load(g)
        c.update(dict(zip(terms.tolist(),tf.tolist())))
        if sid%4==3: print(' merged',sid+1,'union',len(c),flush=True)
    cand=c.most_common(200_000)
    with gzip.open(OUT/'global_candidates_200k.pkl.gz','wb',compresslevel=1) as g:
        pickle.dump(cand,g,protocol=5)
    print('candidate union',len(c),'saved',len(cand),flush=True)
    print('top20',cand[:20],flush=True)
