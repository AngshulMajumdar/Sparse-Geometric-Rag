from __future__ import annotations
import gzip,json,pickle,time,re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT=Path('/mnt/data'); OUT=ROOT/'msmarco_scale_work'; EXACT=OUT/'exact_candidate_counts'; EXACT.mkdir(parents=True,exist_ok=True)
WORKERS=3; N_DOCS=8_841_823; FINAL_K=50_000
pat=re.compile(r'corpus_(\d{4})')

def shard_paths():
 out={}
 for p in ROOT.glob('corpus_*.jsonl*.gz'):
  m=pat.search(p.name)
  if m: out[int(m.group(1))]=p
 return [out[i] for i in sorted(out)]

def load_candidates():
 with gzip.open(OUT/'global_candidates_200k.pkl.gz','rb') as g: cand=pickle.load(g)
 # Candidate vocabulary MUST be alphabetically indexed before the sklearn-style
 # max_features frequency limit is applied.
 terms=sorted([t for t,_ in cand])
 return terms,{t:i for i,t in enumerate(terms)}

def one(arg):
 sid,p=arg; dst=EXACT/f'counts_{sid:04d}.npz'
 if dst.exists(): return sid,'cached'
 terms,vocab=load_candidates()
 texts=[]
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:
   o=json.loads(line); texts.append(((o.get('title') or '')+' '+(o.get('text') or '')).strip())
 cv=CountVectorizer(vocabulary=vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32)
 X=cv.transform(texts).tocsr()
 tf=np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
 # exact document frequency among candidate terms
 X.data[:] = 1
 df=np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
 np.savez(dst,tf=tf,df=df)
 return sid, X.nnz

if __name__=='__main__':
 terms,vocab=load_candidates(); M=len(terms); print('exact candidate terms',M,flush=True)
 paths=shard_paths(); t0=time.time()
 with ProcessPoolExecutor(max_workers=WORKERS) as ex:
  futs={ex.submit(one,(i,p)):i for i,p in enumerate(paths)}; done=0
  for fut in as_completed(futs):
   sid,x=fut.result(); done+=1; print(f'[{done:02d}/36] shard {sid:04d}: {x}',flush=True)
 print('pass seconds',time.time()-t0,flush=True)
 tf=np.zeros(M,np.int64); df=np.zeros(M,np.int64)
 for sid in range(36):
  z=np.load(EXACT/f'counts_{sid:04d}.npz'); tf+=z['tf']; df+=z['df']
 # sklearn CountVectorizer(max_features=K) indexes features alphabetically first,
 # then takes (-tf).argsort()[:K].  Replicate that selection.
 keep=(-tf).argsort()[:FINAL_K]
 keep=np.sort(keep) # final vocabulary coordinates remain alphabetical
 final_terms=np.asarray(terms,dtype=object)[keep]
 final_tf=tf[keep]; final_df=df[keep]
 idf=(np.log((1.0+N_DOCS)/(1.0+final_df.astype(np.float64)))+1.0).astype(np.float32)
 with gzip.open(OUT/'final_vocab_50k.pkl.gz','wb',compresslevel=1) as g:
  pickle.dump({'terms':final_terms,'tf':final_tf,'df':final_df,'idf':idf,'N':N_DOCS},g,protocol=5)
 print('final vocab',len(final_terms), 'df range',int(final_df.min()),int(final_df.max()),flush=True)
 print('top tf terms',sorted(zip(final_terms.tolist(),final_tf.tolist()), key=lambda x:x[1], reverse=True)[:20],flush=True)
