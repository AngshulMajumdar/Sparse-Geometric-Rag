import sys,time,numpy as np,pandas as pd,json
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_fastp import FullIndex,load_query_texts,qrels_from_tsv,eval_run,ROOT,WORK,zscore
NS=500; BIG=1_000_000
LAM=[2.5,5.0]
idx=FullIndex(); print('loaded',flush=True)
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)[:NS]]; del tr
texts=load_query_texts(ids); qrels=qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
w=idx.prepare(texts[ids[0]],hmax=1,pool_max=BIG); del w
runs={l:{} for l in LAM}; times=[]; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=1,pool_max=BIG); times.append((time.perf_counter()-t)*1000)
 if pp is None:
  for l in LAM:runs[l][qid]=[]
 else:
  docs=pp['cand_docs']; cands.append(len(docs)); zt=zscore(pp['cand_tail']); zl=zscore(pp['lex']); zs=zscore(pp['sem'])
  for l in LAM:
   sc=zt+l*zl+0.05*zs; ix=np.argpartition(sc,-min(100,len(sc)))[-min(100,len(sc)):]; ix=ix[np.argsort(sc[ix])[::-1]]; runs[l][qid]=[int(x) for x in docs[ix]]
 if (z+1)%50==0: print(z+1,'median_ms',np.median(times),'p95',np.percentile(times,95),'avgcand',np.mean(cands),flush=True)
for l in LAM: print('LAM',l,eval_run(runs[l],qrels),flush=True)
