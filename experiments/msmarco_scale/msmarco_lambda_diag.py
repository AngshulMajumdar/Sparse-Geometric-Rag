import sys,time,numpy as np,pandas as pd,json
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_fastp as m
from msmarco_full_search_fastp import FullIndex,load_query_texts,qrels_from_tsv,eval_run,ROOT,WORK,zscore
P=8000; NS=1000
LAM=[0,0.25,0.5,1,2.5,5,10,25]
idx=FullIndex(); print('loaded',flush=True)
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=NS,replace=False)]; del tr
texts=load_query_texts(ids); qrels=qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
w=idx.prepare(texts[ids[0]],hmax=1,pool_max=P); del w
runs={x:{} for x in LAM}; purelex={}; puretail={}; times=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=1,pool_max=P); times.append((time.perf_counter()-t)*1000)
 if pp is None:
  for lam in LAM: runs[lam][qid]=[]
  purelex[qid]=[]; puretail[qid]=[]
 else:
  docs=pp['cand_docs'][:P]; zt=zscore(pp['cand_tail'][:P]); zl=zscore(pp['lex'][:P]); zs=zscore(pp['sem'][:P])
  for lam in LAM:
   score=zt+lam*zl+0.05*zs; oo=np.argsort(score)[::-1][:100]; runs[lam][qid]=[int(x) for x in docs[oo]]
  purelex[qid]=[int(x) for x in docs[np.argsort(zl)[::-1][:100]]]
  puretail[qid]=[int(x) for x in docs[np.argsort(zt)[::-1][:100]]]
 if (z+1)%100==0: print('q',z+1,'med',np.median(times),flush=True)
res={}
for lam in LAM:
 res[str(lam)]=eval_run(runs[lam],qrels); print('LAM',lam,res[str(lam)],flush=True)
res['purelex']=eval_run(purelex,qrels); res['puretail']=eval_run(puretail,qrels); print('PURELEX',res['purelex'],flush=True); print('PURETAIL',res['puretail'],flush=True)
json.dump({'P':P,'results':res,'timing':{'median':float(np.median(times)),'p95':float(np.percentile(times,95))}},open(WORK/'lambda_diag.json','w'),indent=2)
