import sys,time,numpy as np,pandas as pd,json
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_fastp import FullIndex,load_query_texts,qrels_from_tsv,eval_run,ROOT,WORK
PGRID=[500,1000,2000,4000,8000,16000]
MAXP=max(PGRID); NS=1000
idx=FullIndex(); print('loaded',flush=True)
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=NS,replace=False)]; del tr
texts=load_query_texts(ids); qrels=qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
# warm
w=idx.prepare(texts[ids[0]],hmax=1,pool_max=MAXP); idx.rank_h(w,0,100,pool=2000); del w
runs={p:{} for p in PGRID}; pool_hit={p:0 for p in PGRID}; route_hit=0; den=0; times=[]; cands=[]
for z,qid in enumerate(ids):
 t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=1,pool_max=MAXP); times.append((time.perf_counter()-t)*1000); cands.append(pp['candidate_docs'] if pp else 0)
 rel=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rel)
 if pp:
  ud=pp['ud']
  for d in rel:
   k=np.searchsorted(ud,d); route_hit+=int(k<len(ud) and int(ud[k])==d)
   for pool in PGRID: pool_hit[pool]+=int(np.any(pp['cand_docs'][:pool]==d))
  for pool in PGRID: runs[pool][qid]=idx.rank_h(pp,0,100,pool=pool)
 else:
  for pool in PGRID: runs[pool][qid]=[]
 if (z+1)%100==0: print('prepared',z+1,'median',np.median(times),'route',route_hit/den,flush=True)
rows={}
for pool in PGRID:
 m=eval_run(runs[pool],qrels); m['pool_relevant_recall']=pool_hit[pool]/den; rows[pool]=m; print('P',pool,m,flush=True)
out={'PGRID':PGRID,'route_relevant_recall':route_hit/den,'metrics':rows,'prepare_maxP_timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))},'avg_candidate_docs':float(np.mean(cands))}
json.dump(out,open(WORK/'msmarco_p_sweep.json','w'),indent=2); print('saved',flush=True)
