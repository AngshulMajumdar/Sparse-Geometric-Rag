import sys,time,json,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_fast import FullIndex,load_query_texts,qrels_from_tsv,eval_run,ROOT,WORK,P
BEST_H=0
idx=FullIndex(); print('loaded',idx.meta,flush=True)
df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=load_query_texts(ids); qrels=qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True)
# warm compile/pages, not timed
w=idx.prepare(texts[ids[0]],hmax=1); idx.rank_h(w,0,100); del w
run={}; times=[]; cands=[]; mems=[]
route_num=pool_num=rel_den=0
for z,qid in enumerate(ids):
 t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=1); rank=idx.rank_h(pp,BEST_H,100); dt=(time.perf_counter()-t)*1000
 run[qid]=rank; times.append(dt); cands.append(pp['candidate_docs'] if pp else 0); mems.append(pp['candidate_memberships'] if pp else 0)
 rel=[int(d) for d,r in qrels[qid].items() if r>0]; rel_den+=len(rel)
 if pp is not None:
  ud=pp['ud']; pool=pp['cand_docs'][:P]
  for d in rel:
   k=np.searchsorted(ud,d); route_num += int(k<len(ud) and int(ud[k])==d); pool_num += int(np.any(pool==d))
 if (z+1)%500==0:
  print('dev',z+1,'median_ms',float(np.median(times)),'p95_ms',float(np.percentile(times,95)),'avgcand',float(np.mean(cands)),'route_rel',route_num/rel_den,'pool_rel',pool_num/rel_den,flush=True)
m=eval_run(run,qrels)
timing={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands)),'median_candidate_docs':float(np.median(cands)),'avg_candidate_memberships':float(np.mean(mems))}
stages={'routed_relevant_recall':route_num/rel_den,'pool_P2000_relevant_recall':pool_num/rel_den,'final_R100':m['R@100'],'total_positive_qrels':rel_den}
out={'protocol':'h=0 selected on deterministic 1000-query TRAIN validation sweep; full 6980-query DEV untouched','best_h':0,'dev_metrics':m,'timing':timing,'stage_diagnostics':stages,'index_meta':idx.meta,'geometry_note':'full 8.84M vocabulary/IDF; geometric codebook calibrated on first 1M passages','physical_optimizations':'branch-sorted postings; query-local dense center/reliability lookup; global compact support CSR; ranking verified identical on regression queries'}
with open(WORK/'full_msmarco_dev_fast_results.json','w') as f: json.dump(out,f,indent=2)
print('DEV_METRICS',m,flush=True); print('TIMING',timing,flush=True); print('STAGES',stages,flush=True); print('saved',WORK/'full_msmarco_dev_fast_results.json',flush=True)
