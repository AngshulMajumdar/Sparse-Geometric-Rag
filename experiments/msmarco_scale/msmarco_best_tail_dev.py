from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK
idx=b.idx

df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True)
_=b.prepare(texts[ids[0]])
run={}; times=[]; cands=[]; routehit=0; poolhit=0; den=0
for z,qid in enumerate(ids):
    t=time.perf_counter(); p=b.prepare(texts[qid]); run[qid]=b.rank_h(p,0,100); times.append((time.perf_counter()-t)*1000)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if p:
        cands.append(p['candidate_docs']); ud=p['ud']; pool=p['cand_docs'][:b.P]
        for d in rels:
            k=np.searchsorted(ud,d); routehit+=int(k<len(ud) and int(ud[k])==d); poolhit+=int(np.any(pool==d))
    if (z+1)%250==0: print('dev',z+1,'median',float(np.median(times)),'p95',float(np.percentile(times,95)),'route',routehit/max(1,den),'pool',poolhit/max(1,den),flush=True)
met=m.eval_run(run,qrels)
out={'protocol':'gamma_tail=0.25, lambda_M=0.125, h=0 locked from deterministic 1000-query TRAIN validation; full DEV untouched','gamma_tail':0.25,'lambda_M':0.125,'h':0,'P':b.P,'dev_metrics':met,'route_relevant_recall':routehit/den,'pool_relevant_recall':poolhit/den,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(out,open(WORK/'best_tail_dev_results.json','w'),indent=2); print('DEV',met,flush=True); print('SUMMARY',out,flush=True)
