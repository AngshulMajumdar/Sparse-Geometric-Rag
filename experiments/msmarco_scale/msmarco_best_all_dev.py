from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx
LLEX=np.float32(4.0); LSEM=np.float32(0.1)

def rank(p,k=100):
    if p is None:return []
    docs=p['cand_docs'][:b.P]; ts=p['cand_tail'][:b.P]; lx=p['lex'][:b.P]; sm=p['sem'][:b.P]
    fin=m.zscore(ts)+LLEX*m.zscore(lx)+LSEM*m.zscore(sm); oo=np.argsort(fin)[::-1][:k]
    return [int(x) for x in docs[oo]]

df=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(df['query-id'].to_numpy())]; del df
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'dev.tsv',ids,positive_only=True)
_=b.prepare(texts[ids[0]])
run={}; times=[]; routehit=0; poolhit=0; den=0; cands=[]
for z,qid in enumerate(ids):
    t=time.perf_counter(); p=b.prepare(texts[qid]); run[qid]=rank(p); times.append((time.perf_counter()-t)*1000)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if p:
        cands.append(p['candidate_docs']); ud=p['ud']; pool=p['cand_docs'][:b.P]
        for d in rels:
            kk=np.searchsorted(ud,d); routehit+=int(kk<len(ud) and int(ud[kk])==d); poolhit+=int(np.any(pool==d))
    if (z+1)%500==0: print('dev',z+1,'median',float(np.median(times)),'p95',float(np.percentile(times,95)),flush=True)
met=m.eval_run(run,qrels)
out={'protocol':'all params locked from TRAIN validation: gamma_tail=.25 lambda_M=.125 h=0 lambda_lex=4 lambda_sem=.1 P=2000; full DEV untouched','params':{'gamma_tail':0.25,'lambda_M':0.125,'h':0,'lambda_lex':4.0,'lambda_sem':0.1,'P':2000,'S':16},'dev_metrics':met,'route_relevant_recall':routehit/den,'pool_relevant_recall':poolhit/den,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(out,open(WORK/'best_all_dev_results.json','w'),indent=2); print('DEV',met,flush=True); print('SUMMARY',out,flush=True)
