from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK
LEX=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.5,10.0]
SEM=[0.0,0.05,0.1,0.25]
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
_=b.prepare(texts[ids[0]])
runs={(ll,ss):{} for ll in LEX for ss in SEM}; times=[]
for z,qid in enumerate(ids):
    t=time.perf_counter(); p=b.prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
    if p is None:
        for k in runs:runs[k][qid]=[]
        continue
    docs=p['cand_docs'][:b.P]; ts=p['cand_tail'][:b.P]; lx=p['lex'][:b.P]; sm=p['sem'][:b.P]
    zt=m.zscore(ts); zl=m.zscore(lx); zs=m.zscore(sm)
    for ll in LEX:
        base=zt+np.float32(ll)*zl
        for ss in SEM:
            fin=base+np.float32(ss)*zs; oo=np.argsort(fin)[::-1][:100]; runs[(ll,ss)][qid]=[int(x) for x in docs[oo]]
    if (z+1)%100==0: print('q',z+1,'median',float(np.median(times)),flush=True)
rows=[]
for ll in LEX:
  for ss in SEM:
    met=m.eval_run(runs[(ll,ss)],qrels); row={'lambda_lex':ll,'lambda_sem':ss,**met}; rows.append(row); print('W',ll,ss,met,flush=True)
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'new tail fixed gamma=.25 lambda_M=.125 P=2000 h=0; final weights tuned on deterministic 1000 TRAIN validation','rows':rows,'best':rows[0],'median_prepare_ms':float(np.median(times))}
json.dump(out,open(WORK/'final_weight_sweep.json','w'),indent=2); print('BEST',rows[0],flush=True)
