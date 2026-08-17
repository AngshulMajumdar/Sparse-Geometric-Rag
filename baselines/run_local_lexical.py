from __future__ import annotations
"""Local lexical references for sanity checking, not the headline speed baseline.

The BM25 implementation here is an effectiveness implementation over the loaded
corpus. Do not present its latency as a production inverted-index latency.
"""
import argparse,json,time,math,re
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from geomretrieval import load_beir_zip,load_beir_directory,evaluate_run

def load_dataset(path,split):
    return load_beir_zip(path,split) if str(path).lower().endswith('.zip') else load_beir_directory(path,split)

def topk(x,k):
    k=min(k,len(x)); ii=np.argpartition(x,-k)[-k:] if k<len(x) else np.arange(len(x)); return ii[np.argsort(x[ii])[::-1]]

def tfidf(ds):
    v=TfidfVectorizer(dtype=np.float32,norm='l2'); X=v.fit_transform(ds.corpus_texts); run={}; times=[]
    qids=[q for q in ds.qrels if q in ds.queries]
    for qid in qids:
        t=time.perf_counter(); q=v.transform([ds.queries[qid]]); s=(X@q.T).toarray().ravel(); ii=topk(s,100); times.append((time.perf_counter()-t)*1000);run[qid]=[ds.corpus_ids[i] for i in ii]
    m=evaluate_run(run,ds.qrels,ks=(10,100),ndcg_k=10,mrr_k=10,exp_gain=False);m['median_ms']=float(np.median(times));m['p95_ms']=float(np.percentile(times,95));return m

def bm25_effectiveness(ds,k1=.9,b=.4):
    tok=lambda s: re.findall(r'(?u)\\b\\w\\w+\\b',s.lower())
    docs=[tok(x) for x in ds.corpus_texts]; N=len(docs); lens=np.array([len(x) for x in docs],np.float32);avg=max(1,float(lens.mean()));df=Counter()
    for x in docs: df.update(set(x))
    postings={}
    for di,x in enumerate(docs):
        c=Counter(x)
        for t,tf in c.items(): postings.setdefault(t,[]).append((di,tf))
    run={};times=[]
    for qid in [q for q in ds.qrels if q in ds.queries]:
        t0=time.perf_counter();score=np.zeros(N,np.float32)
        for term in tok(ds.queries[qid]):
            plist=postings.get(term,()); n=df.get(term,0);idf=math.log(1+(N-n+.5)/(n+.5))
            for d,tf in plist:
                den=tf+k1*(1-b+b*lens[d]/avg);score[d]+=idf*(tf*(k1+1))/den
        ii=topk(score,100);times.append((time.perf_counter()-t0)*1000);run[qid]=[ds.corpus_ids[i] for i in ii]
    m=evaluate_run(run,ds.qrels,ks=(10,100),ndcg_k=10,mrr_k=10,exp_gain=False);m['median_ms_effectiveness_impl']=float(np.median(times));m['p95_ms_effectiveness_impl']=float(np.percentile(times,95));return m

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--split',default='test');p.add_argument('--output',default=None);a=p.parse_args();ds=load_dataset(a.dataset,a.split)
    out={'tfidf':tfidf(ds),'bm25':bm25_effectiveness(ds)};print(json.dumps(out,indent=2));
    if a.output: open(a.output,'w').write(json.dumps(out,indent=2))
if __name__=='__main__':main()
