import sys,time,json,math
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0,'/mnt/data/work_code/geomretrieval_msmarco_scale_codebase/original_v0')
from geomretrieval import GeometricIndex, load_beir_directory, evaluate_run
ROOT='/mnt/data/work_treccovid/trec-covid'; IDX='/mnt/data/treccovid_geom_index'
ds=load_beir_directory(ROOT,'test'); idx=GeometricIndex.load(IDX)
# Exact TF-IDF cosine
run={}; times=[]
for qid in ds.qrels:
    q=idx._query_vector(ds.queries[qid])
    t=time.perf_counter(); s=np.asarray(idx.X @ q.T).ravel(); times.append((time.perf_counter()-t)*1000)
    k=min(100,len(s)); ii=np.argpartition(s,-k)[-k:]; ii=ii[np.argsort(s[ii])[::-1]]
    run[str(qid)]=[str(idx.doc_ids[int(i)]) for i in ii]
mtf=evaluate_run(run,ds.qrels,ks=(10,100),ndcg_k=10,mrr_k=10,exp_gain=False)
mtf['median_ms']=float(np.median(times)); mtf['p95_ms']=float(np.percentile(times,95))
print('TFIDF',mtf,flush=True)
# Count matrix with frozen vocabulary/tokenizer
vec=CountVectorizer(vocabulary=idx.vectorizer.vocabulary_,lowercase=idx.config.lowercase,token_pattern=idx.config.token_pattern,dtype=np.float32)
t=time.perf_counter(); C=vec.transform(ds.corpus_texts).tocsr(); build=time.perf_counter()-t
length=np.asarray(C.sum(axis=1)).ravel().astype(np.float32); avdl=float(length.mean()); Cc=C.tocsc(); N=C.shape[0]
df=np.diff(Cc.indptr).astype(np.float64); idf=np.log((N-df+0.5)/(df+0.5)+1.0).astype(np.float32)
k1=.9;b=.4
run={};times=[]
for qid in ds.qrels:
    q=vec.transform([ds.queries[qid]]).tocsr(); terms=q.indices
    t=time.perf_counter(); score=np.zeros(N,np.float32)
    for term in terms:
        a,bb=Cc.indptr[term],Cc.indptr[term+1]; docs=Cc.indices[a:bb]; tf=Cc.data[a:bb]
        den=tf+k1*(1-b+b*length[docs]/avdl)
        score[docs]+=idf[term]*(tf*(k1+1)/den)
    times.append((time.perf_counter()-t)*1000)
    k=min(100,N);ii=np.argpartition(score,-k)[-k:];ii=ii[np.argsort(score[ii])[::-1]]
    run[str(qid)]=[str(idx.doc_ids[int(i)]) for i in ii]
mb=evaluate_run(run,ds.qrels,ks=(10,100),ndcg_k=10,mrr_k=10,exp_gain=False)
mb['median_ms_effectiveness_impl']=float(np.median(times));mb['p95_ms_effectiveness_impl']=float(np.percentile(times,95));mb['count_build_s']=build
print('BM25',mb,flush=True)
out={'tfidf':mtf,'bm25':mb}
json.dump(out,open('/mnt/data/treccovid_local_baselines.json','w'),indent=2)
