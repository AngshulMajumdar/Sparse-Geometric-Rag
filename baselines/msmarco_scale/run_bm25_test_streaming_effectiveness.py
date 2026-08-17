from __future__ import annotations
import gzip,pickle,json,re,csv,math,time,heapq,glob,os
from pathlib import Path
from collections import Counter,defaultdict
from multiprocessing import Pool, get_context
import numpy as np

ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; OUT=WORK/'baselines'; OUT.mkdir(exist_ok=True)
N=8_841_823; TOPK=1000
# Same token boundary as frozen system / sklearn token_pattern for query terms.
TOKEN_RE=re.compile(r'(?u)\b\w\w+\b')

# Load qids and query texts
qrels_rows=[]; qids=set()
with open(ROOT/'test.tsv',newline='') as f:
    r=csv.DictReader(f,delimiter='\t')
    for x in r:
        q=str(x['query-id']); d=int(x['corpus-id']); rel=float(x['score']); qids.add(q); qrels_rows.append((q,d,rel))
qids=sorted(qids,key=lambda x:int(x)); qindex={q:i for i,q in enumerate(qids)}
texts={}
with open(ROOT/'queries.jsonl') as f:
    for line in f:
        o=json.loads(line); q=str(o['_id'])
        if q in qids: texts[q]=o['text']
assert len(texts)==len(qids)==43
qtoks={q:TOKEN_RE.findall(texts[q].lower()) for q in qids}
allterms=sorted(set(t for z in qtoks.values() for t in z))

# full-corpus known df from the exact 50k vocabulary build
with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as f:z=pickle.load(f)
terms50=z['terms'].tolist(); df50=np.asarray(z['df']); term2df={t:int(df50[i]) for i,t in enumerate(terms50)}
oov=sorted(t for t in allterms if t not in term2df)
print('queries',len(qids),'query terms',len(allterms),'oov',oov,flush=True)

# locate 36 shards in order
def shard_path(i):
    hits=list(ROOT.glob(f'corpus_{i:04d}.jsonl*.gz'))
    if i==35 and len(hits)==0: hits=list((ROOT/'restored').glob(f'corpus_{i:04d}.jsonl*.gz'))
    assert len(hits)==1,(i,hits)
    return str(hits[0])
SHARDS=[shard_path(i) for i in range(36)]

# Exact OOV df with one cheap corpus scan, only 10 rare strings.
OOV_PAT=re.compile(r'(?u)\b(?:'+'|'.join(re.escape(t) for t in sorted(oov,key=len,reverse=True))+r')\b') if oov else None

def count_oov_one(args):
    sid,path=args; c=Counter(); n=0; t0=time.time()
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            o=json.loads(line); tx=((o.get('title') or '')+' '+(o.get('text') or '')).lower(); n+=1
            if OOV_PAT:
                m=set(OOV_PAT.findall(tx))
                for t in m:c[t]+=1
    return sid,n,dict(c),time.time()-t0

if oov:
    t0=time.time();
    with get_context('fork').Pool(processes=8) as p:
        rr=p.map(count_oov_one,list(enumerate(SHARDS)))
    oo=Counter();
    for sid,n,c,sec in rr: oo.update(c)
    for t in oov: term2df[t]=int(oo[t])
    print('oov df',dict(oo),'sec',time.time()-t0,flush=True)

# Robertson BM25 idf; base parameters match common Anserini MS MARCO defaults.
def make_idf():
    return {t:math.log(1.0+(N-term2df[t]+0.5)/(term2df[t]+0.5)) for t in allterms}
idf=make_idf()
# Query term multiplicity mapping term -> [(query index,multiplicity)]
t2q=defaultdict(list)
for q in qids:
    c=Counter(qtoks[q])
    for t,m in c.items(): t2q[t].append((qindex[q],m))
# compiled union matcher exact same word boundaries, avoids tokenizing irrelevant words
PAT=re.compile(r'(?u)\b(?:'+'|'.join(re.escape(t) for t in sorted(allterms,key=len,reverse=True))+r')\b')
avgdl=float(json.load(open(WORK/'full_index/meta.json'))['avg_doc_length'])
# existing exact tokenizer document lengths
DL_PATH=str(WORK/'full_index/doc_lengths.u16')

# Each shard returns top1000 per q. Global top1000 is exactly merge of shard top1000.
def score_shard(args):
    sid,path,k1,b=args
    dlarr=np.memmap(DL_PATH,dtype=np.uint16,mode='r',shape=(N,))
    heaps=[[] for _ in qids]
    n=0; matched=0; t0=time.time()
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            o=json.loads(line); d=int(o['_id']); tx=((o.get('title') or '')+' '+(o.get('text') or '')).lower(); n+=1
            mm=PAT.findall(tx)
            if not mm: continue
            matched+=1; tf=Counter(mm); norm=k1*(1.0-b+b*(float(dlarr[d])/avgdl))
            qs={}
            for term,freq in tf.items():
                w=idf[term]*((freq*(k1+1.0))/(freq+norm))
                for qi,qm in t2q[term]: qs[qi]=qs.get(qi,0.0)+w*qm
            for qi,sc in qs.items():
                h=heaps[qi]; item=(float(sc),-d) # lower docid wins exact tie
                if len(h)<TOPK: heapq.heappush(h,item)
                elif item>h[0]: heapq.heapreplace(h,item)
    return sid,heaps,n,matched,time.time()-t0

def retrieve(k1,b,label):
    t0=time.time()
    with get_context('fork').Pool(processes=8) as p:
        rr=p.map(score_shard,[(i,SHARDS[i],k1,b) for i in range(36)])
    globalh=[[] for _ in qids]
    for sid,heaps,n,matched,sec in rr:
        for qi,h in enumerate(heaps):
            gh=globalh[qi]
            for item in h:
                if len(gh)<TOPK: heapq.heappush(gh,item)
                elif item>gh[0]: heapq.heapreplace(gh,item)
    run={}
    for qi,q in enumerate(qids):
        arr=sorted(globalh[qi],reverse=True); run[q]=[-negd for sc,negd in arr]
    sec=time.time()-t0
    print(label,'retrieval scan sec',sec,flush=True)
    return run,sec

# TREC-compliant eval: passage level 1 = related/not relevant for binary metrics. nDCG remains graded.
qrels=defaultdict(dict)
for q,d,r in qrels_rows:qrels[q][d]=r

def evaluate(run):
    vals=defaultdict(list)
    for q in qids:
        qr=qrels[q]; rank=run.get(q,[])
        binary={d for d,r in qr.items() if r>=2.0}; den=max(1,len(binary))
        h10=sum(d in binary for d in rank[:10]); h100=sum(d in binary for d in rank[:100])
        vals['P@10'].append(h10/10); vals['R@10'].append(h10/den); vals['R@100'].append(h100/den); vals['Hit@10'].append(float(h10>0)); vals['Hit@100'].append(float(h100>0))
        rr=0.0
        for i,d in enumerate(rank[:10],1):
            if d in binary: rr=1.0/i;break
        vals['MRR@10'].append(rr)
        obs=[qr.get(d,0.0) for d in rank[:10]]; ideal=sorted(qr.values(),reverse=True)[:10]
        dcg=sum(r/math.log2(i+2) for i,r in enumerate(obs)); idcg=sum(r/math.log2(i+2) for i,r in enumerate(ideal)); vals['nDCG@10'].append(dcg/idcg if idcg else 0.0)
    return {k:float(np.mean(v)) for k,v in vals.items()} | {'n_queries':len(qids),'n_binary_relevant':sum(r>=2 for _,_,r in qrels_rows)}

if __name__=='__main__':
    # Common MS MARCO/Anserini-like BM25 and a conventional default variant for sensitivity.
    out={'tokenizer':'lowercase regex (?u)\\b\\w\\w+\\b; no stemming','N':N,'avgdl':avgdl,'oov_df':{t:term2df[t] for t in oov}}
    for k1,b,label in [(0.9,0.4,'bm25_k1_0.9_b_0.4'),(1.2,0.75,'bm25_k1_1.2_b_0.75')]:
        run,sec=retrieve(k1,b,label); met=evaluate(run); out[label]={'k1':k1,'b':b,'metrics':met,'full_corpus_parallel_scan_seconds':sec,'run_top1000':run}; print(label,met,flush=True)
    json.dump(out,open(OUT/'bm25_local_test.json','w'),indent=2)
    print('SAVED',OUT/'bm25_local_test.json',flush=True)
