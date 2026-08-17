from __future__ import annotations
import sys,time,json,math
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; M=m.M; S=m.S; P=2000
GAMMA=np.float32(0.25); LAM=np.float32(0.125); HGRID=list(range(0,11))
set_num_threads(5); idx=m.FullIndex(); print('loaded',idx.meta,flush=True)

@njit(parallel=True,cache=False)
def score_components(rslot,mem,rt,sbits,q_dense,rho,cent_local,rel_local):
    K=len(rslot); base=np.zeros(K,np.float32); sig=np.zeros(K,np.float32); cons=np.zeros(K,np.float32)
    for z in prange(K):
        u=int(rslot[z]); local=0.0; sg=0.0; bits=sbits[z]
        for r in range(S):
            t=int(rt[z,r])
            if t==65535: continue
            qv=q_dense[t]; cen=cent_local[u,t]; rel=rel_local[u,t]; rel=1.0 if rel==0.0 else rel
            sgn=1.0 if ((bits>>r)&1)!=0 else -1.0
            local += rel*(qv-cen)*sgn; sg += qv*qv
        c=mem[z]*rho[u]; base[z]=c*local; sig[z]=sg; cons[z]=c
    return base,sig,cons

def prepare(text):
    q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
    spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
    if not spans:return None
    docs=np.concatenate([np.asarray(idx.pd[a:b]) for j,a,b in spans]).astype(np.uint32,copy=False)
    mm=np.concatenate([np.asarray(idx.pm[a:b]) for j,a,b in spans]).astype(np.float32,copy=False)
    rt=np.concatenate([np.asarray(idx.pr[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
    sb=np.concatenate([np.asarray(idx.ps[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
    nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
    for u,(j,a,b) in enumerate(spans):
        rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]
        ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
    rslot=np.concatenate([np.full(b-a,u,dtype=np.uint8) for u,(j,a,b) in enumerate(spans)])
    base,sig,cons=score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
    ud,inv=np.unique(docs,return_inverse=True)
    head=np.bincount(inv,weights=base*np.sqrt(sig),minlength=len(ud)).astype(np.float32)
    tail=np.bincount(inv,weights=base*np.power(sig,GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
    hmax=max(HGRID); hw=min(len(head),max(1,hmax)); hi=np.argpartition(head,-hw)[-hw:] if len(head)>hw else np.arange(len(head)); ho=hi[np.argsort(head[hi])[::-1]]
    want=min(len(tail),P+hmax+8); ci=np.argpartition(tail,-want)[-want:] if len(tail)>want else np.arange(len(tail)); co=ci[np.argsort(tail[ci])[::-1]]
    cand_docs=ud[co]; cand_tail=tail[co]
    lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,b=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:b][:m.SEMK]; sv=idx.A.data[a:b][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
    lex,sem=m.score_support_pool(cand_docs,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
    return {'ud':ud,'head_order':ho,'cand_docs':cand_docs,'cand_tail':cand_tail,'lex':lex,'sem':sem,'candidate_docs':len(ud),'candidate_memberships':len(docs)}

def rank_h(p,h,k=100):
    if p is None:return []
    ud=p['ud']; frozen=ud[p['head_order'][:min(h,len(ud))]] if h else np.empty(0,np.uint32); fs=set(map(int,frozen.tolist()))
    keep=np.asarray([int(d) not in fs for d in p['cand_docs']],bool); docs=p['cand_docs'][keep][:P]; ts=p['cand_tail'][keep][:P]; lx=p['lex'][keep][:P]; sm=p['sem'][keep][:P]
    final=m.zscore(ts)+m.LAMBDA_LEX*m.zscore(lx)+m.LAMBDA_SEM*m.zscore(sm); oo=np.argsort(final)[::-1]
    return [int(x) for x in np.concatenate([frozen,docs[oo]])[:k]]

tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
_=prepare(texts[ids[0]])
runs={h:{} for h in HGRID}; times=[]; cands=[]; poolhit=0; routehit=0; den=0
for z,qid in enumerate(ids):
    t=time.perf_counter(); p=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if p:
        cands.append(p['candidate_docs']); ud=p['ud']; pool=p['cand_docs'][:P]
        for d in rels:
            k=np.searchsorted(ud,d); routehit+=int(k<len(ud) and int(ud[k])==d); poolhit+=int(np.any(pool==d))
    for h in HGRID:runs[h][qid]=rank_h(p,h,100)
    if (z+1)%100==0: print('q',z+1,'median',float(np.median(times)),'route',routehit/max(1,den),'pool',poolhit/max(1,den),flush=True)
rows={}
for h in HGRID:
    met=m.eval_run(runs[h],qrels); rows[h]=met; print('H',h,met,flush=True)
best=max(rows,key=lambda h:(rows[h]['nDCG@10'],rows[h]['MRR@10'],rows[h]['R@100']))
out={'gamma_tail':float(GAMMA),'lambda_M':float(LAM),'P':P,'hgrid':rows,'best_h':int(best),'route_relevant_recall':routehit/den,'pool_relevant_recall':poolhit/den,'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'avg_candidate_docs':float(np.mean(cands))}
json.dump(out,open(WORK/'best_tail_validation.json','w'),indent=2); print('BEST',best,rows[best],flush=True); print('summary',out,flush=True)
