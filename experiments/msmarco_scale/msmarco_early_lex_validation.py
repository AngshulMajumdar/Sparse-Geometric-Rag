from __future__ import annotations
import sys,time,json
import numpy as np, pandas as pd
from numba import set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
LLEX=np.float32(4.0); LSEM=np.float32(0.1)
ETAS=[0.0,0.0625,0.125,0.25,0.5,1.0,2.0,4.0,8.0]
LEX_QUOTAS=[0,100,250,500,750,1000,1250,1500,1750,2000]
set_num_threads(5)

def topk_desc(score,k):
    n=len(score); k=min(k,n)
    if k<=0:return np.empty(0,np.int64)
    if n>k:
        ii=np.argpartition(score,-k)[-k:]
        return ii[np.argsort(score[ii])[::-1]]
    return np.argsort(score)[::-1]

def prepare_all(text):
    q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
    spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
    if not spans:return None
    docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False)
    mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False)
    rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
    sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
    nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
    for u,(j,a,bb) in enumerate(spans):
        rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]
        ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
    rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)])
    base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
    ud,inv=np.unique(docs,return_inverse=True)
    tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
    lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]
    # Whole-document lexical + tiny semantic score for every routed document.
    # This is used only to amortize the validation sweep; the locked deployable
    # implementation below computes semantics only for the selected P.
    semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]
        semvec[nb]+=float(amp)*sv*idx.idf[nb]
    lex,sem=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
    return {'ud':ud,'tail':tail,'lex':lex,'sem':sem,'candidate_docs':len(ud),'candidate_memberships':len(docs)}

def final_rank(p, sel_idx, k=100):
    docs=p['ud'][sel_idx]; ts=p['tail'][sel_idx]; lx=p['lex'][sel_idx]; sm=p['sem'][sel_idx]
    fin=m.zscore(ts)+LLEX*m.zscore(lx)+LSEM*m.zscore(sm)
    oo=np.argsort(fin)[::-1][:k]
    return [int(x) for x in docs[oo]]

def select_direct(p,eta):
    zt=m.zscore(p['tail']); zl=m.zscore(p['lex']); return topk_desc(zt+np.float32(eta)*zl,P)

def select_quota(p,lq):
    # Exactly P slots: preserve P-lq strongest geometric docs, then add strongest
    # lexical docs not already admitted. If duplicates cause shortage, continue
    # down the lexical ordering until P unique docs are selected.
    n=len(p['ud']); k=min(P,n); gq=max(0,k-min(int(lq),k))
    gt=topk_desc(p['tail'],gq)
    if len(gt)==k:return gt
    chosen=np.zeros(n,np.uint8); chosen[gt]=1
    lo=np.argsort(p['lex'])[::-1]
    out=np.empty(k,np.int64); out[:len(gt)]=gt; z=len(gt)
    for ii in lo:
        if chosen[ii]==0:
            chosen[ii]=1; out[z]=ii; z+=1
            if z==k:break
    return out[:z]

# deterministic validation split already used elsewhere
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
# warm up both support kernels and geometric kernel
pp=prepare_all(texts[ids[0]]); _=final_rank(pp,select_direct(pp,0.0)); del pp
runsD={e:{} for e in ETAS}; runsQ={q:{} for q in LEX_QUOTAS}
poolhitD={e:0 for e in ETAS}; poolhitQ={q:0 for q in LEX_QUOTAS}; den=0; routehit=0
times=[]; cands=[]
for z,qid in enumerate(ids):
    t=time.perf_counter(); p=prepare_all(texts[qid]); prepms=(time.perf_counter()-t)*1000; times.append(prepms)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if p is None:
        for e in ETAS:runsD[e][qid]=[]
        for q in LEX_QUOTAS:runsQ[q][qid]=[]
        continue
    cands.append(p['candidate_docs']); ud=p['ud']
    rel_idx=[]
    for d in rels:
        kk=np.searchsorted(ud,d); ok=(kk<len(ud) and int(ud[kk])==d); routehit+=int(ok)
        if ok: rel_idx.append(int(kk))
    relset=set(rel_idx)
    for e in ETAS:
        sel=select_direct(p,e); poolhitD[e]+=sum(int(i) in relset for i in sel); runsD[e][qid]=final_rank(p,sel)
    for qv in LEX_QUOTAS:
        sel=select_quota(p,qv); poolhitQ[qv]+=sum(int(i) in relset for i in sel); runsQ[qv][qid]=final_rank(p,sel)
    if (z+1)%100==0:
        print('q',z+1,'median_prepare_ms',float(np.median(times)),'route',routehit/max(1,den),flush=True)

rowsD=[]
for e in ETAS:
    met=m.eval_run(runsD[e],qrels); row={'eta':e,'pool_relevant_recall':poolhitD[e]/den,**met}; rowsD.append(row); print('DIRECT',row,flush=True)
rowsQ=[]
for qv in LEX_QUOTAS:
    met=m.eval_run(runsQ[qv],qrels); row={'lex_quota':qv,'pool_relevant_recall':poolhitQ[qv]/den,**met}; rowsQ.append(row); print('QUOTA',row,flush=True)
rowsD.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
rowsQ.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'deterministic 1000 TRAIN validation; geometry/tail fixed gamma=.25 lambdaM=.125 S=16 P=2000 h=0; final lambda_lex=4 lambda_sem=.1 locked; whole-document lexical enters routed->P selection','direct_rows':rowsD,'quota_rows':rowsQ,'best_direct':rowsD[0],'best_quota':rowsQ[0],'route_relevant_recall':routehit/den,'timing_prepare_alllex':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))}}
json.dump(out,open(WORK/'early_lex_validation.json','w'),indent=2)
print('BEST_DIRECT',rowsD[0],flush=True); print('BEST_QUOTA',rowsQ[0],flush=True); print('TIMING',out['timing_prepare_alllex'],flush=True)
