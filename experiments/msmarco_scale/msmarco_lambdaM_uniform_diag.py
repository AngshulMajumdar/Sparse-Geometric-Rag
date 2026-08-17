import sys,time,json,math
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; P=m.P; M=m.M
LAMBDAS=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,64.0]
HGRID=[0,1,2,3]
idx=m.FullIndex(); print('loaded',idx.meta,flush=True)
# exact same deterministic validation split
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)

def components(text):
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
    hc,tc,cc=m.score_memberships_local(rslot,mm,rt,sb,qd,rho,cent,rel)
    ud,inv=np.unique(docs,return_inverse=True)
    head=np.bincount(inv,weights=hc,minlength=len(ud)).astype(np.float32)
    local=np.bincount(inv,weights=tc,minlength=len(ud)).astype(np.float32)
    cons=np.bincount(inv,weights=cc,minlength=len(ud)).astype(np.float32)
    # lexical vectors independent of lambda
    lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,b=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:b][:m.SEMK]; sv=idx.A.data[a:b][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
    ho=np.argsort(head)[::-1][:max(HGRID)] if max(HGRID)>0 else np.empty(0,np.int64)
    return q,ud,head,local,cons,lexvec,semvec,ho,len(docs)

runs={(la,h):{} for la in LAMBDAS for h in HGRID}; pool_hit={la:0 for la in LAMBDAS}; rel_den=0; route_hit=0; times=[]; cand_counts=[]
# warm JIT
_ = components(texts[ids[0]])
for z,qid in enumerate(ids):
    t0=time.perf_counter(); c=components(texts[qid]);
    if c is None:
        for key in runs:runs[key][qid]=[]
        continue
    q,ud,head,local,cons,lexvec,semvec,ho,nmem=c; cand_counts.append(len(ud))
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; rel_den+=len(rels)
    for d in rels:
        k=np.searchsorted(ud,d); route_hit+=int(k<len(ud) and int(ud[k])==d)
    # select each lambda pool; union for one support pass
    pools={}; union=[]
    for la in LAMBDAS:
        tail=local+np.float32(la)*cons; want=min(len(tail),P+max(HGRID)+8)
        if len(tail)>want:
            ci=np.argpartition(tail,-want)[-want:]; oo=ci[np.argsort(tail[ci])[::-1]]
        else: oo=np.argsort(tail)[::-1]
        pools[la]=(ud[oo],tail[oo]); union.append(ud[oo])
    udocs=np.unique(np.concatenate(union)); lx,sm=m.score_support_pool(udocs,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl)
    # udocs sorted, so searchsorted maps pool docs
    for la in LAMBDAS:
        docs,ts=pools[la]
        # pool recall before final rerank at h=0 definition
        p0=docs[:P]
        for d in rels: pool_hit[la]+=int(np.any(p0==d))
        pos=np.searchsorted(udocs,docs); lxx=lx[pos]; smm=sm[pos]
        for h in HGRID:
            frozen=ud[ho[:h]] if h else np.empty(0,np.uint32); fs=set(map(int,frozen.tolist()))
            keep=np.asarray([int(d) not in fs for d in docs],bool); dd=docs[keep][:P]; tt=ts[keep][:P]; ll=lxx[keep][:P]; ss=smm[keep][:P]
            fin=m.zscore(tt)+m.LAMBDA_LEX*m.zscore(ll)+m.LAMBDA_SEM*m.zscore(ss); oo=np.argsort(fin)[::-1]; rank=np.concatenate([frozen,dd[oo]])[:100]
            runs[(la,h)][qid]=[int(x) for x in rank]
    times.append((time.perf_counter()-t0)*1000)
    if (z+1)%100==0: print('q',z+1,'median_ms',float(np.median(times)),'route',route_hit/max(1,rel_den),flush=True)

rows=[]
for la in LAMBDAS:
    for h in HGRID:
        met=m.eval_run(runs[(la,h)],qrels); met['pool_relevant_recall']=pool_hit[la]/rel_den; rows.append({'lambda_M':la,'h':h,**met}); print('LAM',la,'H',h,met,'pool',pool_hit[la]/rel_den,flush=True)
best=max(rows,key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']))
out={'protocol':'uniform-1M geometry; deterministic 1000 TRAIN validation; joint diagnostic sweep lambda_M and h','route_relevant_recall':route_hit/rel_den,'timing_median_ms':float(np.median(times)),'avg_candidate_docs':float(np.mean(cand_counts)),'rows':rows,'best':best}
json.dump(out,open(WORK/'lambdaM_uniform1m_diag.json','w'),indent=2); print('BEST',best,flush=True)
