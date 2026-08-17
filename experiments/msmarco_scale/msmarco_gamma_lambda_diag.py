from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; M=m.M; S=m.S; P=2000
GAMMAS=np.array([0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0],np.float32)
LAMBDAS=np.array([0.0,0.125,0.25,0.5,1.0,2.0,4.0],np.float32)
set_num_threads(5)
idx=m.FullIndex(); print('loaded',idx.meta,flush=True)

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
        c=mem[z]*rho[u]
        base[z]=c*local; sig[z]=sg; cons[z]=c
    return base,sig,cons

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
    base,sig,cons=score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
    ud,inv=np.unique(docs,return_inverse=True)
    cdoc=np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
    locals=[]
    for ga in GAMMAS:
        w=base if ga==0 else base*np.power(sig,ga,dtype=np.float32)
        locals.append(np.bincount(inv,weights=w,minlength=len(ud)).astype(np.float32))
    return ud,locals,cdoc,len(docs)

# exact deterministic validation ids; first 300 for grid, all 1000 for finalists
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',ids,positive_only=True)
# warm
_=components(texts[ids[0]])

# Stage A: 300-query grid by pool relevant recall
hits=np.zeros((len(GAMMAS),len(LAMBDAS)),np.int64); den=0; route_hit=0; times=[]
for z,qid in enumerate(ids[:300]):
    t=time.perf_counter(); c=components(texts[qid]); times.append((time.perf_counter()-t)*1000)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if c is None: continue
    ud,locals,cdoc,nmem=c
    relpos=[]
    for d in rels:
        k=np.searchsorted(ud,d); ok=(k<len(ud) and int(ud[k])==d); route_hit+=int(ok); relpos.append(k if ok else -1)
    for gi,loc in enumerate(locals):
        for li,la in enumerate(LAMBDAS):
            tail=loc+la*cdoc; want=min(P,len(tail))
            if len(tail)>want: top=np.argpartition(tail,-want)[-want:]
            else: top=np.arange(len(tail))
            # membership mask avoids repeated np.any scans
            mark=np.zeros(len(ud),np.uint8); mark[top]=1
            for k in relpos:
                if k>=0: hits[gi,li]+=int(mark[k])
    if (z+1)%50==0: print('grid',z+1,'median_ms',float(np.median(times)),'route',route_hit/max(1,den),flush=True)
rec=hits/max(1,den)
flat=[]
for gi,ga in enumerate(GAMMAS):
    for li,la in enumerate(LAMBDAS): flat.append((float(rec[gi,li]),float(ga),float(la)))
flat.sort(reverse=True)
print('TOP GRID',flat[:12],flush=True)
# include historical and take unique top 6
final=[]
for _,ga,la in flat:
    x=(ga,la)
    if x not in final: final.append(x)
    if len(final)>=6: break
if (1.0,2.0) not in final: final.append((1.0,2.0))

# Stage B: exact all-1000 pool recall for finalists
fh={x:0 for x in final}; den=0; rh=0; times2=[]; avgc=[]
for z,qid in enumerate(ids):
    t=time.perf_counter(); c=components(texts[qid]); times2.append((time.perf_counter()-t)*1000)
    rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
    if c is None: continue
    ud,locals,cdoc,nmem=c; avgc.append(len(ud))
    relpos=[]
    for d in rels:
        k=np.searchsorted(ud,d); ok=(k<len(ud) and int(ud[k])==d); rh+=int(ok); relpos.append(k if ok else -1)
    for ga,la in final:
        gi=int(np.where(np.isclose(GAMMAS,ga))[0][0]); tail=locals[gi]+np.float32(la)*cdoc; want=min(P,len(tail))
        top=np.argpartition(tail,-want)[-want:] if len(tail)>want else np.arange(len(tail)); mark=np.zeros(len(ud),np.uint8); mark[top]=1
        for k in relpos:
            if k>=0: fh[(ga,la)]+=int(mark[k])
    if (z+1)%100==0: print('final',z+1,'median_ms',float(np.median(times2)),'route',rh/max(1,den),flush=True)
rows=[]
for ga,la in final:
    rows.append({'gamma':ga,'lambda_M':la,'pool_relevant_recall':fh[(ga,la)]/den}); print('FINAL',rows[-1],flush=True)
rows.sort(key=lambda x:x['pool_relevant_recall'],reverse=True)
out={'stageA_top':flat[:20],'finalists':rows,'route_relevant_recall':rh/den,'median_component_ms':float(np.median(times2)),'avg_candidate_docs':float(np.mean(avgc)),'protocol':'gamma/lambda_M pool shortlist diagnostic; deterministic TRAIN validation sample; P=2000'}
json.dump(out,open(WORK/'gamma_lambda_pool_diag.json','w'),indent=2)
print('BEST',rows[0],flush=True)
