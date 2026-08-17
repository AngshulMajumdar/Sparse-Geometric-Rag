from __future__ import annotations
import sys,time,json,math,os
import numpy as np
sys.path.insert(0,'/mnt/data/work_code/geomretrieval_msmarco_scale_codebase/original_v0')
from geomretrieval import GeometricIndex, load_beir_directory, evaluate_run

IDX='/mnt/data/treccovid_geom_index'
ROOT='/mnt/data/work_treccovid/trec-covid'
idx=GeometricIndex.load(IDX); M=idx.vocab_size
P=2000; GAMMA=.25; LAM_M=.125; PRE_B=.2; FINAL_B=.1; ALPHA=.25
WLEX=4.0; WSEM=.3; WRARE=1.0; SEMK=16; EPS=1e-8
IDF_POWER=2


def zscore(x):
    x=np.asarray(x,np.float32)
    if not len(x): return x
    sd=float(x.std())
    return np.zeros_like(x) if sd<1e-8 else (x-float(x.mean()))/sd

def minmax_hi(x):
    x=np.asarray(x,np.float32)
    if not len(x): return x
    mn=float(x.min()); mx=float(x.max()); den=mx-mn
    return np.ones_like(x) if den<1e-8 else (x-mn)/den

def topk_large(score,k):
    n=len(score); k=min(k,n)
    if k<=0:return np.empty(0,np.int64)
    if n<=k:return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]
    return ii[np.argsort(score[ii])[::-1]]

def center_sparse(j):
    t=idx.center_terms[j]; v=idx.center_values[j]; ok=t>=0
    t=t[ok].astype(np.int32); v=v[ok].astype(np.float32)
    n=float(np.linalg.norm(v))
    if n>0:v=v/n
    oo=np.argsort(t)
    return t[oo],v[oo]

def spdot(a_t,a_v,b_t,b_v):
    i=j=0;s=0.0
    while i<len(a_t) and j<len(b_t):
        if a_t[i]==b_t[j]:s+=float(a_v[i])*float(b_v[j]);i+=1;j+=1
        elif a_t[i]<b_t[j]:i+=1
        else:j+=1
    return s

def prepare(text):
    q=idx._query_vector(text)
    if q.nnz==0:return None
    qd=np.zeros(M,np.float32); qd[q.indices]=q.data
    rt,rv,qt=idx._expanded_route(q)
    if not len(rt):return None
    rd=np.zeros(M,np.float32);rd[rt]=rv
    pieces=[]
    for j in rt:
        a,b=idx.branch_offsets[j],idx.branch_offsets[j+1]
        if b>a:pieces.append(idx.branch_order[a:b])
    if not pieces:return None
    fp=np.concatenate(pieces).astype(np.int64,copy=False)
    docs=(fp//idx.config.F).astype(np.int64); slots=(fp%idx.config.F).astype(np.int64)
    br=idx.branches[docs,slots]
    terms=idx.res_terms[docs,slots];valid=terms>=0;safe=np.where(valid,terms,0);qv=qd[safe]
    local=np.sum(idx.res_reliability[docs,slots].astype(np.float32)*(qv-idx.res_center_values[docs,slots])*idx.res_signs[docs,slots].astype(np.float32)*valid,axis=1)
    sig=np.sum((qv*qv)*valid,axis=1)
    cons=idx.memberships[docs,slots]*rd[br]
    branch_ev=(cons*local*np.power(np.maximum(sig,0),GAMMA)).astype(np.float32)  # E_dj exactly
    base=cons*local
    ud,inv=np.unique(docs,return_inverse=True)
    tail=np.bincount(inv,weights=branch_ev,minlength=len(ud)).astype(np.float32)
    tail += LAM_M*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
    # dominant routed branch retained for diagnostics
    bestv=np.full(len(ud),-1e30,np.float32);db=np.full(len(ud),-1,np.int32)
    for p,u in enumerate(inv):
        if cons[p]>bestv[u]:bestv[u]=cons[p];db[u]=int(br[p])
    # frozen early lexical rescue
    qlex=np.zeros(M,np.float32);qlex[q.indices]=idx.idf[q.indices]
    lex1=np.zeros(len(ud),np.float32)
    for i,d in enumerate(ud):
        a,b=idx.support_indptr[d],idx.support_indptr[d+1];sup=idx.support_indices[a:b]
        raw=float(qlex[sup].sum());den=(1-PRE_B)+PRE_B*(float(idx.doc_lengths[d])/idx.avg_doc_length)
        lex1[i]=raw/(den if den>0 else 1.)
    pre=zscore(tail)+zscore(lex1)
    sel=topk_large(pre,P)
    dd=ud[sel];ts=tail[sel];db=db[sel]
    # mapping routed-doc local index -> pool local index
    poolpos=np.full(len(ud),-1,np.int32); poolpos[sel]=np.arange(len(sel),dtype=np.int32)
    mp=poolpos[inv]
    keep=mp>=0
    mem_pool=mp[keep].astype(np.int32)
    mem_br=br[keep].astype(np.int32)
    mem_ev=branch_ev[keep].astype(np.float32)
    # final current features
    semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,b=idx.A.indptr[t],idx.A.indptr[t+1];nb=idx.A.indices[a:b][:SEMK];sv=idx.A.data[a:b][:SEMK]
        if len(nb):semvec[nb]+=float(amp)*sv*idx.idf[nb]
    qset=set(map(int,q.indices));rare=set(map(int,q.indices[np.argsort(idx.idf[q.indices])[::-1]][:3]));nq=max(1,len(q.indices))
    lx=np.zeros(len(dd),np.float32);sm=np.zeros(len(dd),np.float32);qc=np.zeros(len(dd),np.float32);r3=np.zeros(len(dd),np.float32)
    for i,d in enumerate(dd):
        a,b=idx.support_indptr[d],idx.support_indptr[d+1];sup=idx.support_indices[a:b]
        sm[i]=float(semvec[sup].sum())
        match=[int(t) for t in sup if int(t) in qset]
        raw=sum(float(idx.idf[t])**IDF_POWER for t in match);den=(1-FINAL_B)+FINAL_B*(float(idx.doc_lengths[d])/idx.avg_doc_length)
        lx[i]=raw/(den if den>0 else 1.);qc[i]=len(match);r3[i]=sum(t in rare for t in match)
    cov=qc/nq;ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA);rarecov=r3/max(1,min(3,nq))
    base_score=zscore(ts)+WLEX*zscore(ladj)+WSEM*zscore(sm)+WRARE*zscore(rarecov)

    # Build H_j = mean of top 3 E_dj among selected pool documents for branch j.
    # Multiple memberships of same doc+branch should not occur; if they do, keep max.
    branch_pairs={}
    for pi,b,e in zip(mem_pool,mem_br,mem_ev):
        key=(int(b),int(pi))
        if key not in branch_pairs or e>branch_pairs[key]: branch_pairs[key]=float(e)
    byb={}
    for (b,pi),e in branch_pairs.items(): byb.setdefault(b,[]).append((e,pi))
    H={}; bestdoc={}; docs_by_branch={}
    for b,vals in byb.items():
        vals.sort(key=lambda x:x[0],reverse=True)
        top=vals[:3]
        H[b]=float(np.mean([e for e,_ in top]))
        bestdoc[b]=int(max(vals,key=lambda x: float(base_score[x[1]]))[1])  # best final-relevance doc in branch
        docs_by_branch[b]=np.asarray([pi for _,pi in vals],dtype=np.int32)
    ub=np.asarray(sorted(H.keys()),dtype=np.int32)
    h=np.asarray([H[int(b)] for b in ub],dtype=np.float32)
    hnorm=minmax_hi(h)
    bmap={int(b):i for i,b in enumerate(ub)}
    reps=[center_sparse(int(b)) for b in ub]
    C=np.eye(len(ub),dtype=np.float32)
    for i in range(len(ub)):
        for j in range(i+1,len(ub)):
            C[i,j]=C[j,i]=spdot(*reps[i],*reps[j])
    return {'docs':dd,'base':base_score,'tail':ts,'lex':ladj,'sem':sm,'rare':rarecov,
            'route_docs':ud,'branches_dom':db,'ub':ub,'H':h,'Hn':hnorm,'bmap':bmap,'cos':C,
            'bestdoc':bestdoc,'docs_by_branch':docs_by_branch}

def plain(p,k=100):
    oo=np.argsort(p['base'])[::-1][:k]
    return p['docs'][oo].tolist()

def Dvec(p, selected_bidx):
    C=p['cos']
    if not selected_bidx:return np.zeros(len(C),np.float32)
    si=np.asarray(selected_bidx,np.int32)
    mumun=float(np.mean(C[np.ix_(si,si)]))
    return 1.0+mumun-2.0*np.mean(C[:,si],axis=1)

def select_hq_branches(p, topN=20, lam=1.0, nsel=10, pure_div=False):
    if len(p['ub'])==0:return []
    # eligible branches are the top-N robust-quality H_j branches
    order=np.argsort(p['H'])[::-1]
    elig=order[:min(topN,len(order))]
    # first branch = highest H_j
    selected=[int(elig[0])]
    remaining=set(map(int,elig[1:]))
    while remaining and len(selected)<min(nsel,len(elig)):
        rem=np.asarray(sorted(remaining),dtype=np.int32)
        D=Dvec(p,selected)
        dnorm=minmax_hi(D[rem])
        if pure_div:
            val=dnorm
        else:
            h=p['Hn'][rem]
            val=h+float(lam)*dnorm
        pick=int(rem[int(np.argmax(val))])
        selected.append(pick); remaining.remove(pick)
    return selected

def rank_hq_oneper(p,topN=20,lam=1.0,pure_div=False,k=100):
    # one best final-score document from each selected high-quality/diverse branch for top 10
    sb=select_hq_branches(p,topN,lam,10,pure_div)
    chosen=[]; used=set()
    for bi in sb:
        b=int(p['ub'][bi]); pi=int(p['bestdoc'][b])
        if pi not in used: chosen.append(pi); used.add(pi)
    # if fewer than 10, fill by ordinary score
    for pi in np.argsort(p['base'])[::-1]:
        pi=int(pi)
        if len(chosen)>=10:break
        if pi not in used: chosen.append(pi);used.add(pi)
    # rest by ordinary score
    for pi in np.argsort(p['base'])[::-1]:
        pi=int(pi)
        if len(chosen)>=k:break
        if pi not in used: chosen.append(pi);used.add(pi)
    return p['docs'][np.asarray(chosen[:k],dtype=np.int64)].tolist()

def rank_hq_softdoc(p,topN=20,lam=.25,k=100):
    # restrict diversity bonus to high-quality branches; repeated branches allowed.
    # docs outside HQ set retain pure relevance and are still eligible.
    base=p['base']; n=len(base); order=np.argsort(base)[::-1]
    first=int(order[0]); chosen=[first]; used={first}
    elig_order=np.argsort(p['H'])[::-1][:min(topN,len(p['H']))]
    eligset=set(map(int,elig_order))
    # branch memberships for each pool doc: use all high-quality branches the doc belongs to
    doc_hq=[[] for _ in range(n)]
    for bi in elig_order:
        b=int(p['ub'][bi])
        for pi in p['docs_by_branch'][b]: doc_hq[int(pi)].append(int(bi))
    # selected branch representation starts with highest-quality HQ branch supporting first, if any
    selected=[]
    if doc_hq[first]:
        selected=[max(doc_hq[first], key=lambda bi: float(p['H'][bi]))]
    for _ in range(1,min(10,k,n)):
        rem=np.asarray([i for i in range(n) if i not in used],dtype=np.int32)
        if not len(rem):break
        D=Dvec(p,selected) if selected else np.zeros(len(p['ub']),np.float32)
        # normalize only across eligible branches
        if len(elig_order):
            ed=minmax_hi(D[elig_order]); dn={int(bi):float(v) for bi,v in zip(elig_order,ed)}
        else: dn={}
        # relevance minmax across remaining top pool; high = good
        rn=minmax_hi(base[rem])
        bonus=np.zeros(len(rem),np.float32)
        for k2,pi in enumerate(rem):
            if doc_hq[int(pi)]: bonus[k2]=max(dn.get(bi,0.0) for bi in doc_hq[int(pi)])
        val=rn+float(lam)*bonus
        pi=int(rem[int(np.argmax(val))]); chosen.append(pi); used.add(pi)
        if doc_hq[pi]:
            # add the supporting HQ branch with max diversity, ties quality
            bi=max(doc_hq[pi],key=lambda x:(dn.get(x,0.0),float(p['H'][x])))
            selected.append(int(bi))
    for pi in order:
        pi=int(pi)
        if len(chosen)>=k:break
        if pi not in used:chosen.append(pi);used.add(pi)
    return p['docs'][np.asarray(chosen[:k],dtype=np.int64)].tolist()

def evaluate(ds,packs,ranker,**kw):
    run={};route_num=pool_num=den=0;cands=[]
    for qid,p in packs.items():
        rr=[] if p is None else ranker(p,**kw)
        run[str(qid)]=[str(idx.doc_ids[int(d)]) for d in rr]
        if p is not None:
            route={str(idx.doc_ids[int(d)]) for d in p['route_docs']};pool={str(idx.doc_ids[int(d)]) for d in p['docs']}
            pos={str(d) for d,v in ds.qrels[qid].items() if v>0};den+=len(pos);route_num+=sum(d in route for d in pos);pool_num+=sum(d in pool for d in pos);cands.append(len(route))
    m=evaluate_run(run,ds.qrels,ks=(10,100),ndcg_k=10,mrr_k=10,exp_gain=False)
    m['route_recall_micro']=route_num/max(1,den);m['pool_recall_micro']=pool_num/max(1,den);m['avg_candidates']=float(np.mean(cands)) if cands else 0
    return m


ds=load_beir_directory(ROOT,'test')
allres={'dataset':'TREC-COVID','experiment':'Final whole-document binary lexical IDF power; all else fixed','variants':{}}
for power in [2,3,4]:
    IDF_POWER=power
    packs={};times=[]
    for i,qid in enumerate(ds.qrels):
        t=time.perf_counter(); packs[qid]=prepare(ds.queries[qid]); times.append((time.perf_counter()-t)*1000)
    plain_m=evaluate(ds,packs,plain)
    hq_m=evaluate(ds,packs,rank_hq_softdoc,topN=10,lam=.1)
    allres['variants'][f'idf{power}_plain']=plain_m
    allres['variants'][f'idf{power}_hqdiv']=hq_m
    allres.setdefault('timing',{})[f'idf{power}_prepare_median_ms']=float(np.median(times))
    allres['timing'][f'idf{power}_prepare_p95_ms']=float(np.percentile(times,95))
    print('POWER',power,'PLAIN',plain_m,flush=True)
    print('POWER',power,'HQDIV',hq_m,flush=True)
out='/mnt/data/treccovid_idf_power_2_3_4.json'
json.dump(allres,open(out,'w'),indent=2)
print('saved',out,flush=True)
