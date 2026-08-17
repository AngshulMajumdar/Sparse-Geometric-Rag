from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_early_lex_validation_fast as e
import msmarco_full_search_uniform1m as m
import msmarco_best_tail_core as b

ROOT=m.ROOT; WORK=m.WORK; idx=e.idx; M=m.M; P=2000; S=m.S
set_num_threads(5)
WEIGHTS=[-2.0,-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]
FINAL_B=np.float32(0.1); LEX_ALPHA=np.float32(0.25); WLEX=np.float32(4.0); WSEM=np.float32(0.3)

def topk_desc(score,k):
    n=len(score); k=min(k,n)
    if n<=k:return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]
    return ii[np.argsort(score[ii])[::-1]]

@njit(parallel=True,cache=False)
def selected_lex_features(dd,ip,ids,lexvec,dl,avgdl):
    n=len(dd); lx=np.zeros(n,np.float32); qc=np.zeros(n,np.float32)
    for z in prange(n):
        d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.0; c=0.0
        for k in range(a,bb):
            t=int(ids[k]); v=lexvec[t]
            if v>0:
                raw += v; c += 1.0
        ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
        lx[z]=raw/(den if den>0 else 1.0); qc[z]=c
    return lx,qc

@njit(cache=False)
def find_doc(pd,a,bb,d):
    lo=np.int64(a); hi=np.int64(bb)
    while lo<hi:
        md=(lo+hi)//2; x=int(pd[md])
        if x<d: lo=md+1
        else: hi=md
    if lo<bb and int(pd[lo])==d:return lo
    return -1

@njit(parallel=True,cache=False)
def coherence_features(dd,rterms,rd,offs,pd,pm,pr,ps,qd,ct,cv,rp,ri,rv):
    n=len(dd); geom=np.zeros(n,np.float32); gabs=np.zeros(n,np.float32)
    for zz in prange(n):
        d=int(dd[zz]); gs=0.0; ab=0.0
        for jj in range(len(rterms)):
            j=int(rterms[jj]); a=int(offs[j]); bb=int(offs[j+1]); p=find_doc(pd,a,bb,d)
            if p<0: continue
            c=float(pm[p])*float(rd[j]); local=0.0; sig=0.0; bits=int(ps[p])
            for r in range(S):
                t=int(pr[p,r])
                if t==65535: continue
                qv=float(qd[t]); cen=float(m.lookup_center(ct[j],cv[j],t)); rel=float(m.lookup_rel(rp,ri,rv,j,t)); sgn=1.0 if ((bits>>r)&1) else -1.0
                local += rel*(qv-cen)*sgn; sig += qv*qv
            g=c*local*(sig**0.25 if sig>0 else 0.0)
            gs += g; ab += abs(g)
        geom[zz]=gs; gabs[zz]=ab
    return geom,gabs

def rank100(score):
    n=len(score); k=min(100,n)
    if n<=k: oo=np.argsort(score)[::-1]
    else:
        ii=np.argpartition(score,-k)[-k:]; oo=ii[np.argsort(score[ii])[::-1]]
    return oo

# Exact original fold-0 IDs, plus four new disjoint deterministic folds.
z0=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False)
fold0=[str(x) for x in z0['qids'].tolist()]
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id'])
uq=np.unique(tr['query-id'].to_numpy()); del tr
f0set=set(int(x) for x in fold0)
remaining=np.asarray([x for x in uq if int(x) not in f0set])
rng=np.random.default_rng(20260816)
extra=rng.choice(remaining,size=4000,replace=False)
folds=[fold0]+[[str(x) for x in extra[i*1000:(i+1)*1000]] for i in range(4)]
allids=[q for f in folds for q in f]
texts=m.load_query_texts(allids)
qrels_all=m.qrels_from_tsv(ROOT/'train.tsv',allids,positive_only=True)

# warmup
_=selected_lex_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),idx.dl,idx.avgdl)
q0=idx.query_vec(texts[allids[0]]); qd0=np.zeros(M,np.float32); qd0[q0.indices]=q0.data; rt0,rd0=idx.route(q0)
_=coherence_features(np.array([0],np.uint32),rt0,rd0,idx.offs,idx.pd,idx.pm,idx.pr,idx.ps,qd0,idx.ct,idx.cv,idx.rp,idx.ri,idx.rv)
_=e.prepare_all(texts[allids[0]])

runs=[{w:{} for w in WEIGHTS} for _ in folds]
times=[]
start=time.time()
for fi,ids in enumerate(folds):
    print('FOLD',fi,'START',flush=True)
    for qi,qid in enumerate(ids):
        t0=time.perf_counter(); p=e.prepare_all(texts[qid])
        if p is None:
            for w in WEIGHTS:runs[fi][w][qid]=[]
            continue
        sel=topk_desc(m.zscore(p['tail'])+m.zscore(p['lex']),P)
        dd=p['ud'][sel]; ts=p['tail'][sel]; sm=p['sem'][sel]
        q=idx.query_vec(texts[qid]); lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]
        lx,qc=selected_lex_features(dd,idx.sup_ip,idx.sup_ids,lexvec,idx.dl,idx.avgdl)
        cov=qc/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),LEX_ALPHA)
        qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
        geom,gabs=coherence_features(dd,rterms,rd,idx.offs,idx.pd,idx.pm,idx.pr,idx.ps,qd,idx.ct,idx.cv,idx.rp,idx.ri,idx.rv)
        coh=geom/np.maximum(gabs,1e-6)
        base=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sm)
        for w in WEIGHTS:
            sc=base+np.float32(w)*coh
            oo=rank100(sc); runs[fi][w][qid]=[int(x) for x in dd[oo]]
        times.append((time.perf_counter()-t0)*1000)
        if (qi+1)%250==0:
            print('fold',fi,'q',qi+1,'median_ms',float(np.median(times[-250:])),flush=True)

rows=[]
for fi,ids in enumerate(folds):
    qr={q:qrels_all[q] for q in ids}
    for w in WEIGHTS:
        met=m.eval_run(runs[fi][w],qr); rows.append({'fold':fi,'weight':w,**met})
        print('METRIC fold',fi,'w',w,'ndcg',met['nDCG@10'],'mrr',met['MRR@10'],'r100',met['R@100'],flush=True)
summary=[]
for w in WEIGHTS:
    rr=[r for r in rows if r['weight']==w]
    nd=np.asarray([r['nDCG@10'] for r in rr]); mr=np.asarray([r['MRR@10'] for r in rr]); r100=np.asarray([r['R@100'] for r in rr])
    # Improvement relative to w=0 computed fold-wise.
    base=[next(x for x in rows if x['fold']==fi and x['weight']==0.0) for fi in range(5)]
    delta=np.asarray([rr[fi]['nDCG@10']-base[fi]['nDCG@10'] for fi in range(5)])
    summary.append({'weight':w,'mean_nDCG@10':float(nd.mean()),'std_nDCG@10':float(nd.std(ddof=1)),'mean_MRR@10':float(mr.mean()),'mean_R@100':float(r100.mean()),'mean_delta_nDCG_vs_base':float(delta.mean()),'min_delta_nDCG_vs_base':float(delta.min()),'positive_folds':int(np.sum(delta>0)),'fold_deltas':delta.tolist()})
summary.sort(key=lambda x:(x['positive_folds'],x['min_delta_nDCG_vs_base'],x['mean_delta_nDCG_vs_base']),reverse=True)
out={'protocol':'5 disjoint 1000-query TRAIN folds; fold0 is original validation; folds1-4 are new deterministic samples; same eta=1 P=2000 pools and structural lexical b=.1 alpha=.25 wl4 raw-sem .3; branch coherence only','weights':WEIGHTS,'fold_rows':rows,'summary_ranked_for_robustness':summary,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'seconds':time.time()-start}}
path=WORK/'branch_coherence_multifold.json'; json.dump(out,open(path,'w'),indent=2)
print('SUMMARY'); [print(x) for x in summary]; print('SAVED',path,flush=True)
