from __future__ import annotations
import sys,time,json,math
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
ETA=np.float32(1.0); FINAL_B=np.float32(0.1); ALPHA=np.float32(0.25); WLEX=np.float32(4.0); WSEM=np.float32(0.3); WRARE=np.float32(1.0)
set_num_threads(5)

def topk_desc(score,k):
    n=len(score); k=min(k,n)
    if n<=k:return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

@njit(parallel=True,cache=False)
def final_features(dd,ip,ids,qmask,rarerank,idf,semvec,dl,avgdl):
    n=len(dd); lx=np.zeros(n,np.float32); sm=np.zeros(n,np.float32); qc=np.zeros(n,np.float32); r3=np.zeros(n,np.float32)
    for z in prange(n):
        d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.; ss=0.; c=0.; rr3=0.
        for k in range(a,bb):
            t=int(ids[k]); ss+=semvec[t]
            if qmask[t]:
                x=float(idf[t]); raw+=x*x; c+=1.
                if int(rarerank[t])>0: rr3+=1.
        ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
        if den<=0:den=1.
        lx[z]=raw/den; sm[z]=ss; qc[z]=c; r3[z]=rr3
    return lx,sm,qc,r3

def prepare(text):
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
    # Frozen early rescue: p=1 binary IDF, eta=1, package's weak b=.2 length correction.
    lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32)
    oldlex,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl)
    sel=topk_desc(m.zscore(tail)+ETA*m.zscore(oldlex),P); dd=ud[sel]; ts=tail[sel]
    semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
    qmask=np.zeros(M,np.uint8); qmask[q.indices]=1; rarerank=np.zeros(M,np.uint8)
    ordq=q.indices[np.argsort(idx.idf[q.indices])[::-1]]
    for r,t in enumerate(ordq[:3],start=1): rarerank[t]=r
    lx,sm,qc,r3=final_features(dd,idx.sup_ip,idx.sup_ids,qmask,rarerank,idx.idf,semvec,idx.dl,idx.avgdl)
    cov=qc/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA); rarecov=r3/max(1,min(3,len(q.indices)))
    fin=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sm)+WRARE*m.zscore(rarecov)
    oo=np.argsort(fin)[::-1][:100]
    return [int(x) for x in dd[oo]],ud,sel

def eval_beir(run,qrels):
    metrics={'nDCG@10':[],'nDCG@10_expGain_diag':[],'MRR@10':[],'P@10':[],'R@10':[],'R@100':[],'Hit@10':[],'Hit@100':[]}
    for qid,qr in qrels.items():
        rank=run.get(qid,[]); pos={int(d) for d,r in qr.items() if float(r)>0}; n=max(1,len(pos))
        h10=sum(d in pos for d in rank[:10]); h100=sum(d in pos for d in rank[:100])
        metrics['P@10'].append(h10/10); metrics['R@10'].append(h10/n); metrics['R@100'].append(h100/n); metrics['Hit@10'].append(float(h10>0)); metrics['Hit@100'].append(float(h100>0))
        rr=0.
        for i,d in enumerate(rank[:10],1):
            if d in pos: rr=1/i; break
        metrics['MRR@10'].append(rr)
        obs=[float(qr.get(str(d),qr.get(d,0.0))) for d in rank[:10]]
        ideal=sorted([float(r) for r in qr.values()],reverse=True)[:10]
        dcg_lin=sum(float(r)/math.log2(i+2) for i,r in enumerate(obs)); idcg_lin=sum(float(r)/math.log2(i+2) for i,r in enumerate(ideal))
        dcg_exp=sum((2.0**float(r)-1.0)/math.log2(i+2) for i,r in enumerate(obs)); idcg_exp=sum((2.0**float(r)-1.0)/math.log2(i+2) for i,r in enumerate(ideal))
        metrics['nDCG@10'].append(dcg_lin/idcg_lin if idcg_lin else 0.0)
        metrics['nDCG@10_expGain_diag'].append(dcg_exp/idcg_exp if idcg_exp else 0.0)
    return {k:float(np.mean(v)) for k,v in metrics.items()} | {'n_queries':len(qrels)}

# TEST IDs and full graded qrels (including zero judgments for nDCG ideal ordering).
tdf=pd.read_csv(ROOT/'test.tsv',sep='\t'); ids=[str(x) for x in np.unique(tdf['query-id'].to_numpy())]; del tdf
texts=m.load_query_texts(ids); qrels=m.qrels_from_tsv(ROOT/'test.tsv',ids,positive_only=False)
missing=[q for q in ids if q not in texts]
if missing: raise RuntimeError(f'missing query texts {missing}')
# Warmup excluded.
_=final_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.uint8),np.zeros(M,np.uint8),idx.idf,np.zeros(M,np.float32),idx.dl,idx.avgdl); _=prepare(texts[ids[0]])
run={}; times=[]; routehit=poolhit=den=0; cands=[]
for z,qid in enumerate(ids):
    t=time.perf_counter(); out=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
    if out is None: run[qid]=[]; continue
    rank,ud,sel=out; run[qid]=rank; cands.append(len(ud)); rels=[int(d) for d,r in qrels[qid].items() if float(r)>0]; den+=len(rels); pooldocs=set(map(int,ud[sel].tolist()))
    for d in rels:
        kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; routehit+=int(ok); poolhit+=int(ok and d in pooldocs)
    print('test',z+1,'/',len(ids),'qid',qid,'ms',times[-1],flush=True)
met=eval_beir(run,qrels)
result={'protocol':'FINAL TEST ONCE after TRAIN-only five-fold lock; no test tuning','frozen_params':{'preselection_eta':1.0,'preselection_idf_power':1.0,'P':2000,'gamma_tail':0.25,'lambda_M':0.125,'final_idf_power':2.0,'final_length_b':0.1,'coordination_alpha':0.25,'lambda_lex':4.0,'lambda_sem':0.3,'rare_topk':3,'rare_coverage_weight':1.0,'h':0,'S':16},'test_metrics':met,'route_relevant_recall':routehit/max(1,den),'pool_relevant_recall':poolhit/max(1,den),'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands))},'n_positive_judgments':int(den),'note':'nDCG@10 uses linear graded gain, matching trec_eval/BEIR convention; exponential-gain value retained only as diagnostic.'}
out=WORK/'FINAL_TEST_ONCE_results.json'; json.dump(result,open(out,'w'),indent=2); print('FINAL_RESULT',json.dumps(result,indent=2),flush=True); print('SAVED',out,flush=True)
