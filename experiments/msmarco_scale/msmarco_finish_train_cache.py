from __future__ import annotations
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
OUT=WORK/'finish_train_cache'; OUT.mkdir(exist_ok=True)
set_num_threads(5)

def topk_desc(score,k):
    n=len(score); k=min(k,n)
    if n<=k:return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

@njit(parallel=True,cache=False)
def selected_features(dd,ip,ids,qmask,rarerank,idf,semvec):
    n=len(dd); lex2raw=np.zeros(n,np.float32); cnt=np.zeros(n,np.uint8); sem=np.zeros(n,np.float32); raremask=np.zeros(n,np.uint8)
    for z in prange(n):
        d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); s2=0.; c=0; ss=0.; mask=0
        for kk in range(a,bb):
            t=int(ids[kk]); ss+=semvec[t]
            if qmask[t]:
                x=float(idf[t]); s2+=x*x; c+=1
                rr=int(rarerank[t])
                if rr>0 and rr<=5: mask |= (1 << (rr-1))
        lex2raw[z]=s2; cnt[z]=min(c,255); sem[z]=ss; raremask[z]=mask
    return lex2raw,cnt,sem,raremask

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
    # Current validated early rescue: binary-IDF p=1, default b=.2, eta=1.
    lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32)
    oldlex,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl)
    sel=topk_desc(m.zscore(tail)+m.zscore(oldlex),P); dd=ud[sel]; ts=tail[sel]
    semvec=np.zeros(M,np.float32)
    for t,amp in zip(q.indices,q.data):
        a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
    qmask=np.zeros(M,np.uint8); qmask[q.indices]=1; rarerank=np.zeros(M,np.uint8)
    ordq=q.indices[np.argsort(idx.idf[q.indices])[::-1]]
    for r,t in enumerate(ordq[:5],start=1): rarerank[t]=r
    lex2raw,cnt,sem,raremask=selected_features(dd,idx.sup_ip,idx.sup_ids,qmask,rarerank,idx.idf,semvec)
    return dd,ts,lex2raw,cnt,sem,raremask,len(q.indices),len(ud)

# Exactly the five folds already used in previous robustness experiments.
z0=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); fold0=[str(x) for x in z0['qids'].tolist()]
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr
f0=set(int(x) for x in fold0); rem=np.asarray([x for x in uq if int(x) not in f0]); rng=np.random.default_rng(20260816); extra=rng.choice(rem,size=4000,replace=False)
folds=[fold0]+[[str(x) for x in extra[i*1000:(i+1)*1000]] for i in range(4)]
qids=np.asarray([q for f in folds for q in f],dtype='U32'); fold_id=np.repeat(np.arange(5,dtype=np.uint8),1000)
texts=m.load_query_texts(qids.tolist())
NQ=len(qids); shape=(NQ,P)
docs=np.memmap(OUT/'docs.u32',np.uint32,'w+',shape=shape); docs[:]=np.uint32(0xffffffff)
tail=np.memmap(OUT/'tail.f32',np.float32,'w+',shape=shape); tail[:]=0
lex2=np.memmap(OUT/'lex2raw.f32',np.float32,'w+',shape=shape); lex2[:]=0
sem=np.memmap(OUT/'sem.f32',np.float32,'w+',shape=shape); sem[:]=0
cnt=np.memmap(OUT/'cnt.u8',np.uint8,'w+',shape=shape); cnt[:]=0
rm=np.memmap(OUT/'raremask.u8',np.uint8,'w+',shape=shape); rm[:]=0
nvalid=np.zeros(NQ,np.uint16); qlen=np.zeros(NQ,np.uint16); cand=np.zeros(NQ,np.uint32); times=[]
_=selected_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.uint8),np.zeros(M,np.uint8),idx.idf,np.zeros(M,np.float32)); _=prepare(texts[qids[0]])
start=time.time()
for i,qid in enumerate(qids):
    t=time.perf_counter(); out=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
    if out is not None:
        dd,ts,lx,cc,ss,rr,ql,cd=out; n=min(P,len(dd)); docs[i,:n]=dd[:n]; tail[i,:n]=ts[:n]; lex2[i,:n]=lx[:n]; cnt[i,:n]=cc[:n]; sem[i,:n]=ss[:n]; rm[i,:n]=rr[:n]; nvalid[i]=n; qlen[i]=ql; cand[i]=cd
    if (i+1)%250==0: print('CACHE',i+1,'median250',float(np.median(times[-250:])), 'elapsed',time.time()-start,flush=True)
for a in [docs,tail,lex2,sem,cnt,rm]: a.flush()
np.savez(OUT/'meta.npz',qids=qids,fold_id=fold_id,nvalid=nvalid,qlen=qlen,candidate_docs=cand)
meta={'protocol':'five fixed disjoint 1000-query TRAIN folds; current validated early rescue eta=1 p=1 P=2000; cached final-stage features','n_queries':NQ,'P':P,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'seconds':time.time()-start},'files':{'docs':'docs.u32','tail':'tail.f32','lex2raw':'lex2raw.f32','sem':'sem.f32','cnt':'cnt.u8','raremask':'raremask.u8','meta':'meta.npz'}}
json.dump(meta,open(OUT/'cache_manifest.json','w'),indent=2)
print('DONE',json.dumps(meta,indent=2),flush=True)
