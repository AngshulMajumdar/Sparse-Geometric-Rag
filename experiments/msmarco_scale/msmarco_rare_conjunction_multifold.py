from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
set_num_threads(5)
KS=[1,2,3]
WS=[0.25,0.5,1.0,2.0]
FINAL_B=np.float32(0.1); ALPHA=np.float32(0.25); WLEX=np.float32(4.0); WSEM=np.float32(0.3)
MODELS=[('base',0,0.0)]+[(f'top{k}_w{w}',k,w) for k in KS for w in WS]
def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if n<=k:return np.argsort(score)[::-1]
 ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]
@njit(parallel=True,cache=False)
def selected_features(dd,ip,ids,qmask,rarerank,idf,semvec,dl,avgdl):
 n=len(dd); lex2=np.zeros(n,np.float32); cnt=np.zeros(n,np.float32); sem=np.zeros(n,np.float32); r1=np.zeros(n,np.float32); r2=np.zeros(n,np.float32); r3=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); s2=0.; c=0.; ss=0.; a1=0.; a2=0.; a3=0.
  for kk in range(a,bb):
   t=int(ids[kk]); ss+=semvec[t]
   if qmask[t]:
    x=float(idf[t]); s2+=x*x; c+=1.; rr=int(rarerank[t])
    if rr==1: a1+=1.; a2+=1.; a3+=1.
    elif rr==2: a2+=1.; a3+=1.
    elif rr==3: a3+=1.
  ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
  if den<=0:den=1.
  lex2[z]=s2/den; cnt[z]=c; sem[z]=ss; r1[z]=a1; r2[z]=a2; r3[z]=a3
 return lex2,cnt,sem,r1,r2,r3
def prepare(text):
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
 spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans:return None
 docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False); mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False); rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False); sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]; ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)]); base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
 ud,inv=np.unique(docs,return_inverse=True); tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32); oldlex,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl)
 sel=topk_desc(m.zscore(tail)+m.zscore(oldlex),P); dd=ud[sel]; ts=tail[sel]
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 qmask=np.zeros(M,np.uint8); qmask[q.indices]=1; rarerank=np.zeros(M,np.uint8)
 ordq=q.indices[np.argsort(idx.idf[q.indices])[::-1]]
 for r,t in enumerate(ordq[:3],start=1): rarerank[t]=r
 lx,cnt,sem,r1,r2,r3=selected_features(dd,idx.sup_ip,idx.sup_ids,qmask,rarerank,idx.idf,semvec,idx.dl,idx.avgdl); cov=cnt/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA); basefin=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sem)
 rare=[None,r1/max(1,min(1,len(q.indices))),r2/max(1,min(2,len(q.indices))),r3/max(1,min(3,len(q.indices)))]
 scores=[basefin]
 for name,k,w in MODELS[1:]: scores.append(basefin+np.float32(w)*m.zscore(rare[k]))
 return dd,scores
def top100(sc):
 n=len(sc); k=min(100,n)
 if n<=k:return np.argsort(sc)[::-1]
 ii=np.argpartition(sc,-k)[-k:]; return ii[np.argsort(sc[ii])[::-1]]
z0=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); fold0=[str(x) for x in z0['qids'].tolist()]
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr
f0=set(int(x) for x in fold0); rem=np.asarray([x for x in uq if int(x) not in f0]); rng=np.random.default_rng(20260816); extra=rng.choice(rem,size=4000,replace=False); folds=[fold0]+[[str(x) for x in extra[i*1000:(i+1)*1000]] for i in range(4)]; allids=[q for f in folds for q in f]
texts=m.load_query_texts(allids); qrels_all=m.qrels_from_tsv(ROOT/'train.tsv',allids,positive_only=True)
_=selected_features(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.uint8),np.zeros(M,np.uint8),idx.idf,np.zeros(M,np.float32),idx.dl,idx.avgdl); _=prepare(texts[allids[0]])
rows=[]; times=[]; start=time.time()
for fi,ids in enumerate(folds):
 runs=[{} for _ in MODELS]; print('FOLD',fi,'START',flush=True)
 for qi,qid in enumerate(ids):
  t=time.perf_counter(); out=prepare(texts[qid]); times.append((time.perf_counter()-t)*1000)
  if out is None:
   for r in runs:r[qid]=[]
   continue
  dd,scores=out
  for mi,sc in enumerate(scores):
   oo=top100(sc); runs[mi][qid]=[int(x) for x in dd[oo]]
  if (qi+1)%250==0:print('fold',fi,'q',qi+1,'median_ms',float(np.median(times[-250:])),flush=True)
 qr={q:qrels_all[q] for q in ids}
 for mi,(name,k,w) in enumerate(MODELS):
  met=m.eval_run(runs[mi],qr); row={'fold':fi,'name':name,'topk':k,'weight':w,**met}; rows.append(row); print('METRIC',fi,name,met['nDCG@10'],met['MRR@10'],met['R@100'],flush=True)
summary=[]; base=[next(r for r in rows if r['fold']==fi and r['name']=='base') for fi in range(5)]
for name,k,w in MODELS:
 rr=[r for r in rows if r['name']==name]; delta=np.asarray([rr[fi]['nDCG@10']-base[fi]['nDCG@10'] for fi in range(5)]); summary.append({'name':name,'topk':k,'weight':w,'mean_nDCG@10':float(np.mean([r['nDCG@10'] for r in rr])),'mean_MRR@10':float(np.mean([r['MRR@10'] for r in rr])),'mean_R@100':float(np.mean([r['R@100'] for r in rr])),'mean_delta_nDCG_vs_base':float(delta.mean()),'min_delta_nDCG_vs_base':float(delta.min()),'positive_folds':int(np.sum(delta>0)),'fold_deltas':delta.tolist()})
summary.sort(key=lambda x:(x['positive_folds'],x['min_delta_nDCG_vs_base'],x['mean_delta_nDCG_vs_base']),reverse=True); out={'protocol':'5 disjoint 1000-query TRAIN folds; preselection p1 eta1; final IDF^2 b=.1 coord .25 wl4 sem .3 fixed; bounded coverage of top-1/top-2/top-3 IDF query terms added','models':MODELS,'fold_rows':rows,'summary_ranked_for_robustness':summary,'timing':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'seconds':time.time()-start}}; path=WORK/'rare_conjunction_multifold.json'; json.dump(out,open(path,'w'),indent=2); print('SUMMARY'); [print(x) for x in summary[:10]]; print('SAVED',path,flush=True)
