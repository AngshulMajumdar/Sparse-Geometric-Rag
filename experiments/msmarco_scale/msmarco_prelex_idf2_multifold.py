from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000
set_num_threads(5)
MODELS=[(1.0,0.5),(1.0,1.0),(1.0,2.0),(2.0,0.5),(2.0,1.0),(2.0,2.0)]
PRE_B=np.float32(0.2); FINAL_B=np.float32(0.1); ALPHA=np.float32(0.25); WLEX=np.float32(4.0); WSEM=np.float32(0.3)

def topk_desc(score,k):
 n=len(score); k=min(k,n)
 if n<=k:return np.argsort(score)[::-1]
 ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

@njit(parallel=True,cache=False)
def routed_lex12(dd,ip,ids,qmask,idf,dl,avgdl):
 n=len(dd); l1=np.zeros(n,np.float32); l2=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); s1=0.; s2=0.
  for k in range(a,bb):
   t=int(ids[k])
   if qmask[t]:
    x=float(idf[t]); s1+=x; s2+=x*x
  ratio=float(dl[d])/avgdl; den=(1.0-PRE_B)+PRE_B*ratio
  if den<=0:den=1.0
  l1[z]=s1/den; l2[z]=s2/den
 return l1,l2

@njit(parallel=True,cache=False)
def final_features_idf2(dd,ip,ids,qmask,idf,semvec,dl,avgdl):
 n=len(dd); lx=np.zeros(n,np.float32); sm=np.zeros(n,np.float32); cnt=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); s2=0.; ss=0.; c=0.
  for k in range(a,bb):
   t=int(ids[k]); ss+=semvec[t]
   if qmask[t]:
    x=float(idf[t]); s2+=x*x; c+=1.
  ratio=float(dl[d])/avgdl; den=(1.0-FINAL_B)+FINAL_B*ratio
  if den<=0:den=1.0
  lx[z]=s2/den; sm[z]=ss; cnt[z]=c
 return lx,sm,cnt

def prepare_route(text):
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q)
 spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans:return None
 docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False); mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False); rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False); sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  rowt=np.asarray(idx.ct[j]); ok=rowt!=65535; tids=rowt[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]; ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)]); base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel)
 ud,inv=np.unique(docs,return_inverse=True); tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
 qmask=np.zeros(M,np.uint8); qmask[q.indices]=1
 l1,l2=routed_lex12(ud,idx.sup_ip,idx.sup_ids,qmask,idx.idf,idx.dl,idx.avgdl)
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 return q,ud,tail,l1,l2,qmask,semvec

def final_rank(q,ud,tail,sel,qmask,semvec):
 dd=ud[sel]; ts=tail[sel]; lx,sm,cnt=final_features_idf2(dd,idx.sup_ip,idx.sup_ids,qmask,idx.idf,semvec,idx.dl,idx.avgdl); cov=cnt/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA); fin=m.zscore(ts)+WLEX*m.zscore(ladj)+WSEM*m.zscore(sm); oo=topk_desc(fin,100); return [int(x) for x in dd[oo]]

# Same five TRAIN folds.
z0=np.load(WORK/'amplitude_diag'/'fixed_eta1_pools.npz',allow_pickle=False); fold0=[str(x) for x in z0['qids'].tolist()]
tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr
f0set=set(int(x) for x in fold0); remaining=np.asarray([x for x in uq if int(x) not in f0set]); rng=np.random.default_rng(20260816); extra=rng.choice(remaining,size=4000,replace=False)
folds=[fold0]+[[str(x) for x in extra[i*1000:(i+1)*1000]] for i in range(4)]; allids=[q for f in folds for q in f]
texts=m.load_query_texts(allids); qrels_all=m.qrels_from_tsv(ROOT/'train.tsv',allids,positive_only=True)
_=routed_lex12(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.uint8),idx.idf,idx.dl,idx.avgdl); _=final_features_idf2(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.uint8),idx.idf,np.zeros(M,np.float32),idx.dl,idx.avgdl); _=prepare_route(texts[allids[0]])
rows=[]; times=[]; start=time.time()
for fi,ids in enumerate(folds):
 runs={model:{} for model in MODELS}; pool={model:0 for model in MODELS}; den=routehit=0
 print('FOLD',fi,'START',flush=True)
 for qi,qid in enumerate(ids):
  t=time.perf_counter(); p=prepare_route(texts[qid]); times.append((time.perf_counter()-t)*1000)
  rels=[int(d) for d,r in qrels_all[qid].items() if r>0]; den+=len(rels)
  if p is None:
   for model in MODELS:runs[model][qid]=[]
   continue
  q,ud,tail,l1,l2,qmask,semvec=p; relset=set()
  for d in rels:
   kk=np.searchsorted(ud,d); ok=kk<len(ud) and int(ud[kk])==d; routehit+=int(ok)
   if ok:relset.add(int(kk))
  zt=m.zscore(tail); z1=m.zscore(l1); z2=m.zscore(l2)
  for power,eta in MODELS:
   zl=z1 if power==1.0 else z2; sel=topk_desc(zt+np.float32(eta)*zl,P); pool[(power,eta)]+=sum(int(x) in relset for x in sel); runs[(power,eta)][qid]=final_rank(q,ud,tail,sel,qmask,semvec)
  if (qi+1)%250==0:print('fold',fi,'q',qi+1,'median_route_ms',float(np.median(times[-250:])),flush=True)
 qr={q:qrels_all[q] for q in ids}
 for model in MODELS:
  met=m.eval_run(runs[model],qr); row={'fold':fi,'pre_idf_power':model[0],'eta':model[1],'pool_relevant_recall':pool[model]/den,'route_relevant_recall':routehit/den,**met}; rows.append(row); print('METRIC',row,flush=True)
summary=[]
base_by_fold=[next(r for r in rows if r['fold']==fi and r['pre_idf_power']==1.0 and r['eta']==1.0) for fi in range(5)]
for model in MODELS:
 rr=[r for r in rows if r['pre_idf_power']==model[0] and r['eta']==model[1]]; delta=np.asarray([rr[fi]['nDCG@10']-base_by_fold[fi]['nDCG@10'] for fi in range(5)])
 summary.append({'pre_idf_power':model[0],'eta':model[1],'mean_nDCG@10':float(np.mean([r['nDCG@10'] for r in rr])),'mean_MRR@10':float(np.mean([r['MRR@10'] for r in rr])),'mean_R@100':float(np.mean([r['R@100'] for r in rr])),'mean_pool_relevant_recall':float(np.mean([r['pool_relevant_recall'] for r in rr])),'mean_delta_nDCG_vs_p1_eta1':float(delta.mean()),'min_delta_nDCG_vs_p1_eta1':float(delta.min()),'positive_folds':int(np.sum(delta>0)),'fold_deltas':delta.tolist()})
summary.sort(key=lambda x:(x['positive_folds'],x['min_delta_nDCG_vs_p1_eta1'],x['mean_delta_nDCG_vs_p1_eta1']),reverse=True)
out={'protocol':'5 disjoint 1000-query TRAIN folds; final rule fixed to validated IDF^2 b=.1 coord .25 wl4 sem .3; only early lexical rescue IDF power and eta varied','models':MODELS,'fold_rows':rows,'summary_ranked_for_robustness':summary,'timing':{'median_route_ms':float(np.median(times)),'p95_route_ms':float(np.percentile(times,95)),'seconds':time.time()-start}}
path=WORK/'prelex_idf2_multifold.json'; json.dump(out,open(path,'w'),indent=2); print('SUMMARY'); [print(x) for x in summary]; print('SAVED',path,flush=True)
