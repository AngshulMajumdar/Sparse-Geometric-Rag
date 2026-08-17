from __future__ import annotations
import sys,time,json
import numpy as np,pandas as pd
from numba import njit,prange,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_best_tail_core as b
import msmarco_full_search_uniform1m as m
ROOT=m.ROOT; WORK=m.WORK; idx=b.idx; M=m.M; P=2000; set_num_threads(5)
CONFIGS=[('base',.2,1.),('b15e4',.15,4.),('b20e3',.2,3.),('b20e4',.2,4.)]
FINAL_B=.1; ALPHA=.25; WLEX=4.; WSEM=.3

def topk(score,k):
 n=len(score); k=min(k,n); ii=np.argpartition(score,-k)[-k:] if n>k else np.arange(n); return ii[np.argsort(score[ii])[::-1]]
@njit(parallel=True,cache=False)
def finalfeat(dd,ip,ids,lexvec,semvec,dl,avgdl):
 n=len(dd); lx=np.zeros(n,np.float32); sm=np.zeros(n,np.float32); qc=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(dd[z]); a=int(ip[d]); bb=int(ip[d+1]); raw=0.; s=0.; c=0.
  for kk in range(a,bb):
   t=int(ids[kk]); v=lexvec[t]
   if v>0: raw+=v; c+=1
   s+=semvec[t]
  r=float(dl[d])/avgdl; den=(1-FINAL_B)+FINAL_B*r; lx[z]=raw/den; sm[z]=s; qc[z]=c
 return lx,sm,qc

def prep(text):
 q=idx.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=idx.route(q); spans=[(int(j),int(idx.offs[j]),int(idx.offs[j+1])) for j in rterms if idx.offs[j+1]>idx.offs[j]]
 if not spans:return None
 docs=np.concatenate([np.asarray(idx.pd[a:bb]) for j,a,bb in spans]).astype(np.uint32,copy=False); mm=np.concatenate([np.asarray(idx.pm[a:bb]) for j,a,bb in spans]).astype(np.float32,copy=False); rt=np.concatenate([np.asarray(idx.pr[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False); sb=np.concatenate([np.asarray(idx.ps[a:bb]) for j,a,bb in spans]).astype(np.uint16,copy=False)
 nr=len(spans); cent=np.zeros((nr,M),np.float32); rel=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
 for u,(j,a,bb) in enumerate(spans):
  row=np.asarray(idx.ct[j]); ok=row!=65535; tids=row[ok].astype(np.int32,copy=False); cent[u,tids]=np.asarray(idx.cv[j])[ok]; ra=int(idx.rp[j]); rb=int(idx.rp[j+1]); rel[u,np.asarray(idx.ri[ra:rb],np.int32)]=np.asarray(idx.rv[ra:rb]); rho[u]=rd[j]
 rslot=np.concatenate([np.full(bb-a,u,dtype=np.uint8) for u,(j,a,bb) in enumerate(spans)]); base,sig,cons=b.score_components(rslot,mm,rt,sb,qd,rho,cent,rel); ud,inv=np.unique(docs,return_inverse=True); tail=np.bincount(inv,weights=base*np.power(sig,b.GAMMA,dtype=np.float32),minlength=len(ud)).astype(np.float32)+b.LAM*np.bincount(inv,weights=cons,minlength=len(ud)).astype(np.float32)
 lexvec=np.zeros(M,np.float32); lexvec[q.indices]=idx.idf[q.indices]; zero=np.zeros(M,np.float32); lex02,_=m.score_support_pool(ud,idx.sup_ip,idx.sup_ids,lexvec,zero,idx.dl,idx.avgdl); ratio=np.asarray(idx.dl[ud],np.float32)/idx.avgdl; raw=lex02*((1-.2)+.2*ratio)
 semvec=np.zeros(M,np.float32)
 for t,amp in zip(q.indices,q.data):
  a,bb=idx.A.indptr[t],idx.A.indptr[t+1]; nb=idx.A.indices[a:bb][:m.SEMK]; sv=idx.A.data[a:bb][:m.SEMK]; semvec[nb]+=float(amp)*sv*idx.idf[nb]
 return q,ud,tail,raw,ratio,lexvec,semvec

def rank_cfg(p,preb,eta):
 q,ud,tail,raw,ratio,lexvec,semvec=p; prelex=raw/np.maximum((1-preb)+preb*ratio,1e-6); sel=topk(m.zscore(tail)+eta*m.zscore(prelex),P); dd=ud[sel]; lx,sm,qc=finalfeat(dd,idx.sup_ip,idx.sup_ids,lexvec,semvec,idx.dl,idx.avgdl); cov=qc/max(1,len(q.indices)); ladj=lx*np.power(np.maximum(cov,1e-6),ALPHA); fin=m.zscore(tail[sel])+WLEX*m.zscore(ladj)+WSEM*m.zscore(sm); oo=np.argsort(fin)[::-1][:100]; return [int(x) for x in dd[oo]],sel

tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); del tr; rng=np.random.default_rng(20260815); qids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; texts=m.load_query_texts(qids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)
_=finalfeat(np.array([0],np.uint32),idx.sup_ip,idx.sup_ids,np.zeros(M,np.float32),np.zeros(M,np.float32),idx.dl,idx.avgdl); _=prep(texts[qids[0]])
runs={n:{} for n,_,_ in CONFIGS}; hits={n:0 for n,_,_ in CONFIGS}; den=0; times=[]
for zi,qid in enumerate(qids):
 t=time.perf_counter(); p=prep(texts[qid]); times.append((time.perf_counter()-t)*1000); rels=[int(d) for d,r in qrels[qid].items() if r>0]; den+=len(rels)
 if p is None:
  for n,_,_ in CONFIGS:runs[n][qid]=[]
  continue
 ud=p[1]
 for n,bb,eta in CONFIGS:
  rank,sel=rank_cfg(p,bb,eta); runs[n][qid]=rank; pool=set(map(int,ud[sel].tolist())); hits[n]+=sum(d in pool for d in rels)
 if (zi+1)%100==0:print('q',zi+1,'medianprep',float(np.median(times)),flush=True)
rows=[]
for n,bb,eta in CONFIGS:
 met=m.eval_run(runs[n],qrels); row={'name':n,'pre_b':bb,'eta':eta,'pool_relevant_recall':hits[n]/den,**met}; rows.append(row); print(row,flush=True)
rows.sort(key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True); out={'protocol':'preselection finalists selected from pool-survival validation, then structural final rule b=.1 alpha=.25 wl4 ws.3 evaluated on same deterministic TRAIN validation','rows':rows,'best':rows[0]}; json.dump(out,open(WORK/'preselection_finalists_validation.json','w'),indent=2); print('BEST',rows[0])
