from __future__ import annotations
import gzip,json,pickle,time,math,random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from numba import njit, prange, set_num_threads

ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; GEOM=WORK/'geometry_1m'; IDX=WORK/'full_index'
N=8_841_823; M=50_000; F=4; S=16; SENT=np.uint16(65535)
ROUTE_ALPHA=.10; ROUTE_BUDGET=32; GAMMA_HEAD=.5; GAMMA_TAIL=1.; LAMBDA_M=2.; P=2000; LAMBDA_LEX=2.5; LENGTH_B=.2; SEMK=16; LAMBDA_SEM=.05
HGRID=list(range(0,11))+[15,20]
set_num_threads(5)

@njit(cache=False)
def lookup_center(ct,cv,t):
 lo=0; hi=ct.size
 while lo<hi:
  mid=(lo+hi)//2; x=ct[mid]
  if x==65535 or x>=t: hi=mid
  else: lo=mid+1
 if lo<ct.size and ct[lo]==t: return cv[lo]
 return 0.0

@njit(cache=False)
def lookup_rel(indptr,indices,data,j,t):
 lo=np.int64(indptr[j]); hi=np.int64(indptr[j+1]); b=hi
 while lo<hi:
  mid=np.int64((lo+hi)//2); x=indices[mid]
  if x>=t: hi=mid
  else: lo=mid+1
 if lo<b and indices[lo]==t: return data[lo]
 return 1.0

@njit(parallel=True,cache=False)
def score_memberships(br,mem,rt,sbits,q_dense,route_dense,ct,cv,rp,ri,rv):
 K=len(br); hc=np.zeros(K,np.float32); tc=np.zeros(K,np.float32); cc=np.zeros(K,np.float32)
 for z in prange(K):
  j=int(br[z]); local=0.0; sig=0.0; bits=sbits[z]
  for r in range(S):
   t=int(rt[z,r])
   if t==65535: continue
   qv=q_dense[t]; cen=lookup_center(ct[j],cv[j],t); rel=lookup_rel(rp,ri,rv,j,t); sgn=1.0 if ((bits>>r)&1)!=0 else -1.0
   local += rel*(qv-cen)*sgn; sig += qv*qv
  rho=route_dense[j]; m=mem[z]
  hc[z]=m*rho*local*(sig**0.5 if sig>0 else 0.0)
  tc[z]=m*rho*local*sig
  cc[z]=m*rho
 return hc,tc,cc

def zscore(x):
 x=np.asarray(x,np.float32); s=float(x.std()); return np.zeros_like(x) if s<1e-8 else (x-float(x.mean()))/(s+1e-8)

def dcg(vals):
 return sum((2.0**float(r)-1.0)/math.log2(i+2) for i,r in enumerate(vals))

def eval_run(run,qrels):
 metrics={'nDCG@10':[],'MRR@10':[],'P@10':[],'R@10':[],'R@100':[],'Hit@10':[],'Hit@100':[]}
 for qid,qr in qrels.items():
  rank=run[qid]; pos={int(d) for d,r in qr.items() if r>0}; n=max(1,len(pos))
  h10=sum(d in pos for d in rank[:10]); h100=sum(d in pos for d in rank[:100]); metrics['P@10'].append(h10/10); metrics['R@10'].append(h10/n); metrics['R@100'].append(h100/n); metrics['Hit@10'].append(float(h10>0)); metrics['Hit@100'].append(float(h100>0))
  rr=0
  for i,d in enumerate(rank[:10],1):
   if d in pos: rr=1/i; break
  metrics['MRR@10'].append(rr)
  obs=[float(qr.get(str(d),qr.get(d,0.0))) for d in rank[:10]]; ideal=sorted([float(r) for r in qr.values()],reverse=True)[:10]; idc=dcg(ideal); metrics['nDCG@10'].append(dcg(obs)/idc if idc else 0)
 return {k:float(np.mean(v)) for k,v in metrics.items()} | {'n_queries':len(qrels)}

class FullIndex:
 def __init__(self):
  with gzip.open(WORK/'final_vocab_50k.pkl.gz','rb') as g: z=pickle.load(g)
  self.terms=z['terms'].tolist(); self.idf=np.asarray(z['idf'],np.float32); self.vocab={t:i for i,t in enumerate(self.terms)}
  self.cvq=CountVectorizer(vocabulary=self.vocab,lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.int32)
  self.ct=np.load(GEOM/'center_terms.npy',mmap_mode='r'); self.cv=np.load(GEOM/'center_values.npy',mmap_mode='r'); self.A=sparse.load_npz(GEOM/'assoc_ppmi.npz').tocsr(); self.G=sparse.load_npz(GEOM/'context_similarity.npz').tocsr(); self.rp=np.load(GEOM/'rel_indptr.npy',mmap_mode='r'); _rm=json.load(open(GEOM/'rel_meta.json')); _rn=int(_rm['nnz']); self.ri=np.memmap(GEOM/'rel_indices.u16',np.uint16,'r',shape=(_rn,)); self.rv=np.memmap(GEOM/'rel_data.f32',np.float32,'r',shape=(_rn,))
  self.branches=np.memmap(IDX/'branches.u16',np.uint16,'r',shape=(N,F)); self.mem=np.memmap(IDX/'memberships.f32',np.float32,'r',shape=(N,F)); self.rt=np.memmap(IDX/'res_terms.u16',np.uint16,'r',shape=(N,F,S)); self.sb=np.memmap(IDX/'signbits.u16',np.uint16,'r',shape=(N,F)); self.dl=np.memmap(IDX/'doc_lengths.u16',np.uint16,'r',shape=(N,)); self.bo=np.memmap(IDX/'branch_order.u32',np.uint32,'r'); self.offs=np.load(IDX/'branch_offsets.npy',mmap_mode='r')
  with open(IDX/'meta.json') as f: self.meta=json.load(f)
  self.avgdl=float(self.meta['avg_doc_length']); self.sup=[]
  for sid in range(36):
   with open(IDX/f'shard_{sid:04d}.json') as f: sm=json.load(f)
   ii=np.memmap(IDX/f'support_{sid:04d}.u16',np.uint16,'r',shape=(sm['nnz'],)); ip=np.memmap(IDX/f'support_indptr_{sid:04d}.u32',np.uint32,'r',shape=(sm['n']+1,)); self.sup.append((sm['offset'],sm['n'],ip,ii))
 def query_vec(self,text):
  q=self.cvq.transform([text]).tocsr().astype(np.float32); q.data*=self.idf[q.indices]; normalize(q,norm='l2',axis=1,copy=False); return q
 def route(self,q):
  qt=q.indices; qv=q.data; dense=np.zeros(M,np.float32); dense[qt]=qv
  for t,v in zip(qt,qv):
   a,b=self.G.indptr[t],self.G.indptr[t+1]; dense[self.G.indices[a:b]] += ROUTE_ALPHA*float(v)*self.G.data[a:b]
  nz=np.flatnonzero(dense>0); orig=set(map(int,qt.tolist()))
  if len(nz)>ROUTE_BUDGET:
   inf=np.asarray([i for i in nz if int(i) not in orig],np.int32); budget=max(0,ROUTE_BUDGET-len(orig))
   if budget and len(inf)>budget: inf=inf[np.argpartition(dense[inf],-budget)[-budget:]]
   elif budget==0: inf=np.empty(0,np.int32)
   nz=np.concatenate([np.asarray(sorted(orig),np.int32),inf])
  nz=nz[np.argsort(dense[nz])[::-1]]; return nz.astype(np.int32),dense
 def support(self,d):
  sid=min(35,int(d)//250_000); off,n,ip,ii=self.sup[sid]; ld=int(d)-off; a=int(ip[ld]); b=int(ip[ld+1]); return ii[a:b]
 def prepare(self,text,hmax=20):
  q=self.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=self.route(q)
  pieces=[self.bo[int(self.offs[j]):int(self.offs[j+1])] for j in rterms if self.offs[j+1]>self.offs[j]]
  if not pieces: return None
  fp=np.concatenate(pieces).astype(np.uint32,copy=False); docs=(fp//F).astype(np.uint32); slots=(fp%F).astype(np.uint8); br=self.branches[docs,slots]; mm=self.mem[docs,slots]; rt=self.rt[docs,slots]; sb=self.sb[docs,slots]
  hc,tc,cc=score_memberships(br,mm,rt,sb,qd,rd,self.ct,self.cv,self.rp,self.ri,self.rv)
  ud,inv=np.unique(docs,return_inverse=True); head=np.bincount(inv,weights=hc,minlength=len(ud)).astype(np.float32); tail=np.bincount(inv,weights=tc,minlength=len(ud)).astype(np.float32)+LAMBDA_M*np.bincount(inv,weights=cc,minlength=len(ud)).astype(np.float32)
  ho=np.argsort(head)[::-1]; cheap=np.argsort(tail)[::-1]; want=min(len(cheap),P+hmax+8); cand_idx=cheap[:want]; cand_docs=ud[cand_idx]; cand_tail=tail[cand_idx]
  lexvec=np.zeros(M,np.float32); lexvec[q.indices]=self.idf[q.indices]; semvec=np.zeros(M,np.float32)
  for t,amp in zip(q.indices,q.data):
   a,b=self.A.indptr[t],self.A.indptr[t+1]; nb=self.A.indices[a:b][:SEMK]; sv=self.A.data[a:b][:SEMK]; semvec[nb]+=float(amp)*sv*self.idf[nb]
  lex=np.zeros(want,np.float32); sem=np.zeros(want,np.float32)
  for i,d in enumerate(cand_docs):
   sp=self.support(int(d)); lex[i]=float(lexvec[sp].sum()); denom=(1-LENGTH_B)+LENGTH_B*(float(self.dl[int(d)])/self.avgdl); lex[i]/=denom if denom>0 else 1.; sem[i]=float(semvec[sp].sum())
  return {'ud':ud,'head':head,'head_order':ho,'tail':tail,'cheap_order':cheap,'cand_docs':cand_docs,'cand_tail':cand_tail,'lex':lex,'sem':sem,'candidate_memberships':len(fp),'candidate_docs':len(ud)}
 def rank_h(self,p,h,k=100):
  if p is None:return []
  ud=p['ud']; frozen=ud[p['head_order'][:min(h,len(ud))]] if h else np.empty(0,np.uint32); fs=set(map(int,frozen.tolist()))
  # Candidate rerank arrays were computed for top P+hmax; exclude frozen and take P.
  keep=np.asarray([int(d) not in fs for d in p['cand_docs']],bool); docs=p['cand_docs'][keep][:P]; ts=p['cand_tail'][keep][:P]; lx=p['lex'][keep][:P]; sm=p['sem'][keep][:P]
  final=zscore(ts)+LAMBDA_LEX*zscore(lx)+LAMBDA_SEM*zscore(sm); oo=np.argsort(final)[::-1]; taildocs=docs[oo]
  ranked=np.concatenate([frozen,taildocs])[:k]; return [int(x) for x in ranked]

def load_query_texts(want):
 want=set(map(str,want)); out={}
 with open(ROOT/'queries.jsonl','r',encoding='utf-8') as f:
  for line in f:
   o=json.loads(line); qid=str(o['_id'])
   if qid in want: out[qid]=o.get('text','') or ''
 return out

def qrels_from_tsv(path,qids=None,positive_only=False):
 df=pd.read_csv(path,sep='\t'); qset=None if qids is None else set(map(str,qids)); out={}
 for q,d,s in zip(df['query-id'],df['corpus-id'],df['score']):
  q=str(q)
  if qset is not None and q not in qset: continue
  if positive_only and float(s)<=0: continue
  out.setdefault(q,{})[str(d)]=float(s)
 return out

if __name__=='__main__':
 idx=FullIndex(); print('loaded full index',idx.meta,flush=True)
 # validation: deterministic 1000 train queries, selected without touching dev qrels
 tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); val_ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
 devdf=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); dev_ids=[str(x) for x in np.unique(devdf['query-id'].to_numpy())]; del devdf
 texts=load_query_texts(val_ids+dev_ids); valq=qrels_from_tsv(ROOT/'train.tsv',val_ids,positive_only=True); devq=qrels_from_tsv(ROOT/'dev.tsv',dev_ids,positive_only=True)
 # Prepare validation queries once, sweep h without repeating retrieval.
 pre={}; times=[]; cands=[]
 for z,qid in enumerate(val_ids):
  t=time.perf_counter(); pre[qid]=idx.prepare(texts[qid],hmax=max(HGRID)); times.append((time.perf_counter()-t)*1000); p=pre[qid]; cands.append(p['candidate_docs'] if p else 0)
  if (z+1)%100==0: print('val prepared',z+1,'median ms',float(np.median(times)),'avg candidates',float(np.mean(cands)),flush=True)
 rows=[]
 for h in HGRID:
  run={qid:idx.rank_h(pre[qid],h,100) for qid in val_ids}; m=eval_run(run,valq); rows.append((h,m)); print('H',h,m,flush=True)
 best=max(rows,key=lambda x:(x[1]['nDCG@10'],x[1]['MRR@10'],x[1]['R@100']))[0]; print('BEST_H',best,flush=True)
 # Release validation intermediates before full dev.
 del pre
 run={}; times=[]; cands=[]
 for z,qid in enumerate(dev_ids):
  t=time.perf_counter(); p=idx.prepare(texts[qid],hmax=best); run[qid]=idx.rank_h(p,best,100); times.append((time.perf_counter()-t)*1000); cands.append(p['candidate_docs'] if p else 0)
  if (z+1)%250==0: print('dev',z+1,'median',float(np.median(times)),'p95',float(np.percentile(times,95)),'avgcand',float(np.mean(cands)),flush=True)
 m=eval_run(run,devq); timing={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands)),'median_candidate_docs':float(np.median(cands))}
 out={'best_h':best,'validation':{str(h):mm for h,mm in rows},'dev_metrics':m,'timing':timing,'index_meta':idx.meta}
 with open(WORK/'full_msmarco_results.json','w') as f: json.dump(out,f,indent=2)
 print('DEV_METRICS',m,flush=True); print('TIMING',timing,flush=True); print('saved',WORK/'full_msmarco_results.json',flush=True)
