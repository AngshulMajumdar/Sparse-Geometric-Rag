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

@njit(parallel=True,cache=False)
def score_memberships_local(rslot,mem,rt,sbits,q_dense,rho,cent_local,rel_local):
 K=len(rslot); hc=np.zeros(K,np.float32); tc=np.zeros(K,np.float32); cc=np.zeros(K,np.float32)
 for z in prange(K):
  u=int(rslot[z]); local=0.0; sig=0.0; bits=sbits[z]
  for r in range(S):
   t=int(rt[z,r])
   if t==65535: continue
   qv=q_dense[t]; cen=cent_local[u,t]; rel=rel_local[u,t]; rel=1.0 if rel==0.0 else rel; sgn=1.0 if ((bits>>r)&1)!=0 else -1.0
   local += rel*(qv-cen)*sgn; sig += qv*qv
  rh=rho[u]; m=mem[z]
  hc[z]=m*rh*local*(sig**0.5 if sig>0 else 0.0)
  tc[z]=m*rh*local*sig
  cc[z]=m*rh
 return hc,tc,cc

@njit(parallel=True,cache=False)
def score_support_pool(cand_docs,ip,ids,lexvec,semvec,dl,avgdl):
 n=len(cand_docs); lex=np.zeros(n,np.float32); sem=np.zeros(n,np.float32)
 for z in prange(n):
  d=int(cand_docs[z]); a=int(ip[d]); b=int(ip[d+1]); lx=0.0; sm=0.0
  for k in range(a,b):
   t=int(ids[k]); lx+=lexvec[t]; sm+=semvec[t]
  denom=(1.0-LENGTH_B)+LENGTH_B*(float(dl[d])/avgdl)
  lex[z]=lx/(denom if denom>0 else 1.0); sem[z]=sm
 return lex,sem

@njit(cache=False)
def aggregate_by_doc(docs,hc,tc,cc,mark,head_acc,tail_acc,gen):
 # Dense generation-mark accumulator avoids sorting every membership record.
 # The returned document IDs are sorted afterwards to preserve deterministic
 # tie behaviour of the original np.unique path.
 seen=np.empty(len(docs),np.uint32); nseen=0
 for z in range(len(docs)):
  d=int(docs[z])
  if mark[d]!=gen:
   mark[d]=gen; head_acc[d]=0.0; tail_acc[d]=0.0; seen[nseen]=d; nseen+=1
  head_acc[d]+=float(hc[z])
  tail_acc[d]+=float(tc[z])+LAMBDA_M*float(cc[z])
 return seen[:nseen]

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
  self.dl=np.memmap(IDX/'doc_lengths.u16',np.uint16,'r',shape=(N,)); self.offs=np.load(IDX/'branch_offsets.npy',mmap_mode='r'); _np=int(self.offs[-1]); self.pd=np.memmap(IDX/'post_doc.u32',np.uint32,'r',shape=(_np,)); self.pm=np.memmap(IDX/'post_membership.f32',np.float32,'r',shape=(_np,)); self.pr=np.memmap(IDX/'post_res_terms.u16',np.uint16,'r',shape=(_np,S)); self.ps=np.memmap(IDX/'post_signbits.u16',np.uint16,'r',shape=(_np,))
  with open(IDX/'meta.json') as f: self.meta=json.load(f)
  self.avgdl=float(self.meta['avg_doc_length']); self.sup_ip=np.memmap(IDX/'support_all_indptr.u32',np.uint32,'r',shape=(N+1,)); self.sup_ids=np.memmap(IDX/'support_all.u16',np.uint16,'r',shape=(329617090,))
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
  a=int(self.sup_ip[int(d)]); b=int(self.sup_ip[int(d)+1]); return self.sup_ids[a:b]
 def prepare(self,text,hmax=20):
  q=self.query_vec(text); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; rterms,rd=self.route(q)
  spans=[(int(j),int(self.offs[j]),int(self.offs[j+1])) for j in rterms if self.offs[j+1]>self.offs[j]]
  if not spans: return None
  docs=np.concatenate([np.asarray(self.pd[a:b]) for j,a,b in spans]).astype(np.uint32,copy=False)
  mm=np.concatenate([np.asarray(self.pm[a:b]) for j,a,b in spans]).astype(np.float32,copy=False)
  rt=np.concatenate([np.asarray(self.pr[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
  sb=np.concatenate([np.asarray(self.ps[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
  # Query-local dense lookup tables: only <=32 routed branches are materialized.
  # This replaces millions of tiny binary searches into center/reliability CSR rows.
  nr=len(spans); cent_local=np.zeros((nr,M),np.float32); rel_local=np.zeros((nr,M),np.float32); rho=np.empty(nr,np.float32)
  for u,(j,a,b) in enumerate(spans):
   rowt=np.asarray(self.ct[j]); ok=rowt!=65535; ids=rowt[ok].astype(np.int32,copy=False); cent_local[u,ids]=np.asarray(self.cv[j])[ok]
   ra=int(self.rp[j]); rb=int(self.rp[j+1]); rel_local[u,np.asarray(self.ri[ra:rb],np.int32)]=np.asarray(self.rv[ra:rb])
   rho[u]=rd[j]
  rslot=np.concatenate([np.full(b-a,u,dtype=np.uint8) for u,(j,a,b) in enumerate(spans)])
  hc,tc,cc=score_memberships_local(rslot,mm,rt,sb,qd,rho,cent_local,rel_local)
  ud,inv=np.unique(docs,return_inverse=True)
  head=np.bincount(inv,weights=hc,minlength=len(ud)).astype(np.float32)
  tail=np.bincount(inv,weights=tc,minlength=len(ud)).astype(np.float32)+LAMBDA_M*np.bincount(inv,weights=cc,minlength=len(ud)).astype(np.float32)
  # Only the top hmax head documents and top P+hmax tail documents are ever used.
  # Partial selection avoids O(C log C) full sorts when C is 10^5--10^6.
  hwant=min(len(head),max(1,hmax))
  if len(head)>hwant:
   hi=np.argpartition(head,-hwant)[-hwant:]; ho=hi[np.argsort(head[hi])[::-1]]
  else: ho=np.argsort(head)[::-1]
  want=min(len(tail),P+hmax+8)
  if len(tail)>want:
   ci=np.argpartition(tail,-want)[-want:]; cheap=ci[np.argsort(tail[ci])[::-1]]
  else: cheap=np.argsort(tail)[::-1]
  cand_idx=cheap[:want]; cand_docs=ud[cand_idx]; cand_tail=tail[cand_idx]
  lexvec=np.zeros(M,np.float32); lexvec[q.indices]=self.idf[q.indices]; semvec=np.zeros(M,np.float32)
  for t,amp in zip(q.indices,q.data):
   a,b=self.A.indptr[t],self.A.indptr[t+1]; nb=self.A.indices[a:b][:SEMK]; sv=self.A.data[a:b][:SEMK]; semvec[nb]+=float(amp)*sv*self.idf[nb]
  lex,sem=score_support_pool(cand_docs,self.sup_ip,self.sup_ids,lexvec,semvec,self.dl,self.avgdl)
  return {'ud':ud,'head_order':ho,'cand_docs':cand_docs,'cand_tail':cand_tail,'lex':lex,'sem':sem,'candidate_memberships':len(docs),'candidate_docs':len(ud)}
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
 # Validation protocol: tune h on a deterministic 1000-query sample from TRAIN,
 # then lock h and evaluate the entire 6,980-query DEV split untouched.
 tr=pd.read_csv(ROOT/'train.tsv',sep='\t',usecols=['query-id']); uq=np.unique(tr['query-id'].to_numpy()); rng=np.random.default_rng(20260815); val_ids=[str(x) for x in rng.choice(uq,size=1000,replace=False)]; del tr
 devdf=pd.read_csv(ROOT/'dev.tsv',sep='\t',usecols=['query-id']); dev_ids=[str(x) for x in np.unique(devdf['query-id'].to_numpy())]; del devdf
 texts=load_query_texts(val_ids+dev_ids); valq=qrels_from_tsv(ROOT/'train.tsv',val_ids,positive_only=True); devq=qrels_from_tsv(ROOT/'dev.tsv',dev_ids,positive_only=True)
 missing=[q for q in val_ids+dev_ids if q not in texts]
 if missing: raise RuntimeError(f'missing query texts: {missing[:10]} ({len(missing)} total)')
 # JIT and I/O warmup; excluded from timing.
 _w=idx.prepare(texts[val_ids[0]],hmax=max(HGRID)); _=idx.rank_h(_w,0,100); del _w
 # Prepare each validation query ONCE, immediately materialize all h rankings,
 # and discard the large candidate arrays. This keeps RAM bounded at scale.
 vruns={h:{} for h in HGRID}; times=[]; cands=[]; memc=[]
 for z,qid in enumerate(val_ids):
  t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=max(HGRID)); times.append((time.perf_counter()-t)*1000)
  cands.append(pp['candidate_docs'] if pp else 0); memc.append(pp['candidate_memberships'] if pp else 0)
  for h in HGRID: vruns[h][qid]=idx.rank_h(pp,h,100)
  if (z+1)%100==0: print('val',z+1,'median_ms',float(np.median(times)),'p95_ms',float(np.percentile(times,95)),'avg_candidate_docs',float(np.mean(cands)),flush=True)
 rows=[]
 for h in HGRID:
  mm=eval_run(vruns[h],valq); rows.append((h,mm)); print('H',h,mm,flush=True)
 best=max(rows,key=lambda x:(x[1]['nDCG@10'],x[1]['MRR@10'],x[1]['R@100']))[0]; print('BEST_H',best,flush=True)
 val_timing={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands)),'avg_candidate_memberships':float(np.mean(memc))}
 del vruns
 # Full untouched DEV evaluation with h locked.
 run={}; times=[]; cands=[]; memc=[]
 for z,qid in enumerate(dev_ids):
  t=time.perf_counter(); pp=idx.prepare(texts[qid],hmax=max(best,1)); run[qid]=idx.rank_h(pp,best,100); times.append((time.perf_counter()-t)*1000); cands.append(pp['candidate_docs'] if pp else 0); memc.append(pp['candidate_memberships'] if pp else 0)
  if (z+1)%250==0: print('dev',z+1,'median_ms',float(np.median(times)),'p95_ms',float(np.percentile(times,95)),'avg_candidate_docs',float(np.mean(cands)),flush=True)
 mm=eval_run(run,devq); timing={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'mean_ms':float(np.mean(times)),'qps':1000/float(np.mean(times)),'avg_candidate_docs':float(np.mean(cands)),'median_candidate_docs':float(np.median(cands)),'avg_candidate_memberships':float(np.mean(memc))}
 out={'protocol':'h tuned on deterministic 1000-query TRAIN sample; full DEV untouched','best_h':best,'validation':{str(h):vv for h,vv in rows},'validation_timing':val_timing,'dev_metrics':mm,'timing':timing,'index_meta':idx.meta,'geometry_note':'full 8.84M vocabulary/IDF; geometric codebook calibrated on first 1M passages'}
 with open(WORK/'full_msmarco_results.json','w') as f: json.dump(out,f,indent=2)
 print('DEV_METRICS',mm,flush=True); print('TIMING',timing,flush=True); print('saved',WORK/'full_msmarco_results.json',flush=True)
