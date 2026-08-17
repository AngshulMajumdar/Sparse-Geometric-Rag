from __future__ import annotations
import sys,json,math,time
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit,set_num_threads
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_uniform1m as m

ROOT=m.ROOT; WORK=m.WORK; OUT=WORK/'amplitude_diag'; IDX=m.IDX; GEOM=m.GEOM
N=m.N; M=m.M; F=4; S=16; P=2000
set_num_threads(5)
idx=m.FullIndex()

z=np.load(OUT/'fixed_eta1_pools.npz',allow_pickle=False)
qids=[str(x) for x in z['qids'].tolist()]; valid=z['valid'].astype(np.int32); docs=z['docs']; tail=z['tail']; lex=z['lex']; sem=z['sem']
union=np.load(OUT/'union_docs.npy',mmap_mode='r'); uip=np.load(OUT/'exact_tfidf_indptr.npy',mmap_mode='r'); exact=np.memmap(OUT/'exact_tfidf_data.f32',np.float32,'r',shape=(int(uip[-1]),))
# document-order geometry arrays
base=WORK/'full_index'
branches=np.memmap(base/'branches.u16',np.uint16,'r',shape=(N,F))
memberships=np.memmap(base/'memberships.f32',np.float32,'r',shape=(N,F))
res_terms=np.memmap(IDX/'res_terms.u16',np.uint16,'r',shape=(N,F,S))
# query texts/qrels
texts=m.load_query_texts(qids); qrels=m.qrels_from_tsv(ROOT/'train.tsv',qids,positive_only=True)

@njit(cache=False)
def bsearch_u16(arr,a,b,t):
    lo=np.int64(a); hi=np.int64(b)
    while lo<hi:
        mid=np.int64((lo+hi)//2)
        x=int(arr[mid])
        if x<t: lo=mid+1
        else: hi=mid
    if lo<b and int(arr[lo])==t:return lo
    return -1

@njit(cache=False)
def center_lookup(ctrow,cvrow,t):
    lo=0; hi=len(ctrow)
    while lo<hi:
        md=(lo+hi)//2; x=int(ctrow[md])
        if x==65535 or x>=t: hi=md
        else: lo=md+1
    if lo<len(ctrow) and int(ctrow[lo])==t:return float(cvrow[lo])
    return 0.0

@njit(cache=False)
def rel_lookup(rp,ri,rv,j,t):
    lo=np.int64(rp[j]); hi=np.int64(rp[j+1]); end=hi
    while lo<hi:
        md=np.int64((lo+hi)//2); x=int(ri[md])
        if x<t: lo=md+1
        else: hi=md
    if lo<end and int(ri[lo])==t:return float(rv[lo])
    return 1.0

@njit(cache=False)
def score_pool(cdocs, union, uip, exact, sup_ip, sup_ids, branches, memberships, res_terms,
               qdense, route_dense, ct, cv, rp, ri, rv):
    n=len(cdocs); tfcos=np.zeros(n,np.float32); amp_tail=np.zeros(n,np.float32)
    for zz in range(n):
        d=int(cdocs[zz])
        ur=np.searchsorted(union,np.uint32(d))
        if ur>=len(union) or int(union[ur])!=d: continue
        ga=int(sup_ip[d]); gb=int(sup_ip[d+1]); ua=int(uip[ur])
        # exact normalized tf-idf cosine
        s=0.0
        for kk in range(ga,gb):
            t=int(sup_ids[kk]); s += float(exact[ua+(kk-ga)])*float(qdense[t])
        tfcos[zz]=s
        # exact retained residual amplitudes, same current gamma=.25 and lambdaM=.125
        tsum=0.0; csum=0.0
        for f in range(F):
            j=int(branches[d,f])
            if j==65535: continue
            rho=float(route_dense[j])
            if rho<=0.0: continue
            mem=float(memberships[d,f]); local=0.0; sig=0.0
            for r in range(S):
                t=int(res_terms[d,f,r])
                if t==65535: continue
                pos=bsearch_u16(sup_ids,ga,gb,t)
                if pos<0: continue
                xv=float(exact[ua+(pos-ga)])
                cen=center_lookup(ct[j],cv[j],t)
                rel=rel_lookup(rp,ri,rv,j,t)
                qv=float(qdense[t]); rr=xv-cen
                local += rel*(qv-cen)*rr
                sig += qv*qv
            c=mem*rho
            tsum += c*local*(sig**0.25 if sig>0 else 0.0)
            csum += c
        amp_tail[zz]=tsum + 0.125*csum
    return tfcos,amp_tail

# feature matrices for reproducibility
TF=np.zeros((len(qids),P),np.float32); AMP=np.zeros((len(qids),P),np.float32)
start=time.time()
for qi,qid in enumerate(qids):
    k=int(valid[qi]); q=idx.query_vec(texts[qid]); qd=np.zeros(M,np.float32); qd[q.indices]=q.data; _,rd=idx.route(q)
    tf,amp=score_pool(docs[qi,:k],union,uip,exact,idx.sup_ip,idx.sup_ids,branches,memberships,res_terms,qd,rd,idx.ct,idx.cv,idx.rp,idx.ri,idx.rv)
    TF[qi,:k]=tf; AMP[qi,:k]=amp
    if (qi+1)%100==0: print('features',qi+1,'elapsed',time.time()-start,flush=True)
np.save(OUT/'exact_tfidf_cos.npy',TF); np.save(OUT/'exact_residual_amp_tail.npy',AMP)

def rank_score(score,k=100):
    if len(score)<=k:return np.argsort(score)[::-1]
    ii=np.argpartition(score,-k)[-k:]; return ii[np.argsort(score[ii])[::-1]]

def evaluate(kind, a=0.0, x=0.0, use_cur=True, use_amp=False):
    run={}
    for qi,qid in enumerate(qids):
        k=int(valid[qi]); dd=docs[qi,:k]
        if kind=='tfidf_only': sc=TF[qi,:k]
        elif kind=='amp_only': sc=AMP[qi,:k]
        elif kind=='binary_only': sc=lex[qi,:k]
        else:
            sc=np.zeros(k,np.float32)
            if use_cur: sc += m.zscore(tail[qi,:k])
            if use_amp: sc += np.float32(a)*m.zscore(AMP[qi,:k])
            sc += np.float32(4.0)*m.zscore(lex[qi,:k]) + np.float32(.1)*m.zscore(sem[qi,:k])
            if x!=0: sc += np.float32(x)*m.zscore(TF[qi,:k])
        oo=rank_score(sc,100); run[qid]=[int(v) for v in dd[oo]]
    return m.eval_run(run,qrels)

rows=[]
for kind in ['binary_only','tfidf_only','amp_only']:
    met=evaluate(kind); rows.append({'model':kind,**met}); print(kind,met,flush=True)
# Baseline locked final score and additions of exact tfidf
met=evaluate('fusion',0,0,True,False); rows.append({'model':'current_final','amp_coef':0,'tfidf_coef':0,**met}); print('current',met,flush=True)
for x in [0.125,0.25,0.5,1.0,2.0,4.0,8.0]:
    met=evaluate('fusion',0,x,True,False); rows.append({'model':'current_plus_tfidf','amp_coef':0,'tfidf_coef':x,**met}); print('tf',x,met,flush=True)
# Replace current sign-tail by amplitude-tail
for x in [0.0,0.25,0.5,1.0,2.0,4.0]:
    met=evaluate('fusion',1.0,x,False,True); rows.append({'model':'amp_tail_plus_final','amp_coef':1.0,'tfidf_coef':x,**met}); print('amp_replace tf',x,met,flush=True)
# retain current tail and add amplitude as extra feature, plus optional exact tfidf
for a in [0.125,0.25,0.5,1.0,2.0,4.0]:
  for x in [0.0,0.25,0.5,1.0,2.0]:
    met=evaluate('fusion',a,x,True,True); rows.append({'model':'current_plus_amp_plus_tfidf','amp_coef':a,'tfidf_coef':x,**met})
rows_sorted=sorted(rows,key=lambda r:(r['nDCG@10'],r['MRR@10'],r['R@100']),reverse=True)
out={'protocol':'same fixed eta=1 P=2000 pools from deterministic 1000 TRAIN validation; raw corpus reread only to reconstruct exact amplitudes; no candidate-generation changes','rows':rows_sorted,'best':rows_sorted[0],'feature_seconds':time.time()-start}
json.dump(out,open(OUT/'amplitude_diagnostic_results.json','w'),indent=2)
print('BEST',rows_sorted[0],flush=True)
print('TOP10')
for r in rows_sorted[:10]:print(r,flush=True)
