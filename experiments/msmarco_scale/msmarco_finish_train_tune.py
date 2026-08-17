from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit,set_num_threads
ROOT=Path('/mnt/data'); WORK=ROOT/'msmarco_scale_work'; C=WORK/'finish_train_cache'; P=2000; NQ=5000
meta=np.load(C/'meta.npz',allow_pickle=False); qids=meta['qids'].astype(str); fold_id=meta['fold_id']; nvalid=meta['nvalid'].astype(np.int32); qlen=meta['qlen'].astype(np.int32)
docs=np.memmap(C/'docs.u32',np.uint32,'r',shape=(NQ,P)); tail=np.memmap(C/'tail.f32',np.float32,'r',shape=(NQ,P)); lex2=np.memmap(C/'lex2raw.f32',np.float32,'r',shape=(NQ,P)); sem=np.memmap(C/'sem.f32',np.float32,'r',shape=(NQ,P)); cnt=np.memmap(C/'cnt.u8',np.uint8,'r',shape=(NQ,P)); rm=np.memmap(C/'raremask.u8',np.uint8,'r',shape=(NQ,P))
IDX=WORK/'full_index_uniform1m'; N=8841823; dl=np.memmap(IDX/'doc_lengths.u16',np.uint16,'r',shape=(N,)); avgdl=float(json.load(open(IDX/'meta.json'))['avg_doc_length'])
# relevance padded (TRAIN is binary, max 7 positives/query)
df=pd.read_csv(ROOT/'train.tsv',sep='\t'); wanted={q:i for i,q in enumerate(qids)}; rel_lists=[[] for _ in range(NQ)]
for q,d,s in zip(df['query-id'].astype(str),df['corpus-id'],df['score']):
    i=wanted.get(q)
    if i is not None and float(s)>0: rel_lists[i].append(int(d))
maxr=max(map(len,rel_lists)); rel=np.full((NQ,maxr),np.uint32(0xffffffff),np.uint32); nr=np.zeros(NQ,np.int32)
for i,r in enumerate(rel_lists): nr[i]=len(r); rel[i,:len(r)]=r
set_num_threads(5)

@njit(cache=False)
def popcnt5(x):
    c=0
    for b in range(5): c += (x>>b)&1
    return c

@njit(cache=False)
def eval_params(docs,tail,lex2,sem,cnt,rm,nvalid,qlen,dl,avgdl,rel,nr,fold_id,b,alpha,wlex,wsem,rarek,wrare):
    # sums: ndcg,mrr,p10,r10,r100,hit10,hit100,count per fold
    sums=np.zeros((5,8),np.float64)
    for qi in range(docs.shape[0]):
        n=int(nvalid[qi]); f=int(fold_id[qi]); ql=max(1,int(qlen[qi])); nrel=max(1,int(nr[qi]))
        if n<=0:
            sums[f,7]+=1.0
            continue
        # component means
        mt=0.; ml=0.; ms=0.; mr=0.
        # scratch raw lexical and rare values
        lv=np.empty(n,np.float32); rv=np.empty(n,np.float32)
        masklim=(1<<rarek)-1 if rarek>0 else 0
        denomrare=max(1,min(rarek,ql)) if rarek>0 else 1
        for j in range(n):
            d=int(docs[qi,j]); den=(1.0-b)+b*(float(dl[d])/avgdl)
            if den<=0: den=1.0
            cov=float(cnt[qi,j])/ql
            if cov<1e-6: cov=1e-6
            l=float(lex2[qi,j])/den*(cov**alpha)
            if rarek>0:
                rr=popcnt5(int(rm[qi,j]) & masklim)/denomrare
            else: rr=0.0
            lv[j]=l; rv[j]=rr; mt+=float(tail[qi,j]); ml+=l; ms+=float(sem[qi,j]); mr+=rr
        mt/=n; ml/=n; ms/=n; mr/=n
        vt=0.; vl=0.; vs=0.; vr=0.
        for j in range(n):
            x=float(tail[qi,j])-mt; vt+=x*x
            x=float(lv[j])-ml; vl+=x*x
            x=float(sem[qi,j])-ms; vs+=x*x
            x=float(rv[j])-mr; vr+=x*x
        st=math.sqrt(vt/n)+1e-8; sl=math.sqrt(vl/n)+1e-8; ss=math.sqrt(vs/n)+1e-8; sr=math.sqrt(vr/n)+1e-8
        sc=np.empty(n,np.float32)
        for j in range(n):
            zt=(float(tail[qi,j])-mt)/st; zl=(float(lv[j])-ml)/sl; zs=(float(sem[qi,j])-ms)/ss
            zr=0.0 if rarek<=0 or sr<1e-7 else (float(rv[j])-mr)/sr
            sc[j]=zt+wlex*zl+wsem*zs+wrare*zr
        order=np.argsort(sc)[::-1]
        h10=0; h100=0; rrmetric=0.; dc=0.
        for rnk in range(min(100,n)):
            d=int(docs[qi,order[rnk]]); hit=False
            for k in range(int(nr[qi])):
                if d==int(rel[qi,k]): hit=True; break
            if hit:
                h100+=1
                if rnk<10:
                    h10+=1; dc += 1.0/math.log2(rnk+2.0)
                    if rrmetric==0.: rrmetric=1.0/(rnk+1.0)
        ideal=0.
        for rnk in range(min(10,int(nr[qi]))): ideal += 1.0/math.log2(rnk+2.0)
        ndcg=dc/ideal if ideal>0 else 0.
        sums[f,0]+=ndcg; sums[f,1]+=rrmetric; sums[f,2]+=h10/10.; sums[f,3]+=h10/nrel; sums[f,4]+=h100/nrel; sums[f,5]+=1.0 if h10>0 else 0.; sums[f,6]+=1.0 if h100>0 else 0.; sums[f,7]+=1.
    return sums

def metrics(s):
    out=[]
    for f in range(5):
        n=s[f,7]; out.append({'fold':f,'nDCG@10':float(s[f,0]/n),'MRR@10':float(s[f,1]/n),'P@10':float(s[f,2]/n),'R@10':float(s[f,3]/n),'R@100':float(s[f,4]/n),'Hit@10':float(s[f,5]/n),'Hit@100':float(s[f,6]/n)})
    return out

def run_grid(name,params,baseline_key=None):
    rows=[]; start=time.time()
    for z,p in enumerate(params):
        s=eval_params(docs,tail,lex2,sem,cnt,rm,nvalid,qlen,dl,avgdl,rel,nr,fold_id,*p['args']); fm=metrics(s); nd=np.array([x['nDCG@10'] for x in fm]); mr=np.array([x['MRR@10'] for x in fm]); r100=np.array([x['R@100'] for x in fm]); row={**p['meta'],'fold_metrics':fm,'mean_nDCG@10':float(nd.mean()),'mean_MRR@10':float(mr.mean()),'mean_R@100':float(r100.mean())}; rows.append(row); print(name,z+1,'/',len(params),p['meta'],'mean',row['mean_nDCG@10'],flush=True)
    # deltas vs designated current baseline
    if baseline_key is not None:
        base=next(r for r in rows if all(r.get(k)==v for k,v in baseline_key.items())); bnd=np.array([x['nDCG@10'] for x in base['fold_metrics']])
        for r in rows:
            d=np.array([x['nDCG@10'] for x in r['fold_metrics']])-bnd; r['mean_delta_vs_baseline']=float(d.mean()); r['min_delta_vs_baseline']=float(d.min()); r['positive_folds_vs_baseline']=int((d>0).sum()); r['fold_deltas_vs_baseline']=d.tolist()
    path=WORK/f'finish_{name}.json'; json.dump({'name':name,'rows':rows,'seconds':time.time()-start},open(path,'w'),indent=2); print('SAVED',path,flush=True)
    return rows

# JIT
_=eval_params(docs[:1],tail[:1],lex2[:1],sem[:1],cnt[:1],rm[:1],nvalid[:1],qlen[:1],dl,avgdl,rel[:1],nr[:1],fold_id[:1],.1,.25,4.,.3,3,1.)
# A: length/coordination robustness, current rest fixed.
A=[]
for bb in [0.0,0.05,0.1,0.15,0.2]:
  for aa in [0.0,0.125,0.25,0.375,0.5]: A.append({'args':(bb,aa,4.,.3,3,1.),'meta':{'b':bb,'alpha':aa}})
ra=run_grid('length_coord_multifold',A,{'b':0.1,'alpha':0.25})
# choose by all-fold robustness first; otherwise max mean.
cand=[r for r in ra if r.get('positive_folds_vs_baseline',0)==5 and r.get('min_delta_vs_baseline',-1)>0]
if cand: besta=max(cand,key=lambda r:(r['mean_nDCG@10'],r['min_delta_vs_baseline']))
else: besta=max(ra,key=lambda r:r['mean_nDCG@10'])
bb=float(besta['b']); aa=float(besta['alpha']); print('SELECT_A',bb,aa,besta['mean_nDCG@10'],flush=True)
# B: final weights around current.
B=[]
for wl in [3.,4.,5.,6.]:
  for ws in [0.,0.1,0.2,0.3,0.4,0.5]: B.append({'args':(bb,aa,wl,ws,3,1.),'meta':{'wlex':wl,'wsem':ws}})
rb=run_grid('final_weights_multifold',B,{'wlex':4.0,'wsem':0.3})
cand=[r for r in rb if r.get('positive_folds_vs_baseline',0)==5 and r.get('min_delta_vs_baseline',-1)>0]
if cand: bestb=max(cand,key=lambda r:(r['mean_nDCG@10'],r['min_delta_vs_baseline']))
else: bestb=max(rb,key=lambda r:r['mean_nDCG@10'])
wl=float(bestb['wlex']); ws=float(bestb['wsem']); print('SELECT_B',wl,ws,bestb['mean_nDCG@10'],flush=True)
# C: rare conjunction depth/weight, include current.
C=[]
for k in [2,3,4,5]:
  for wr in [0.5,0.75,1.0,1.25,1.5]: C.append({'args':(bb,aa,wl,ws,k,wr),'meta':{'rarek':k,'wrare':wr}})
rc=run_grid('rare_final_multifold',C,{'rarek':3,'wrare':1.0})
cand=[r for r in rc if r.get('positive_folds_vs_baseline',0)==5 and r.get('min_delta_vs_baseline',-1)>0]
if cand: bestc=max(cand,key=lambda r:(r['mean_nDCG@10'],r['min_delta_vs_baseline']))
else: bestc=max(rc,key=lambda r:r['mean_nDCG@10'])
k=int(bestc['rarek']); wr=float(bestc['wrare']); print('SELECT_C',k,wr,bestc['mean_nDCG@10'],flush=True)
final={'protocol':'finish remaining TRAIN-only development on five fixed disjoint 1000-query folds; no DEV/test selection','selected':{'final_length_b':bb,'coordination_alpha':aa,'lambda_lex':wl,'lambda_sem':ws,'rare_topk':k,'rare_weight':wr,'final_idf_power':2.0,'preselection_eta':1.0,'preselection_idf_power':1.0,'P':2000,'gamma_tail':0.25,'lambda_M':0.125,'h':0,'S':16},'stageA_selected':besta,'stageB_selected':bestb,'stageC_selected':bestc}
json.dump(final,open(WORK/'finish_train_selected.json','w'),indent=2); print('FINAL_SELECTED',json.dumps(final['selected'],indent=2),flush=True)
