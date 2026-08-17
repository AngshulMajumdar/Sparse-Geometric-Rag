from __future__ import annotations
import json, sys, math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

# Load function definitions only, avoiding the experiment main block.
path='/mnt/data/treccovid_hq_branch_compact.py'
src=open(path,'r',encoding='utf-8').read()
prefix=src.split("ds=load_beir_directory(ROOT,'test')",1)[0]
ns={}
exec(compile(prefix,path,'exec'),ns)
idx=ns['idx']; ROOT=ns['ROOT']; prepare=ns['prepare']; rank_hq_softdoc=ns['rank_hq_softdoc']; load_beir_directory=ns['load_beir_directory']

ds=load_beir_directory(ROOT,'test')

def minmax_cols(X):
    X=np.asarray(X,np.float64)
    mn=X.min(axis=0); mx=X.max(axis=0); den=mx-mn
    den=np.where(den<1e-12,1.0,den)
    return (X-mn)/den

def doc_support_mask(d, qids, qpos):
    a,b=idx.support_indptr[d],idx.support_indptr[d+1]
    sup=idx.support_indices[a:b]
    mask=0
    # qpos dict is tiny
    for t in sup:
        p=qpos.get(int(t))
        if p is not None: mask |= (1<<p)
    return int(mask)

def dom_slot(d, b):
    slots=np.where(idx.branches[int(d)]==int(b))[0]
    return int(slots[0]) if len(slots) else -1

def residual_keys(d, b, qpos):
    sl=dom_slot(d,b)
    if sl<0: return (),()
    terms=idx.res_terms[int(d),sl]
    signs=idx.res_signs[int(d),sl]
    full=[]; proj=[]
    for t,s in zip(terms,signs):
        ti=int(t)
        if ti<0: continue
        si=int(s)
        full.append((ti,si))
        p=qpos.get(ti)
        if p is not None: proj.append((int(p),si))
    return tuple(full),tuple(proj)

def mixed_class_stats(keys, labels):
    groups=defaultdict(lambda:[0,0])
    for k,y in zip(keys,labels): groups[k][int(y)]+=1
    mixed={k:v for k,v in groups.items() if v[0]>0 and v[1]>0}
    n=len(keys); nrel=int(np.sum(labels)); nnon=n-nrel
    rel_mixed=sum(v[1] for v in mixed.values()); non_mixed=sum(v[0] for v in mixed.values())
    docs_mixed=rel_mixed+non_mixed
    return {
        'n_docs':n,'n_classes':len(groups),'n_mixed_classes':len(mixed),
        'docs_in_mixed_classes':docs_mixed,
        'doc_mixed_fraction':docs_mixed/max(1,n),
        'relevant_docs':nrel,'relevant_in_mixed_classes':rel_mixed,
        'relevant_mixed_fraction':rel_mixed/max(1,nrel),
        'nonrelevant_in_mixed_classes':non_mixed,
        'largest_class':max((sum(v) for v in groups.values()), default=0),
        'largest_mixed_class':max((sum(v) for v in mixed.values()), default=0),
    }

def near_stats(X,y):
    X=np.asarray(X,np.float64); y=np.asarray(y,np.int8)
    rel=np.where(y==1)[0]; non=np.where(y==0)[0]
    if len(rel)==0 or len(non)==0:
        return {'n_rel':len(rel),'n_nonrel':len(non)}
    tree=cKDTree(X[non])
    dist,_=tree.query(X[rel],k=1)
    out={'n_rel':int(len(rel)),'n_nonrel':int(len(non)),
         'nearest_nonrel_median':float(np.median(dist)),
         'nearest_nonrel_p25':float(np.percentile(dist,25)),
         'nearest_nonrel_p75':float(np.percentile(dist,75)),
         'nearest_nonrel_p90':float(np.percentile(dist,90))}
    for th in [0.005,0.01,0.02,0.05,0.10,0.20]:
        out[f'frac_rel_nn_le_{th:g}']=float(np.mean(dist<=th))
    return out

def quantized_mixed(X,y,bins):
    # X already in [0,1]; map each dimension into 0..bins-1
    Q=np.minimum(bins-1,np.floor(np.asarray(X)*bins).astype(np.int16))
    keys=[tuple(row.tolist()) for row in Q]
    return mixed_class_stats(keys,y)

perq={}
agg_light=[]; agg_full=[]
# aggregate counters manually by concatenating keys with qid prefix to avoid cross-query collisions
all_light_keys=[]; all_full_keys=[]; all_labels=[]
all_X=[]; all_y=[]
all_pool_X=[]; all_pool_y=[]
q_oracle=[]
actual_top10_rel=[]
base_top10_rel=[]
rel_pool_total=rel_hq_total=0

for qi,qid in enumerate(ds.qrels):
    p=prepare(ds.queries[qid])
    if p is None: continue
    q=idx._query_vector(ds.queries[qid])
    qids=list(map(int,q.indices)); qpos={t:i for i,t in enumerate(qids)}
    rare_ids=set(map(int,q.indices[np.argsort(idx.idf[q.indices])[::-1]][:3]))
    # labels for 2k
    pos={str(d) for d,v in ds.qrels[qid].items() if v>0}
    labels=np.asarray([1 if str(idx.doc_ids[int(d)]) in pos else 0 for d in p['docs']],dtype=np.int8)
    # high-quality top10 branches
    elig_order=np.argsort(p['H'])[::-1][:min(10,len(p['H']))]
    hq_positions=set()
    for bi in elig_order:
        b=int(p['ub'][int(bi)])
        hq_positions.update(map(int,p['docs_by_branch'][b]))
    hq=np.asarray(sorted(hq_positions),dtype=np.int32)
    if len(hq)==0: continue
    yh=labels[hq]
    # feature vectors: tail, lex, sem, rare. normalize over full 2k, then subset HQ
    V=np.column_stack([p['tail'],p['lex'],p['sem'],p['rare']]).astype(np.float64)
    Vn=minmax_cols(V)
    Xh=Vn[hq]
    # symbolic keys
    light=[]; full=[]
    for pi in hq:
        d=int(p['docs'][pi]); b=int(p['branches_dom'][pi])
        smask=doc_support_mask(d,qids,qpos)
        rmask=0
        for t in rare_ids:
            qp=qpos.get(t)
            if qp is not None and (smask & (1<<qp)): rmask |= (1<<qp)
        fkey,pkey=residual_keys(d,b,qpos)
        light.append((smask,rmask,b,pkey))
        full.append((smask,rmask,b,fkey))
    ls=mixed_class_stats(light,yh); fs=mixed_class_stats(full,yh); ns4=near_stats(Xh,yh)
    qmix={str(b):quantized_mixed(Xh,yh,b) for b in [10,20,50,100]}
    # ceilings
    nrel_pool=int(labels.sum()); nrel_hq=int(yh.sum())
    rel_pool_total += nrel_pool; rel_hq_total += nrel_hq
    oracle_top10=min(10,nrel_hq)
    q_oracle.append(oracle_top10)
    # current HQ-div rank and plain rank rel count
    rr=rank_hq_softdoc(p,topN=10,lam=.1,k=10)
    rr_ext={str(idx.doc_ids[int(d)]) for d in rr}
    ar=sum(d in pos for d in rr_ext); actual_top10_rel.append(ar)
    bo=np.argsort(p['base'])[::-1][:10]
    br=sum(labels[bo]); base_top10_rel.append(int(br))
    perq[str(qid)]={'pool_rel':nrel_pool,'hq_docs':int(len(hq)),'hq_rel':nrel_hq,
                    'actual_hqdiv_rel10':int(ar),'plain_rel10':int(br),'oracle_rel10_hq':int(oracle_top10),
                    'light_collision':ls,'full_collision':fs,'near4d':ns4,'quantized4d':qmix}
    # aggregate with qid prefix
    all_light_keys.extend([(str(qid),)+tuple(k) for k in light])
    all_full_keys.extend([(str(qid),)+tuple(k) for k in full])
    all_labels.extend(yh.tolist())
    all_X.append(Xh); all_y.append(yh)
    all_pool_X.append(Vn); all_pool_y.append(labels)

Y=np.asarray(all_labels,np.int8); X=np.vstack(all_X); PY=np.concatenate(all_pool_y); PX=np.vstack(all_pool_X)
result={
    'dataset':'TREC-COVID',
    'diagnostic':'2k -> 10 discrimination inside top-10 high-quality branches',
    'definitions':{
        'hq_branches':'top 10 branches by H_j = mean top-3 branch-specific E_dj',
        'light_collision':'same whole-query binary support mask + same rare-term mask + same dominant branch + same query-projected residual sign signature',
        'full_collision':'same support/rare masks + same dominant branch + identical full 16-coordinate residual (term,sign) code',
        'near4d':'Euclidean distance after per-query min-max normalization of [tail, IDF^2 lexical+coordination, semantic support, rare3 coverage]',
    },
    'aggregate':{
        'queries':len(perq),'pool_relevant_total':int(rel_pool_total),'hq_relevant_total':int(rel_hq_total),
        'hq_share_of_pool_relevant':float(rel_hq_total/max(1,rel_pool_total)),
        'mean_plain_relevant_at10':float(np.mean(base_top10_rel)),
        'mean_hqdiv_relevant_at10':float(np.mean(actual_top10_rel)),
        'mean_oracle_relevant_at10_if_perfect_inside_hq':float(np.mean(q_oracle)),
        'light_collision':mixed_class_stats(all_light_keys,Y),
        'full_collision':mixed_class_stats(all_full_keys,Y),
        'near4d_hq':near_stats(X,Y),
        'near4d_full_pool':near_stats(PX,PY),
        'quantized4d_hq':{str(b):quantized_mixed(X,Y,b) for b in [10,20,50,100]},
        'quantized4d_full_pool':{str(b):quantized_mixed(PX,PY,b) for b in [10,20,50,100]},
    },
    'per_query':perq,
}
out='/mnt/data/treccovid_collision_diagnostic.json'
json.dump(result,open(out,'w'),indent=2)
print(json.dumps(result['aggregate'],indent=2))
print('saved',out)
