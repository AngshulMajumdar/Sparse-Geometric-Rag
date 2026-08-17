from pathlib import Path
import json,time,gc,os
import numpy as np
from scipy import sparse
import sys
sys.path.insert(0,'/mnt/data')
import msmarco_build_geometry as base
W=Path('/mnt/data/msmarco_scale_work'); SRC=W/'geometry_uniform1m'; G=W/'geometry_uniform1m_s32'; G.mkdir(exist_ok=True)
N=1_000_000; M=50_000; F=4; S=32; TAU=20.; BETA=-.2; EPS=1e-6; SENT=np.uint16(65535)
# geometry-independent files are symlinked
for name in ['idf.npy','terms.pkl.gz','center_terms.npy','center_values.npy','assoc_ppmi.npz','context_similarity.npz','sample_ids.npy','cal_X.npz','cal_branches.npy','cal_memberships.npy','cal_topL.npy']:
 p=G/name
 if not p.exists(): p.symlink_to(SRC/name)
X=sparse.load_npz(SRC/'cal_X.npz').tocsr(); branches=np.load(SRC/'cal_branches.npy'); ct=np.load(SRC/'center_terms.npy',mmap_mode='r'); cv=np.load(SRC/'center_values.npy',mmap_mode='r')
t=time.time(); rt,rs=base.residual_codes(X.indptr.astype(np.int64),X.indices.astype(np.int32),X.data.astype(np.float32),branches,ct,cv,S); print('s32 residual sec',time.time()-t,flush=True); np.save(G/'cal_res_terms.npy',rt); np.save(G/'cal_res_signs.npy',rs)
# reliability exact same estimator with S32 observations
total_memberships=int(np.sum(branches!=SENT)); gcnt=np.zeros(M,np.float64); gsum=np.zeros(M,np.float64)
for d0 in range(0,N,50_000):
 tt=rt[d0:d0+50_000].ravel(); zz=rs[d0:d0+50_000].ravel().astype(np.float64); ok=tt!=SENT
 gcnt += np.bincount(tt[ok].astype(np.int64),minlength=M); gsum += np.bincount(tt[ok].astype(np.int64),weights=zz[ok],minlength=M)
ge2=gcnt/max(1,total_memberships); ge1=gsum/max(1,total_memberships); gvar=np.maximum(ge2-ge1*ge1,0.)
flat=branches.ravel(); valid=np.flatnonzero(flat!=SENT); order=valid[np.argsort(flat[valid],kind='stable')]; sorted_br=flat[order].astype(np.int64); counts=np.bincount(sorted_br,minlength=M); offs=np.zeros(M+1,np.int64); np.cumsum(counts,out=offs[1:]); f_rt=rt.reshape(N*F,S); f_rs=rs.reshape(N*F,S)
max_rel=N*F*S; rel_i=np.memmap(G/'rel_indices.u16',np.uint16,'w+',shape=(max_rel,)); rel_v=np.memmap(G/'rel_data.f32',np.float32,'w+',shape=(max_rel,)); rel_p=np.zeros(M+1,np.uint64); pos=0; t=time.time()
for j in range(M):
 a,b=offs[j],offs[j+1]; mpos=order[a:b]; nj=len(mpos)
 if nj:
  tj=f_rt[mpos].ravel(); sj=f_rs[mpos].ravel().astype(np.float64); ok=tj!=SENT
  if np.any(ok):
   u,inv=np.unique(tj[ok],return_inverse=True); cnt=np.bincount(inv).astype(np.float64); sm=np.bincount(inv,weights=sj[ok]).astype(np.float64); e2=cnt/nj; e1=sm/nj; lv=np.maximum(e2-e1*e1,0.); shr=(cnt/(cnt+TAU))*lv+(TAU/(cnt+TAU))*gvar[u.astype(np.int64)]; ww=np.power(shr+EPS,BETA)
   if len(ww) and np.isfinite(ww).all() and ww.mean()>0: ww=ww/ww.mean()
   n=len(u); rel_i[pos:pos+n]=u.astype(np.uint16); rel_v[pos:pos+n]=ww.astype(np.float32); pos+=n
 rel_p[j+1]=pos
 if j and j%5000==0: print('rel',j,pos,flush=True)
rel_i.flush(); rel_v.flush(); np.save(G/'rel_indptr.npy',rel_p); np.save(G/'global_sign_var.npy',gvar.astype(np.float32)); json.dump({'nnz':int(pos),'max_entries':int(max_rel)},open(G/'rel_meta.json','w'))
meta=json.load(open(SRC/'meta.json')); meta['S']=32; meta['capacity_test']='same uniform1m centers and graph; only residual width/reliability changed 16->32'; json.dump(meta,open(G/'meta.json','w'),indent=2)
print('S32 REL DONE nnz',pos,'sec',time.time()-t,flush=True)
