from pathlib import Path
import numpy as np,time,json,os
IDX=Path('/mnt/data/msmarco_scale_work/full_index'); N=8_841_823; F=4; S=16
bo=np.memmap(IDX/'branch_order.u32',np.uint32,'r'); n=len(bo)
mem=np.memmap(IDX/'memberships.f32',np.float32,'r',shape=(N,F)); rt=np.memmap(IDX/'res_terms.u16',np.uint16,'r',shape=(N,F,S)); sb=np.memmap(IDX/'signbits.u16',np.uint16,'r',shape=(N,F))
pd=np.memmap(IDX/'post_doc.u32',np.uint32,'w+',shape=(n,)); pm=np.memmap(IDX/'post_membership.f32',np.float32,'w+',shape=(n,)); pr=np.memmap(IDX/'post_res_terms.u16',np.uint16,'w+',shape=(n,S)); ps=np.memmap(IDX/'post_signbits.u16',np.uint16,'w+',shape=(n,))
t=time.time(); block=500_000
for a in range(0,n,block):
 b=min(n,a+block); fp=np.asarray(bo[a:b],np.uint32); docs=fp//F; slots=(fp%F).astype(np.uint8)
 pd[a:b]=docs; pm[a:b]=mem[docs,slots]; pr[a:b]=rt[docs,slots]; ps[a:b]=sb[docs,slots]
 if (a//block)%10==0: print(a,b,'/',n,'sec',time.time()-t,flush=True)
pd.flush();pm.flush();pr.flush();ps.flush()
print('DONE',n,'seconds',time.time()-t,flush=True)
