from pathlib import Path
import numpy as np, json, time, os
I=Path('/mnt/data/msmarco_scale_work/full_index'); N=8841823
total=0; metas=[]
for sid in range(36):
 m=json.load(open(I/f'shard_{sid:04d}.json')); metas.append(m); total+=int(m['nnz'])
print('total',total,flush=True)
ids=np.memmap(I/'support_all.u16',np.uint16,'w+',shape=(total,))
ip=np.memmap(I/'support_all_indptr.u32',np.uint32,'w+',shape=(N+1,))
pos=0; dpos=0; t=time.time(); ip[0]=0
for sid,m in enumerate(metas):
 n=int(m['n']); nn=int(m['nnz'])
 si=np.memmap(I/f'support_{sid:04d}.u16',np.uint16,'r',shape=(nn,)); sp=np.memmap(I/f'support_indptr_{sid:04d}.u32',np.uint32,'r',shape=(n+1,))
 ids[pos:pos+nn]=si
 ip[dpos+1:dpos+n+1]=np.asarray(sp[1:],np.uint64)+pos
 pos+=nn; dpos+=n
 print(sid,n,nn,pos,dpos,time.time()-t,flush=True)
ids.flush();ip.flush(); print('done',pos,dpos,time.time()-t,flush=True)
