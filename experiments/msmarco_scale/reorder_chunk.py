import sys,time
from pathlib import Path
import numpy as np
I=Path('/mnt/data/msmarco_scale_work/full_index'); N=8841823; F=4; S=16
a=int(sys.argv[1]); b=int(sys.argv[2]); npost=35365827
bo=np.memmap(I/'branch_order.u32',np.uint32,'r',shape=(npost,)); mem=np.memmap(I/'memberships.f32',np.float32,'r',shape=(N,F)); rt=np.memmap(I/'res_terms.u16',np.uint16,'r',shape=(N,F,S)); sb=np.memmap(I/'signbits.u16',np.uint16,'r',shape=(N,F))
pd=np.memmap(I/'post_doc.u32',np.uint32,'r+',shape=(npost,)); pm=np.memmap(I/'post_membership.f32',np.float32,'r+',shape=(npost,)); pr=np.memmap(I/'post_res_terms.u16',np.uint16,'r+',shape=(npost,S)); ps=np.memmap(I/'post_signbits.u16',np.uint16,'r+',shape=(npost,))
t=time.time(); fp=np.asarray(bo[a:b],np.uint32); docs=fp//F; slots=(fp%F).astype(np.uint8); pd[a:b]=docs; pm[a:b]=mem[docs,slots]; pr[a:b]=rt[docs,slots]; ps[a:b]=sb[docs,slots]; pd.flush();pm.flush();pr.flush();ps.flush(); print(a,b,time.time()-t,flush=True)
