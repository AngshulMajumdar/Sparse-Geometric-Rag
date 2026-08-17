from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp,time,os
import numpy as np
W=Path('/mnt/data/msmarco_scale_work'); OLD=W/'full_index'; SRC=W/'full_index_uniform1m_s32'; NEW=W/'full_index_uniform1m_s32_post'
NEW.mkdir(exist_ok=True)
N=8_841_823; F=4; S=32
npost=int(np.load(OLD/'branch_offsets.npy',mmap_mode='r')[-1])

def job(a,b):
    t=time.time(); bo=np.memmap(OLD/'branch_order.u32',np.uint32,'r',shape=(npost,)); rt=np.memmap(SRC/'res_terms.u16',np.uint16,'r',shape=(N,F,S)); sb=np.memmap(SRC/'signbits.u32',np.uint32,'r',shape=(N,F)); pr=np.memmap(NEW/'post_res_terms.u16',np.uint16,'r+',shape=(npost,S)); ps=np.memmap(NEW/'post_signbits.u32',np.uint32,'r+',shape=(npost,))
    block=250_000
    for x in range(a,b,block):
        y=min(b,x+block); fp=np.asarray(bo[x:y],np.uint32); docs=fp//F; slots=(fp%F).astype(np.uint8); pr[x:y]=rt[docs,slots]; ps[x:y]=sb[docs,slots]
    pr.flush(); ps.flush(); return a,b,time.time()-t

if __name__=='__main__':
    if not (NEW/'post_res_terms.u16').exists(): np.memmap(NEW/'post_res_terms.u16',np.uint16,'w+',shape=(npost,S)).flush()
    if not (NEW/'post_signbits.u32').exists(): np.memmap(NEW/'post_signbits.u32',np.uint32,'w+',shape=(npost,)).flush()
    for name in ['branch_offsets.npy','post_doc.u32','post_membership.f32','doc_lengths.u16','support_all_indptr.u32','support_all.u16','meta.json']:
        dst=NEW/name
        if not dst.exists(): dst.symlink_to(OLD/name)
    bounds=np.linspace(0,npost,13,dtype=np.int64); chunks=[(int(bounds[i]),int(bounds[i+1])) for i in range(12)]
    t=time.time()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as ex:
        fs=[ex.submit(job,a,b) for a,b in chunks]
        for k,fu in enumerate(as_completed(fs),1):
            a,b,sec=fu.result(); print(f'[{k:02d}/12] {a:,}:{b:,} sec={sec:.1f}',flush=True)
    print('S32 REORDER DONE npost',npost,'sec',time.time()-t,flush=True)
