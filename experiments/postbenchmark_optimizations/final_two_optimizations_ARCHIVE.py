from __future__ import annotations
import sys,time,json,math,argparse,gc
from pathlib import Path
import numpy as np
from numba import njit,set_num_threads
set_num_threads(5)
sys.path.insert(0,'/mnt/data/exp')
import eval_scale_rag as e
from opt_two_variants import FastScaleIndex, agg_stamp

@njit(cache=True)
def compact_pool_evidence_and_code(docs,rslot,mem,rt,sbits,ev,spanbr,rho,poolmap,poolstamp,token,nmax,qct,qcw,qcbits):
    cap=nmax*e.F
    mp=np.empty(cap,np.int32);mb=np.empty(cap,np.int32);me=np.empty(cap,np.float32);cm=np.zeros(nmax,np.float32);n=0
    for z in range(len(docs)):
        d=np.int64(docs[z])
        if poolstamp[d]!=token: continue
        pi=poolmap[d]
        if pi<0 or pi>=nmax: continue
        u=np.int64(rslot[z]); b=spanbr[u]
        mp[n]=pi;mb[n]=b;me[n]=ev[z];n+=1
        bits=np.uint16(sbits[z]);s=0.;den=0.
        for a in range(e.S):
            qt=np.int64(qct[u,a])
            if qt==65535: continue
            w=qcw[u,a];den+=w
            for r in range(e.S):
                if np.int64(rt[z,r])==qt:
                    qs=1 if ((qcbits[u]>>a)&1)!=0 else -1
                    ds=1 if ((bits>>r)&1)!=0 else -1
                    s += w*(1. if qs==ds else -1.)
                    break
        if den>0.: cm[pi]+=mem[z]*rho[u]*(s/den)
    return mp[:n],mb[:n],me[:n],cm

def qcodes_sparse(idx,spans,q,qd,rel):
    nr=len(spans);qt=np.full((nr,e.S),65535,np.uint16);qw=np.zeros((nr,e.S),np.float32);qb=np.zeros(nr,np.uint16)
    qterms=np.asarray(q.indices,np.int32)
    for u,(j,_,_) in enumerate(spans):
        rowt=np.asarray(idx.ct[j]);ok=rowt!=e.SENT;cids=rowt[ok].astype(np.int32,copy=False)
        cvals=np.asarray(idx.cv[j],np.float32)[ok]
        cand=np.unique(np.concatenate([cids,qterms]))
        # center lookup only for <=80 coordinates
        cvmap={int(t):float(v) for t,v in zip(cids,cvals)}
        diff=np.asarray([float(qd[t])-cvmap.get(int(t),0.0) for t in cand],np.float32);aa=np.abs(diff)
        if len(cand)>e.S:
            ii=np.argpartition(aa,-e.S)[-e.S:];ii=ii[np.argsort(aa[ii])[::-1]]
        else:ii=np.argsort(aa)[::-1]
        for a,k in enumerate(ii[:e.S]):
            t=int(cand[k]);qt[u,a]=t;rr=float(rel[u,t]) if rel[u,t]!=0 else 1.0;qw[u,a]=float(aa[k])*rr
            if diff[k]>=0: qb[u]|=np.uint16(1<<a)
    return qt,qw,qb

class FinalOptimizedIndex(FastScaleIndex):
    def search_optimized(self,text,P,gate=None,wcode=.25):
        t0=time.perf_counter();self.token+=1;token=self.token
        q=self.qvec(text)
        if q.nnz==0:return [],{'total_ms':(time.perf_counter()-t0)*1000,'route_docs':0,'gate_docs':0}
        qd=np.zeros(e.M,np.float32);qd[q.indices]=q.data;rterms,rd=self.route(q)
        spans=[(int(j),int(self.offs[j]),int(self.offs[j+1])) for j in rterms if self.offs[j+1]>self.offs[j]]
        if not spans:return [],{'total_ms':(time.perf_counter()-t0)*1000,'route_docs':0,'gate_docs':0}
        docs=np.concatenate([np.asarray(self.pd[a:b]) for j,a,b in spans]).astype(np.uint32,copy=False)
        mem=np.concatenate([np.asarray(self.pm[a:b]) for j,a,b in spans]).astype(np.float32,copy=False)
        rt=np.concatenate([np.asarray(self.pr[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
        sb=np.concatenate([np.asarray(self.ps[a:b]) for j,a,b in spans]).astype(np.uint16,copy=False)
        nr=len(spans);cent=np.zeros((nr,e.M),np.float32);rel=np.zeros((nr,e.M),np.float32);rho=np.empty(nr,np.float32);spanbr=np.empty(nr,np.int32);rs=[]
        for u,(j,a,b) in enumerate(spans):
            spanbr[u]=j;rowt=np.asarray(self.ct[j]);ok=rowt!=e.SENT;ids=rowt[ok].astype(np.int32,copy=False);cent[u,ids]=np.asarray(self.cv[j])[ok]
            ra=int(self.rp[j]);rb=int(self.rp[j+1]);rel[u,np.asarray(self.ri[ra:rb],np.int32)]=np.asarray(self.rv[ra:rb]);rho[u]=rd[j];rs.append(np.full(b-a,u,dtype=np.uint8))
        rslot=np.concatenate(rs);ev,cons=e.score_memberships_local(rslot,mem,rt,sb,qd,rho,cent,rel)
        # Optimization 1a: O(K) stamp aggregation instead of np.unique sort.
        n=agg_stamp(docs,ev,cons,self.stamp,self.taila,self.consa,self.touched,token)
        ud=np.asarray(self.touched[:n],np.uint32).copy();tail=(self.taila[ud]+e.LAM_M*self.consa[ud]).astype(np.float32)
        # Optimization 1b: small tail gate before whole-chunk lexical scan.
        if gate is None: gate=min(len(ud),max(10000,40*int(P)))
        if len(ud)>gate:
            gi=e.topk_sorted(tail,gate);cand=ud[gi];ct=tail[gi]
        else:cand=ud;ct=tail
        qlex=np.zeros(e.M,np.float32);qlex[q.indices]=self.idf[q.indices];lex1=e.score_pre(cand,self.ip,self.ids,qlex,self.dl,self.avgdl)
        pre=e.zscore(ct)+e.zscore(lex1);sel=e.topk_sorted(pre,P);pooldocs=cand[sel];pooltail=ct[sel]
        # exact pool map with stamps
        self.poolstamp[pooldocs]=token;self.poolmap[pooldocs]=np.arange(len(pooldocs),dtype=np.int32)
        # Optimization 2: query-conditioned full 16-coordinate residual-code match.
        qct,qcw,qcb=qcodes_sparse(self,spans,q,qd,rel)
        mp,mb,me,cm=compact_pool_evidence_and_code(docs,rslot,mem,rt,sb,ev,spanbr,rho,self.poolmap,self.poolstamp,token,len(pooldocs),qct,qcw,qcb)
        semv=np.zeros(e.M,np.float32)
        for t,amp in zip(q.indices,q.data):
            a,b=self.A.indptr[t],self.A.indptr[t+1];nb=self.A.indices[a:b][:e.SEMK];sv=self.A.data[a:b][:e.SEMK]
            if len(nb):semv[nb]+=float(amp)*sv*self.idf[nb]
        lex2v=np.zeros(e.M,np.float32);lex2v[q.indices]=self.idf[q.indices]**2;qmask=np.zeros(e.M,np.float32);qmask[q.indices]=1.;rarem=np.zeros(e.M,np.float32);rareidx=q.indices[np.argsort(self.idf[q.indices])[::-1]][:e.RAREK];rarem[rareidx]=1.
        p={'q':q,'pooldocs':pooldocs,'pooltail':pooltail,'mem_pool':mp,'mem_branch':mb,'mem_ev':me,'semv':semv,'lex2v':lex2v,'qmask':qmask,'rarem':rarem,'ressem_signed':cm,'route_docs':ud}
        r,rtm=self.rank_variant(p,P,wres=wcode,absres=False)
        return r,{'total_ms':(time.perf_counter()-t0)*1000,'rank_ms':rtm['rank_ms'],'route_docs':len(ud),'gate_docs':len(cand),'pool_size':len(pooldocs)}

def full_run(dataset,work,qpath,rpath,P,outfile,gate=None,wcode=.25,start=0,limit=None):
    gc.disable()
    idx=FinalOptimizedIndex(work,dataset);qr=e.load_qrels(rpath);qids=sorted(qr);qs=e.load_queries(qpath,qids);sub=qids[start:] if limit is None else qids[start:start+limit]
    # compile/warm
    if sub:idx.search_optimized(qs[sub[0]],P,gate,wcode)
    run={};times=[];routes=[];gates=[]
    for z,qid in enumerate(sub):
        r,t=idx.search_optimized(qs[qid],P,gate,wcode);run[qid]=r;times.append(t['total_ms']);routes.append(t['route_docs']);gates.append(t['gate_docs'])
        if (z+1)%250==0:
            print(dataset,'done',z+1,'med_ms',float(np.median(times)),'route_med',float(np.median(routes)),flush=True)
            Path(outfile+'.checkpoint').write_text(json.dumps({'done':z+1,'run':run,'times':times,'routes':routes,'gates':gates}))
    qrs={q:qr[q] for q in sub};m=e.metrics_for(run,qrs);a=np.asarray(times)
    m.update({'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'mean_ms':float(np.mean(a)),'qps':float(1000/np.mean(a)),'median_route_docs':float(np.median(routes)),'median_gate_docs':float(np.median(gates)),'P':P,'wcode':wcode,'gate_rule':'min(route,max(10000,40P))' if gate is None else gate})
    out={'dataset':dataset,'metrics':m,'run':run};Path(outfile).write_text(json.dumps(out,indent=2));print(json.dumps(m,indent=2));return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['nq','hotpot']);ap.add_argument('--limit',type=int,default=None);ap.add_argument('--out',required=True);a=ap.parse_args()
    if a.dataset=='nq':full_run('nq','/mnt/data/exp/nq_work','/mnt/data/queries(3).jsonl','/mnt/data/test(3).tsv',100,a.out,limit=a.limit)
    else:full_run('hotpot','/mnt/data/exp/hotpot_work','/mnt/data/queries(2).jsonl','/mnt/data/test(2).tsv',500,a.out,limit=a.limit)
