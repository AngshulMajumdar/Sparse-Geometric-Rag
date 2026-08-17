from __future__ import annotations
import argparse,json
from pathlib import Path
from geomretrieval import GeometricIndex,RAGTop10Config,RAGTop10Ranker,load_beir_zip,load_beir_directory

def load_dataset(path,split):
    return load_beir_zip(path,split) if str(path).lower().endswith('.zip') else load_beir_directory(path,split)

def main():
    p=argparse.ArgumentParser(description='Top-10 RAG shortlist sweep. Deep-recall metrics are intentionally not used for model selection.')
    p.add_argument('dataset'); p.add_argument('index'); p.add_argument('--split',default='test')
    p.add_argument('--pools',type=int,nargs='+',default=[25,50,100,200,500])
    p.add_argument('--output',default='pool_sweep.json')
    a=p.parse_args(); ds=load_dataset(a.dataset,a.split); idx=GeometricIndex.load(a.index)
    out={'dataset':ds.name,'split':a.split,'protocol':'top-10 RAG only','pools':{}}
    for P in a.pools:
        ranker=RAGTop10Ranker(idx,RAGTop10Config(pool_size=P))
        metrics,_=ranker.evaluate(ds,k=10)
        out['pools'][str(P)]=metrics
        print('P=',P,json.dumps(metrics,sort_keys=True))
    Path(a.output).write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
