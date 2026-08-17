from __future__ import annotations
import argparse, json
from pathlib import Path

from geomretrieval import GeometricIndex, RAGTop10Config, RAGTop10Ranker, load_beir_zip, load_beir_directory


def load_dataset(path, split):
    return load_beir_zip(path, split) if str(path).lower().endswith('.zip') else load_beir_directory(path, split)


def main():
    p=argparse.ArgumentParser(description='Evaluate the current top-10 RAG protocol on a BEIR dataset.')
    p.add_argument('dataset', help='BEIR zip or extracted dataset directory')
    p.add_argument('index', help='saved GeometricIndex directory')
    p.add_argument('--split', default='test')
    p.add_argument('--pool', type=int, default=100)
    p.add_argument('--hq-branches', type=int, default=10)
    p.add_argument('--lambda-diversity', type=float, default=0.1)
    p.add_argument('--output', default=None)
    a=p.parse_args()
    ds=load_dataset(a.dataset,a.split)
    idx=GeometricIndex.load(a.index)
    cfg=RAGTop10Config(pool_size=a.pool,hq_top_branches=a.hq_branches,lambda_diversity=a.lambda_diversity)
    ranker=RAGTop10Ranker(idx,cfg)
    metrics,run=ranker.evaluate(ds,k=10)
    result={'dataset':ds.name,'split':a.split,'config':cfg.__dict__,'metrics':metrics}
    print(json.dumps(result,indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps({'summary':result,'run':run},indent=2))

if __name__=='__main__': main()
