import numpy as np
from geomretrieval.rag_top10 import _minmax_hi, _zscore

def test_minmax_hi_is_bounded_and_monotone():
    x=np.array([2.0,5.0,11.0],dtype=np.float32)
    y=_minmax_hi(x)
    assert np.allclose(y,[0.0,1/3,1.0])

def test_zscore_constant_is_zero():
    assert np.allclose(_zscore(np.ones(4,dtype=np.float32)),0)

from geomretrieval import FrozenConfig, GeometricIndex, RAGTop10Config, RAGTop10Ranker

def test_rag_top10_ranker_runs_on_toy_index():
    docs=[
        'car automobile engine road vehicle',
        'automobile vehicle insurance motor road',
        'river bank flood erosion water',
        'bank account credit loan interest',
        'vitamin respiratory infection clinical study',
    ]
    ids=[f'd{i}' for i in range(len(docs))]
    idx=GeometricIndex.build(docs,ids,FrozenConfig(max_features=100,F=2,B=8,S=4,L=4,assoc_k=6,route_k=4,route_budget=6,rerank_pool=5,semantic_k=3,output_k=5),verbose=False)
    ranker=RAGTop10Ranker(idx,RAGTop10Config(pool_size=5,semantic_k=3,hq_top_branches=3,branch_quality_top_docs=2))
    out=ranker.search('automobile road insurance',k=3)
    assert len(out)>=1
    assert out[0] in {'d0','d1'}
