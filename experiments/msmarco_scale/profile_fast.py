import sys,cProfile,pstats,io
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_fast import FullIndex,load_query_texts
q='9083'; txt=load_query_texts([q])[q]; idx=FullIndex(); p=idx.prepare(txt,20); idx.rank_h(p,0,100)
pr=cProfile.Profile(); pr.enable(); p=idx.prepare(txt,20); idx.rank_h(p,0,100); pr.disable(); s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumtime').print_stats(30); print(s.getvalue())
