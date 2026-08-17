import sys,time,json, numpy as np
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_post import FullIndex, load_query_texts
ids=['300674','125705','94798','9083','174249']
txt=load_query_texts(ids)
idx=FullIndex(); print('loaded')
# compile/warm
for rep in range(2):
 for q in ids:
  t=time.perf_counter(); p=idx.prepare(txt[q],hmax=20); r=idx.rank_h(p,0,100); dt=(time.perf_counter()-t)*1000
  print(rep,q,dt,p['candidate_memberships'],p['candidate_docs'],r[:5],flush=True)
