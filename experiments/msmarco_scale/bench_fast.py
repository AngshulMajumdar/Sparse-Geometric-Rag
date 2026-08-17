import sys,time
sys.path.insert(0,'/mnt/data')
from msmarco_full_search_fast import FullIndex,load_query_texts
ids=['300674','125705','94798','9083','174249']; txt=load_query_texts(ids); idx=FullIndex(); print('loaded')
for rep in range(2):
 for q in ids:
  t=time.perf_counter(); p=idx.prepare(txt[q],20); r=idx.rank_h(p,0,100); print(rep,q,(time.perf_counter()-t)*1000,p['candidate_memberships'],p['candidate_docs'],r[:5],flush=True)
