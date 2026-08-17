import sys,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
import msmarco_full_search_post as slow
import msmarco_full_search_fast as fast
tr=pd.read_csv('/mnt/data/dev.tsv',sep='\t',usecols=['query-id']); ids=[str(x) for x in np.unique(tr['query-id'].to_numpy())[:10]]; del tr
txt=fast.load_query_texts(ids)
a=slow.FullIndex(); b=fast.FullIndex()
for q in ids:
 pa=a.prepare(txt[q],20); pb=b.prepare(txt[q],20)
 for h in [0,1,5,10,20]:
  ra=a.rank_h(pa,h,100); rb=b.rank_h(pb,h,100)
  if ra!=rb:
   print('MISMATCH',q,h,next((i for i,(x,y) in enumerate(zip(ra,rb)) if x!=y),None)); break
 else: print('OK',q)
