# Full baseline suite and provenance policy

Every benchmark table in this project keeps the complete comparison suite visible, even when a particular baseline has not yet been rerun on the current dataset.

| Baseline | Representation / search | Speed quantity that must be reported |
|---|---|---|
| TF-IDF | sparse lexical, exact scan or inverted index | end-to-end query + search |
| BM25 | sparse lexical inverted index | end-to-end query + postings traversal |
| FAISS Flat | fixed dense embeddings, exact inner product/cosine | query encoder + exact vector search |
| FAISS HNSW | fixed dense embeddings, HNSW | query encoder + ANN search |
| FAISS IVF-Flat | fixed dense embeddings, IVF | query encoder + ANN search |
| FAISS IVF-PQ | fixed dense embeddings, IVF + product quantization | query encoder + ANN search |
| hnswlib HNSW | fixed dense embeddings, HNSW | query encoder + ANN search |
| ScaNN | fixed dense embeddings, pruning/quantization | query encoder + ANN search |
| Contriever | neural dense retrieval + ANN | query encoder + ANN search |
| SPLADE++ | neural sparse expansion + inverted index | sparse query encoder + search |
| BGE-base | neural dense retrieval + ANN | query encoder + ANN search |
| Modern ColBERT | neural multi-vector late interaction | query encoder + candidate generation + late interaction |
| OURS | sparse TF-IDF geometry + signed residuals | TF-IDF query construction + routing + shortlist + top-10 selection |

## Missing values are shown, not hidden

A dash in a result table means the baseline has not yet been rerun under the same representation/hardware/protocol. It is intentionally left visible. We do not fill missing latency with a number from a different CPU/GPU and we do not compare ANN-only latency against our end-to-end latency.

## Current provenance classes

- **local**: executed on the current dataset by the code in this repository/session;
- **historical local**: executed in an earlier frozen version of the same project;
- **published/context ledger**: retained from the baseline ledger used during the experimental campaign; not presented as a same-hardware speed result;
- **pending same-representation rerun**: part of the required suite but deliberately blank until a controlled experiment exists.
