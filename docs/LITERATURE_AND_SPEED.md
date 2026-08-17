# Retrieval literature through the speed lens

Speed is not a secondary metric in this project. A retriever that produces an excellent rank at a cost incompatible with interactive RAG has solved a different problem.

## The computational boundary we measure

A RAG request pays for **query representation + retrieval + shortlist scoring + top-10 selection**. For ANN baselines, the literature often reports ANN search after the dense query embedding has already been computed. We therefore keep two latency columns whenever possible:

1. **ANN/search-only latency** — useful for comparing indexes.
2. **End-to-end query latency** — the number relevant to an actual RAG request.

The two must never be silently mixed.

| Family | Query-time representation | Search object | Speed implication |
|---|---|---|---|
| BM25 | tokenization only | inverted postings | no neural query inference; strong classical latency reference |
| FAISS Flat / IVF / PQ | dense query embedding | dense vectors / quantized vectors | optimized vector search; encoder cost is normally outside ANN timing |
| HNSW | dense query embedding | graph over dense vectors | very fast ANN search, but memory-heavy graph and query encoder remain |
| ScaNN | dense query embedding | partitioned/quantized dense vectors | search is optimized around MIPS/quantization; encoder cost is separate |
| Contriever / BGE | neural dense encoder | ANN dense index | representation quality is strong, but query inference is part of deployed RAG cost |
| SPLADE | neural sparse encoder | sparse inverted index | sparse search, but query sparse vector is produced by a transformer |
| ColBERTv2 | neural token encoder | compressed multi-vector index + late interaction | excellent quality, but multiple query vectors and late interaction increase work |
| **Sparse Geometric RAG (this repo)** | **TF-IDF query vector; no neural inference** | **fuzzy sparse branches + 16-coordinate signed residuals** | **query formation is cheap; routing and local scoring touch tiny sparse structures; only a small shortlist reaches chunk-level scoring** |

## Primary references

- Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs* (FAISS): https://arxiv.org/abs/1702.08734
- Douze et al., *The Faiss Library*: https://arxiv.org/abs/2401.08281
- Malkov and Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*: https://arxiv.org/abs/1603.09320
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN): https://arxiv.org/abs/1908.10396
- Izacard et al., *Unsupervised Dense Information Retrieval with Contrastive Learning* (Contriever): https://arxiv.org/abs/2112.09118
- Formal et al., *SPLADE v2*: https://arxiv.org/abs/2109.10086
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*: https://arxiv.org/abs/2112.01488
- Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE family): https://arxiv.org/abs/2309.07597
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*: https://arxiv.org/abs/2104.08663

## Why this method is different computationally

FAISS, HNSW, and ScaNN solve the problem “search a database of dense vectors quickly.” This project asks a prior question: **does the retrieval database need a dense vector per chunk at all?** The stored object is instead a coarse fuzzy location plus a very small signed departure from its local center. Query-time work follows that sparse geometry.

The intended speed story is therefore not “we wrote a faster HNSW.” It is: **avoid most of the arithmetic and memory traffic that make ANN necessary in the first place.**
