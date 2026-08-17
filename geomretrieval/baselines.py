"""Optional dense ANN baseline helpers.

These deliberately accept PRECOMPUTED embeddings. They never run a transformer.
Install with: pip install 'geomretrieval[ann]'
"""
from __future__ import annotations
import time
import numpy as np


def faiss_flat_ip(corpus: np.ndarray, queries: np.ndarray, k: int = 100):
    import faiss
    xb = np.ascontiguousarray(corpus.astype(np.float32))
    xq = np.ascontiguousarray(queries.astype(np.float32))
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    ms = (time.perf_counter() - t0) * 1000.0 / len(xq)
    return I, D, ms


def faiss_hnsw_ip(corpus: np.ndarray, queries: np.ndarray, k: int = 100, M: int = 32, ef_search: int = 128):
    import faiss
    xb = np.ascontiguousarray(corpus.astype(np.float32))
    xq = np.ascontiguousarray(queries.astype(np.float32))
    index = faiss.IndexHNSWFlat(xb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(xb)
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    ms = (time.perf_counter() - t0) * 1000.0 / len(xq)
    return I, D, ms


def faiss_ivf_flat_ip(corpus: np.ndarray, queries: np.ndarray, k: int = 100, nlist: int = 4096, nprobe: int = 64):
    import faiss
    xb = np.ascontiguousarray(corpus.astype(np.float32))
    xq = np.ascontiguousarray(queries.astype(np.float32))
    quant = faiss.IndexFlatIP(xb.shape[1])
    index = faiss.IndexIVFFlat(quant, xb.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(xb)
    index.add(xb)
    index.nprobe = nprobe
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    ms = (time.perf_counter() - t0) * 1000.0 / len(xq)
    return I, D, ms
