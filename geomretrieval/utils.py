from __future__ import annotations
import numpy as np
from scipy import sparse


def topk_sparse_row(indices: np.ndarray, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return up to k largest values from one CSR row, descending by value."""
    if len(values) <= k:
        order = np.argsort(values)[::-1]
    else:
        pick = np.argpartition(values, -k)[-k:]
        order = pick[np.argsort(values[pick])[::-1]]
    return indices[order], values[order]


def topk_abs_sparse_row(indices: np.ndarray, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return up to k largest absolute values from one sparse row."""
    av = np.abs(values)
    if len(values) <= k:
        order = np.argsort(av)[::-1]
    else:
        pick = np.argpartition(av, -k)[-k:]
        order = pick[np.argsort(av[pick])[::-1]]
    return indices[order], values[order]


def csr_row_topk_matrix(X: sparse.csr_matrix, k: int, binary: bool = False) -> sparse.csr_matrix:
    """Keep top-k entries in every CSR row.

    Used only offline to construct the term-geometry estimation matrix.
    """
    rows, cols, vals = [], [], []
    for r in range(X.shape[0]):
        a, b = X.indptr[r], X.indptr[r + 1]
        idx, dat = X.indices[a:b], X.data[a:b]
        if len(idx) == 0:
            continue
        ii, vv = topk_sparse_row(idx, dat, k)
        rows.extend([r] * len(ii))
        cols.extend(ii.tolist())
        vals.extend(([1.0] * len(ii)) if binary else vv.tolist())
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=X.shape)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    s = float(x.std())
    if s < eps:
        return np.zeros_like(x)
    return (x - float(x.mean())) / (s + eps)


def padded_topk_dense_from_csr_row(row: sparse.csr_matrix, k: int, width: int | None = None):
    """Top-k of a single CSR row, returned padded and term-id sorted.

    Sorting term IDs rather than scores is useful for fast center lookup later.
    """
    width = width or k
    idx, dat = row.indices, row.data
    if len(dat):
        ii, vv = topk_sparse_row(idx, dat, k)
        order = np.argsort(ii)
        ii, vv = ii[order], vv[order]
    else:
        ii = np.empty(0, dtype=np.int32)
        vv = np.empty(0, dtype=np.float32)
    out_i = np.full(width, -1, dtype=np.int32)
    out_v = np.zeros(width, dtype=np.float32)
    n = min(width, len(ii))
    out_i[:n] = ii[:n]
    out_v[:n] = vv[:n]
    return out_i, out_v


def lookup_sorted(keys: np.ndarray, vals: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    """Lookup query keys in sorted padded key/value arrays; absent -> 0."""
    valid = keys >= 0
    k = keys[valid]
    v = vals[valid]
    if len(k) == 0 or len(query_keys) == 0:
        return np.zeros(len(query_keys), dtype=np.float32)
    pos = np.searchsorted(k, query_keys)
    out = np.zeros(len(query_keys), dtype=np.float32)
    ok = pos < len(k)
    ok_idx = np.flatnonzero(ok)
    if len(ok_idx):
        p = pos[ok_idx]
        same = k[p] == query_keys[ok_idx]
        chosen = ok_idx[same]
        out[chosen] = v[pos[chosen]]
    return out
