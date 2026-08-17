from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .config import FrozenConfig
from .utils import csr_row_topk_matrix


def _keep_top_sparse_rows(mat: sparse.csr_matrix, k: int, exclude_diagonal_offset: int | None = None) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for r in range(mat.shape[0]):
        a, b = mat.indptr[r], mat.indptr[r + 1]
        idx, dat = mat.indices[a:b], mat.data[a:b]
        if exclude_diagonal_offset is not None:
            diag = exclude_diagonal_offset + r
            mask = idx != diag
            idx, dat = idx[mask], dat[mask]
        if len(dat) == 0:
            continue
        kk = min(k, len(dat))
        pick = np.argpartition(dat, -kk)[-kk:]
        pick = pick[np.argsort(dat[pick])[::-1]]
        rows.extend([r] * kk)
        cols.extend(idx[pick].tolist())
        vals.extend(dat[pick].astype(np.float32).tolist())
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=mat.shape)


def build_term_graphs(X: sparse.csr_matrix, cfg: FrozenConfig) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build first-order significance-shrunk PPMI A and second-order context graph G.

    Both are built blockwise; a dense vocabulary x vocabulary matrix is never
    instantiated.
    """
    N, M = X.shape
    T = csr_row_topk_matrix(X, cfg.L, binary=True)
    n_i = np.asarray(T.sum(axis=0)).ravel().astype(np.float64)

    A_rows, A_cols, A_vals = [], [], []
    bs = cfg.graph_block_size
    for start in range(0, M, bs):
        end = min(M, start + bs)
        # co[r,j] = number of documents in which term start+r and j both appear
        # among the document's top-L TF-IDF coordinates.
        co = (T[:, start:end].T @ T).tocsr()
        for local in range(end - start):
            i = start + local
            a, b = co.indptr[local], co.indptr[local + 1]
            js = co.indices[a:b]
            nij = co.data[a:b].astype(np.float64)
            mask = (js != i) & (nij > 0) & (n_i[js] > 0) & (n_i[i] > 0)
            js, nij = js[mask], nij[mask]
            if not len(js):
                continue
            ppmi = np.log((nij * float(N) + 1e-12) / (n_i[i] * n_i[js] + 1e-12))
            ppmi = np.maximum(ppmi, 0.0)
            score = (nij / (nij + cfg.graph_significance_tau)) * ppmi
            pos = score > 0
            js, score = js[pos], score[pos]
            if not len(score):
                continue
            kk = min(cfg.assoc_k, len(score))
            pick = np.argpartition(score, -kk)[-kk:]
            pick = pick[np.argsort(score[pick])[::-1]]
            A_rows.extend([i] * kk)
            A_cols.extend(js[pick].tolist())
            A_vals.extend(score[pick].astype(np.float32).tolist())

    A = sparse.csr_matrix((np.asarray(A_vals, np.float32), (A_rows, A_cols)), shape=(M, M))
    A.eliminate_zeros()

    An = normalize(A, norm="l2", axis=1, copy=True)
    G_rows, G_cols, G_vals = [], [], []
    for start in range(0, M, bs):
        end = min(M, start + bs)
        sim = (An[start:end] @ An.T).tocsr()
        for local in range(end - start):
            i = start + local
            a, b = sim.indptr[local], sim.indptr[local + 1]
            js = sim.indices[a:b]
            vv = sim.data[a:b]
            mask = (js != i) & (vv > 0)
            js, vv = js[mask], vv[mask]
            if not len(vv):
                continue
            kk = min(cfg.route_k, len(vv))
            pick = np.argpartition(vv, -kk)[-kk:]
            pick = pick[np.argsort(vv[pick])[::-1]]
            G_rows.extend([i] * kk)
            G_cols.extend(js[pick].tolist())
            G_vals.extend(vv[pick].astype(np.float32).tolist())
    G = sparse.csr_matrix((np.asarray(G_vals, np.float32), (G_rows, G_cols)), shape=(M, M))
    G.eliminate_zeros()
    return A, G
