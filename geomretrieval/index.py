from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import FrozenConfig
from .geometry import build_term_graphs
from .utils import topk_sparse_row, zscore


class GeometricIndex:
    """Frozen sparse geometric retrieval index.

    The implementation follows the final handoff architecture:
      TF-IDF -> F=4 fuzzy routing -> B=64 sparse centers -> S=16 signed
      residuals -> inverse local sign-variance reliability -> significance
      scoring -> second-order vocabulary routing -> binary whole-document
      support reranking.
    """

    def __init__(self, config: FrozenConfig | None = None):
        self.config = config or FrozenConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_ids: np.ndarray | None = None
        self.X: sparse.csr_matrix | None = None

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        texts: list[str],
        doc_ids: list[str] | None = None,
        config: FrozenConfig | None = None,
        verbose: bool = True,
    ) -> "GeometricIndex":
        self = cls(config)
        cfg = self.config
        N = len(texts)
        if doc_ids is None:
            doc_ids = [str(i) for i in range(N)]
        if len(doc_ids) != N:
            raise ValueError("doc_ids and texts must have identical length")
        self.doc_ids = np.asarray(doc_ids, dtype=object)

        def log(msg):
            if verbose:
                print(msg, flush=True)

        t0 = time.perf_counter()
        log(f"[1/8] TF-IDF: N={N:,}, max_features={cfg.max_features:,}")
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            min_df=cfg.min_df,
            lowercase=cfg.lowercase,
            token_pattern=cfg.token_pattern,
            norm="l2",
            dtype=np.float32,
            smooth_idf=True,
            sublinear_tf=False,
        )
        X = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        X.sort_indices()
        self.X = X
        self.idf = np.asarray(self.vectorizer.idf_, dtype=np.float32)
        self.vocab_size = X.shape[1]
        M = self.vocab_size
        log(f"      shape={X.shape}, nnz={X.nnz:,}, {time.perf_counter()-t0:.2f}s")

        # Whole-document binary support is simply the CSR sparsity pattern.
        # Keep a separate compact CSR with uint8 data so query reranking never
        # needs the TF-IDF amplitudes.
        self.support_indptr = X.indptr.astype(np.int64, copy=True)
        self.support_indices = X.indices.astype(np.int32, copy=True)

        analyzer = self.vectorizer.build_analyzer()
        self.doc_lengths = np.asarray([len(analyzer(t)) for t in texts], dtype=np.int32)
        self.avg_doc_length = float(max(1.0, self.doc_lengths.mean()))

        # ---------------- Fuzzy memberships ----------------
        log(f"[2/8] Fuzzy memberships F={cfg.F}")
        branches = np.full((N, cfg.F), -1, dtype=np.int32)
        memberships = np.zeros((N, cfg.F), dtype=np.float32)
        for d in range(N):
            a, b = X.indptr[d], X.indptr[d+1]
            idx, dat = X.indices[a:b], X.data[a:b]
            if not len(idx):
                continue
            ii, vv = topk_sparse_row(idx, dat, cfg.F)
            n = len(ii)
            branches[d, :n] = ii
            den = float(vv.sum())
            memberships[d, :n] = vv / den if den > 0 else 1.0 / n
        self.branches = branches
        self.memberships = memberships

        # Flatten memberships and sort by branch. This one structure serves as
        # the branch inverted index while preserving the document/slot identity.
        flat_branch = branches.ravel()
        valid_flat = np.flatnonzero(flat_branch >= 0).astype(np.int64)
        order = np.argsort(flat_branch[valid_flat], kind="stable")
        self.branch_order = valid_flat[order]
        sorted_br = flat_branch[self.branch_order]
        counts = np.bincount(sorted_br, minlength=M)
        self.branch_offsets = np.zeros(M + 1, dtype=np.int64)
        np.cumsum(counts, out=self.branch_offsets[1:])

        # ---------------- Sparse shared centers ----------------
        log(f"[3/8] Sparse branch centers B={cfg.B}")
        wr = np.repeat(np.arange(N, dtype=np.int32), cfg.F)
        wc = branches.ravel()
        wd = memberships.ravel()
        valid = wc >= 0
        W = sparse.csr_matrix((wd[valid], (wr[valid], wc[valid])), shape=(N, M), dtype=np.float32)
        branch_mass = np.asarray(W.sum(axis=0)).ravel().astype(np.float32)
        center_terms = np.full((M, cfg.B), -1, dtype=np.int32)
        center_values = np.zeros((M, cfg.B), dtype=np.float32)
        block = 256
        for start in range(0, M, block):
            end = min(M, start + block)
            C = (W[:, start:end].T @ X).tocsr()
            for local in range(end-start):
                j = start + local
                if branch_mass[j] <= 0:
                    continue
                a, b = C.indptr[local], C.indptr[local+1]
                idx = C.indices[a:b]
                dat = C.data[a:b] / branch_mass[j]
                if not len(dat):
                    continue
                kk = min(cfg.B, len(dat))
                pick = np.argpartition(dat, -kk)[-kk:]
                ii, vv = idx[pick], dat[pick]
                # Sorted term IDs make residual construction and later lookup cheap.
                oo = np.argsort(ii)
                ii, vv = ii[oo], vv[oo]
                center_terms[j, :kk] = ii
                center_values[j, :kk] = vv
        self.center_terms = center_terms
        self.center_values = center_values
        del W

        # ---------------- Signed residual codes ----------------
        log(f"[4/8] Signed residuals S={cfg.S} (document-present coordinates only)")
        res_terms = np.full((N, cfg.F, cfg.S), -1, dtype=np.int32)
        res_signs = np.zeros((N, cfg.F, cfg.S), dtype=np.int8)
        res_center = np.zeros((N, cfg.F, cfg.S), dtype=np.float32)

        for d in range(N):
            a, b = X.indptr[d], X.indptr[d+1]
            didx, dval = X.indices[a:b], X.data[a:b]
            if not len(didx):
                continue
            for s in range(cfg.F):
                j = int(branches[d, s])
                if j < 0:
                    continue
                cidx = center_terms[j]
                cval = center_values[j]
                maskc = cidx >= 0
                ck, cv = cidx[maskc], cval[maskc]
                c_at_doc = np.zeros(len(didx), dtype=np.float32)
                if len(ck):
                    pos = np.searchsorted(ck, didx)
                    ok = pos < len(ck)
                    oi = np.flatnonzero(ok)
                    if len(oi):
                        p = pos[oi]
                        same = ck[p] == didx[oi]
                        chosen = oi[same]
                        c_at_doc[chosen] = cv[pos[chosen]]
                residual = dval - c_at_doc
                kk = min(cfg.S, len(residual))
                pick = np.argpartition(np.abs(residual), -kk)[-kk:]
                pick = pick[np.argsort(np.abs(residual[pick]))[::-1]]
                res_terms[d, s, :kk] = didx[pick]
                res_signs[d, s, :kk] = np.where(residual[pick] >= 0, 1, -1).astype(np.int8)
                res_center[d, s, :kk] = c_at_doc[pick]
        self.res_terms = res_terms
        self.res_signs = res_signs
        self.res_center_values = res_center

        # ---------------- Reliability ----------------
        log("[5/8] Zero-inclusive local sign reliability")
        rel = np.ones((N, cfg.F, cfg.S), dtype=np.float16)
        # Global sign variance: zeros are implicit over all valid memberships.
        n_memberships_total = max(1, len(self.branch_order))
        global_count = np.zeros(M, dtype=np.float64)
        global_sum = np.zeros(M, dtype=np.float64)
        for d0 in range(0, N, 50_000):
            tt = res_terms[d0:d0+50_000].ravel()
            zz = res_signs[d0:d0+50_000].ravel().astype(np.float64)
            ok = tt >= 0
            global_count += np.bincount(tt[ok], minlength=M)
            global_sum += np.bincount(tt[ok], weights=zz[ok], minlength=M)
        g_e2 = global_count / n_memberships_total
        g_e1 = global_sum / n_memberships_total
        global_var = np.maximum(g_e2 - g_e1 * g_e1, 0.0)
        self.global_sign_var = global_var.astype(np.float32)

        # Process one branch at a time. Each branch sees only its own memberships,
        # so np.unique operates on a small local residual set rather than a giant
        # vocabulary x vocabulary table.
        flat_rel = rel.reshape(N * cfg.F, cfg.S)
        flat_terms = res_terms.reshape(N * cfg.F, cfg.S)
        flat_signs = res_signs.reshape(N * cfg.F, cfg.S)
        for j in range(M):
            a, b = self.branch_offsets[j], self.branch_offsets[j+1]
            mpos = self.branch_order[a:b]
            nj = len(mpos)
            if nj == 0:
                continue
            terms_j = flat_terms[mpos].ravel()
            signs_j = flat_signs[mpos].ravel().astype(np.float64)
            ok = terms_j >= 0
            if not np.any(ok):
                continue
            u, inv = np.unique(terms_j[ok], return_inverse=True)
            cnt = np.bincount(inv).astype(np.float64)
            sm = np.bincount(inv, weights=signs_j[ok]).astype(np.float64)
            e2 = cnt / nj
            e1 = sm / nj
            lv = np.maximum(e2 - e1 * e1, 0.0)
            shr = (cnt / (cnt + cfg.tau)) * lv + (cfg.tau / (cnt + cfg.tau)) * global_var[u]
            w = np.power(shr + cfg.reliability_eps, cfg.beta)
            # Keep the mean branch weight near one to avoid branch-scale artifacts.
            if len(w) and np.isfinite(w).all() and w.mean() > 0:
                w = w / w.mean()
            lookup = {int(t): float(v) for t, v in zip(u, w)}
            # Offline dictionary use is acceptable; query-time retrieval remains vectorized.
            for p in mpos:
                for r in range(cfg.S):
                    t = int(flat_terms[p, r])
                    if t >= 0:
                        flat_rel[p, r] = np.float16(lookup.get(t, 1.0))
        self.res_reliability = rel

        # ---------------- Term geometry ----------------
        log(f"[6/8] Term geometry L={cfg.L}, PPMI top={cfg.assoc_k}, context top={cfg.route_k}")
        self.A, self.G = build_term_graphs(X, cfg)

        # Index no longer requires TF-IDF corpus amplitudes for normal querying.
        # Retain X only in-memory for diagnostics; save() omits it by default.
        log("[7/8] Finalizing compact index")
        self._fitted = True
        self.build_seconds = time.perf_counter() - t0
        log(f"[8/8] DONE in {self.build_seconds:.2f}s")
        return self

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    def _query_vector(self, text: str) -> sparse.csr_matrix:
        if self.vectorizer is None:
            raise RuntimeError("Index is not fitted")
        q = self.vectorizer.transform([text]).tocsr().astype(np.float32)
        q.sort_indices()
        return q

    def _expanded_route(self, q: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        M = self.vocab_size
        qa, qb = q.indptr[0], q.indptr[1]
        q_terms = q.indices[qa:qb]
        q_vals = q.data[qa:qb]
        route = np.zeros(M, dtype=np.float32)
        route[q_terms] = q_vals

        # Weak second-order semantic routing. Original coordinates are preserved.
        for t, qv in zip(q_terms, q_vals):
            a, b = self.G.indptr[t], self.G.indptr[t+1]
            nb = self.G.indices[a:b]
            sv = self.G.data[a:b]
            route[nb] += cfg.route_alpha * float(qv) * sv

        nonzero = np.flatnonzero(route > 0)
        originals = set(map(int, q_terms.tolist()))
        if len(nonzero) > cfg.route_budget:
            # Preserve every literal query term; fill the remaining budget with
            # strongest inferred coordinates.
            inferred = np.asarray([i for i in nonzero if int(i) not in originals], dtype=np.int32)
            budget = max(0, cfg.route_budget - len(originals))
            if budget and len(inferred) > budget:
                pick = np.argpartition(route[inferred], -budget)[-budget:]
                inferred = inferred[pick]
            elif budget == 0:
                inferred = np.empty(0, dtype=np.int32)
            chosen = np.asarray(sorted(originals), dtype=np.int32)
            nonzero = np.concatenate([chosen, inferred])
        # strongest first is convenient but not required for union routing
        order = np.argsort(route[nonzero])[::-1]
        return nonzero[order].astype(np.int32), route[nonzero[order]], q_terms

    def search(self, text: str, k: int | None = None, return_scores: bool = False):
        cfg = self.config
        k = int(k or cfg.output_k)
        q = self._query_vector(text)
        q_dense = np.zeros(self.vocab_size, dtype=np.float32)
        q_dense[q.indices] = q.data
        route_terms, route_vals, q_terms = self._expanded_route(q)
        if len(route_terms) == 0:
            return ([], np.empty(0, np.float32)) if return_scores else []

        route_dense = np.zeros(self.vocab_size, dtype=np.float32)
        route_dense[route_terms] = route_vals

        # Retrieve matching membership positions, not just docs, because fuzzy
        # multi-branch evidence is part of the score.
        pieces = []
        for j in route_terms:
            a, b = self.branch_offsets[j], self.branch_offsets[j+1]
            if b > a:
                pieces.append(self.branch_order[a:b])
        if not pieces:
            return ([], np.empty(0, np.float32)) if return_scores else []
        flatpos = np.concatenate(pieces).astype(np.int64, copy=False)
        docs = (flatpos // cfg.F).astype(np.int64)
        slots = (flatpos % cfg.F).astype(np.int64)
        br = self.branches[docs, slots]

        terms = self.res_terms[docs, slots]
        valid = terms >= 0
        safe_terms = np.where(valid, terms, 0)
        qv = q_dense[safe_terms]
        local = np.sum(
            self.res_reliability[docs, slots].astype(np.float32)
            * (qv - self.res_center_values[docs, slots])
            * self.res_signs[docs, slots].astype(np.float32)
            * valid,
            axis=1,
        )
        significance = np.sum((qv * qv) * valid, axis=1)
        m = self.memberships[docs, slots]
        rho = route_dense[br]

        unique_docs, inv = np.unique(docs, return_inverse=True)
        head_contrib = m * rho * local * np.power(np.maximum(significance, 0.0), cfg.gamma_head)
        tail_contrib = m * rho * local * np.power(np.maximum(significance, 0.0), cfg.gamma_tail)
        consensus_contrib = m * rho
        head = np.bincount(inv, weights=head_contrib, minlength=len(unique_docs)).astype(np.float32)
        tail = np.bincount(inv, weights=tail_contrib, minlength=len(unique_docs)).astype(np.float32)
        consensus = np.bincount(inv, weights=consensus_contrib, minlength=len(unique_docs)).astype(np.float32)
        tail = tail + cfg.lambda_membership * consensus

        # Freeze precision head.
        hk = min(cfg.head_k, len(unique_docs))
        hidx = np.argpartition(head, -hk)[-hk:]
        hidx = hidx[np.argsort(head[hidx])[::-1]]
        frozen_docs = unique_docs[hidx]
        frozen_set = set(map(int, frozen_docs.tolist()))

        # Recall-oriented tail shortlist.
        mask_tail = np.asarray([int(d) not in frozen_set for d in unique_docs], dtype=bool)
        td = unique_docs[mask_tail]
        ts = tail[mask_tail]
        if len(td):
            P = min(cfg.rerank_pool, len(td))
            pidx = np.argpartition(ts, -P)[-P:]
            shortlist_docs = td[pidx]
            shortlist_tail = ts[pidx]

            # Whole-document binary lexical support. This is deliberately term
            # presence only; exact within-document TF was found unnecessary.
            lex_vec = np.zeros(self.vocab_size, dtype=np.float32)
            lex_vec[q.indices] = self.idf[q.indices]
            lex = np.zeros(P, dtype=np.float32)

            sem_vec = np.zeros(self.vocab_size, dtype=np.float32)
            for t, qamp in zip(q.indices, q.data):
                a, b = self.A.indptr[t], self.A.indptr[t+1]
                nb = self.A.indices[a:b][:cfg.semantic_k]
                sv = self.A.data[a:b][:cfg.semantic_k]
                if len(nb):
                    sem_vec[nb] += float(qamp) * sv * self.idf[nb]
            sem = np.zeros(P, dtype=np.float32)

            for i, d in enumerate(shortlist_docs):
                a, b = self.support_indptr[d], self.support_indptr[d+1]
                support = self.support_indices[a:b]
                lex[i] = float(lex_vec[support].sum())
                if cfg.length_b != 0:
                    denom = (1.0 - cfg.length_b) + cfg.length_b * (float(self.doc_lengths[d]) / self.avg_doc_length)
                    if denom > 0:
                        lex[i] /= denom
                sem[i] = float(sem_vec[support].sum())

            final = zscore(shortlist_tail) + cfg.lambda_lex * zscore(lex) + cfg.lambda_sem * zscore(sem)
            oo = np.argsort(final)[::-1]
            ranked_tail = shortlist_docs[oo]
            ranked_tail_scores = final[oo]

            # If caller asks beyond the reranking pool, append remaining tail by
            # the cheap score. This does not affect the usual top-100 evaluation.
            shortlist_set = set(map(int, shortlist_docs.tolist()))
            rest_mask = np.asarray([int(d) not in shortlist_set for d in td], dtype=bool)
            rest_docs = td[rest_mask]
            rest_scores = ts[rest_mask]
            if len(rest_docs):
                ro = np.argsort(rest_scores)[::-1]
                ranked_tail = np.concatenate([ranked_tail, rest_docs[ro]])
                ranked_tail_scores = np.concatenate([ranked_tail_scores, rest_scores[ro]])
        else:
            ranked_tail = np.empty(0, dtype=np.int64)
            ranked_tail_scores = np.empty(0, dtype=np.float32)

        ranked = np.concatenate([frozen_docs, ranked_tail])[:k]
        # Head and tail score scales differ; scores are only for diagnostics.
        hs = head[hidx]
        scores = np.concatenate([hs, ranked_tail_scores])[:k]
        ids = self.doc_ids[ranked].tolist()
        if return_scores:
            return ids, scores
        return ids

    def batch_search(self, queries: dict[str, str], k: int | None = None, timing: bool = False):
        run: dict[str, list[str]] = {}
        times_ms = []
        for qid, text in queries.items():
            t0 = time.perf_counter()
            run[str(qid)] = self.search(text, k=k)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        if timing:
            arr = np.asarray(times_ms, dtype=np.float64)
            return run, {
                "median_ms": float(np.median(arr)),
                "mean_ms": float(np.mean(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "qps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else float("inf"),
            }
        return run

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------
    def save(self, path: str | os.PathLike, include_tfidf_matrix: bool = False):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, p / "vectorizer.joblib")
        with (p / "config.json").open("w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        meta = {
            "vocab_size": int(self.vocab_size),
            "avg_doc_length": float(self.avg_doc_length),
            "build_seconds": float(getattr(self, "build_seconds", 0.0)),
        }
        with (p / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        np.savez_compressed(
            p / "arrays.npz",
            doc_ids=self.doc_ids,
            idf=self.idf,
            support_indptr=self.support_indptr,
            support_indices=self.support_indices,
            doc_lengths=self.doc_lengths,
            branches=self.branches,
            memberships=self.memberships,
            branch_order=self.branch_order,
            branch_offsets=self.branch_offsets,
            center_terms=self.center_terms,
            center_values=self.center_values,
            res_terms=self.res_terms,
            res_signs=self.res_signs,
            res_center_values=self.res_center_values,
            res_reliability=self.res_reliability,
            global_sign_var=self.global_sign_var,
        )
        sparse.save_npz(p / "assoc_ppmi.npz", self.A)
        sparse.save_npz(p / "context_similarity.npz", self.G)
        if include_tfidf_matrix and self.X is not None:
            sparse.save_npz(p / "tfidf_corpus.npz", self.X)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "GeometricIndex":
        p = Path(path)
        with (p / "config.json").open() as f:
            cfg = FrozenConfig.from_dict(json.load(f))
        self = cls(cfg)
        self.vectorizer = joblib.load(p / "vectorizer.joblib")
        with (p / "meta.json").open() as f:
            meta = json.load(f)
        a = np.load(p / "arrays.npz", allow_pickle=True)
        for name in a.files:
            setattr(self, name, a[name])
        self.vocab_size = int(meta["vocab_size"])
        self.avg_doc_length = float(meta["avg_doc_length"])
        self.build_seconds = float(meta.get("build_seconds", 0.0))
        self.A = sparse.load_npz(p / "assoc_ppmi.npz").tocsr()
        self.G = sparse.load_npz(p / "context_similarity.npz").tocsr()
        tfidf = p / "tfidf_corpus.npz"
        self.X = sparse.load_npz(tfidf).tocsr() if tfidf.exists() else None
        self._fitted = True
        return self
