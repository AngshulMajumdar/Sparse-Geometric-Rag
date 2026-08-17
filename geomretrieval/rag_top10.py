from __future__ import annotations

"""Top-10 RAG ranking layer for the sparse geometric index.

This module is the cleaned, path-independent version of the exact SciFact and
TREC-COVID experiment scripts preserved under ``experiments/beir/*history.py``.
It keeps the geometric index fixed and changes only the shortlist size P and
the final top-10 set construction.

Important implementation choices
--------------------------------
* Early rescue: binary whole-chunk IDF^1 support.
* Final lexical signal: binary whole-chunk IDF^2 support, not TF^2.
* Final components retain the validated per-query z-normalization.
* Branch quality H_j is the mean of the top three branch-specific evidences.
* Diversity is available only to the ten highest-quality branches.
* Rank 1 is pure relevance; ranks 2..10 receive a soft diversity correction.
* Repeated branches are allowed. There is no one-document-per-branch rule.
"""

from dataclasses import dataclass
import time
import numpy as np

from .metrics import evaluate_run


def _zscore(x):
    x = np.asarray(x, np.float32)
    if not len(x):
        return x
    sd = float(x.std())
    return np.zeros_like(x) if sd < 1e-8 else (x - float(x.mean())) / sd


def _minmax_hi(x):
    """Map scores monotonically to [0, 1], high remains good."""
    x = np.asarray(x, np.float32)
    if not len(x):
        return x
    lo, hi = float(x.min()), float(x.max())
    den = hi - lo
    return np.ones_like(x) if den < 1e-8 else (x - lo) / den


def _topk_large(score, k):
    score = np.asarray(score)
    n = len(score)
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, np.int64)
    if n <= k:
        return np.argsort(score)[::-1]
    ii = np.argpartition(score, -k)[-k:]
    return ii[np.argsort(score[ii])[::-1]]


@dataclass(frozen=True)
class RAGTop10Config:
    pool_size: int = 100
    gamma: float = 0.25
    lambda_membership: float = 0.125
    pre_length_b: float = 0.2
    final_length_b: float = 0.1
    coordination_alpha: float = 0.25
    lambda_lex: float = 4.0
    lambda_sem: float = 0.3
    lambda_rare: float = 1.0
    semantic_k: int = 16
    rare_topk: int = 3
    hq_top_branches: int = 10
    branch_quality_top_docs: int = 3
    lambda_diversity: float = 0.1


class RAGTop10Ranker:
    """RAG-oriented shortlist and top-10 set selector.

    ``GeometricIndex`` performs corpus indexing and stores the compact geometry.
    This class consumes that frozen representation. It does not train a model,
    rebuild centers, or alter residual codes.
    """

    def __init__(self, index, config: RAGTop10Config | None = None):
        self.idx = index
        self.cfg = config or RAGTop10Config()
        self.M = int(index.vocab_size)

    def _center_sparse(self, branch):
        t = self.idx.center_terms[branch]
        v = self.idx.center_values[branch]
        ok = t >= 0
        t = t[ok].astype(np.int32)
        v = v[ok].astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        order = np.argsort(t)
        return t[order], v[order]

    @staticmethod
    def _spdot(a_t, a_v, b_t, b_v):
        i = j = 0
        s = 0.0
        while i < len(a_t) and j < len(b_t):
            if a_t[i] == b_t[j]:
                s += float(a_v[i]) * float(b_v[j]); i += 1; j += 1
            elif a_t[i] < b_t[j]:
                i += 1
            else:
                j += 1
        return s

    def prepare(self, text: str):
        """Retrieve geometric candidates and create the P-sized chunk shortlist."""
        idx, cfg, M = self.idx, self.cfg, self.M
        q = idx._query_vector(text)
        if q.nnz == 0:
            return None

        q_dense = np.zeros(M, np.float32)
        q_dense[q.indices] = q.data
        route_terms, route_values, _ = idx._expanded_route(q)
        if not len(route_terms):
            return None
        route_dense = np.zeros(M, np.float32)
        route_dense[route_terms] = route_values

        pieces = []
        for j in route_terms:
            a, b = idx.branch_offsets[j], idx.branch_offsets[j + 1]
            if b > a:
                pieces.append(idx.branch_order[a:b])
        if not pieces:
            return None

        flat_pos = np.concatenate(pieces).astype(np.int64, copy=False)
        docs = (flat_pos // idx.config.F).astype(np.int64)
        slots = (flat_pos % idx.config.F).astype(np.int64)
        branches = idx.branches[docs, slots]

        terms = idx.res_terms[docs, slots]
        valid = terms >= 0
        safe_terms = np.where(valid, terms, 0)
        qv = q_dense[safe_terms]
        local = np.sum(
            idx.res_reliability[docs, slots].astype(np.float32)
            * (qv - idx.res_center_values[docs, slots])
            * idx.res_signs[docs, slots].astype(np.float32)
            * valid,
            axis=1,
        )
        significance = np.sum((qv * qv) * valid, axis=1)
        consensus = idx.memberships[docs, slots] * route_dense[branches]
        branch_ev = (
            consensus * local * np.power(np.maximum(significance, 0), cfg.gamma)
        ).astype(np.float32)

        unique_docs, inverse = np.unique(docs, return_inverse=True)
        tail = np.bincount(inverse, weights=branch_ev, minlength=len(unique_docs)).astype(np.float32)
        tail += cfg.lambda_membership * np.bincount(
            inverse, weights=consensus, minlength=len(unique_docs)
        ).astype(np.float32)

        # Stage 1: cheap whole-chunk binary IDF^1 rescue before expensive final scoring.
        qlex = np.zeros(M, np.float32)
        qlex[q.indices] = idx.idf[q.indices]
        lex1 = np.zeros(len(unique_docs), np.float32)
        for i, d in enumerate(unique_docs):
            a, b = idx.support_indptr[d], idx.support_indptr[d + 1]
            support = idx.support_indices[a:b]
            raw = float(qlex[support].sum())
            den = (1 - cfg.pre_length_b) + cfg.pre_length_b * (
                float(idx.doc_lengths[d]) / idx.avg_doc_length
            )
            lex1[i] = raw / (den if den > 0 else 1.0)

        pre = _zscore(tail) + _zscore(lex1)
        selected = _topk_large(pre, cfg.pool_size)
        pool_docs = unique_docs[selected]
        pool_tail = tail[selected]

        # Preserve branch-specific evidence for robust branch-quality estimation.
        pool_position = np.full(len(unique_docs), -1, np.int32)
        pool_position[selected] = np.arange(len(selected), dtype=np.int32)
        mapped = pool_position[inverse]
        keep = mapped >= 0
        mem_pool = mapped[keep].astype(np.int32)
        mem_branch = branches[keep].astype(np.int32)
        mem_ev = branch_ev[keep].astype(np.float32)

        # Stage 2: final chunk evidence. No document TF is used here.
        semvec = np.zeros(M, np.float32)
        for t, amp in zip(q.indices, q.data):
            a, b = idx.A.indptr[t], idx.A.indptr[t + 1]
            nb = idx.A.indices[a:b][: cfg.semantic_k]
            sv = idx.A.data[a:b][: cfg.semantic_k]
            if len(nb):
                semvec[nb] += float(amp) * sv * idx.idf[nb]

        qset = set(map(int, q.indices))
        rare = set(map(int, q.indices[np.argsort(idx.idf[q.indices])[::-1]][: cfg.rare_topk]))
        nq = max(1, len(q.indices))
        lex2 = np.zeros(len(pool_docs), np.float32)
        sem = np.zeros(len(pool_docs), np.float32)
        matched_count = np.zeros(len(pool_docs), np.float32)
        rare_count = np.zeros(len(pool_docs), np.float32)

        for i, d in enumerate(pool_docs):
            a, b = idx.support_indptr[d], idx.support_indptr[d + 1]
            support = idx.support_indices[a:b]
            sem[i] = float(semvec[support].sum())
            match = [int(t) for t in support if int(t) in qset]
            raw = sum(float(idx.idf[t]) ** 2 for t in match)
            den = (1 - cfg.final_length_b) + cfg.final_length_b * (
                float(idx.doc_lengths[d]) / idx.avg_doc_length
            )
            lex2[i] = raw / (den if den > 0 else 1.0)
            matched_count[i] = len(match)
            rare_count[i] = sum(t in rare for t in match)

        coverage = matched_count / nq
        lex_adjusted = lex2 * np.power(np.maximum(coverage, 1e-6), cfg.coordination_alpha)
        rare_coverage = rare_count / max(1, min(cfg.rare_topk, nq))
        relevance = (
            _zscore(pool_tail)
            + cfg.lambda_lex * _zscore(lex_adjusted)
            + cfg.lambda_sem * _zscore(sem)
            + cfg.lambda_rare * _zscore(rare_coverage)
        )

        # Robust high-quality branch score H_j: mean top-r branch-specific evidence.
        branch_pairs = {}
        for pi, b, e in zip(mem_pool, mem_branch, mem_ev):
            key = (int(b), int(pi))
            if key not in branch_pairs or e > branch_pairs[key]:
                branch_pairs[key] = float(e)
        by_branch = {}
        for (b, pi), e in branch_pairs.items():
            by_branch.setdefault(b, []).append((e, pi))

        H, docs_by_branch = {}, {}
        for b, vals in by_branch.items():
            vals.sort(key=lambda x: x[0], reverse=True)
            r = min(cfg.branch_quality_top_docs, len(vals))
            H[b] = float(np.mean([e for e, _ in vals[:r]]))
            docs_by_branch[b] = np.asarray([pi for _, pi in vals], dtype=np.int32)

        unique_branches = np.asarray(sorted(H.keys()), dtype=np.int32)
        h = np.asarray([H[int(b)] for b in unique_branches], dtype=np.float32)
        centers = [self._center_sparse(int(b)) for b in unique_branches]
        cosine = np.eye(len(unique_branches), dtype=np.float32)
        for i in range(len(unique_branches)):
            for j in range(i + 1, len(unique_branches)):
                cosine[i, j] = cosine[j, i] = self._spdot(*centers[i], *centers[j])

        return {
            "docs": pool_docs,
            "relevance": relevance,
            "route_docs": unique_docs,
            "branches": unique_branches,
            "branch_quality": h,
            "cosine": cosine,
            "docs_by_branch": docs_by_branch,
        }

    @staticmethod
    def _deviation(cosine, selected_branch_indices):
        """Squared distance of each unit branch center from selected-center centroid."""
        if not selected_branch_indices:
            return np.zeros(len(cosine), np.float32)
        si = np.asarray(selected_branch_indices, np.int32)
        centroid_norm_sq = float(np.mean(cosine[np.ix_(si, si)]))
        return 1.0 + centroid_norm_sq - 2.0 * np.mean(cosine[:, si], axis=1)

    def rank(self, packet, k: int = 10):
        """Construct top-k; branch diversity affects only the first ten ranks."""
        if packet is None or not len(packet["docs"]):
            return []
        cfg = self.cfg
        base = packet["relevance"]
        n = len(base)
        order = np.argsort(base)[::-1]
        first = int(order[0])
        chosen = [first]
        used = {first}

        # Only top query-specific branches are eligible to receive a diversity bonus.
        h = packet["branch_quality"]
        eligible = np.argsort(h)[::-1][: min(cfg.hq_top_branches, len(h))]
        doc_hq = [[] for _ in range(n)]
        branch_to_index = {int(b): i for i, b in enumerate(packet["branches"])}
        for bi in eligible:
            b = int(packet["branches"][bi])
            for pi in packet["docs_by_branch"].get(b, []):
                doc_hq[int(pi)].append(int(bi))

        selected_branches = []
        if doc_hq[first]:
            selected_branches = [max(doc_hq[first], key=lambda bi: float(h[bi]))]

        # Ranks 2..10: relevance plus soft central-deviation bonus.
        for _ in range(1, min(10, k, n)):
            rem = np.asarray([i for i in range(n) if i not in used], dtype=np.int32)
            if not len(rem):
                break
            dev = self._deviation(packet["cosine"], selected_branches)
            if len(eligible):
                dev_values = _minmax_hi(dev[eligible])
                dev_map = {int(bi): float(v) for bi, v in zip(eligible, dev_values)}
            else:
                dev_map = {}
            rel = _minmax_hi(base[rem])
            bonus = np.zeros(len(rem), np.float32)
            for ii, pi in enumerate(rem):
                if doc_hq[int(pi)]:
                    bonus[ii] = max(dev_map.get(bi, 0.0) for bi in doc_hq[int(pi)])
            value = rel + cfg.lambda_diversity * bonus
            pi = int(rem[int(np.argmax(value))])
            chosen.append(pi)
            used.add(pi)
            if doc_hq[pi]:
                bi = max(doc_hq[pi], key=lambda x: (dev_map.get(x, 0.0), float(h[x])))
                selected_branches.append(int(bi))

        # Beyond rank 10, ordinary relevance. This keeps the top-10 RAG mechanism isolated.
        for pi in order:
            pi = int(pi)
            if len(chosen) >= min(k, n):
                break
            if pi not in used:
                chosen.append(pi)
                used.add(pi)
        return packet["docs"][np.asarray(chosen[:k], dtype=np.int64)].tolist()

    def search(self, text: str, k: int = 10, timing: bool = False):
        t0 = time.perf_counter()
        packet = self.prepare(text)
        t1 = time.perf_counter()
        local_ids = self.rank(packet, k=k)
        t2 = time.perf_counter()
        doc_ids = [str(self.idx.doc_ids[int(d)]) for d in local_ids]
        if not timing:
            return doc_ids
        return doc_ids, {
            "prepare_ms": (t1 - t0) * 1000.0,
            "rank_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
            "route_size": 0 if packet is None else len(packet["route_docs"]),
            "pool_size": 0 if packet is None else len(packet["docs"]),
        }

    def evaluate(self, dataset, k: int = 10):
        run = {}
        times = []
        for qid in dataset.qrels:
            docs, timing = self.search(dataset.queries[qid], k=k, timing=True)
            run[str(qid)] = docs
            times.append(timing)
        metrics = evaluate_run(run, dataset.qrels, ks=(10,), ndcg_k=10, mrr_k=10, exp_gain=False)
        metrics.update({
            "median_total_ms": float(np.median([x["total_ms"] for x in times])),
            "p95_total_ms": float(np.percentile([x["total_ms"] for x in times], 95)),
            "median_rank_ms": float(np.median([x["rank_ms"] for x in times])),
            "median_route_size": float(np.median([x["route_size"] for x in times])),
            "median_pool_size": float(np.median([x["pool_size"] for x in times])),
        })
        return metrics, run
