from __future__ import annotations
import math
import numpy as np


def _dcg(rels: list[float], exp_gain: bool = True) -> float:
    total = 0.0
    for rank, rel in enumerate(rels, start=1):
        gain = (2.0 ** rel - 1.0) if exp_gain else rel
        total += gain / math.log2(rank + 1.0)
    return total


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    ks: tuple[int, ...] = (10, 100),
    ndcg_k: int = 10,
    mrr_k: int = 10,
    exp_gain: bool = True,
) -> dict[str, float]:
    """Binary top-K metrics plus graded nDCG.

    'Accuracy' for retrieval is reported as Hit@K rather than ordinary
    classification accuracy, which is meaningless with millions of negatives.
    """
    qids = [q for q in qrels if q in run]
    if not qids:
        raise ValueError("No query IDs overlap between run and qrels")

    vals: dict[str, list[float]] = {}
    for k in ks:
        vals[f"P@{k}"] = []
        vals[f"R@{k}"] = []
        vals[f"Hit@{k}"] = []
    vals[f"MRR@{mrr_k}"] = []
    vals[f"nDCG@{ndcg_k}"] = []

    for qid in qids:
        ranked = run[qid]
        qr = qrels[qid]
        positive = {d for d, r in qr.items() if r > 0}
        npos = max(1, len(positive))

        for k in ks:
            top = ranked[:k]
            hits = sum(1 for d in top if d in positive)
            vals[f"P@{k}"].append(hits / float(k))
            vals[f"R@{k}"].append(hits / float(npos))
            vals[f"Hit@{k}"].append(float(hits > 0))

        rr = 0.0
        for rank, d in enumerate(ranked[:mrr_k], start=1):
            if d in positive:
                rr = 1.0 / rank
                break
        vals[f"MRR@{mrr_k}"].append(rr)

        observed = [float(qr.get(d, 0.0)) for d in ranked[:ndcg_k]]
        ideal = sorted((float(r) for r in qr.values()), reverse=True)[:ndcg_k]
        idcg = _dcg(ideal, exp_gain=exp_gain)
        vals[f"nDCG@{ndcg_k}"].append(_dcg(observed, exp_gain=exp_gain) / idcg if idcg > 0 else 0.0)

    out = {k: float(np.mean(v)) for k, v in vals.items()}
    out["n_queries"] = float(len(qids))
    return out
