from geomretrieval import FrozenConfig, GeometricIndex, evaluate_run


def test_toy_build_and_search():
    docs = [
        "car automobile engine road vehicle",
        "automobile vehicle insurance motor road",
        "river bank flood erosion water",
        "bank account credit loan interest",
        "vitamin d respiratory infection clinical study",
        "random unrelated astronomy galaxy star",
    ]
    ids = [f"d{i}" for i in range(len(docs))]
    # Small toy corpus cannot support the production widths; keep the same
    # architecture while mechanically reducing vocabulary-dependent dimensions.
    cfg = FrozenConfig(
        max_features=100,
        F=2,
        B=8,
        S=4,
        L=4,
        assoc_k=6,
        route_k=4,
        route_budget=6,
        rerank_pool=5,
        semantic_k=3,
        output_k=5,
    )
    idx = GeometricIndex.build(docs, ids, cfg, verbose=False)
    out = idx.search("automobile road insurance", k=3)
    assert 1 <= len(out) <= 3
    assert out[0] in {"d0", "d1"}

    run = {"q1": out}
    qrels = {"q1": {"d0": 1.0, "d1": 1.0}}
    m = evaluate_run(run, qrels, ks=(1, 3), ndcg_k=3, mrr_k=3)
    assert m["Hit@1"] == 1.0
    assert m["MRR@3"] == 1.0
