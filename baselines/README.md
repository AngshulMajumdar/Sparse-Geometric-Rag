# Baselines

The result tables keep all baseline families used during development: TF-IDF, BM25, MiniLM/FAISS, Contriever, SPLADE++, BGE, ColBERT, and the historical version of OURS.

`run_local_lexical.py` reproduces local TF-IDF and an effectiveness-oriented BM25 sanity check. Its BM25 latency is **not** a production inverted-index benchmark and is labelled accordingly.

For neural/vector baselines, use the authors' released models and index libraries. Their effectiveness values in `results/beir/rag_top10_pool_sweep_scifact_treccovid.json` are retained as the comparison ledger from the experiment campaign. Before a publication-quality speed claim, rerun every baseline on the same CPU and report both:

- query encoder time;
- index/search time;
- end-to-end query time.

Do not compare an end-to-end sparse retrieval number with ANN-only latency that excludes query embedding inference.
