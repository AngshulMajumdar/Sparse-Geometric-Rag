# Post-benchmark optimization branch

`final_two_optimizations_ARCHIVE.py` is the exact recovered source supplied after the frozen six-dataset `P=100 -> top-10` benchmark was produced.

It is preserved for provenance, not used to generate `RESULTS_FINAL_6DATASETS.md`. Its original one-off scale harness imported `eval_scale_rag` and `opt_two_variants` from an experiment working directory; those two transient modules did not survive as standalone files. The reusable/frozen benchmark implementation is therefore the package code in `geomretrieval/`, especially `geomretrieval/rag_top10.py`, plus the runners in `experiments/beir/` and the full-scale scripts in `experiments/msmarco_scale/`.

The recovered optimization branch introduced:

1. O(K) generation-stamp aggregation instead of `np.unique` sorting.
2. A pre-lexical tail gate `min(route_docs, max(10000, 40*P))`.
3. Query-conditioned 16-coordinate residual-code sign matching with default weight `wcode=0.25`.

Recorded post-benchmark outputs are stored in `results/final_100_to_10/nq_two_optimizations_merged.json` and `hotpotqa_two_optimizations_merged.json`.
