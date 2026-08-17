# Baseline notes — full MS MARCO / TREC-DL19

`run_bm25_test_streaming_effectiveness.py` is a local effectiveness sanity check over all 8,841,823 passages. It is deliberately **not** reported as an indexed BM25 latency measurement because it scans the corpus.

`BASELINE_COMPARISON.json` records the locally measured frozen-system result and published same-test reference rows. Published rows should be cited to NIST/Pyserini/the original dense-retrieval papers in manuscripts.

TREC-DL19 passage judgments require two evaluation conventions:
- `nDCG@10`: retain graded qrel values (trec_eval `ndcg_cut` uses raw relevance values as gains);
- binary MRR/P/Recall/Hit: grade 1 is only *Related*, not relevant, therefore use relevance >= 2.

The local frozen-system evaluator in `../scale_experiments/msmarco_test_baseline_compare.py` returns top 1000 and applies the correct binary threshold.
