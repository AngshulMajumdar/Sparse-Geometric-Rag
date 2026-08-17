# Practical RAG evaluation protocol: measure the first ten chunks

## Why K=10 is the deployment target

A RAG system does not consume an abstract recall curve. It consumes a small context set. Chunks ranked at 100, 500, or 1000 are normally never passed to the generator. Optimizing deep recall can therefore reward a retriever for spending CPU, memory bandwidth, and storage on results that have zero downstream utility.

For this repository, the primary protocol is deliberately strict:

- retrieve a ranked **top 10**;
- report **nDCG@10, MRR@10, Precision@10, Recall@10, Hit@10**;
- measure **CPU median latency, p95 latency, and QPS**;
- report the number of routed candidates and the chunk shortlist size `P`;
- use deep-recall metrics only as diagnostics, never as the main optimization objective.

This is the only protocol in the repository used to decide whether a change helps practical RAG.

## Why shortlist size P is a RAG parameter

The historical system used `P=2000` because it was selected for Recall@100. Once the objective became top-10 RAG quality, the relevant question changed to:

> How many retrieved geometric representations should be exposed to chunk-level scoring before choosing ten chunks?

The current sweep is therefore `P in {25, 50, 100, 200, 500}` plus the historical `P=2000` reference. TREC-COVID has a large route and peaks sharply at `P=100`; SciFact has a tiny route (~157 candidates/query) and improves as the artificial pruning is removed. The lesson is not that 100 is universal. The lesson is that **the shortlist must be evaluated for the top-10 deployment objective rather than inherited from a deep-recall benchmark.**

## Timing discipline

Speed is a first-class result.

1. CPU is the principal deployment regime.
2. Warm the index before timing.
3. Record median and p95, not only an average.
4. Report query representation time separately when a baseline uses a neural encoder.
5. Never compare our end-to-end latency with an ANN-only latency that silently excludes dense query encoding.
6. Mark diagnostic Python implementations as diagnostic; do not present them as optimized production latency.
