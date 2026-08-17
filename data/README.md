# Data

Datasets are intentionally not committed to the repository.

The code accepts either an extracted standard BEIR directory or a standard BEIR `.zip` containing:

```text
corpus.jsonl
queries.jsonl
qrels/test.tsv
```

Examples used in the current repository:

```text
data/scifact.zip
data/trec-covid.zip
```

For full MS MARCO scale reproduction, follow `docs/REPRODUCIBILITY.md` and the shard manifest in `manifests/`.
