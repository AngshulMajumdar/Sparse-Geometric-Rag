# Repository manifest

- `README.md` — project front page: architecture, complexity, low-cost CPU positioning, full benchmark suite, and reproducibility guidance.
- `figures/` — ten publication-quality architecture, complexity, deployment, storage, repository, shortlist, and benchmark figures used by the README.
- `geomretrieval/` — reusable sparse geometric index and frozen top-10 RAG ranker.
- `experiments/beir/` — generic BEIR runners and exact SciFact/TREC-COVID experiment-history scripts.
- `experiments/msmarco_scale/` — complete full-scale MS MARCO experimental campaign.
- `experiments/postbenchmark_optimizations/` — preserved later optimization branch; not substituted into the frozen six-dataset row.
- `results/final_100_to_10/` — frozen final six-dataset result artifacts.
- `results/beir/` — BEIR sweeps, diagnostics, and local benchmark outputs.
- `results/msmarco_scale/` — full-scale MS MARCO result JSONs.
- `results/reproduced_reference/` — clean-runner reference checks.
- `baselines/` — local lexical reference code and baseline-policy material.
- `configs/` — frozen configuration handles.
- `docs/` — method, RAG protocol, baseline suite, speed/literature, and reproducibility notes.
- `docs/images/` — historical sweep/latency figures referenced by documentation.
- `data/` — data acquisition/use notes; corpora themselves are not bundled.
- `scripts/` — reproduction helpers.
- `manifests/` — corpus/run manifests.
- `tests/` — core and RAG smoke tests.
- `requirements.txt` — core CPU-first dependencies.
- `requirements-baselines.txt` — optional dependencies for baseline reproduction.
- `pyproject.toml` — package metadata.
- `SHA256SUMS.txt` — checksums for the shipped repository files.

No `.gitignore`, `.github/`, `.git/`, `.gitattributes`, caches, or other dot-files/directories are included in this package.
