# Reproducibility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional ANN/neural baselines:

```bash
pip install -r requirements-baselines.txt
```

## SciFact

Place a standard BEIR archive at `data/scifact.zip` and run:

```bash
./scripts/reproduce_scifact.sh data/scifact.zip artifacts/scifact_index
```

## TREC-COVID

Place a standard BEIR archive at `data/trec-covid.zip` and run:

```bash
./scripts/reproduce_treccovid.sh data/trec-covid.zip artifacts/treccovid_index
```

## One configuration

```bash
python experiments/beir/run_rag_top10.py   data/trec-covid.zip artifacts/treccovid_index   --pool 100 --hq-branches 10 --lambda-diversity 0.1   --output results/reproduced/treccovid_p100.json
```

## Full MS MARCO

The exact historical full-scale scripts are preserved under `experiments/msmarco_scale/`. They are intentionally kept close to the scripts that produced the recorded JSON files. The 8.84M corpus shards and multi-GB generated arrays are not in this repository. Use `manifests/msmarco_manifest.json` to verify the shard set, then follow the build order described in the root README.

## Exact experiment history vs cleaned runner

`experiments/beir/*exact_history.py` contains the scripts as executed in the current session, including their original local paths. `experiments/beir/run_rag_top10.py` and `run_pool_sweep.py` are cleaned path-independent runners using the same formulas.
