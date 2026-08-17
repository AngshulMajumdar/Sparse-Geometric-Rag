#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-data/scifact.zip}
INDEX=${2:-artifacts/scifact_index}
mkdir -p "$(dirname "$INDEX")" results/reproduced
python -m geomretrieval.cli build "$DATA" "$INDEX" --split test
python experiments/beir/run_pool_sweep.py "$DATA" "$INDEX" --split test --pools 25 50 100 200 500 --output results/reproduced/scifact_pool_sweep.json
