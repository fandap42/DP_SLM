#!/usr/bin/env bash
CONFIG=${1:-"experiments/configs/poc_quick.json"}
echo "Running SLM Toki Pona Experiments with config: $CONFIG"
source .venv/bin/activate 2>/dev/null || true
python experiments/scripts/run_experiments.py --config "$CONFIG"
