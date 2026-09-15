"""
Convenience entry point for running SLM experiments.
Delegates to experiments/scripts/run_experiments.py.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from experiments.scripts.run_experiments import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diplomka SLM Runner")
    parser.add_argument("--config", type=str, default="experiments/configs/poc_quick.json")
    args = parser.parse_args()

    run_pipeline(root / args.config, root)
