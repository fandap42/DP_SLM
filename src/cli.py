"""
Command-line interface (CLI) entry points for the diplomka-slm monorepo.
Can be invoked directly or via package console scripts:
  - slm-run
  - slm-analyze
  - slm-demo
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_project_in_path() -> Path:
    root = get_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scripts_dir = root / "experiments" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return root


def run_main():
    root = _ensure_project_in_path()
    from experiments.scripts.run_experiments import run_pipeline
    parser = argparse.ArgumentParser(description="Run SLM Toki Pona experiments")
    parser.add_argument("--config", type=str, default="experiments/configs/poc_quick.json", help="Path to JSON config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    run_pipeline(cfg_path, root)


def analyze_main():
    root = _ensure_project_in_path()
    from experiments.scripts.analyze_results import run_statistical_analysis
    run_statistical_analysis(
        root / "results" / "metrics" / "results.csv",
        root / "results" / "metrics",
        root / "results" / "figures",
    )


def demo_main():
    root = _ensure_project_in_path()
    from experiments.scripts.demo_inference import load_model_and_tokenizer, generate_text
    parser = argparse.ArgumentParser(description="Toki Pona SLM Inference")
    parser.add_argument("--run", type=str, default="exp_012_d75_mini_word_s42")
    parser.add_argument("--prompt", type=str, default="jan pona mi li")
    args = parser.parse_args()

    model, tokenizer, device, meta = load_model_and_tokenizer(args.run, root)
    print(f"\nID: {meta['exp_id']} | Tokenizer: {meta['tokenizer'].upper()} | Val BPC: {meta['val_bpc']}")
    print(f"Prompt: '{args.prompt}'")
    for i in range(3):
        out = generate_text(model, tokenizer, args.prompt, device, max_new_tokens=20)
        print(f"  Sample {i+1}: {out}")


def pack_main():
    root = _ensure_project_in_path()
    from experiments.scripts.pack_release_assets import main as pack_assets
    pack_assets()


if __name__ == "__main__":
    run_main()

