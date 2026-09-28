"""
Command-line interface (CLI) entry points for the diplomka-slm monorepo.
Can be invoked directly or via package console scripts:
  - slm-run
  - slm-analyze
  - slm-eval
  - slm-generate-tests
  - slm-demo
  - slm-chat
  - slm-pack
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


def eval_main():
    root = _ensure_project_in_path()
    from src.evaluation.evaluator import UnifiedEvaluator
    parser = argparse.ArgumentParser(description="Unified SLM Evaluator across all 4 thesis pillars")
    parser.add_argument("--run", type=str, default=None, help="Name of specific run to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all model checkpoints in results/models")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for test evaluation")
    args = parser.parse_args()

    evaluator = UnifiedEvaluator(root)
    if args.run:
        evaluator.evaluate_checkpoint(args.run, batch_size=args.batch_size)
    else:
        evaluator.evaluate_all_runs()


def generate_tests_main():
    root = _ensure_project_in_path()
    from src.evaluation.grammar_generator import generate_all_grammar_pairs, save_grammar_suite
    from src.evaluation.compositional import save_compositional_suite
    target_dir = root / "data" / "evaluation"
    print(f"[CLI] Generating synthetic test suites into {target_dir}...")
    save_grammar_suite(generate_all_grammar_pairs(), target_dir)
    save_compositional_suite(target_dir)
    print("[CLI] All test suites successfully generated!")


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


def chat_main():
    root = _ensure_project_in_path()
    from experiments.scripts.chat import main as chat_cli
    chat_cli()


def pack_main():
    root = _ensure_project_in_path()
    from experiments.scripts.pack_release_assets import main as pack_assets
    pack_assets()


if __name__ == "__main__":
    run_main()
