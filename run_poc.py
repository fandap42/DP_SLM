"""
Master Proof of Concept Runner for Diploma Thesis:
'Small Language Models (SLM) na malých datech: statistická analýza výkonu a škálování'
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch

from slm_tokipona.corpus.downloader import build_and_save_corpus
from slm_tokipona.experiments.runner import ExperimentSpec, ExperimentRunner
from analyze_results import run_statistical_analysis


def main():
    parser = argparse.ArgumentParser(description="SLM Toki Pona Proof-of-Concept Pipeline")
    parser.add_argument("--quick", action="store_true", default=True, help="Run fast PoC grid (default: True)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per run (default: 2)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset rebuild if files exist")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    exp_output_dir = project_root / "experiments" / "outputs"
    figures_dir = project_root / "figures"

    print("=" * 70)
    print("  DIPLOMA THESIS PROOF-OF-CONCEPT:")
    print("  Small Language Models (SLM) na malých datech:")
    print("  statistická analýza výkonu a škálování (Toki Pona)")
    print("=" * 70)

    # 1. Environment verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Setup] Compute Device: {device} ({device_name})")
    print(f"[Setup] PyTorch Version: {torch.__version__}")

    # 2. Corpus preparation
    meta_file = processed_data_dir / "metadata.json"
    if args.skip_data and meta_file.exists():
        print(f"[Corpus] Skipping rebuild; loading metadata from {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        print("\n--- PHASE 1: Corpus Acquisition, Cleaning, and Source-Stratified Split ---")
        meta = build_and_save_corpus(
            raw_dir=raw_data_dir,
            output_dir=processed_data_dir,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            seed=42,
        )

    # Read a sample of train sentences to train tokenizers
    sample_texts = []
    with open(processed_data_dir / "train.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 5000:  # 5,000 sentences is plenty for vocabulary induction in Toki Pona
                break
            sample_texts.append(json.loads(line)["text"])

    # 3. Initialize Runner and Tokenizers
    print("\n--- PHASE 2: Tokenization Induction (Char, BPE, Word) ---")
    runner = ExperimentRunner(
        data_dir=processed_data_dir,
        output_dir=exp_output_dir,
        corpus_meta=meta,
    )
    runner.init_tokenizers(sample_texts)

    # 4. Define Factorial Experiment Grid (DoE)
    print("\n--- PHASE 3: Factorial Experiment Design (DoE Matrix) ---")
    if args.quick:
        # Fast PoC grid: 2 data sizes x 2 models x 3 tokenizers x 1 seed = 12 runs
        # Designed to finish quickly and deliver rich statistical variance
        data_fractions = [0.25, 0.75]
        model_sizes = ["micro", "mini"]
        tokenizers = ["char", "bpe", "word"]
        seeds = [42]
        epochs = args.epochs
    else:
        # Full grid
        data_fractions = [0.10, 0.25, 0.50, 1.0]
        model_sizes = ["micro", "mini", "small"]
        tokenizers = ["char", "bpe", "word"]
        seeds = [42, 1337]
        epochs = args.epochs

    specs = []
    run_idx = 1
    for df in data_fractions:
        for ms in model_sizes:
            for tok in tokenizers:
                for s in seeds:
                    exp_id = f"exp_{run_idx:03d}_d{int(df*100):02d}_{ms}_{tok}_s{s}"
                    specs.append(
                        ExperimentSpec(
                            exp_id=exp_id,
                            data_fraction=df,
                            model_size=ms,
                            tokenizer_type=tok,
                            seed=s,
                            epochs=epochs,
                            batch_size=args.batch_size,
                            learning_rate=5e-4,
                        )
                    )
                    run_idx += 1

    print(f"[DoE] Total experimental runs in design matrix: {len(specs)}")
    print(f"  - Data Fractions: {data_fractions}")
    print(f"  - Model Sizes:    {model_sizes}")
    print(f"  - Tokenizers:     {tokenizers}")
    print(f"  - Seeds:          {seeds}")

    # 5. Run Factorial Grid
    print("\n--- PHASE 4: Factorial Training & Multi-Metric Evaluation ---")
    results_df = runner.run_grid(specs)

    # 6. Statistical Modeling, Scaling Laws & Publication Visualization
    print("\n--- PHASE 5: Statistical Analysis & Scaling Law Estimation ---")
    run_statistical_analysis(
        results_csv=exp_output_dir / "results.csv",
        output_dir=exp_output_dir,
        figures_dir=figures_dir,
    )

    print("\n" + "=" * 70)
    print("  PROOF-OF-CONCEPT EXECUTION FINISHED SUCCESSFULLY!")
    print(f"  - Results CSV:     {exp_output_dir / 'results.csv'}")
    print(f"  - Stats Report:    {exp_output_dir / 'statistical_report.md'}")
    print(f"  - Figures Saved:   {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
