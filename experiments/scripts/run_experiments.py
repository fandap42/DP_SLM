"""
Orchestration script for Factorial SLM Experiments on Toki Pona.
Can be executed locally (Windows/Linux) or submitted as a SLURM job.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import pandas as pd
import torch

from src.data.dataset import create_dataloader
from src.data.downloader import build_and_save_corpus
from src.tokenization.char_tokenizer import CharTokenizer
from src.tokenization.word_tokenizer import WordTokenizer
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.models.configs import TransformerConfig
from src.models.transformer import TokiPonaTransformer
from src.training.trainer import SLMTrainer, TrainingConfig
from src.evaluation.stats import fit_mixed_effects_model, compute_factorial_anova, fit_scaling_law, fit_tokenizer_scaling_curves


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(config_path: Path | str, project_root: Path | str):
    project_root = Path(project_root)
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "results" / "metrics"
    models_dir = project_root / "results" / "models"
    figures_dir = project_root / "results" / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corpus verification
    meta_file = data_dir / "metadata.json"
    if not meta_file.exists():
        print("[Corpus] metadata.json not found in data/processed, downloading and preparing corpus...")
        meta = build_and_save_corpus(
            raw_dir=project_root / "data" / "raw",
            output_dir=data_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
    else:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 2. Tokenizers
    sample_texts = []
    with open(data_dir / "train.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 5000: break
            sample_texts.append(json.loads(line)["text"])

    print("[Tokenizers] Initializing Char, BPE, and Word tokenizers...")
    tokenizers = {
        "char": CharTokenizer(sample_texts),
        "word": WordTokenizer(sample_texts),
        "bpe": BPETokenizer(sample_texts, target_vocab_size=80),
    }

    tok_dir = metrics_dir / "tokenizers"
    tok_dir.mkdir(exist_ok=True)
    tokenizers["char"].save(str(tok_dir / "char_tok.json"))
    tokenizers["word"].save(str(tok_dir / "word_tok.json"))
    tokenizers["bpe"].save(str(tok_dir / "bpe_tok.json"))

    # 3. Factorial execution
    data_fractions = cfg["data_fractions"]
    model_sizes = cfg["model_sizes"]
    tok_types = cfg["tokenizers"]
    seeds = cfg["seeds"]
    epochs = cfg.get("epochs", 2)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("learning_rate", 5e-4)

    total_runs = len(data_fractions) * len(model_sizes) * len(tok_types) * len(seeds)
    print(f"\n[DoE] Factorial Experiment: {total_runs} total runs planned from config '{config_path.name}'")

    results = []
    run_idx = 1
    total_train_words = meta["train"]["total_words"]

    for df in data_fractions:
        for ms in model_sizes:
            for tok_type in tok_types:
                for s in seeds:
                    exp_id = f"exp_{run_idx:03d}_d{int(df*100):02d}_{ms}_{tok_type}_s{s}"
                    print(f"\n--- [{run_idx}/{total_runs}] Running {exp_id} ---")
                    set_seed(s)

                    tokenizer = tokenizers[tok_type]
                    data_words = int(total_train_words * df)

                    train_loader = create_dataloader(
                        data_path=data_dir / "train.jsonl",
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        fraction=df,
                        seed=s,
                        shuffle=True,
                    )
                    val_loader = create_dataloader(
                        data_path=data_dir / "val.jsonl",
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        fraction=0.15,
                        shuffle=False,
                    )

                    model_cfg = TransformerConfig.get_preset(ms, vocab_size=tokenizer.vocab_size)
                    model = TokiPonaTransformer(model_cfg)
                    param_counts = model.get_parameter_counts()

                    train_cfg = TrainingConfig(
                        batch_size=batch_size,
                        learning_rate=lr,
                        max_epochs=epochs,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    run_save_dir = models_dir / exp_id
                    trainer = SLMTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        config=train_cfg,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        output_dir=run_save_dir,
                        seed=s,
                    )
                    summary = trainer.train()

                    # Save lightweight training_log into metrics
                    metrics_runs_dir = metrics_dir / "runs" / exp_id
                    metrics_runs_dir.mkdir(parents=True, exist_ok=True)
                    with open(metrics_runs_dir / "training_log.json", "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                    row = {
                        "exp_id": exp_id,
                        "data_fraction": df,
                        "data_words": data_words,
                        "model_size": ms,
                        "tokenizer": tok_type,
                        "seed": s,
                        "epochs": epochs,
                        "vocab_size": tokenizer.vocab_size,
                        "total_params": param_counts["total_params"],
                        "non_embedding_params": param_counts["non_embedding_params"],
                        "embedding_params": param_counts["embedding_params"],
                        "val_bpc": summary["best_val_bpc"],
                        "final_val_bpc": summary["final_val_bpc"],
                        "grammar_acc": summary["final_grammar_acc"],
                        "total_training_time_sec": summary["total_training_time_sec"],
                        "total_tokens_seen": summary["total_tokens_seen"],
                    }
                    results.append(row)

                    # Update results.csv
                    df_current = pd.DataFrame(results)
                    df_current.to_csv(metrics_dir / "results.csv", index=False)
                    with open(metrics_dir / "results.json", "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)

                    del model, trainer, train_loader, val_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    run_idx += 1

    print("\n[Analysis] Generating statistical analysis and figures...")
    try:
        from experiments.scripts.analyze_results import run_statistical_analysis
    except ModuleNotFoundError:
        from analyze_results import run_statistical_analysis
    run_statistical_analysis(
        results_csv=metrics_dir / "results.csv",
        output_dir=metrics_dir,
        figures_dir=figures_dir,
    )
    print("\n[Done] Pipeline execution successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/poc_quick.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(project_root / args.config, project_root)
