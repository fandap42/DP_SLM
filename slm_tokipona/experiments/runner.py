"""
Factorial Experiment Runner for SLM Toki Pona.
Orchestrates training across Data Sizes x Model Sizes x Tokenizers x Seeds.
Saves logs, checkpoints, and consolidated results for statistical analysis.
"""

from __future__ import annotations
import gc
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from ..corpus.dataset import create_dataloader
from ..models.configs import TransformerConfig
from ..models.transformer import TokiPonaTransformer
from ..tokenizers.base import BaseTokenizer
from ..tokenizers.char_tokenizer import CharTokenizer
from ..tokenizers.word_tokenizer import WordTokenizer
from ..tokenizers.bpe_tokenizer import BPETokenizer
from ..training.trainer import SLMTrainer, TrainingConfig


@dataclass
class ExperimentSpec:
    exp_id: str
    data_fraction: float
    model_size: str
    tokenizer_type: str
    seed: int
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentRunner:
    def __init__(
        self,
        data_dir: Path | str,
        output_dir: Path | str,
        corpus_meta: Dict,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_meta = corpus_meta

        # Cache trained tokenizers
        self.tokenizers: Dict[str, BaseTokenizer] = {}

    def init_tokenizers(self, train_samples: List[str]):
        print("[Runner] Initializing and training tokenizers on corpus...")
        self.tokenizers["char"] = CharTokenizer(train_samples)
        self.tokenizers["word"] = WordTokenizer(train_samples)
        self.tokenizers["bpe"] = BPETokenizer(train_samples, target_vocab_size=80)

        print(f"  - Char Vocab Size: {self.tokenizers['char'].vocab_size}")
        print(f"  - Word Vocab Size: {self.tokenizers['word'].vocab_size}")
        print(f"  - BPE  Vocab Size: {self.tokenizers['bpe'].vocab_size}")

        # Save tokenizers
        tok_dir = self.output_dir / "tokenizers"
        tok_dir.mkdir(exist_ok=True)
        self.tokenizers["char"].save(str(tok_dir / "char_tok.json"))
        self.tokenizers["word"].save(str(tok_dir / "word_tok.json"))
        self.tokenizers["bpe"].save(str(tok_dir / "bpe_tok.json"))

    def run_single_experiment(self, spec: ExperimentSpec) -> Dict[str, any]:
        print(f"\n=================================================================")
        print(f"  RUNNING EXP: {spec.exp_id}")
        print(f"  Data: {spec.data_fraction*100:.0f}% | Model: {spec.model_size} | "
              f"Tokenizer: {spec.tokenizer_type} | Seed: {spec.seed} | Epochs: {spec.epochs}")
        print(f"=================================================================")

        set_seed(spec.seed)
        exp_dir = self.output_dir / "runs" / spec.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = self.tokenizers[spec.tokenizer_type]

        # Calculate estimated data words
        total_train_words = self.corpus_meta["train"]["total_words"]
        data_words = int(total_train_words * spec.data_fraction)

        # Create DataLoaders
        train_loader = create_dataloader(
            data_path=self.data_dir / "train.jsonl",
            tokenizer=tokenizer,
            batch_size=spec.batch_size,
            fraction=spec.data_fraction,
            seed=spec.seed,
            shuffle=True,
        )

        val_loader = create_dataloader(
            data_path=self.data_dir / "val.jsonl",
            tokenizer=tokenizer,
            batch_size=spec.batch_size,
            fraction=0.15,
            shuffle=False,
        )

        # Model instantiation
        cfg = TransformerConfig.get_preset(
            spec.model_size,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=128,
        )
        model = TokiPonaTransformer(cfg)
        param_counts = model.get_parameter_counts()

        train_cfg = TrainingConfig(
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            max_epochs=spec.epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        trainer = SLMTrainer(
            model=model,
            tokenizer=tokenizer,
            config=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=exp_dir,
            seed=spec.seed,
        )

        train_summary = trainer.train()

        result_row = {
            "exp_id": spec.exp_id,
            "data_fraction": spec.data_fraction,
            "data_words": data_words,
            "model_size": spec.model_size,
            "tokenizer": spec.tokenizer_type,
            "seed": spec.seed,
            "epochs": spec.epochs,
            "vocab_size": tokenizer.vocab_size,
            "total_params": param_counts["total_params"],
            "non_embedding_params": param_counts["non_embedding_params"],
            "embedding_params": param_counts["embedding_params"],
            "val_bpc": train_summary["best_val_bpc"],
            "final_val_bpc": train_summary["final_val_bpc"],
            "grammar_acc": train_summary["final_grammar_acc"],
            "total_training_time_sec": train_summary["total_training_time_sec"],
            "total_tokens_seen": train_summary["total_tokens_seen"],
        }

        # Clean GPU memory between runs
        del model, trainer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result_row

    def run_grid(self, specs: List[ExperimentSpec]) -> pd.DataFrame:
        results = []
        res_file = self.output_dir / "results.csv"

        print(f"[Runner] Launching factorial execution of {len(specs)} experiment runs...")
        start_all = time.time()

        for idx, spec in enumerate(specs, 1):
            print(f"\n>>> Progress: [{idx}/{len(specs)}] <<<")
            row = self.run_single_experiment(spec)
            results.append(row)

            # Persist checkpoint results after every run
            df_current = pd.DataFrame(results)
            df_current.to_csv(res_file, index=False)
            with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        total_elapsed = time.time() - start_all
        print(f"\n[Runner] Grid completed in {total_elapsed/60:.2f} minutes!")
        return pd.DataFrame(results)
