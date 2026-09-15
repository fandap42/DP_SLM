"""
Trainer module for SLM experiments on Toki Pona.
Supports ROCm GPU acceleration, AMP (Automatic Mixed Precision),
learning rate scheduling, and comprehensive metric logging.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from ..evaluation.metrics import compute_loss_and_bpc
from ..evaluation.grammar_suite import evaluate_grammar_suite
from ..tokenization.base import BaseTokenizer


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_epochs: int = 5
    warmup_steps: int = 50
    grad_clip: float = 1.0
    mixed_precision: bool = True
    eval_every_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class SLMTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        output_dir: Path | str,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(config.warmup_steps, max(1, total_steps // 10)),
            num_training_steps=max(1, total_steps),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.mixed_precision and self.device.type == "cuda"))
        self.history: List[Dict] = []
        self.best_bpc = float("inf")

    def train(self) -> Dict[str, any]:
        start_time = time.time()
        print(f"  [Trainer] Beginning training for {self.config.max_epochs} epochs on {self.device}...")

        step = 0
        total_tokens_seen = 0

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                step += 1
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                valid_tokens = (target_ids != -100).sum().item()
                total_tokens_seen += valid_tokens
                epoch_tokens += valid_tokens

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler.is_enabled():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _, loss = self.model(input_ids, target_ids)
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, loss = self.model(input_ids, target_ids)
                    loss.backward()
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                self.scheduler.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / max(1, len(self.train_loader))
            epoch_duration = time.time() - epoch_start

            # Evaluate
            val_metrics = compute_loss_and_bpc(self.model, self.val_loader, self.device)
            grammar_results = evaluate_grammar_suite(self.model, self.tokenizer, self.device)

            val_bpc = val_metrics["bpc"]
            val_loss = val_metrics["loss_nats"]
            val_ppl_char = val_metrics["ppl_char"]
            grammar_acc = grammar_results["overall_accuracy"]

            is_best = val_bpc < self.best_bpc
            if is_best:
                self.best_bpc = val_bpc
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")

            log_entry = {
                "epoch": epoch,
                "step": step,
                "total_tokens": total_tokens_seen,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_bpc": round(val_bpc, 4),
                "val_ppl_char": round(val_ppl_char, 2),
                "val_ppl_tok": round(val_metrics["ppl_tok"], 2),
                "grammar_acc": round(grammar_acc, 4),
                "grammar_categories": grammar_results["category_accuracies"],
                "epoch_time_sec": round(epoch_duration, 2),
                "lr": round(self.optimizer.param_groups[0]["lr"], 7),
            }
            self.history.append(log_entry)

            print(
                f"  Epoch {epoch:2d}/{self.config.max_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val BPC: {val_bpc:.4f} | "
                f"Val PPL (char): {val_ppl_char:.2f} | "
                f"Grammar: {grammar_acc * 100:.1f}% | "
                f"Time: {epoch_duration:.1f}s"
            )

        total_time = time.time() - start_time
        summary = {
            "total_epochs": self.config.max_epochs,
            "total_steps": step,
            "total_tokens_seen": total_tokens_seen,
            "total_training_time_sec": round(total_time, 2),
            "best_val_bpc": round(self.best_bpc, 4),
            "final_val_bpc": round(self.history[-1]["val_bpc"], 4),
            "final_grammar_acc": round(self.history[-1]["grammar_acc"], 4),
            "history": self.history,
        }

        # Save history log
        import json
        with open(self.output_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
