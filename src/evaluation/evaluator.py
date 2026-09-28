"""
Unified Evaluation Framework for Toki Pona SLM Models.
Evaluates model checkpoints across all 4 thesis pillars:
1. Standard test set BPC, Character-normalized loss, and Domain Split (In-domain vs Out-of-domain)
2. Synthetic grammatical acceptability minimal pairs (overall and per rule category)
3. Compositional generalization (held-out compounds vs seen compounds and Generalization Gap)
4. Training dynamics, architectural parameter counts, and FLOPs estimation
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch

from ..models.configs import TransformerConfig
from ..models.transformer import TokiPonaTransformer
from ..tokenization import load_tokenizer
from ..tokenization.char_tokenizer import CharTokenizer
from ..tokenization.word_tokenizer import WordTokenizer
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..data.dataset import create_dataloader
from .metrics import compute_loss_and_bpc
from .grammar_suite import evaluate_grammar_suite, get_default_test_suite
from .compositional import evaluate_compositional_suite


class UnifiedEvaluator:
    def __init__(self, project_root: Optional[Path | str] = None, device: Optional[str] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data" / "processed"
        self.metrics_dir = self.project_root / "results" / "metrics"
        self.models_dir = self.project_root / "results" / "models"
        self.eval_data_dir = self.project_root / "data" / "evaluation"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.test_data_path = self.data_dir / "test.jsonl"
        self.grammar_test_suite = get_default_test_suite(self.project_root)

    def load_tokenizer_for_run(self, tok_type: str):
        tok_path = self.metrics_dir / "tokenizers" / f"{tok_type}_tok.json"
        if tok_path.exists():
            return load_tokenizer(str(tok_path))

        # Fallback to sample training
        train_file = self.data_dir / "train.jsonl"
        sample_texts = []
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 5000:
                        break
                    sample_texts.append(json.loads(line)["text"])

        if tok_type == "char":
            return CharTokenizer(sample_texts)
        elif tok_type == "word":
            return WordTokenizer(sample_texts)
        elif tok_type == "bpe":
            return BPETokenizer(sample_texts, target_vocab_size=80)
        else:
            raise ValueError(f"Unknown tokenizer type: {tok_type}")

    def load_checkpoint(self, exp_id: str, model_size: str, tokenizer) -> TokiPonaTransformer:
        cfg = TransformerConfig.get_preset(model_size, vocab_size=tokenizer.vocab_size)
        model = TokiPonaTransformer(cfg).to(self.device)

        candidate_paths = [
            self.models_dir / exp_id / "best_model.pt",
            self.metrics_dir / "runs" / exp_id / "best_model.pt",
            self.models_dir / "best_model.pt",
        ]
        weights_path = next((p for p in candidate_paths if p.exists()), None)
        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(f"  [Warning] Checkpoint weights for {exp_id} not found. Running initialized model.")

        model.eval()
        return model

    def evaluate_checkpoint(self, exp_id: str, batch_size: int = 64) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation for a single model run.
        """
        print(f"\n[Evaluator] Evaluating run '{exp_id}' on {self.device}...")

        # Parse run configuration from name or training_log
        training_log_path = self.models_dir / exp_id / "training_log.json"
        if not training_log_path.exists():
            training_log_path = self.metrics_dir / "runs" / exp_id / "training_log.json"

        training_log = {}
        if training_log_path.exists():
            with open(training_log_path, "r", encoding="utf-8") as f:
                training_log = json.load(f)

        # Look up existing row in results.csv if present
        csv_path = self.metrics_dir / "results.csv"
        prev_row = {}
        if csv_path.exists():
            try:
                df_prev = pd.read_csv(csv_path)
                match = df_prev[df_prev["exp_id"] == exp_id]
                if not match.empty:
                    prev_row = match.iloc[0].to_dict()
            except Exception:
                pass

        # Infer config
        tok_type = prev_row.get("tokenizer")
        model_size = prev_row.get("model_size")
        data_fraction = prev_row.get("data_fraction")
        seed = prev_row.get("seed", 42)

        # Parse from exp_id if missing: exp_001_d25_micro_char_s42
        parts = exp_id.split("_")
        if tok_type is None:
            for p in ["char", "word", "bpe"]:
                if p in parts:
                    tok_type = p
                    break
            tok_type = tok_type or "word"

        if model_size is None:
            for p in ["micro", "mini", "small", "medium", "large"]:
                if p in parts:
                    model_size = p
                    break
            model_size = model_size or "mini"

        if data_fraction is None:
            for p in parts:
                if p.startswith("d") and p[1:].isdigit():
                    data_fraction = float(p[1:]) / 100.0
                    break
            data_fraction = data_fraction or 1.0

        if seed == 42:
            for p in parts:
                if p.startswith("s") and p[1:].isdigit():
                    seed = int(p[1:])
                    break

        tokenizer = self.load_tokenizer_for_run(tok_type)
        model = self.load_checkpoint(exp_id, model_size, tokenizer)
        cfg = model.config
        param_counts = model.get_parameter_counts()

        # 1. Pillar 1: Full Test Set & Domain Split
        test_metrics = {}
        if self.test_data_path.exists():
            test_loader = create_dataloader(
                data_path=self.test_data_path,
                tokenizer=tokenizer,
                batch_size=batch_size,
                shuffle=False,
            )
            test_metrics = compute_loss_and_bpc(model, test_loader, self.device, in_domain_sources=("tatoeba",))

        # 2. Pillar 2: Automatically Generated Grammar Suite
        grammar_metrics = evaluate_grammar_suite(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            test_suite=self.grammar_test_suite,
        )

        # 3. Pillar 3: Compositional Generalization Suite
        comp_metrics = evaluate_compositional_suite(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            data_dir=self.eval_data_dir,
        )

        # 4. Pillar 4: Training Dynamics & Compute Estimates
        history = training_log.get("history", [])
        final_epoch_entry = history[-1] if history else {}
        first_epoch_entry = history[0] if history else {}

        train_loss = final_epoch_entry.get("train_loss", prev_row.get("train_loss", None))
        val_loss = final_epoch_entry.get("val_loss", prev_row.get("val_loss", None))
        val_bpc = final_epoch_entry.get("val_bpc", prev_row.get("val_bpc", None))
        overfitting_gap = round(val_loss - train_loss, 4) if (val_loss is not None and train_loss is not None) else None

        total_tokens_seen = training_log.get(
            "total_tokens_seen",
            prev_row.get("total_tokens_seen", 0)
        )
        total_training_time_sec = training_log.get(
            "total_training_time_sec",
            prev_row.get("total_training_time_sec", 0.0)
        )

        # Standard forward+backward FLOPs estimate: ~6 * N_non_emb * tokens_seen
        non_emb = param_counts["non_embedding_params"]
        flops_est = 6 * non_emb * total_tokens_seen if total_tokens_seen > 0 else 0

        # Construct comprehensive run record
        rec = {
            # Independent Variables (Factors)
            "exp_id": exp_id,
            "tokenizer": tok_type,
            "model_size": model_size,
            "data_fraction": float(data_fraction),
            "data_words": int(prev_row.get("data_words", int(1415500 * float(data_fraction)))),
            "seed": int(seed),
            "vocab_size": tokenizer.vocab_size,
            "total_params": param_counts["total_params"],
            "non_embedding_params": non_emb,
            "embedding_params": param_counts["embedding_params"],
            "d_model": cfg.d_model,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "d_ff": cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
            "total_tokens_seen": total_tokens_seen,
            "flops_est": flops_est,
            "total_training_time_sec": total_training_time_sec,

            # Pillar 1: Language Modeling Performance & Domain Generalization
            "test_bpc": test_metrics.get("bpc", None),
            "test_loss_char": test_metrics.get("loss_char", None),
            "test_loss_nats": test_metrics.get("loss_nats", None),
            "test_ppl_tok": test_metrics.get("ppl_tok", None),
            "test_ppl_char": test_metrics.get("ppl_char", None),
            "bpc_in_domain": test_metrics.get("bpc_in_domain", None),
            "bpc_out_domain": test_metrics.get("bpc_out_domain", None),
            "ppl_tok_in": test_metrics.get("ppl_tok_in", None),
            "ppl_char_in": test_metrics.get("ppl_char_in", None),
            "ppl_tok_out": test_metrics.get("ppl_tok_out", None),
            "ppl_char_out": test_metrics.get("ppl_char_out", None),
            "domain_gap": test_metrics.get("domain_gap", None),
            "domain_ratio": test_metrics.get("domain_ratio", None),

            # Per-source BPCs
            "bpc_tatoeba": test_metrics.get("sources", {}).get("tatoeba", {}).get("bpc", None),
            "bpc_wikipesija": test_metrics.get("sources", {}).get("wikipesija", {}).get("bpc", None),
            "bpc_poki_lapo": test_metrics.get("sources", {}).get("poki_lapo", {}).get("bpc", None),
            "bpc_lipu_sewi": test_metrics.get("sources", {}).get("lipu_sewi", {}).get("bpc", None),

            # Pillar 2: Grammatical Acceptability (BLiMP minimal pairs)
            "grammar_acc": grammar_metrics["overall_accuracy"],
            "acc_particle_li": grammar_metrics["category_accuracies"].get("particle_li", None),
            "acc_direct_object_e": grammar_metrics["category_accuracies"].get("direct_object_e", None),
            "acc_modifier_order": grammar_metrics["category_accuracies"].get("modifier_order", None),
            "acc_modifier_pi": grammar_metrics["category_accuracies"].get("modifier_pi", None),
            "acc_context_la": grammar_metrics["category_accuracies"].get("context_la", None),
            "acc_lexical_validity": grammar_metrics["category_accuracies"].get("lexical_validity", None),

            # Pillar 3: Compositional Generalization
            "comp_bpc_heldout": comp_metrics["comp_bpc_heldout"],
            "comp_bpc_seen": comp_metrics["comp_bpc_seen"],
            "comp_ppl_heldout": comp_metrics["comp_ppl_heldout"],
            "comp_ppl_seen": comp_metrics["comp_ppl_seen"],
            "comp_generalization_gap": comp_metrics["comp_generalization_gap"],
            "comp_gap_ratio": comp_metrics["comp_gap_ratio"],
            "comp_pair_accuracy": comp_metrics["comp_pair_accuracy"],
            "comp_pair_acc_heldout": comp_metrics["comp_pair_acc_heldout"],
            "comp_pair_acc_seen": comp_metrics["comp_pair_acc_seen"],

            # Training dynamics & validation
            "val_bpc": val_bpc if val_bpc is not None else test_metrics.get("bpc", None),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "overfitting_gap": overfitting_gap,
            "best_val_bpc": training_log.get("best_val_bpc", val_bpc),
            "epochs": training_log.get("total_epochs", prev_row.get("epochs", 2)),
        }

        # Save per-run evaluation json
        run_metric_dir = self.metrics_dir / "runs" / exp_id
        run_metric_dir.mkdir(parents=True, exist_ok=True)
        with open(run_metric_dir / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        # Save item-level grammar predictions for GLMM analysis
        item_csv = run_metric_dir / "grammar_item_predictions.csv"
        with open(item_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "exp_id", "tokenizer", "model_size", "non_embedding_params", "seed",
                "item_id", "rule_id", "category", "grammatical", "ungrammatical",
                "logp_grammatical", "logp_ungrammatical", "margin", "correct"
            ])
            writer.writeheader()
            for d in grammar_metrics["details"]:
                writer.writerow({
                    "exp_id": exp_id,
                    "tokenizer": tok_type,
                    "model_size": model_size,
                    "non_embedding_params": non_emb,
                    "seed": seed,
                    "item_id": d["item_id"],
                    "rule_id": d["rule_id"],
                    "category": d["category"],
                    "grammatical": d["grammatical"],
                    "ungrammatical": d["ungrammatical"],
                    "logp_grammatical": d["logp_grammatical"],
                    "logp_ungrammatical": d["logp_ungrammatical"],
                    "margin": d["margin"],
                    "correct": d["correct"],
                })

        print(f"  Test BPC: {rec['test_bpc']} (In: {rec['bpc_in_domain']} | Out: {rec['bpc_out_domain']} | Gap: {rec['domain_gap']})")
        print(f"  Grammar Acc: {rec['grammar_acc'] * 100:.1f}% | Comp Gap: {rec['comp_generalization_gap']} | Comp Pair Acc: {rec['comp_pair_accuracy'] * 100:.1f}%")

        return rec

    def evaluate_all_runs(self) -> pd.DataFrame:
        """
        Discovers all trained model runs in results/models and evaluates them all.
        Updates results.csv, results.json, and results/metrics/grammar_item_eval.csv.
        """
        run_dirs = [p for p in self.models_dir.iterdir() if p.is_dir() and (p / "best_model.pt").exists()]
        run_dirs.sort(key=lambda p: p.name)

        if not run_dirs:
            print("[Evaluator] No trained checkpoints found in results/models.")
            return pd.DataFrame()

        print(f"[Evaluator] Found {len(run_dirs)} trained models to evaluate.")
        records = []
        all_item_rows = []

        for r_dir in run_dirs:
            exp_id = r_dir.name
            try:
                rec = self.evaluate_checkpoint(exp_id)
                records.append(rec)

                # Accumulate item-level predictions
                item_csv = self.metrics_dir / "runs" / exp_id / "grammar_item_predictions.csv"
                if item_csv.exists():
                    with open(item_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        all_item_rows.extend(list(reader))
            except Exception as e:
                print(f"  [Error] Failed to evaluate {exp_id}: {e}")

        df = pd.DataFrame(records)

        # Save results.csv and results.json
        csv_path = self.metrics_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        with open(self.metrics_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

        # Save cumulative item-level dataset for GLMM
        if all_item_rows:
            cum_item_csv = self.metrics_dir / "grammar_item_eval.csv"
            with open(cum_item_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_item_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_item_rows)
            print(f"[Evaluator] Saved cumulative item-level GLMM dataset: {len(all_item_rows)} rows -> {cum_item_csv}")

        print(f"\n[Evaluator] Successfully updated {csv_path} with {len(df)} rows.")
        return df


def main():
    parser = argparse.ArgumentParser(description="Unified SLM Evaluator across all 4 pillars")
    parser.add_argument("--run", type=str, default=None, help="Name of specific run to evaluate (e.g. exp_012_d75_mini_word_s42)")
    parser.add_argument("--all", action="store_true", help="Evaluate all model checkpoints in results/models")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for test dataloader")
    args = parser.parse_args()

    evaluator = UnifiedEvaluator()

    if args.run:
        evaluator.evaluate_checkpoint(args.run, batch_size=args.batch_size)
    elif args.all or not args.run:
        evaluator.evaluate_all_runs()


if __name__ == "__main__":
    main()
