"""
Grammatical Acceptability Test Suite for Toki Pona.
Inspired by BLiMP (Benchmark of Linguistic Minimal Pairs).
Evaluates whether a language model assigns higher sequence log-likelihood
to grammatical vs ungrammatical sentences: log P(S_correct) > log P(S_incorrect).
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch

from ..tokenization.base import BaseTokenizer
from .metrics import score_sentence
from .grammar_generator import (
    GrammarMinimalPair,
    generate_all_grammar_pairs,
    load_grammar_suite,
    save_grammar_suite,
)

# For backward compatibility
MinimalPair = GrammarMinimalPair


def get_default_test_suite(project_root: Optional[Path | str] = None) -> List[GrammarMinimalPair]:
    """
    Returns the comprehensive synthetic minimal pairs test suite.
    Prefers loading from versioned data/evaluation/grammar_pairs.json if present;
    otherwise generates all pairs on the fly.
    """
    if project_root is not None:
        p = Path(project_root) / "data" / "evaluation" / "grammar_pairs.json"
        if p.exists():
            return load_grammar_suite(p)

    # Check relative to this file
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "data" / "evaluation" / "grammar_pairs.json"
    if data_path.exists():
        return load_grammar_suite(data_path)

    # Generate and save
    pairs = generate_all_grammar_pairs()
    try:
        save_grammar_suite(pairs, repo_root / "data" / "evaluation")
    except Exception:
        pass
    return pairs


def evaluate_grammar_suite(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    test_suite: Optional[List[GrammarMinimalPair]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model across the test suite of minimal pairs.

    Returns:
      - overall_accuracy: % of pairs where log P(grammatical) > log P(ungrammatical)
      - category_accuracies: breakdown per linguistic rule category
      - sub_rule_accuracies: fine-grained breakdown
      - total_pairs, total_correct
      - details: per-item evaluation records for GLMM analysis
    """
    if test_suite is None:
        test_suite = get_default_test_suite()

    model.eval()
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}
    sub_correct: Dict[str, int] = {}
    sub_total: Dict[str, int] = {}
    details: List[Dict[str, Any]] = []

    total_correct = 0

    for pair in test_suite:
        logp_g = score_sentence(model, tokenizer, pair.grammatical, device)
        logp_u = score_sentence(model, tokenizer, pair.ungrammatical, device)

        is_correct = int(logp_g > logp_u)
        total_correct += is_correct

        cat = pair.category
        cat_correct[cat] = cat_correct.get(cat, 0) + is_correct
        cat_total[cat] = cat_total.get(cat, 0) + 1

        sub = getattr(pair, "sub_rule", cat)
        sub_correct[sub] = sub_correct.get(sub, 0) + is_correct
        sub_total[sub] = sub_total.get(sub, 0) + 1

        details.append({
            "item_id": getattr(pair, "item_id", f"pair_{len(details):03d}"),
            "rule_id": getattr(pair, "rule_id", cat),
            "sub_rule": sub,
            "category": cat,
            "description": pair.description,
            "grammatical": pair.grammatical,
            "ungrammatical": pair.ungrammatical,
            "logp_grammatical": round(logp_g, 3),
            "logp_ungrammatical": round(logp_u, 3),
            "margin": round(logp_g - logp_u, 3),
            "correct": is_correct,
        })

    cat_acc = {
        cat: round(cat_correct[cat] / cat_total[cat], 4)
        for cat in sorted(cat_total)
    }
    sub_acc = {
        sub: round(sub_correct[sub] / sub_total[sub], 4)
        for sub in sorted(sub_total)
    }
    overall_acc = round(total_correct / max(len(test_suite), 1), 4)

    return {
        "overall_accuracy": overall_acc,
        "category_accuracies": cat_acc,
        "sub_rule_accuracies": sub_acc,
        "total_pairs": len(test_suite),
        "total_correct": total_correct,
        "details": details,
    }


def save_grammar_details_csv(details: List[Dict[str, Any]], filepath: Path | str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not details:
        return

    fieldnames = list(details[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
