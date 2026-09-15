"""
Model configuration dataclass and presets for SLM scaling experiments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class TransformerConfig:
    name: str
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    bias: bool = False

    @classmethod
    def get_preset(cls, name: str, vocab_size: int, max_seq_len: int = 128) -> "TransformerConfig":
        presets: Dict[str, Dict] = {
            # ~120k params (useful for fast DoE validation & tests)
            "micro": {
                "d_model": 64,
                "n_layer": 2,
                "n_head": 2,
                "d_ff": 256,
            },
            # ~500k params
            "mini": {
                "d_model": 128,
                "n_layer": 4,
                "n_head": 4,
                "d_ff": 512,
            },
            # ~2.5M params
            "small": {
                "d_model": 256,
                "n_layer": 6,
                "n_head": 8,
                "d_ff": 1024,
            },
            # ~7.5M params
            "medium": {
                "d_model": 384,
                "n_layer": 8,
                "n_head": 8,
                "d_ff": 1536,
            },
            # ~20M params
            "large": {
                "d_model": 512,
                "n_layer": 12,
                "n_head": 16,
                "d_ff": 2048,
            },
        }

        name_lower = name.lower()
        if name_lower not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

        kwargs = presets[name_lower]
        return cls(
            name=name_lower,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            **kwargs,
        )
