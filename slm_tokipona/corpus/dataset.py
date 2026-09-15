"""
PyTorch Dataset and DataLoader collation for Toki Pona SLM training.
Preserves character lengths alongside tokens for exact Bits-Per-Character (BPC) calculation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from ..tokenizers.base import BaseTokenizer


class TokiPonaDataset(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        tokenizer: BaseTokenizer,
        max_seq_len: int = 128,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict] = []

        data_path = Path(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        if fraction < 1.0:
            import random
            rng = random.Random(seed)
            sample_size = max(1, int(len(self.samples) * fraction))
            self.samples = rng.sample(self.samples, sample_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.samples[idx]
        text = item["text"]
        char_len = len(text)

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        # Truncate if exceeds max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]

        return {
            "tokens": tokens,
            "char_len": char_len,
            "source": item.get("source", "unknown"),
        }


def collate_fn(batch: List[Dict], pad_token_id: int):
    """
    Pad sequences dynamically to the longest sequence in the batch.
    Creates input x and target y (shifted by 1).
    """
    batch_tokens = [b["tokens"] for b in batch]
    char_lens = [b["char_len"] for b in batch]
    sources = [b["source"] for b in batch]

    max_len = max(len(toks) for toks in batch_tokens)
    # Require at least 2 tokens for autoregressive modeling
    max_len = max(max_len, 2)

    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len - 1), pad_token_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)  # -100 is ignored by CrossEntropyLoss

    for i, toks in enumerate(batch_tokens):
        if len(toks) < 2:
            continue
        seq_in = toks[:-1]
        seq_out = toks[1:]
        input_ids[i, : len(seq_in)] = torch.tensor(seq_in, dtype=torch.long)
        target_ids[i, : len(seq_out)] = torch.tensor(seq_out, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "char_lens": torch.tensor(char_lens, dtype=torch.long),
        "sources": sources,
    }


def create_dataloader(
    data_path: Path | str,
    tokenizer: BaseTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 128,
    fraction: float = 1.0,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokiPonaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        fraction=fraction,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
