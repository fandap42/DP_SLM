"""
Base Tokenizer abstract class and utilities for character-normalized metric tracking.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseTokenizer(ABC):
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.pad_token_id: int = 0
        self.bos_token_id: int = 1
        self.eos_token_id: int = 2
        self.unk_token_id: int = 3
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode string to list of token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode list of token IDs back to string."""
        pass

    def get_char_length(self, text: str) -> int:
        """
        Returns number of characters in the original text (excluding BOS/EOS)
        for fair cross-tokenizer bits-per-character (BPC) calculation.
        """
        return len(text)

    def save(self, filepath: str) -> None:
        import json
        data = {
            "type": self.__class__.__name__,
            "token_to_id": self.token_to_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_tokenizer(filepath: str) -> BaseTokenizer:
    import json
    from pathlib import Path
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    t_type = data.get("type")
    if t_type == "CharTokenizer":
        from .char_tokenizer import CharTokenizer
        tok = CharTokenizer.__new__(CharTokenizer)
    elif t_type == "WordTokenizer":
        from .word_tokenizer import WordTokenizer
        tok = WordTokenizer.__new__(WordTokenizer)
    elif t_type == "BPETokenizer":
        from .bpe_tokenizer import BPETokenizer
        tok = BPETokenizer.__new__(BPETokenizer)
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.bpe_ranks = {tuple(m): idx for idx, m in enumerate(tok.merges)}
    else:
        raise ValueError(f"Unknown tokenizer type: {t_type}")

    tok.pad_token_id = data["pad_token_id"]
    tok.bos_token_id = data["bos_token_id"]
    tok.eos_token_id = data["eos_token_id"]
    tok.unk_token_id = data["unk_token_id"]
    tok.token_to_id = data["token_to_id"]
    tok.id_to_token = {int(idx): token for token, idx in tok.token_to_id.items()}
    return tok

