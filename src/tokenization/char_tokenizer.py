"""
Character-level tokenizer for Toki Pona.
Vocab size: ~30 tokens.
"""

from __future__ import annotations
from typing import List
from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self, corpus_texts: List[str] | None = None):
        super().__init__()

        # Pre-seed special tokens
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, s in enumerate(specials):
            self.token_to_id[s] = i
            self.id_to_token[i] = s

        # Standard Toki Pona alphabet + punctuation
        base_chars = list("aeijklmnopstuw .,!?:;'-")

        # Discover any additional characters if corpus provided
        extra_chars = set()
        if corpus_texts:
            for text in corpus_texts:
                for ch in text.lower():
                    if ch not in base_chars and ch not in specials:
                        extra_chars.add(ch)

        all_chars = base_chars + sorted(list(extra_chars))
        for ch in all_chars:
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        text = text.lower()
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for ch in text:
            tokens.append(self.token_to_id.get(ch, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        res = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for tid in tokens:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid == self.unk_token_id and skip_special_tokens:
                res.append("?")
            else:
                res.append(self.id_to_token.get(tid, "?"))
        return "".join(res)
