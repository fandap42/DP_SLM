"""
Word-level tokenizer for Toki Pona.
Vocab size: ~150-180 tokens (standard Toki Pona pu/ku dictionary + punctuation).
"""

from __future__ import annotations
import re
from typing import List, Set
from .base import BaseTokenizer

# Official 120-137 pu words + standard ku additions
PU_WORDS = [
    "a", "akesi", "ala", "alasa", "ale", "ali", "anpa", "ante", "anu", "awen",
    "e", "en", "esun", "ijo", "ilo", "insa", "jaki", "jan", "jelo", "jo",
    "kala", "kalama", "kama", "kasi", "ken", "kepeken", "kili", "kiwen", "ko",
    "kon", "kule", "kulupu", "kute", "la", "lape", "laso", "lawa", "leko",
    "len", "lete", "li", "lili", "linja", "lipu", "loje", "lon", "luka",
    "lukin", "lupa", "ma", "mama", "mani", "meli", "mi", "mije", "moku",
    "moli", "monsi", "mu", "mun", "musi", "mute", "namako", "nanpa", "nasa",
    "nasin", "nena", "ni", "nimi", "noka", "o", "olin", "ona", "open",
    "pakala", "pali", "palisa", "pan", "pana", "pi", "pilin", "pimeja", "pini",
    "pipi", "poka", "poki", "pona", "pu", "sama", "seli", "selo", "sewi",
    "sijelo", "sike", "sin", "sina", "sinpin", "sitelen", "sona", "soweli",
    "supa", "suwi", "tan", "taso", "tawa", "telo", "tenpo", "toki", "tomo",
    "tonsi", "tu", "unpa", "utala", "walo", "wan", "waso", "wawa", "weka",
    "wile"
]

KU_EXTENSIONS = [
    "epiku", "jasima", "kijetesantakalu", "kin", "kipisi", "kokoju", "ku",
    "lanpan", "leko", "meso", "misikeke", "monsuta", "n", "namako", "oko",
    "soko", "tonsi"
]

PUNCTUATION = [".", ",", "!", "?", ":", ";", "-"]


class WordTokenizer(BaseTokenizer):
    def __init__(self, corpus_texts: List[str] | None = None, min_freq: int = 2):
        super().__init__()

        # Pre-seed special tokens
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, s in enumerate(specials):
            self.token_to_id[s] = i
            self.id_to_token[i] = s

        # Preload pu + ku + punctuation
        core_vocab: Set[str] = set(PU_WORDS) | set(KU_EXTENSIONS) | set(PUNCTUATION)

        # Count frequencies in corpus to include frequent proper names or loanwords if any
        if corpus_texts:
            freqs = {}
            for text in corpus_texts:
                for token in self._tokenize_regex(text):
                    freqs[token] = freqs.get(token, 0) + 1
            for token, cnt in freqs.items():
                if cnt >= min_freq:
                    core_vocab.add(token)

        for w in sorted(list(core_vocab)):
            if w not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[w] = idx
                self.id_to_token[idx] = w

    @staticmethod
    def _tokenize_regex(text: str) -> List[str]:
        """Split text into words and punctuation tokens."""
        text = text.lower()
        # Find sequences of letters or individual punctuation marks
        return re.findall(r"[a-z]+|[.,!?:;\-]", text)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raw_tokens = self._tokenize_regex(text)
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for t in raw_tokens:
            tokens.append(self.token_to_id.get(t, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for tid in tokens:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid == self.unk_token_id and skip_special_tokens:
                words.append("<unk>")
            else:
                words.append(self.id_to_token.get(tid, "<unk>"))

        # Reconstruct natural spacing (no space before punctuation)
        out = ""
        for w in words:
            if w in PUNCTUATION:
                out = out.rstrip() + w + " "
            else:
                out += w + " "
        return out.strip()
