"""
Byte-Pair Encoding (BPE) Subword Tokenizer for Toki Pona.
Vocab size: configurable (typically 64-100 tokens, bridging char and word levels).
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Set, Tuple
from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, corpus_texts: List[str] | None = None, target_vocab_size: int = 80):
        super().__init__()

        # Pre-seed special tokens
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, s in enumerate(specials):
            self.token_to_id[s] = i
            self.id_to_token[i] = s

        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        if corpus_texts:
            self.train(corpus_texts, target_vocab_size=target_vocab_size)
        else:
            # Default minimal alphabet
            base_chars = list("aeijklmnopstuw .,!?:;'-")
            for ch in base_chars:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

    def train(self, corpus_texts: List[str], target_vocab_size: int = 80) -> None:
        """
        Train BPE merges on Toki Pona text corpus until target_vocab_size is reached.
        """
        # Step 1: Pre-tokenize into words with word-boundary markers
        word_freqs: Dict[Tuple[str, ...], int] = {}
        initial_chars: Set[str] = set()

        for text in corpus_texts:
            text = text.lower()
            # Split into words and punctuation
            tokens = re.findall(r"[a-z]+|[.,!?:;\-]", text)
            for t in tokens:
                chars = tuple(list(t) + ["</w>"])
                word_freqs[chars] = word_freqs.get(chars, 0) + 1
                for c in t:
                    initial_chars.add(c)

        initial_chars.add("</w>")
        # Also add punctuation
        for p in " .,!?:;'-":
            initial_chars.add(p)

        # Populate base vocab
        for ch in sorted(list(initial_chars)):
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        # Step 2: Iteratively find best pair and merge
        num_merges = max(0, target_vocab_size - len(self.token_to_id))

        for rank in range(num_merges):
            pair_counts: Dict[Tuple[str, str], int] = {}
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                # No frequent pairs left
                break

            new_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = rank

            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # Update word frequencies
            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                new_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                        new_tuple.append(new_token)
                        i += 2
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_tuple)] = freq
            word_freqs = new_word_freqs

    def _bpe_segment_word(self, word: str) -> List[str]:
        """Segment a single word using learned BPE ranks."""
        pieces = list(word) + ["</w>"]
        if len(pieces) == 1:
            return pieces

        while len(pieces) > 1:
            # Find pairs and their ranks
            min_rank = float("inf")
            best_pair_idx = -1
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair_idx = i

            if min_rank == float("inf"):
                break  # No more mergeable pairs

            # Merge best pair
            merged = pieces[best_pair_idx] + pieces[best_pair_idx + 1]
            pieces = pieces[:best_pair_idx] + [merged] + pieces[best_pair_idx + 2 :]

        return pieces

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        raw_words = re.findall(r"[a-z]+|[.,!?:;\-]", text.lower())
        for w in raw_words:
            if re.match(r"[.,!?:;\-]", w):
                tokens.append(self.token_to_id.get(w, self.unk_token_id))
            else:
                subwords = self._bpe_segment_word(w)
                for sw in subwords:
                    tokens.append(self.token_to_id.get(sw, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for tid in tokens:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid == self.unk_token_id:
                pieces.append("<unk>")
            else:
                pieces.append(self.id_to_token.get(tid, "<unk>"))

        text = "".join(pieces).replace("</w>", " ")
        # Clean up punctuation spacing
        text = re.sub(r"\s+([.,!?:;\-])", r"\1", text)
        return text.strip()

    def save(self, filepath: str) -> None:
        data = {
            "type": self.__class__.__name__,
            "token_to_id": self.token_to_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "merges": self.merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
