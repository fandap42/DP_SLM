"""
Corpus acquisition, cleaning, deduplication, and source-stratified splitting for Toki Pona.
Supports downloading from Tatoeba, Toki Pona Monolingual (Wikipesija), Lipu Sewi, and Poki Lapo.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


VALID_CHARS_PATTERN = re.compile(r"[^a-z\s.,!?:;'\-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CorpusItem:
    id: str
    text: str
    source: str
    char_len: int
    word_len: int


def clean_toki_pona_text(text: str) -> str:
    """
    Clean and canonicalize Toki Pona sentence:
    - Lowercase
    - Replace curly quotes / weird punctuation
    - Strip unsupported non-latin/non-punctuation characters
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = VALID_CHARS_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def load_raw_sources(cache_dir: Path) -> List[Tuple[str, str]]:
    """
    Downloads or reads cached raw datasets from Hugging Face:
    - Tatoeba (conversational)
    - Wikipesija / Monolingual (encyclopedic / news)
    - Lipu Sewi (literary / biblical narrative)
    - Poki Lapo (poetry and stories)
    Returns a list of tuples: (cleaned_text, source_name)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[str, str]] = []

    # 1. Tatoeba
    tatoeba_cache = cache_dir / "tatoeba.parquet"
    if not tatoeba_cache.exists():
        print("  [Corpus] Downloading Tatoeba...")
        df_tat = pd.read_parquet("https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona/resolve/main/data/train-00000-of-00001.parquet")
        df_tat.to_parquet(tatoeba_cache)
    else:
        df_tat = pd.read_parquet(tatoeba_cache)

    if "tok" in df_tat.columns:
        for t in df_tat["tok"].dropna():
            cleaned = clean_toki_pona_text(t)
            if cleaned and len(cleaned) >= 5:
                items.append((cleaned, "tatoeba"))

    # 2. Toki Pona Monolingual / Wikipesija
    mono_cache = cache_dir / "monolingual.parquet"
    if not mono_cache.exists():
        print("  [Corpus] Downloading Toki Pona Monolingual...")
        df_mono = pd.read_parquet("https://huggingface.co/datasets/NetherQuartz/tokipona-monolingual/resolve/main/data/train-00000-of-00001.parquet")
        df_mono.to_parquet(mono_cache)
    else:
        df_mono = pd.read_parquet(mono_cache)

    if "text" in df_mono.columns:
        for t in df_mono["text"].dropna():
            cleaned = clean_toki_pona_text(t)
            if cleaned and len(cleaned) >= 5:
                items.append((cleaned, "wikipesija"))

    # 3. Lipu Sewi
    sewi_cache = cache_dir / "lipu_sewi.parquet"
    if not sewi_cache.exists():
        print("  [Corpus] Downloading Lipu Sewi...")
        df_sewi = pd.read_parquet("https://huggingface.co/datasets/NetherQuartz/lipu-sewi/resolve/main/data/train-00000-of-00001.parquet")
        df_sewi.to_parquet(sewi_cache)
    else:
        df_sewi = pd.read_parquet(sewi_cache)

    if "tok" in df_sewi.columns:
        for t in df_sewi["tok"].dropna():
            cleaned = clean_toki_pona_text(t)
            if cleaned and len(cleaned) >= 5:
                items.append((cleaned, "lipu_sewi"))

    # 4. Poki Lapo
    poki_cache = cache_dir / "poki_lapo.parquet"
    if not poki_cache.exists():
        print("  [Corpus] Downloading Poki Lapo...")
        df_poki = pd.read_parquet("https://huggingface.co/datasets/NetherQuartz/poki-lapo/resolve/main/data/train-00000-of-00001.parquet")
        df_poki.to_parquet(poki_cache)
    else:
        df_poki = pd.read_parquet(poki_cache)

    if "text" in df_poki.columns:
        for doc in df_poki["text"].dropna():
            # Poki Lapo contains full paragraphs / multi-line poems; split by sentences / lines
            for line in str(doc).split("\n"):
                cleaned = clean_toki_pona_text(line)
                if cleaned and len(cleaned) >= 10:
                    items.append((cleaned, "poki_lapo"))

    return items


def deduplicate_and_filter(items: List[Tuple[str, str]]) -> List[CorpusItem]:
    """
    Perform exact deduplication using SHA256 hashes of the cleaned text.
    Filters out extreme outliers (e.g. fewer than 2 words or single repetitive characters).
    """
    seen_hashes = set()
    deduped: List[CorpusItem] = []

    for text, source in items:
        # Require at least 2 words
        words = text.split()
        if len(words) < 2 or len(words) > 128:
            continue

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        deduped.append(
            CorpusItem(
                id=h[:12],
                text=text,
                source=source,
                char_len=len(text),
                word_len=len(words),
            )
        )

    return deduped


def split_by_source(
    items: List[CorpusItem],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[CorpusItem], List[CorpusItem], List[CorpusItem]]:
    """
    Source-stratified split: each source is partitioned independently according
    to the requested ratios to guarantee identical source proportions across splits.
    """
    import random

    rng = random.Random(seed)
    source_groups: Dict[str, List[CorpusItem]] = {}
    for item in items:
        source_groups.setdefault(item.source, []).append(item)

    train_set: List[CorpusItem] = []
    val_set: List[CorpusItem] = []
    test_set: List[CorpusItem] = []

    for source, group in source_groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_part = group[:n_train]
        val_part = group[n_train : n_train + n_val]
        test_part = group[n_train + n_val :]

        train_set.extend(train_part)
        val_set.extend(val_part)
        test_set.extend(test_part)

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    return train_set, val_set, test_set


def build_and_save_corpus(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict:
    """
    Complete end-to-end corpus building pipeline.
    Saves:
      - train.jsonl
      - val.jsonl
      - test.jsonl
      - metadata.json
    """
    print("[Corpus] Loading raw sources...")
    raw_items = load_raw_sources(raw_dir)
    print(f"[Corpus] Total raw instances extracted: {len(raw_items):,}")

    print("[Corpus] Deduplicating and filtering...")
    clean_items = deduplicate_and_filter(raw_items)
    print(f"[Corpus] Unique items after deduplication: {len(clean_items):,}")

    print("[Corpus] Performing source-stratified train/val/test split...")
    train_items, val_items, test_items = split_by_source(
        clean_items, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(path: Path, dataset: List[CorpusItem]):
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    save_jsonl(output_dir / "train.jsonl", train_items)
    save_jsonl(output_dir / "val.jsonl", val_items)
    save_jsonl(output_dir / "test.jsonl", test_items)

    # Compute detailed corpus statistics
    def get_stats(subset: List[CorpusItem]) -> Dict:
        src_counts = {}
        for x in subset:
            src_counts[x.source] = src_counts.get(x.source, 0) + 1
        total_chars = sum(x.char_len for x in subset)
        total_words = sum(x.word_len for x in subset)
        return {
            "num_sentences": len(subset),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_sent": round(total_chars / max(len(subset), 1), 2),
            "avg_words_per_sent": round(total_words / max(len(subset), 1), 2),
            "source_distribution": src_counts,
        }

    meta = {
        "total_instances": len(clean_items),
        "train": get_stats(train_items),
        "val": get_stats(val_items),
        "test": get_stats(test_items),
        "seed": seed,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Corpus] Pipeline complete! Saved to {output_dir}")
    print(f"  Train: {len(train_items):,} items ({meta['train']['total_words']:,} words)")
    print(f"  Val:   {len(val_items):,} items ({meta['val']['total_words']:,} words)")
    print(f"  Test:  {len(test_items):,} items ({meta['test']['total_words']:,} words)")

    return meta
