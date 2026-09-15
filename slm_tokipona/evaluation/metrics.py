"""
Evaluation metrics for fair cross-tokenizer comparison:
- Bits-Per-Character (BPC)
- Character-normalized Perplexity (PPL_char)
- Token Perplexity (PPL_token)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from ..tokenizers.base import BaseTokenizer


@torch.no_grad()
def compute_loss_and_bpc(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Computes exact total cross-entropy, token perplexity, and character-normalized BPC.
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        char_lens = batch["char_lens"]

        logits, _ = model(input_ids)
        vocab_size = logits.size(-1)

        # Compute unreduced cross-entropy loss (in nats)
        loss_unreduced = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid_tokens = (target_ids != -100).sum().item()

        total_nats += loss_unreduced.item()
        total_tokens += valid_tokens
        total_chars += char_lens.sum().item()

    if total_tokens == 0 or total_chars == 0:
        return {"loss_nats": 0.0, "bpc": 0.0, "ppl_tok": 1.0, "ppl_char": 1.0}

    mean_loss_nats = total_nats / total_tokens
    # BPC = Total Bits / Total Characters = (Total Nats / ln 2) / Total Chars
    total_bits = total_nats / math.log(2.0)
    bpc = total_bits / total_chars

    ppl_token = math.exp(min(mean_loss_nats, 50.0))
    ppl_char = 2.0 ** min(bpc, 50.0)

    return {
        "loss_nats": mean_loss_nats,
        "bpc": bpc,
        "ppl_tok": ppl_token,
        "ppl_char": ppl_char,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


@torch.no_grad()
def score_sentence(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    sentence: str,
    device: torch.device,
) -> float:
    """
    Calculates the sequence log-likelihood log P(s) in nats for a given sentence.
    """
    model.eval()
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    if len(tokens) < 2:
        return -9999.0

    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

    logits, _ = model(input_ids)
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1),
        ignore_index=-100,
        reduction="sum",
    )

    # Return total sequence log-likelihood: -sum(cross_entropy)
    return -loss.item()
