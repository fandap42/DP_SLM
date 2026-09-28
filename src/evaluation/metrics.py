"""
Evaluation metrics for fair cross-tokenizer comparison:
- Bits-Per-Character (BPC)
- Character-normalized Loss (Loss per Char in nats)
- Character-normalized Perplexity (PPL_char)
- Token Perplexity (PPL_token)
- Domain-split BPC and Generalization Gap (In-Domain vs Out-of-Domain)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from ..tokenization.base import BaseTokenizer


@torch.no_grad()
def compute_loss_and_bpc(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    in_domain_sources: Tuple[str, ...] = ("tatoeba",),
) -> Dict[str, Any]:
    """
    Computes exact total cross-entropy, token perplexity, character-normalized BPC,
    and source-stratified domain metrics (in-domain vs out-of-domain).

    Loss_char = (-sum_t ln P(x_t | x_<t)) / N_chars
    BPC       = Loss_char / ln(2)
    PPL_token = exp(Loss_token)
    PPL_char  = 2^(BPC) = exp(Loss_char)
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0

    # Domain-specific accumulators
    source_stats: Dict[str, Dict[str, float]] = {}
    in_domain_nats = 0.0
    in_domain_tokens = 0
    in_domain_chars = 0
    out_domain_nats = 0.0
    out_domain_tokens = 0
    out_domain_chars = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        char_lens = batch["char_lens"]
        sources = batch.get("sources", ["unknown"] * input_ids.size(0))

        logits, _ = model(input_ids)
        vocab_size = logits.size(-1)

        # Per-token cross entropy in nats
        loss_unreduced = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(input_ids.size(0), -1)

        valid_tokens_mask = (target_ids != -100)
        sample_nats = (loss_unreduced * valid_tokens_mask).sum(dim=1).cpu()
        sample_tokens = valid_tokens_mask.sum(dim=1).cpu()

        for i, src in enumerate(sources):
            s_nats = float(sample_nats[i].item())
            s_toks = int(sample_tokens[i].item())
            s_chars = int(char_lens[i].item())

            total_nats += s_nats
            total_tokens += s_toks
            total_chars += s_chars

            if src not in source_stats:
                source_stats[src] = {"nats": 0.0, "tokens": 0, "chars": 0}
            source_stats[src]["nats"] += s_nats
            source_stats[src]["tokens"] += s_toks
            source_stats[src]["chars"] += s_chars

            if src in in_domain_sources:
                in_domain_nats += s_nats
                in_domain_tokens += s_toks
                in_domain_chars += s_chars
            else:
                out_domain_nats += s_nats
                out_domain_tokens += s_toks
                out_domain_chars += s_chars

    if total_tokens == 0 or total_chars == 0:
        return {
            "loss_nats": 0.0,
            "loss_char": 0.0,
            "bpc": 0.0,
            "ppl_tok": 1.0,
            "ppl_char": 1.0,
            "total_tokens": 0,
            "total_chars": 0,
            "bpc_in_domain": 0.0,
            "bpc_out_domain": 0.0,
            "domain_gap": 0.0,
            "domain_ratio": 1.0,
            "sources": {},
        }

    ln2 = math.log(2.0)
    mean_loss_nats = total_nats / total_tokens
    loss_char = total_nats / total_chars
    bpc = (total_nats / ln2) / total_chars
    ppl_token = math.exp(min(mean_loss_nats, 50.0))
    ppl_char = 2.0 ** min(bpc, 50.0)

    # In-domain metrics
    bpc_in = ((in_domain_nats / ln2) / in_domain_chars) if in_domain_chars > 0 else bpc
    ppl_tok_in = math.exp(min(in_domain_nats / max(1, in_domain_tokens), 50.0)) if in_domain_tokens > 0 else ppl_token
    ppl_char_in = 2.0 ** min(bpc_in, 50.0)

    # Out-of-domain metrics
    bpc_out = ((out_domain_nats / ln2) / out_domain_chars) if out_domain_chars > 0 else bpc
    ppl_tok_out = math.exp(min(out_domain_nats / max(1, out_domain_tokens), 50.0)) if out_domain_tokens > 0 else ppl_token
    ppl_char_out = 2.0 ** min(bpc_out, 50.0)

    domain_gap = bpc_out - bpc_in
    domain_ratio = bpc_out / max(bpc_in, 1e-6)

    # Source breakdown
    detailed_sources = {}
    for src, stats in source_stats.items():
        s_chars = stats["chars"]
        s_toks = stats["tokens"]
        s_nats = stats["nats"]
        s_bpc = ((s_nats / ln2) / s_chars) if s_chars > 0 else 0.0
        s_loss_nats = (s_nats / s_toks) if s_toks > 0 else 0.0
        s_loss_char = (s_nats / s_chars) if s_chars > 0 else 0.0
        detailed_sources[src] = {
            "bpc": round(s_bpc, 4),
            "loss_nats": round(s_loss_nats, 4),
            "loss_char": round(s_loss_char, 4),
            "ppl_tok": round(math.exp(min(s_loss_nats, 50.0)), 2),
            "ppl_char": round(2.0 ** min(s_bpc, 50.0), 2),
            "tokens": s_toks,
            "chars": s_chars,
        }

    return {
        "loss_nats": round(mean_loss_nats, 4),
        "loss_char": round(loss_char, 4),
        "bpc": round(bpc, 4),
        "ppl_tok": round(ppl_token, 2),
        "ppl_char": round(ppl_char, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "bpc_in_domain": round(bpc_in, 4),
        "bpc_out_domain": round(bpc_out, 4),
        "ppl_tok_in": round(ppl_tok_in, 2),
        "ppl_char_in": round(ppl_char_in, 2),
        "ppl_tok_out": round(ppl_tok_out, 2),
        "ppl_char_out": round(ppl_char_out, 2),
        "domain_gap": round(domain_gap, 4),
        "domain_ratio": round(domain_ratio, 4),
        "sources": detailed_sources,
    }


@torch.no_grad()
def score_sentence(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    sentence: str,
    device: torch.device,
) -> float:
    """
    Calculates the sequence log-likelihood log P(s) in nats for a given sentence:
    sum_{t} ln P(x_t | x_{<t}).
    Returns negative infinity for empty / unencodeable sequences.
    """
    if not sentence or not sentence.strip():
        return -9999.0

    model.eval()
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    if len(tokens) < 2:
        return -9999.0

    max_len = getattr(getattr(model, "config", None), "max_seq_len", 128)
    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + [tokenizer.eos_token_id]

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

    # Return total sequence log-likelihood: -sum(cross_entropy in nats)
    return -loss.item()


@torch.no_grad()
def score_sentence_bpc(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    sentence: str,
    device: torch.device,
) -> float:
    """
    Computes Bits-Per-Character for an individual sentence.
    """
    logp = score_sentence(model, tokenizer, sentence, device)
    if logp <= -9000.0:
        return 99.0

    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    max_len = getattr(getattr(model, "config", None), "max_seq_len", 128)
    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + [tokenizer.eos_token_id]
        char_len = max(1, len(tokenizer.decode(tokens, skip_special_tokens=True)))
    else:
        char_len = max(1, len(sentence))

    nats = -logp
    bpc = (nats / math.log(2.0)) / char_len
    return float(bpc)
