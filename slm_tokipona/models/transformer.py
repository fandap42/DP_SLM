"""
PyTorch Causal Decoder-Only Transformer implementation optimized for SLM scaling experiments.
Uses RMSNorm, modern MLP activation, and PyTorch scaled_dot_product_attention.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"

        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal attention via PyTorch SDPA (native ROCm/CUDA acceleration)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TokiPonaTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if target_ids is not None:
            # Flatten to compute cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

        return logits, loss

    def get_parameter_counts(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        emb_params = sum(p.numel() for p in self.tok_embeddings.parameters()) + sum(
            p.numel() for p in self.pos_embeddings.parameters()
        )
        non_emb_params = total_params - emb_params
        return {
            "total_params": total_params,
            "non_embedding_params": non_emb_params,
            "embedding_params": emb_params,
        }
