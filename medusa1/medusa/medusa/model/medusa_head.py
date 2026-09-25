"""Medusa heads (Architecture.md §3.2, §4.1).

Head k computes  p_t^(k) = softmax( W_2^(k) · (SiLU(W_1^(k) h_t) + h_t) ).

The ResBlock is the part in brackets. W_1 is zero-initialised and W_2 is copied from the
backbone's LM head, so at initialisation every head reproduces the base model's
next-token distribution (§4.1.1). The heads return *logits*; softmax is applied by the
loss (cross-entropy) or by the decoder, never stored.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """x -> SiLU(W_1 x) + x, with W_1 ∈ R^{d×d} initialised to zero."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # No bias: Architecture.md §4.1 writes the head as SiLU(W_1 h) with no bias term.
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.linear(x)) + x


class MedusaHead(nn.Module):
    """One head: `num_layers` ResBlocks followed by a vocabulary projection W_2 (d -> V)."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # W_2^(k)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.blocks(hidden))


class MedusaHeads(nn.Module):
    """K independent heads. Head index k (0-based here) predicts the token at offset t+k+2,
    i.e. the paper's head k+1 predicting y_{t+(k+1)+1}; the original LM head covers t+1."""

    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, vocab_size, num_layers) for _ in range(num_heads)]
        )

    @torch.no_grad()
    def init_from_lm_head(self, lm_head_weight: torch.Tensor) -> None:
        """§4.1.1: W_2^(k) <- original LM head weight, W_1^(k) <- 0, for every k.
        `copy_` gives each head its own storage, so heads stay independent parameters."""
        for head in self.heads:
            head.lm_head.weight.copy_(lm_head_weight.to(head.lm_head.weight.dtype))
            for block in head.blocks:
                block.linear.weight.zero_()

    def forward_head(self, k: int, hidden: torch.Tensor) -> torch.Tensor:
        """Logits of head k only. Used by the trainer to back-propagate one head at a time."""
        return self.heads[k](hidden.to(self.heads[k].lm_head.weight.dtype))

    def forward(self, hidden: torch.Tensor, heads: Optional[List[int]] = None) -> torch.Tensor:
        """hidden: [..., d]  ->  logits: [K, ..., V] (stacked along a new leading head dim)."""
        idx = range(self.num_heads) if heads is None else heads
        return torch.stack([self.forward_head(k, hidden) for k in idx], dim=0)
