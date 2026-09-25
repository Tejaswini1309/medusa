"""Token-selection utilities (Architecture.md §3.8): temperature, top-k, top-p, and top-s_k candidates.

Kept independent from the decoding engine so candidate generation can be reconfigured without
touching decode.py.
"""
from typing import Tuple

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 here; use greedy selection for temperature 0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits (last dim), set the rest to -inf. k <= 0 disables."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probability mass >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum - torch.softmax(sorted_logits, dim=-1) >= p   # keep the token that crosses p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    x = apply_temperature(logits.float(), temperature)
    x = top_p_filter(top_k_filter(x, top_k), top_p)
    return F.softmax(x, dim=-1)


def select_token(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Greedy (temperature 0) or sampled next token. logits [..., V] -> ids [...]."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = logits_to_probs(logits, temperature, top_k, top_p)
    flat = probs.reshape(-1, probs.size(-1))
    ids = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return ids.reshape(probs.shape[:-1])


def topk_candidates(logits: torch.Tensor, s: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-s candidate tokens and their probabilities per row. logits [..., V] -> ([..., s], [..., s]).
    Ranking is temperature-invariant; the probabilities use the given temperature (1.0 if <= 0)."""
    t = temperature if temperature > 0 else 1.0
    probs = F.softmax(logits.float() / t, dim=-1)
    vals, ids = torch.topk(probs, s, dim=-1)
    return ids, vals
