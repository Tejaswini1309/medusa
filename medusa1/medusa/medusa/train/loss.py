"""Training losses (Architecture.md §4.9, §4.12, §4.14).

Index convention (0-based head k in code  <->  paper's head k+1):
  hidden state h_t at position t  ->  head k predicts token y_{t+k+2}.
  Paper: L_k = -log p_t^(k)(y_{t+k+1}) with k = 1..K, i.e. the same thing.
The original LM head predicts y_{t+1} (shift 1).
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def head_loss_weights(num_heads: int, base: float = 0.8) -> List[float]:
    """λ_k = base^k for paper heads k = 1..K (paper App. B.2 uses 0.8^k)."""
    return [base ** (k + 1) for k in range(num_heads)]


def head_shift(k: int) -> int:
    """How far ahead 0-based head k predicts."""
    return k + 2


def shifted_targets(labels: torch.Tensor, k: int) -> torch.Tensor:
    """Targets for head k aligned with positions 0..T-shift-1: labels[:, t+shift]."""
    return labels[:, head_shift(k):]


def ce_sum(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summed cross-entropy over valid targets (fp32 for stability), plus #valid and #top-1 correct.
    logits [..., V], targets [...] with IGNORE_INDEX for positions that carry no loss."""
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_t = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_t, ignore_index=IGNORE_INDEX, reduction="sum")
    valid = flat_t != IGNORE_INDEX
    with torch.no_grad():
        correct = ((flat_logits.argmax(-1) == flat_t) & valid).sum()
    return loss, valid.sum(), correct


def medusa1_loss(medusa_logits: torch.Tensor, labels: torch.Tensor,
                 weights: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_Medusa-1 = Σ_k λ_k · mean_t[-log p_t^(k)(y_{t+k+1})]   (§4.9)
    medusa_logits [K, B, T, V], labels [B, T]. Reference (non-chunked) implementation; the trainer
    computes the same quantity chunk by chunk to bound memory."""
    total = medusa_logits.new_zeros((), dtype=torch.float32)
    logs = {}
    T = labels.shape[1]
    for k in range(medusa_logits.shape[0]):
        s = head_shift(k)
        if T <= s:
            continue
        loss_sum, n, correct = ce_sum(medusa_logits[k][:, : T - s], shifted_targets(labels, k))
        if n == 0:
            continue
        mean = loss_sum / n
        total = total + weights[k] * mean
        logs[f"medusa{k}_loss"] = mean.item()
        logs[f"medusa{k}_top1"] = (correct.float() / n).item()
    return total, logs


def lm_loss(lm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy of the backbone (Medusa-2 with ground-truth data)."""
    loss_sum, n, _ = ce_sum(lm_logits[:, :-1], labels[:, 1:])
    return loss_sum / n.clamp(min=1)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """L_LM-distill = KL(p_teacher || p_student), averaged over positions where mask is True (§4.12).
    The teacher is detached here as well, so no gradient can reach it even if the caller forgets."""
    t_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)          # [B, T]
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp(min=1)


def medusa2_loss(backbone_loss: torch.Tensor, medusa1: torch.Tensor, lambda0: float) -> torch.Tensor:
    """L_Medusa-2 = L_LM-distill (or L_LM) + λ_0 · L_Medusa-1   (§4.14)."""
    return backbone_loss + lambda0 * medusa1
