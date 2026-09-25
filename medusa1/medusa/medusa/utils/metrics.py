"""Speed and acceptance metrics (Architecture.md Â§3.11).

Definitions follow the paper, App. B.1:
  acceleration rate = tokens generated / decoding steps          (1.0 for plain decoding)
  overhead          = Medusa latency per step / baseline latency per step
  speedup           = acceleration rate / overhead  (= wall-clock tokens/s ratio)
Prefill (the prompt forward pass) is timed separately and excluded from per-step latency.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


class CudaTimer:
    """Wall-clock timer that synchronises the GPU, so queued kernels are included."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


@dataclass
class DecodingStats:
    num_heads: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0
    steps: int = 0                                    # decoding forward passes after prefill
    prefill_time: float = 0.0
    decode_time: float = 0.0
    accept_lengths: List[int] = field(default_factory=list)   # Medusa tokens accepted per step (excl. root)
    head_accepts: List[int] = field(default_factory=list)     # per head: steps where depth k was accepted

    def record_step(self, accepted_beyond_root: int) -> None:
        self.steps += 1
        self.accept_lengths.append(accepted_beyond_root)
        if not self.head_accepts:
            self.head_accepts = [0] * self.num_heads
        for k in range(accepted_beyond_root):
            self.head_accepts[k] += 1

    @property
    def acceleration_rate(self) -> float:
        # Each step yields its accepted Medusa tokens + 1 token from the LM head.
        return (sum(self.accept_lengths) + self.steps) / max(self.steps, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / max(self.decode_time + self.prefill_time, 1e-9)

    @property
    def latency_per_step(self) -> float:
        return self.decode_time / max(self.steps, 1)

    def head_acceptance_rates(self) -> List[float]:
        """P(head k's token accepted in a step). Accepting head k requires heads 1..k-1 too."""
        return [a / max(self.steps, 1) for a in self.head_accepts]

    def as_dict(self) -> Dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "steps": self.steps,
            "prefill_time_s": self.prefill_time,
            "decode_time_s": self.decode_time,
            "tokens_per_second": self.tokens_per_second,
            "acceleration_rate": self.acceleration_rate,
            "latency_per_step_ms": 1e3 * self.latency_per_step,
            "head_acceptance": self.head_acceptance_rates(),
        }


def aggregate(stats: List[DecodingStats]) -> Dict[str, float]:
    """Pool over prompts: sums first, ratios after (a per-prompt mean would over-weight short outputs)."""
    toks = sum(s.new_tokens for s in stats)
    steps = sum(s.steps for s in stats)
    dec = sum(s.decode_time for s in stats)
    pre = sum(s.prefill_time for s in stats)
    heads = max((len(s.head_accepts) for s in stats), default=0)
    head_acc = [sum(s.head_accepts[k] for s in stats if len(s.head_accepts) > k) / max(steps, 1)
                for k in range(heads)]
    return {
        "prompts": len(stats),
        "new_tokens": toks,
        "steps": steps,
        "tokens_per_second": toks / max(dec + pre, 1e-9),
        "decode_tokens_per_second": toks / max(dec, 1e-9),
        "acceleration_rate": (sum(sum(s.accept_lengths) for s in stats) + steps) / max(steps, 1),
        "latency_per_step_ms": 1e3 * dec / max(steps, 1),
        "head_acceptance": head_acc,
    }


def compare(baseline: Dict[str, float], medusa: Dict[str, float]) -> Dict[str, Optional[float]]:
    overhead = medusa["latency_per_step_ms"] / max(baseline["latency_per_step_ms"], 1e-9)
    return {
        "acceleration_rate": medusa["acceleration_rate"],
        "overhead": overhead,
        "speedup_from_rates": medusa["acceleration_rate"] / max(overhead, 1e-9),
        "speedup_wallclock_decode": medusa["decode_tokens_per_second"] / max(baseline["decode_tokens_per_second"], 1e-9),
        "speedup_wallclock_total": medusa["tokens_per_second"] / max(baseline["tokens_per_second"], 1e-9),
    }
