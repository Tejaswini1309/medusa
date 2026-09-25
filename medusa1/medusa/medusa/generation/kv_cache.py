"""KV cache for tree-structured verification (Architecture.md §3.5, §3.6).

Why a custom cache: HF's DynamicCache grows with torch.cat on every step (a reallocation per
layer per step) and cannot drop the rejected tree branches. This cache preallocates one
contiguous fp16 buffer for all layers, [layers, 2 (k/v), batch, kv_heads, max_len, head_dim],
so that:
  * writes are slice assignments (no reallocation);
  * after acceptance, a single gather/scatter over all layers keeps only the accepted path
    (`compact`).

Memory for Vicuna-7B at max_len=2048+tree: 32*2*2109*4096*2 B ≈ 1.1 GB, all in VRAM.
Only the hot path lives here; putting the KV cache in shared (system) memory would make every
decoding step wait on PCIe.

It implements the parts of the `transformers.Cache` interface used by LlamaModel in
transformers==4.46.x (the pinned version): `update`, `get_seq_length`, `get_max_length`.
"""
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class TreeKVCache(Cache):
    def __init__(self, config, max_len: int, batch_size: int = 1,
                 device: torch.device = None, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        n_kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.max_len = max_len
        self.kv = torch.zeros(self.num_layers, 2, batch_size, n_kv, max_len, head_dim,
                              device=device, dtype=dtype)
        self.length = 0        # number of committed (valid) positions
        self._pending_end = 0  # end of the slots written by the forward pass in progress

    # ---------------------------------------------------------------- Cache API
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write the new keys/values into their cache slots and return all keys/values up to
        the end of those slots. New tokens always go to slots [length, length+q); the decoder
        passes the matching `cache_position`, which is deliberately not read back here (reading a
        CUDA tensor on the host would force a GPU sync in every layer).
        Slots are *storage* indices; RoPE positions come from position_ids and may differ
        (tree nodes at the same depth share a position but own separate slots)."""
        q = key_states.shape[-2]
        start = self.length
        end = start + q
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {end} slots, capacity {self.max_len}")
        self.kv[layer_idx, 0, :, :, start:end] = key_states
        self.kv[layer_idx, 1, :, :, start:end] = value_states
        if layer_idx == self.num_layers - 1:
            self._pending_end = end
        return self.kv[layer_idx, 0, :, :, :end], self.kv[layer_idx, 1, :, :, :end]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def get_max_length(self) -> int:
        return self.max_len

    def get_max_cache_shape(self) -> int:
        return self.max_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.length

    def __len__(self) -> int:
        return self.num_layers

    # ---------------------------------------------------------------- Medusa-specific
    def commit(self, num_new: Optional[int] = None) -> None:
        """Mark positions written by the last forward pass as valid (all of them, or the first
        `num_new`). Used after prefill and after a plain autoregressive step."""
        self.length = self._pending_end if num_new is None else self.length + num_new

    def compact(self, accepted_slots: torch.Tensor) -> None:
        """Keep only the accepted tree nodes (§4.17 step 8).
        accepted_slots: absolute slot indices (length L+1, root first) written by the tree pass.
        They are moved to [length, length+L+1) for every layer in one gather + one copy."""
        n_new = accepted_slots.numel()
        dst = self.length
        gathered = self.kv.index_select(4, accepted_slots)          # [layers,2,B,H,L+1,D]
        self.kv[:, :, :, :, dst:dst + n_new] = gathered
        self.length = dst + n_new
        # Rejected-branch entries beyond `length` are stale but harmless: they get overwritten
        # and the attention mask never exposes slots >= the current write range.

    def reset(self) -> None:
        self.length = 0
        self._pending_end = 0


# ---------------------------------------------------------------------------- masks
def build_tree_attention_mask(ancestor_mask: torch.Tensor, past_len: int,
                              dtype: torch.dtype) -> torch.Tensor:
    """4-D additive mask for the verification pass: [1, 1, N, past_len + N].
    Columns [0, past_len): cached context, visible to every node (§4.4.1 rule 1).
    Columns [past_len, past_len+N): the tree, visible only for self + ancestors (rule 2).
    0 = attend, finfo.min = blocked (the 'inverted' form HF passes straight to SDPA)."""
    n = ancestor_mask.shape[0]
    visible = torch.ones(n, past_len + n, dtype=torch.bool, device=ancestor_mask.device)
    visible[:, past_len:] = ancestor_mask
    mask = torch.zeros(n, past_len + n, dtype=dtype, device=ancestor_mask.device)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask[None, None]
