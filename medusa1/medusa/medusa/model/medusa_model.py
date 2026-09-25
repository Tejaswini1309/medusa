"""Backbone LLM + Medusa heads in one module (Architecture.md §3.1, §3.3, §4.13).

The forward pass calls the *full* backbone with `output_hidden_states=True` and reads
  * the last hidden-state layer  (h_t, already passed through the final RMSNorm), and
  * the LM head's own logits     (the base next-token prediction)
from that single call (§3.3). Nothing bypasses the LM head, so the same code works for a
plain HF model and for a PEFT/LoRA-wrapped one.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .medusa_head import MedusaHeads


@dataclass
class MedusaOutput:
    lm_logits: torch.Tensor                    # [B, T', V]  (T' = T, or fewer if lm_logits_to_keep > 0)
    hidden_states: torch.Tensor                # [B, T, d]   last layer, post-norm
    medusa_logits: Optional[torch.Tensor]      # [K, B, T, V] or None
    past_key_values: Optional[object] = None


class MedusaModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_heads: int, num_layers: int = 1):
        super().__init__()
        self.backbone = backbone
        cfg = backbone.config
        self.heads = MedusaHeads(cfg.hidden_size, cfg.vocab_size, num_heads, num_layers)
        self.heads.init_from_lm_head(self.get_lm_head().weight)
        self.lora_attached = False

    # ------------------------------------------------------------------ helpers
    @property
    def config(self):
        return self.backbone.config

    def get_lm_head(self) -> nn.Module:
        # Works for both plain HF models and PEFT-wrapped ones (PEFT forwards the call).
        head = self.backbone.get_output_embeddings()
        return getattr(head, "base_layer", head)   # un-wrap a LoRA layer if present

    def freeze_backbone(self) -> None:
        """Stage 1 (§3.1): backbone frozen, heads trainable."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        for p in self.heads.parameters():
            p.requires_grad_(True)

    def attach_lora(self, lora_config) -> None:
        """§4.13: wrap the backbone with PEFT. Base weights are frozen and shared; only the
        LoRA adapter is trainable. Teacher = `with self.backbone.disable_adapter():`."""
        from peft import get_peft_model

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.lora_attached = True

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        medusa_forward: bool = True,
        lm_logits_to_keep: int = 0,
    ) -> MedusaOutput:
        """lm_logits_to_keep=0 returns LM logits for every position; =1 only for the last one
        (saves a [T, 32000] matmul when only h_t is needed, e.g. Medusa-1 training)."""
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            num_logits_to_keep=lm_logits_to_keep,
        )
        hidden = out.hidden_states[-1]
        medusa_logits = self.heads(hidden) if medusa_forward else None
        return MedusaOutput(
            lm_logits=out.logits,
            hidden_states=hidden,
            medusa_logits=medusa_logits,
            past_key_values=out.past_key_values if use_cache else None,
        )
