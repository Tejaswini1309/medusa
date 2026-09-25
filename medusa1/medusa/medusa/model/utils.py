"""Config loading, 4-bit weight loading and Medusa model initialisation (Architecture.md §3.3, §3.10)."""
import copy
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .medusa_model import MedusaModel

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------- config
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config. A `base:` key (relative path) is loaded first and deep-merged under it."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), base))
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. Install the CUDA build of PyTorch (see README.md) and check `nvidia-smi`."
        )
    return torch.device("cuda:0")


def gpu_memory_report(tag: str = "") -> str:
    if not torch.cuda.is_available():
        return f"[mem {tag}] cpu only"
    alloc = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    return (f"[mem {tag}] allocated={alloc:.2f} GiB peak={peak:.2f} GiB "
            f"free={free / 2**30:.2f}/{total / 2**30:.2f} GiB")


# ---------------------------------------------------------------------------- loading
def load_tokenizer(cfg: Dict[str, Any]):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token   # Vicuna has no pad token; unk (id 0) is the usual choice
    return tok


def build_bnb_config(cfg: Dict[str, Any]):
    q = cfg["model"].get("quantization") or {}
    if not q.get("load_in_4bit", False):
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=_DTYPES[q.get("bnb_4bit_compute_dtype", "float16")],
    )


def load_backbone(cfg: Dict[str, Any], device: Optional[torch.device] = None):
    """Load the target LLM. With 4-bit NF4 every linear layer of the 32 decoder blocks is
    quantised (~3.6 GB); bitsandbytes keeps `lm_head` and the embeddings in fp16."""
    from transformers import AutoModelForCausalLM

    device = device or get_device()
    mcfg = cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name_or_path"],
        torch_dtype=_DTYPES[mcfg.get("torch_dtype", "float16")],
        quantization_config=build_bnb_config(cfg),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
        device_map={"": device.index or 0},   # everything on the GPU: no silent CPU offload of layers
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def build_medusa_model(
    cfg: Dict[str, Any],
    heads_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    backbone=None,
) -> MedusaModel:
    """Backbone + K heads. Heads are initialised from the LM head (§4.1.1) and optionally
    overwritten with trained weights from `heads_path`. Heads live on the GPU in the backbone's
    compute dtype (fp16): this is the working copy used for inference and for training forward/backward."""
    device = device or get_device()
    backbone = backbone if backbone is not None else load_backbone(cfg, device)
    model = MedusaModel(backbone, cfg["medusa"]["num_heads"], cfg["medusa"].get("num_layers", 1))
    dtype = model.get_lm_head().weight.dtype
    model.heads.to(device=device, dtype=dtype)
    if heads_path:
        load_heads(model, heads_path)
    return model


# ---------------------------------------------------------------------------- head I/O
def save_heads(state_dict: Dict[str, torch.Tensor], path: str, meta: Optional[Dict[str, str]] = None) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tensors = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in state_dict.items()}
    save_file(tensors, path, metadata={k: str(v) for k, v in (meta or {}).items()})


def load_heads(model: MedusaModel, path: str) -> None:
    from safetensors.torch import load_file

    sd = load_file(path)
    ref = next(model.heads.parameters())
    missing, unexpected = model.heads.load_state_dict(
        {k: v.to(ref.device, ref.dtype) for k, v in sd.items()}, strict=False
    )
    if missing or unexpected:
        raise ValueError(f"Head checkpoint mismatch. missing={missing} unexpected={unexpected}. "
                         "Check medusa.num_heads / num_layers in the config.")
