"""Training data (Architecture.md §3.9, §4.11).

ShareGPT conversations are rendered with the Vicuna v1.1 chat template (the format Vicuna was
fine-tuned on, so the heads see the same distribution they will see at inference):

    "<system> USER: <q1> ASSISTANT: <a1></s>USER: <q2> ASSISTANT: <a2></s>"

Only assistant tokens carry a loss (user turns and the system prompt get IGNORE_INDEX), matching
the official Medusa training script. Tokenisation runs once; the result is cached to
`data.cache_dir` as a .npz and reused by every run.
"""
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .loss import IGNORE_INDEX

VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")
_ROLE = {"human": "USER", "user": "USER", "gpt": "ASSISTANT", "chatgpt": "ASSISTANT",
         "bard": "ASSISTANT", "bing": "ASSISTANT", "assistant": "ASSISTANT"}


# ---------------------------------------------------------------------------- template
def clean_turns(conversation: Sequence[Dict]) -> List[Tuple[str, str]]:
    """ShareGPT turns -> alternating [(USER, text), (ASSISTANT, text), ...]. Leading assistant turns
    are dropped and the conversation is cut at the first role violation or unknown role."""
    turns = []
    for t in conversation:
        role = _ROLE.get(str(t.get("from", "")).lower())
        if role is None:
            break
        if not turns and role != "USER":
            continue
        expected = "USER" if len(turns) % 2 == 0 else "ASSISTANT"
        if role != expected:
            break
        turns.append((role, str(t.get("value", ""))))
    return turns


def render_vicuna(turns: Sequence[Tuple[str, Optional[str]]], eos: str = "</s>") -> Tuple[str, List[Tuple[int, int]]]:
    """Render turns; return (text, char spans of assistant replies incl. their </s>).
    A final (ASSISTANT, None) turn renders as the open prompt "... ASSISTANT:" for generation."""
    text = VICUNA_SYSTEM + " "
    spans = []
    for role, msg in turns:
        if msg is None:
            text += f"{role}:"
            break
        if role == "USER":
            text += f"USER: {msg} "
        else:
            text += "ASSISTANT: "
            start = len(text)
            text += msg + eos
            spans.append((start, len(text)))
    return text, spans


def build_vicuna_prompt(history: Sequence[Tuple[str, str]], user_msg: str) -> str:
    """Inference prompt: previous (user, assistant) pairs + the new user message."""
    turns: List[Tuple[str, Optional[str]]] = []
    for u, a in history:
        turns += [("USER", u), ("ASSISTANT", a)]
    turns += [("USER", user_msg), ("ASSISTANT", None)]
    return render_vicuna(turns)[0]


def tokenize_batch(tokenizer, conversations: Sequence[Sequence[Dict]], max_len: int):
    """Returns a list of (input_ids int32, loss_mask bool) per conversation (None if unusable)."""
    rendered = [render_vicuna(clean_turns(c), tokenizer.eos_token) for c in conversations]
    enc = tokenizer([r[0] for r in rendered], return_offsets_mapping=True, truncation=True,
                    max_length=max_len, add_special_tokens=True)
    out = []
    for (text, spans), ids, offs in zip(rendered, enc["input_ids"], enc["offset_mapping"]):
        if not spans:
            out.append(None)
            continue
        offs = np.asarray(offs, dtype=np.int64)
        starts, ends = offs[:, 0], offs[:, 1]
        mask = np.zeros(len(ids), dtype=bool)
        for s, e in spans:                       # token overlaps an assistant reply -> trainable
            mask |= (ends > s) & (starts < e)
        mask &= ends > starts                    # special tokens with empty span (BOS) excluded
        # `</s>` is a special token: its offsets are its own chars, so it is included above.
        out.append((np.asarray(ids, dtype=np.int32), mask) if mask.any() else None)
    return out


# ---------------------------------------------------------------------------- ShareGPT
def download_sharegpt(cfg) -> str:
    from huggingface_hub import hf_hub_download

    d = cfg["data"]
    return hf_hub_download(repo_id=d["dataset_repo"], filename=d["dataset_file"], repo_type="dataset")


def load_conversations(path: str) -> List[List[Dict]]:
    """Reads ShareGPT .json (list) or .jsonl (one record per line, e.g. self-distillation output)."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
    return [r["conversations"] for r in records if isinstance(r, dict) and r.get("conversations")]


def cache_path(cfg, tag: str = "sharegpt") -> str:
    d = cfg["data"]
    name = cfg["model"]["name_or_path"].replace("/", "__")
    return os.path.join(d["cache_dir"], f"{tag}_{name}_len{d['max_seq_len']}.npz")


def prepare_dataset(cfg, tokenizer, source_path: Optional[str] = None, tag: str = "sharegpt",
                    force: bool = False, batch: int = 512) -> str:
    """Tokenise once and store: concatenated ids/masks + offsets. Returns the cache path."""
    from tqdm import tqdm

    out = cache_path(cfg, tag)
    if os.path.exists(out) and not force:
        return out
    source_path = source_path or download_sharegpt(cfg)
    convs = load_conversations(source_path)
    ids_all, mask_all, lengths = [], [], []
    for i in tqdm(range(0, len(convs), batch), desc="tokenizing"):
        for item in tokenize_batch(tokenizer, convs[i:i + batch], cfg["data"]["max_seq_len"]):
            if item is None:
                continue
            ids_all.append(item[0])
            mask_all.append(item[1])
            lengths.append(len(item[0]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tmp = out + ".tmp.npz"
    np.savez(tmp, ids=np.concatenate(ids_all), mask=np.concatenate(mask_all), offsets=offsets)
    os.replace(tmp, out)
    n_tok = int(offsets[-1])
    n_train = int(sum(m.sum() for m in mask_all))
    print(f"[data] {len(lengths)} usable / {len(convs)} conversations, {n_tok:,} tokens, "
          f"{n_train:,} ({100 * n_train / max(n_tok, 1):.1f}%) carry loss -> {out}")
    return out


class TokenizedConversations(Dataset):
    def __init__(self, npz_path: str, indices: Optional[Sequence[int]] = None):
        data = np.load(npz_path)
        self.ids, self.mask, self.offsets = data["ids"], data["mask"], data["offsets"]
        self.indices = np.arange(len(self.offsets) - 1) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j = self.indices[i]
        s, e = self.offsets[j], self.offsets[j + 1]
        ids = torch.from_numpy(self.ids[s:e].astype(np.int64))
        labels = ids.clone()
        labels[~torch.from_numpy(self.mask[s:e])] = IGNORE_INDEX
        return {"input_ids": ids, "labels": labels}


def split_dataset(npz_path: str, eval_samples: int, max_train_samples: Optional[int], seed: int):
    """Deterministic shuffle -> last `eval_samples` are held out, never trained on."""
    n = len(np.load(npz_path)["offsets"]) - 1
    perm = np.random.RandomState(seed).permutation(n)
    eval_idx, train_idx = perm[:eval_samples], perm[eval_samples:]
    if max_train_samples:
        train_idx = train_idx[:max_train_samples]
    return TokenizedConversations(npz_path, train_idx), TokenizedConversations(npz_path, eval_idx)


def collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    T = max(b["input_ids"].numel() for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].numel()
        ids[i, :n], labels[i, :n], attn[i, :n] = b["input_ids"], b["labels"], 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


class EpochSampler(Sampler):
    """Seeded permutation per epoch, starting at `start` (so a resumed run continues exactly)."""

    def __init__(self, n: int, seed: int, epoch: int, start: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def __iter__(self) -> Iterable[int]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start


# ---------------------------------------------------------------------------- self-distillation (§4.11)
@torch.inference_mode()
def generate_self_distillation_data(backbone, tokenizer, seed_conversations: Sequence[Sequence[Dict]],
                                    out_path: str, temperature: float = 0.3, top_p: float = 1.0,
                                    max_new_tokens: int = 512, max_context: int = 2048,
                                    max_conversations: Optional[int] = None) -> str:
    """Feed each seed conversation's user prompts to the *original* model one after another and
    record the model's own replies (§4.11.2). Output: ShareGPT-style .jsonl, appended so an
    interrupted run resumes. Kept separate from the optimisation loop (§4.11.3); train on the
    result with `prepare_dataset(cfg, tok, source_path=out_path, tag="selfdistill")`."""
    from tqdm import tqdm

    done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
    device = next(backbone.parameters()).device
    convs = list(seed_conversations)[: max_conversations] if max_conversations else list(seed_conversations)
    with open(out_path, "a", encoding="utf-8") as f:
        for conv in tqdm(convs[done:], desc="self-distillation", initial=done, total=len(convs)):
            prompts = [msg for role, msg in clean_turns(conv) if role == "USER"]
            history: List[Tuple[str, str]] = []
            for p in prompts:
                text = build_vicuna_prompt(history, p)
                ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                if ids.shape[1] + max_new_tokens > max_context:
                    break
                gen = backbone.generate(ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                        temperature=max(temperature, 1e-5), top_p=top_p,
                                        pad_token_id=tokenizer.pad_token_id)
                history.append((p, tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()))
            record = {"conversations": [t for u, a in history
                                        for t in ({"from": "human", "value": u}, {"from": "gpt", "value": a})]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    return out_path
