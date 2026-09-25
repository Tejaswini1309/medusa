"""Training loop (Architecture.md §3.9, §4.9, §4.14, §4.16) for an 8 GB GPU + system RAM.

Memory plan (Vicuna-7B, 5 heads ≈ 739M head parameters, 89% of them in the W_2 vocab projections):

  GPU (VRAM, fast)                         System RAM ("shared memory", over PCIe)
  ------------------------------------     ------------------------------------------
  4-bit backbone              ~3.9 GB      fp32 master copy of the heads    ~3.0 GB
  fp16 working copy of heads  ~1.5 GB      fp32 accumulated head gradients  ~3.0 GB
  one head's fp16 gradient    ~0.3 GB      AdamW moments (fp32, 2 per param) ~5.9 GB
  activations (chunked)       <1 GB        pinned fp16 staging buffers      ~1.5 GB

Stage 1 per micro-batch:
  1. backbone forward under no_grad -> h (frozen backbone, §4.9: no graph is kept).
  2. for each head k: forward+backward on chunks of positions (bounded logits memory), then
     stream that head's fp16 gradient to pinned RAM (async DMA) and free it on the GPU. A
     background thread adds it into the fp32 gradient accumulator while the GPU moves on.
  3. every `global_batch/micro_batch` micro-batches: clip, AdamW step on CPU, and copy the updated
     weights back to the GPU fp16 working copy.
fp16 gradients underflow easily, so the loss is multiplied by a dynamic loss scale and divided
back out in fp32 on the CPU; a step with inf/nan gradients is skipped and the scale halved.
"""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..model.medusa_model import MedusaModel
from ..model.utils import gpu_memory_report, save_heads
from .dataset import EpochSampler, collate
from .loss import IGNORE_INDEX, ce_sum, distillation_loss, head_loss_weights, head_shift, lm_loss

STAGES = ("medusa1", "medusa2_joint", "medusa2_distill")


class DynamicLossScaler:
    def __init__(self, init_scale: float = 2.0 ** 16, growth_interval: int = 2000,
                 factor: float = 2.0, min_scale: float = 1.0):
        self.scale, self.growth_interval, self.factor, self.min_scale = init_scale, growth_interval, factor, min_scale
        self._good = 0

    def update(self, overflow: bool) -> None:
        if overflow:
            self.scale = max(self.scale / self.factor, self.min_scale)
            self._good = 0
        else:
            self._good += 1
            if self._good % self.growth_interval == 0:
                self.scale *= self.factor

    def state_dict(self):
        return {"scale": self.scale, "good": self._good}

    def load_state_dict(self, sd):
        self.scale, self._good = sd["scale"], sd["good"]


class OffloadedHeadOptimizer:
    """fp32 master weights + AdamW in system RAM; fp16 working copy of the heads on the GPU."""

    def __init__(self, heads, lr: float, betas, weight_decay: float, max_grad_norm: float,
                 scaler: DynamicLossScaler):
        self.scaler = scaler
        self.max_grad_norm = max_grad_norm
        self.cuda = next(heads.parameters()).is_cuda
        self.gpu: List[List[torch.nn.Parameter]] = [list(h.parameters()) for h in heads.heads]
        self.names: List[List[str]] = [[f"heads.{k}.{n}" for n, _ in h.named_parameters()]
                                       for k, h in enumerate(heads.heads)]
        self.master: List[List[torch.Tensor]] = []
        self.staging: List[List[torch.Tensor]] = []
        for group in self.gpu:
            m = [p.detach().to("cpu", torch.float32).clone() for p in group]
            for t in m:
                t.grad = torch.zeros_like(t)
            self.master.append(m)
            self.staging.append([torch.empty(p.shape, dtype=p.dtype, pin_memory=self.cuda) for p in group])
        flat = [t for g in self.master for t in g]
        try:
            self.optimizer = torch.optim.AdamW(flat, lr=lr, betas=tuple(betas), weight_decay=weight_decay, fused=True)
        except (RuntimeError, TypeError):
            self.optimizer = torch.optim.AdamW(flat, lr=lr, betas=tuple(betas), weight_decay=weight_decay, foreach=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = [None] * len(self.gpu)

    # -------------------------------------------------------------- gradients
    def offload_head(self, k: int) -> bool:
        """Move head k's (scaled, fp16) GPU gradient into the fp32 CPU accumulator.
        Returns False if it contains inf/nan (the caller then skips the step)."""
        if self.pending[k] is not None:
            self.pending[k].result()                 # staging buffer k must be free again
        grads = [p.grad for p in self.gpu[k]]
        if any(g is None for g in grads):
            return True
        finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        if finite:
            for g, s in zip(grads, self.staging[k]):
                s.copy_(g, non_blocking=self.cuda)   # async DMA into pinned RAM
            event = None
            if self.cuda:
                event = torch.cuda.Event()
                event.record()
            self.pending[k] = self.pool.submit(self._accumulate, k, event, 1.0 / self.scaler.scale)
        for p in self.gpu[k]:
            p.grad = None       # safe: the allocator only reuses it for kernels queued after the copy
        return finite

    def _accumulate(self, k: int, event, inv_scale: float) -> None:
        if event is not None:
            event.synchronize()
        for m, s in zip(self.master[k], self.staging[k]):
            m.grad.add_(s, alpha=inv_scale)         # fp32 += fp16 * (1/scale)

    def wait(self) -> None:
        for i, f in enumerate(self.pending):
            if f is not None:
                f.result()
                self.pending[i] = None

    def zero_grad(self) -> None:
        self.wait()
        for g in self.master:
            for t in g:
                t.grad.zero_()

    # -------------------------------------------------------------- update
    def grad_norm_and_clip(self) -> float:
        self.wait()
        params = [t for g in self.master for t in g]
        return float(torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm))

    def step(self) -> None:
        self.optimizer.step()
        self.zero_grad()
        self.push_to_gpu()

    @torch.no_grad()
    def push_to_gpu(self) -> None:
        """fp32 master -> fp16 staging (CPU) -> GPU working copy (async DMA)."""
        for gpu, master, staging in zip(self.gpu, self.master, self.staging):
            for p, m, s in zip(gpu, master, staging):
                s.copy_(m)
                p.data.copy_(s, non_blocking=self.cuda)
        if self.cuda:
            torch.cuda.synchronize()               # staging buffers are reused by the next backward

    # -------------------------------------------------------------- state
    def master_state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: t for names, g in zip(self.names, self.master) for n, t in zip(names, g)}

    def state_dict(self):
        return {"master": self.master_state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, sd) -> None:
        with torch.no_grad():
            for names, g in zip(self.names, self.master):
                for n, t in zip(names, g):
                    t.copy_(sd["master"][n])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.push_to_gpu()


def _cosine(optimizer, warmup: int, total: int):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(optimizer, warmup, max(total, 1))


class MedusaTrainer:
    def __init__(self, model: MedusaModel, cfg: Dict, train_ds, eval_ds, pad_id: int,
                 stage: Optional[str] = None):
        t = cfg["train"]
        self.cfg, self.t = cfg, t
        self.stage = stage or t["stage"]
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.model, self.train_ds, self.eval_ds, self.pad_id = model, train_ds, eval_ds, pad_id
        self.device = next(model.heads.parameters()).device
        self.K = model.heads.num_heads
        self.weights = head_loss_weights(self.K, t["lambda_base"])
        if t["global_batch_size"] % t["micro_batch_size"]:
            raise ValueError("global_batch_size must be a multiple of micro_batch_size")
        self.accum = t["global_batch_size"] // t["micro_batch_size"]
        self.steps_per_epoch = len(train_ds) // t["global_batch_size"]
        self.total_steps = self.steps_per_epoch * t["epochs"]
        if self.total_steps == 0:
            raise ValueError("dataset smaller than one global batch")
        self.chunk = t["head_chunk_tokens"]
        self.out_dir = cfg["output_dir"]
        os.makedirs(self.out_dir, exist_ok=True)

        self.scaler = DynamicLossScaler(t["loss_scale_init"], t["loss_scale_growth_interval"])
        self.head_opt = OffloadedHeadOptimizer(model.heads, t["heads_lr"], t["adam_betas"],
                                               t["weight_decay"], t["max_grad_norm"], self.scaler)
        self.head_sched = _cosine(self.head_opt.optimizer, t["warmup_steps"], self.total_steps)

        self.lora_params: List[torch.nn.Parameter] = []
        self.lora_opt = self.lora_sched = None
        if self.stage == "medusa1":
            model.freeze_backbone()                                   # §4.16 mode 1, steps 2 & 13
        else:
            self._setup_medusa2()

        self.step = 0
        self.epoch = 0
        self.epoch_offset = 0          # samples of the current epoch already consumed
        self.log_path = os.path.join(self.out_dir, "train_log.jsonl")

    # ------------------------------------------------------------------ Medusa-2 setup (§4.13)
    def _setup_medusa2(self) -> None:
        from peft import LoraConfig, prepare_model_for_kbit_training

        lc = self.cfg["medusa2"]["lora"]
        if getattr(self.model.backbone, "is_loaded_in_4bit", False):
            self.model.backbone = prepare_model_for_kbit_training(
                self.model.backbone, use_gradient_checkpointing=self.t["gradient_checkpointing"])
        elif self.t["gradient_checkpointing"]:
            self.model.backbone.gradient_checkpointing_enable()
        self.model.attach_lora(LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                                          target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
        self.lora_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        try:   # paper App. B.2: 8-bit AdamW; the paged variant spills its state to system RAM under pressure
            import bitsandbytes as bnb
            self.lora_opt = bnb.optim.PagedAdamW8bit(self.lora_params, lr=self.t["backbone_lr"],
                                                     betas=tuple(self.t["adam_betas"]), weight_decay=self.t["weight_decay"])
        except Exception:
            self.lora_opt = torch.optim.AdamW(self.lora_params, lr=self.t["backbone_lr"],
                                              betas=tuple(self.t["adam_betas"]), weight_decay=self.t["weight_decay"])
        self.lora_sched = _cosine(self.lora_opt, self.t["warmup_steps"], self.total_steps)
        for p in self.model.heads.parameters():
            p.requires_grad_(True)

    def _lambda0(self) -> float:
        m2 = self.cfg["medusa2"]
        if m2.get("lambda0_schedule", "constant") == "sine":      # paper App. B.4
            return m2["lambda0"] * math.sin(0.5 * math.pi * min(self.step / self.total_steps, 1.0))
        return m2["lambda0"]

    # ------------------------------------------------------------------ head losses (chunked)
    def _heads_pass(self, h: torch.Tensor, labels: torch.Tensor, loss_mult: float,
                    backward: bool, stats: Dict[str, float]) -> bool:
        """Σ_k λ_k · mean CE of head k, computed chunk-by-chunk over positions.
        backward=True: each chunk's loss is back-propagated immediately (only one chunk of logits
        is ever alive) and head k's gradient is offloaded before head k+1 starts.
        Returns False if a gradient overflowed."""
        T = labels.shape[1]
        scale = self.scaler.scale if backward else 1.0
        finite = True
        for k in range(self.K):
            s = head_shift(k)
            if T <= s:
                continue
            tgt = labels[:, s:]
            valid = tgt != IGNORE_INDEX
            n = int(valid.sum())
            if n == 0:
                continue
            coef = self.weights[k] * loss_mult / n
            hk = h[:, : T - s]
            for c0 in range(0, T - s, self.chunk):
                c1 = min(c0 + self.chunk, T - s)
                if not bool(valid[:, c0:c1].any()):
                    continue                                          # user-turn chunk: no loss, skip compute
                logits = self.model.heads.forward_head(k, hk[:, c0:c1])
                loss_sum, nv, correct = ce_sum(logits, tgt[:, c0:c1])
                if backward:
                    (loss_sum * (coef * scale)).backward()
                stats[f"medusa{k}_loss_sum"] = stats.get(f"medusa{k}_loss_sum", 0.0) + float(loss_sum)
                stats[f"medusa{k}_n"] = stats.get(f"medusa{k}_n", 0) + int(nv)
                stats[f"medusa{k}_correct"] = stats.get(f"medusa{k}_correct", 0) + int(correct)
                del logits, loss_sum
            if backward:
                finite &= self.head_opt.offload_head(k)
        return finite

    # ------------------------------------------------------------------ micro-steps
    def _micro_medusa1(self, batch, stats) -> bool:
        with torch.no_grad():                                         # frozen backbone: no graph
            h = self.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                           medusa_forward=False, lm_logits_to_keep=1).hidden_states.detach()
        return self._heads_pass(h, batch["labels"], 1.0 / self.accum, backward=True, stats=stats)

    def _micro_medusa2(self, batch, stats) -> bool:
        ids, attn, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        self.model.backbone.train()
        teacher = None
        if self.stage == "medusa2_distill":                          # teacher = LoRA OFF, no grad (§4.13)
            with torch.no_grad(), self.model.backbone.disable_adapter():
                teacher = self.model(ids, attention_mask=attn, medusa_forward=False).lm_logits.detach()
        out = self.model(ids, attention_mask=attn, medusa_forward=False)   # student = LoRA ON
        h = out.hidden_states
        h_leaf = h.detach().requires_grad_(True)
        lam0 = self._lambda0()
        # Heads: backward into h_leaf (collects dL/dh) + head params; head grads offloaded per head.
        finite = self._heads_pass(h_leaf, labels, lam0 / self.accum, backward=True, stats=stats)
        if teacher is not None:
            bb = distillation_loss(out.lm_logits, teacher, attn.bool())
        else:
            bb = lm_loss(out.lm_logits, labels)
        stats["backbone_loss_sum"] = stats.get("backbone_loss_sum", 0.0) + float(bb)
        stats["backbone_n"] = stats.get("backbone_n", 0) + 1
        roots = [bb * (self.scaler.scale / self.accum)]
        grads = [None]
        if h_leaf.grad is not None:                                   # route head gradients into the backbone
            roots.append(h)
            grads.append(h_leaf.grad)
        torch.autograd.backward(roots, grads)
        return finite

    # ------------------------------------------------------------------ optimizer step
    def _lora_unscale_and_check(self) -> bool:
        if not self.lora_params:
            return True
        inv = 1.0 / self.scaler.scale
        finite = True
        for p in self.lora_params:
            if p.grad is not None:
                p.grad.mul_(inv)
                finite &= bool(torch.isfinite(p.grad).all())
        return finite

    def _zero_all(self) -> None:
        self.head_opt.zero_grad()
        if self.lora_opt:
            self.lora_opt.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> Dict[str, float]:
        lora_ok = self._lora_unscale_and_check()
        if not lora_ok:
            self._zero_all()
            self.scaler.update(overflow=True)
            return {"skipped": 1}
        head_norm = self.head_opt.grad_norm_and_clip()
        self.head_opt.step()
        self.head_sched.step()
        out = {"grad_norm_heads": head_norm, "lr_heads": self.head_sched.get_last_lr()[0]}
        if self.lora_opt:
            out["grad_norm_lora"] = float(torch.nn.utils.clip_grad_norm_(self.lora_params, self.t["max_grad_norm"]))
            self.lora_opt.step()
            self.lora_opt.zero_grad(set_to_none=True)
            self.lora_sched.step()
            out["lr_backbone"] = self.lora_sched.get_last_lr()[0]
        self.scaler.update(overflow=False)
        return out

    # ------------------------------------------------------------------ loop
    def _loader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.t["micro_batch_size"],
                          sampler=EpochSampler(len(self.train_ds), self.cfg["seed"], self.epoch, self.epoch_offset),
                          collate_fn=partial(collate, pad_id=self.pad_id),   # picklable for Windows workers
                          num_workers=self.cfg["data"].get("num_workers", 0),
                          pin_memory=self.device.type == "cuda", drop_last=False)

    def train(self) -> str:
        from tqdm import tqdm

        micro_fn = self._micro_medusa1 if self.stage == "medusa1" else self._micro_medusa2
        print(gpu_memory_report("start"))
        bar = tqdm(total=self.total_steps, initial=self.step, desc=self.stage)
        while self.epoch < self.t["epochs"]:
            steps_this_epoch = self.epoch_offset // self.t["global_batch_size"]
            micro, stats, tokens, t0 = 0, {}, 0, time.time()
            for batch in self._loader():
                if steps_this_epoch >= self.steps_per_epoch:
                    break
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                self.epoch_offset += batch["input_ids"].shape[0]
                tokens += int(batch["attention_mask"].sum())
                if not micro_fn(batch, stats):                        # overflow: drop this window
                    self._zero_all()
                    self.scaler.update(overflow=True)
                    self._log({"step": self.step, "event": "overflow", "new_scale": self.scaler.scale})
                    micro, stats, tokens = 0, {}, 0
                    continue
                micro += 1
                if micro < self.accum:
                    continue
                upd = self._optimizer_step()
                if upd.get("skipped"):
                    self._log({"step": self.step, "event": "overflow", "new_scale": self.scaler.scale})
                    micro, stats, tokens = 0, {}, 0
                    continue
                self.step += 1
                steps_this_epoch += 1
                bar.update(1)
                if self.step % self.t["log_every"] == 0:
                    rec = {"step": self.step, "epoch": self.epoch, "loss_scale": self.scaler.scale,
                           "tokens_per_s": tokens / max(time.time() - t0, 1e-9), **upd, **self._summarise(stats)}
                    rec["gpu_peak_gib"] = (torch.cuda.max_memory_allocated() / 2**30) if self.device.type == "cuda" else 0.0
                    self._log(rec)
                    bar.set_postfix(loss=f"{rec.get('medusa0_loss', float('nan')):.3f}",
                                    top1=f"{rec.get('medusa0_top1', float('nan')):.3f}")
                micro, stats, tokens, t0 = 0, {}, 0, time.time()
                if self.step % self.t["eval_every"] == 0:
                    self._log({"step": self.step, "eval": self.evaluate()})
                if self.step % self.t["save_every"] == 0:
                    self.save_checkpoint()
            self.epoch += 1
            self.epoch_offset = 0
            self._log({"step": self.step, "epoch_end": self.epoch, "eval": self.evaluate()})
            self.save_checkpoint()
        bar.close()
        return self.save_final()

    def _summarise(self, stats: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k in range(self.K):
            n = stats.get(f"medusa{k}_n", 0)
            if n:
                out[f"medusa{k}_loss"] = stats[f"medusa{k}_loss_sum"] / n
                out[f"medusa{k}_top1"] = stats[f"medusa{k}_correct"] / n
        if stats.get("backbone_n"):
            out["backbone_loss"] = stats["backbone_loss_sum"] / stats["backbone_n"]
        return out

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Per-head loss / top-1 accuracy on held-out conversations (never trained on)."""
        if len(self.eval_ds) == 0:
            return {}
        self.model.eval()
        stats: Dict[str, float] = {}
        for i in range(len(self.eval_ds)):
            batch = collate([self.eval_ds[i]], self.pad_id)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            h = self.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                           medusa_forward=False, lm_logits_to_keep=1).hidden_states
            self._heads_pass(h, batch["labels"], 1.0, backward=False, stats=stats)
        if self.stage != "medusa1":
            self.model.backbone.train()
        return self._summarise(stats)

    # ------------------------------------------------------------------ checkpoints
    def _log(self, rec: Dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def save_checkpoint(self) -> None:
        d = os.path.join(self.out_dir, "checkpoint")
        os.makedirs(d, exist_ok=True)
        self.head_opt.wait()
        save_heads(self.head_opt.master_state_dict(), os.path.join(d, "medusa_heads.safetensors"),
                   {"step": self.step, "stage": self.stage})
        if self.t.get("save_optimizer_state", True):
            state = {"step": self.step, "epoch": self.epoch, "epoch_offset": self.epoch_offset,
                     "stage": self.stage, "scaler": self.scaler.state_dict(),
                     "head_opt": self.head_opt.state_dict(), "head_sched": self.head_sched.state_dict()}
            if self.lora_opt:
                state["lora_opt"] = self.lora_opt.state_dict()
                state["lora_sched"] = self.lora_sched.state_dict()
                state["lora"] = {n: p.detach().cpu() for n, p in self.model.backbone.named_parameters() if p.requires_grad}
            tmp = os.path.join(d, "trainer_state.pt.tmp")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(d, "trainer_state.pt"))   # atomic: a crash never leaves half a file

    def load_checkpoint(self) -> bool:
        path = os.path.join(self.out_dir, "checkpoint", "trainer_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["stage"] != self.stage:
            raise ValueError(f"checkpoint is for stage {state['stage']}, not {self.stage}")
        self.step, self.epoch, self.epoch_offset = state["step"], state["epoch"], state["epoch_offset"]
        self.scaler.load_state_dict(state["scaler"])
        self.head_opt.load_state_dict(state["head_opt"])
        self.head_sched.load_state_dict(state["head_sched"])
        if self.lora_opt and "lora_opt" in state:
            named = dict(self.model.backbone.named_parameters())
            with torch.no_grad():
                for n, v in state["lora"].items():
                    named[n].copy_(v)
            self.lora_opt.load_state_dict(state["lora_opt"])
            self.lora_sched.load_state_dict(state["lora_sched"])
        print(f"[resume] step {self.step}, epoch {self.epoch}, offset {self.epoch_offset}")
        return True

    def save_final(self) -> str:
        path = os.path.join(self.out_dir, "medusa_heads.safetensors")
        self.head_opt.wait()
        save_heads(self.head_opt.master_state_dict(), path, {"step": self.step, "stage": self.stage})
        if self.lora_opt:
            self.model.backbone.save_pretrained(os.path.join(self.out_dir, "lora_adapter"))
        return path
