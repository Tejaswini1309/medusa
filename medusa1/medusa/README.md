# Medusa-1 on Vicuna-7B (4-bit) for an RTX 2070 Super (8 GB)

Re-implementation of **MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding
Heads** (Cai et al., 2024, arXiv:2401.10774), following `../Architecture.md`.

## What it does, in brief
A normal LLM produces one token per forward pass. Medusa attaches K small **heads** to the last hidden
state. Head k guesses the token k+1 steps ahead. Each step, the heads' top guesses are arranged as a
**tree** of candidate continuations, and the base model checks the whole tree in **one** forward pass
using a **tree attention mask** (each candidate sees only its own ancestors). The longest candidate
prefix that the base model agrees with is accepted, so one forward pass can yield several tokens.

## Settings (paper → this repo)
| item | paper | here |
|---|---|---|
| backbone | Vicuna-7B v1.5 (§3.1) | `lmsys/vicuna-7b-v1.5`, 4-bit NF4 (App. B.3 also uses a 4-bit backbone) |
| data | ShareGPT, 2 epochs | `Aeala/ShareGPT_Vicuna_unfiltered`, 2 epochs |
| heads | 5 heads × 1 layer, λ_k = 0.8^k | same |
| optimiser | 8-bit AdamW, cosine, warmup 40, lr 2e-3, batch 64 | **32-bit AdamW on the CPU**, cosine, warmup 40, lr 2e-3, batch 64 (1 × 64 accumulation) |
| seq length | not stated for 7B | 2048 |
| precision | bf16 (A100) | fp16 + dynamic loss scaling (Turing has no bf16) |
| tree | optimised sparse tree, 64 nodes | Cartesian tree `[4,2,2,1,1]` → 61 nodes (Architecture.md §4.2) |
| acceptance | typical acceptance ε, δ | ε = 0.09, δ = 0.3 (= √ε, paper §3.3.2); temperature 0 → greedy |

## GPU and shared-memory plan
Windows' "shared GPU memory" is system RAM that the GPU reaches over PCIe, roughly 10–30× slower
than VRAM. So only the per-token hot path stays in VRAM, and bulky state that is touched once per
optimiser step lives in RAM:

| VRAM (8 GB) | system RAM |
|---|---|
| 4-bit backbone ≈ 3.9 GB | fp32 master weights of the heads ≈ 3.0 GB |
| fp16 heads (working copy) ≈ 1.5 GB | fp32 gradient accumulator ≈ 3.0 GB |
| one head's fp16 gradient ≈ 0.3 GB | AdamW moments ≈ 5.9 GB |
| chunked activations < 1 GB | pinned staging buffers (async DMA) ≈ 1.5 GB |

Training needs about **14 GB of free system RAM**. At inference, everything (backbone, heads, a
preallocated ≈1.1 GB KV cache) fits in VRAM.

**Turn off the NVIDIA "sysmem fallback".** If VRAM overflows, the Windows driver silently spills
into shared memory and everything slows down with no error. In *NVIDIA Control Panel → Manage 3D
settings → CUDA – Sysmem Fallback Policy*, choose **Prefer No Sysmem Fallback**. You then get a clear
out-of-memory error instead, and the benchmark numbers stay honest.

## Setup (Windows, PowerShell)
```powershell
cd "C:\Users\SAI TEJASWINI\OneDrive\Desktop\llm inference\medusa1\medusa"
py -3.11 -m venv .venv            # Python 3.10/3.11 (your system Python 3.14 has no matching wheels)
.venv\Scripts\Activate.ps1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                  # makes `import medusa` work for tests and scripts
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
* This folder is inside **OneDrive**. `outputs/` (≈9 GB checkpoints) and `data/` would get synced.
  Pause OneDrive, or point `output_dir` / `data.cache_dir` in `configs/base_config.yaml` to a
  folder outside OneDrive (e.g. `D:/medusa_runs`).
* The model (≈13.5 GB fp16, quantised to 4-bit while loading) and ShareGPT (≈0.7 GB) are downloaded
  into the Hugging Face cache (`%USERPROFILE%\.cache\huggingface`) the first time.

## Order in which to run the files
Run everything from this `medusa/` folder, with the venv active.

| # | command | what it does |
|---|---|---|
| 1 | `python -m pytest tests -q` | unit tests on a tiny random model (CPU is fine): head init, tree/mask, KV cache, and **greedy Medusa output == greedy baseline** |
| 2 | `python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml --prepare-data-only --inspect 2` | downloads ShareGPT, tokenises it with the Vicuna template, caches `data/*.npz`, prints two samples with the tokens that carry loss |
| 3 | `python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml --max-train-samples 640 --output-dir outputs/pilot` | **pilot** (2 epochs × 10 optimiser steps): checks memory and speed before committing days of GPU time |
| 4 | `python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml` | **Medusa-1 training** (2 epochs). If interrupted, run again with `--resume` |
| 5 | `python scripts/evaluate.py --config configs/vicuna_7b_medusa.yaml --heads outputs/vicuna_7b_medusa1/medusa_heads.safetensors` | baseline vs Medusa on MT-Bench first turns: tokens/s, acceleration rate, overhead, speedup, per-head acceptance |
| 6 | `python scripts/run_inference.py --config configs/vicuna_7b_medusa.yaml --heads outputs/vicuna_7b_medusa1/medusa_heads.safetensors` | interactive chat (`--baseline` for comparison, `--temperature 0.7` for typical acceptance) |

Files the scripts call internally (you don't run these directly):
`configs/*.yaml → medusa/model/utils.py (4-bit load) → medusa/model/medusa_model.py + medusa_head.py →
medusa/train/dataset.py → loss.py → trainer.py` for training, and
`medusa/generation/tree.py → kv_cache.py → decode.py (+ utils/sampling.py, utils/metrics.py)` for inference.

## What to check
* **Pilot run:** `outputs/pilot/train_log.jsonl` should show `gpu_peak_gib` below ~7 GB and
  `medusa0_top1` rising. Multiply the time per step by `steps` (printed at start) to get the
  full-run time. Rough guess (not measured): the paper reports ~5 h on one A100 for 60k samples,
  and a 2070 Super is several times slower, so expect **days** for 2 epochs on the full split. Use
  `--max-train-samples 60000` to match the paper's sample count.
* **Head accuracy:** the eval entries in the log report top-1 accuracy per head on 200 held-out
  conversations. Head 0 (paper's head 1) should be the most accurate, and accuracy should fall as k
  grows.
* **Greedy correctness:** at temperature 0, `evaluate.py` reports how many outputs are identical to
  the baseline. They should almost all match. A few can differ because fp16 kernels round differently
  for 61-token and 1-token inputs, which can flip near-ties. Many mismatches mean a bug.
* **If the speedup is low:** bitsandbytes 4-bit has a fast kernel for 1-token decoding, but the
  61-token tree pass dequantises the weights, so the per-step *overhead* is higher than in the fp16
  A100 setting of the paper. Try a smaller tree, e.g. `generation.tree_topk: [3, 2, 1, 1]`.

## Medusa-2 / self-distillation (library support only)
`trainer.py` also implements `medusa2_joint` (QLoRA backbone + heads, L = L_LM + λ₀·L_Medusa-1)
and `medusa2_distill` (teacher = LoRA off, student = LoRA on, L = KL + λ₀·L_Medusa-1).
`dataset.generate_self_distillation_data` builds the synthetic dataset. They were smoke-tested on a
tiny CPU model during development (not part of `tests/`) and were not the target of this setup. With a 4-bit backbone, the teacher is itself
quantised, which the paper warns can hurt quality (§2.3.2).
