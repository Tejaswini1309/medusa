"""Benchmark Medusa vs. standard autoregressive decoding (Architecture.md §3.11).

  python scripts/evaluate.py --config configs/vicuna_7b_medusa.yaml \
      --heads outputs/vicuna_7b_medusa1/medusa_heads.safetensors

Prompts: first turns of the 80 MT-Bench questions (the paper's benchmark), downloaded once;
falls back to a small built-in list if offline. Both methods use the same 4-bit model, the same
KV cache implementation and batch size 1; prefill is timed separately.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from medusa.generation.decode import MedusaDecoder  # noqa: E402
from medusa.model.utils import build_medusa_model, gpu_memory_report, load_config, load_tokenizer, set_seed  # noqa: E402
from medusa.train.dataset import build_vicuna_prompt  # noqa: E402
from medusa.utils.metrics import aggregate, compare  # noqa: E402

MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
FALLBACK = [
    "Write a short story about a robot learning to paint.",
    "Explain the difference between a list and a tuple in Python, with examples.",
    "What are the main causes of inflation? Answer in five bullet points.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me a 3-day travel itinerary for Paris.",
    "Explain how a transformer language model generates text.",
    "Translate 'Knowledge is power' into French, Spanish and German.",
]


def load_prompts(cache_dir: str, n: int):
    path = os.path.join(cache_dir, "mt_bench_questions.jsonl")
    try:
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(MT_BENCH_URL, path)
        with open(path, encoding="utf-8") as f:
            qs = [json.loads(l) for l in f if l.strip()]
        return [q["turns"][0] for q in qs][:n], "mt_bench"
    except Exception as e:  # offline
        print(f"[prompts] MT-Bench download failed ({e}); using built-in prompts")
        return FALLBACK[:n], "builtin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vicuna_7b_medusa.yaml")
    ap.add_argument("--heads", required=True)
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=None, help="override generation.temperature")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.temperature is not None:
        cfg["generation"]["temperature"] = args.temperature
    set_seed(cfg["seed"])
    tok = load_tokenizer(cfg)
    model = build_medusa_model(cfg, heads_path=args.heads).eval()
    dec = MedusaDecoder.from_config(model, cfg, tok.eos_token_id)
    print(gpu_memory_report("loaded"), f"| tree nodes={dec.tree.num_nodes} branches={dec.tree.num_branches}")

    prompts, source = load_prompts(cfg["data"]["cache_dir"], args.num_prompts)
    encoded = [tok(build_vicuna_prompt([], p), return_tensors="pt").input_ids for p in prompts]

    for ids in encoded[: args.warmup]:                     # CUDA/cuBLAS warm-up, not timed
        dec.baseline_generate(ids, 16)
        dec.generate(ids, 16)

    base_stats, med_stats, same = [], [], 0
    for i, ids in enumerate(encoded):
        b_out, b = dec.baseline_generate(ids, args.max_new_tokens)
        m_out, m = dec.generate(ids, args.max_new_tokens)
        base_stats.append(b)
        med_stats.append(m)
        same += int(b_out == m_out)
        print(f"[{i + 1}/{len(encoded)}] base {b.tokens_per_second:6.1f} tok/s | medusa {m.tokens_per_second:6.1f} tok/s"
              f" | accel {m.acceleration_rate:.2f}")

    base, med = aggregate(base_stats), aggregate(med_stats)
    result = {"prompt_source": source, "temperature": cfg["generation"]["temperature"],
              "tree_topk": cfg["generation"]["tree_topk"], "baseline": base, "medusa": med,
              "comparison": compare(base, med),
              "identical_outputs": f"{same}/{len(encoded)}",
              "gpu": torch.cuda.get_device_name(0)}
    out = args.out or os.path.join(cfg["output_dir"], "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    c = result["comparison"]
    print(f"\nacceleration rate {c['acceleration_rate']:.2f} | overhead {c['overhead']:.2f} | "
          f"speedup (decode) {c['speedup_wallclock_decode']:.2f}x | head acceptance "
          f"{[round(a, 3) for a in med['head_acceptance']]}")
    if cfg["generation"]["temperature"] == 0:
        print(f"greedy outputs identical to baseline: {result['identical_outputs']} "
              "(small fp16 differences between batched tree and sequential kernels can flip near-ties)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
