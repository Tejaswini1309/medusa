"""Interactive chat with Medusa decoding (Architecture.md §3.12).

  python scripts/run_inference.py --heads outputs/vicuna_7b_medusa1/medusa_heads.safetensors
  commands inside the chat:  /reset  (clear history)   /exit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medusa.generation.decode import MedusaDecoder  # noqa: E402
from medusa.model.utils import build_medusa_model, load_config, load_tokenizer, set_seed  # noqa: E402
from medusa.train.dataset import build_vicuna_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vicuna_7b_medusa.yaml")
    ap.add_argument("--heads", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--baseline", action="store_true", help="plain autoregressive decoding for comparison")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg["generation"]
    if args.temperature is not None:
        g["temperature"] = args.temperature
    max_new = args.max_new_tokens or g["max_new_tokens"]
    set_seed(cfg["seed"])
    tok = load_tokenizer(cfg)
    model = build_medusa_model(cfg, heads_path=args.heads).eval()
    dec = MedusaDecoder.from_config(model, cfg, tok.eos_token_id)
    run = dec.baseline_generate if args.baseline else dec.generate
    print(f"Loaded. mode={'baseline' if args.baseline else 'medusa'} temperature={g['temperature']}. /reset, /exit")

    history = []
    while True:
        try:
            user = input("\nUSER: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/exit":
            break
        if user == "/reset":
            history = []
            continue
        if not user:
            continue
        ids = tok(build_vicuna_prompt(history, user), return_tensors="pt").input_ids
        if ids.shape[1] + max_new + dec.tree.num_nodes > dec.cache.max_len:
            print("[history too long for the KV cache; resetting]")
            history = []
            ids = tok(build_vicuna_prompt(history, user), return_tensors="pt").input_ids

        generated, printed = [], 0
        print("ASSISTANT: ", end="", flush=True)

        def stream(new_tokens):                     # decode the whole reply, print only the new suffix
            nonlocal printed
            generated.extend(new_tokens)
            text = tok.decode(generated, skip_special_tokens=True)
            print(text[printed:], end="", flush=True)
            printed = len(text)

        out, stats = run(ids, max_new, callback=stream)
        reply = tok.decode(out, skip_special_tokens=True).strip()
        history.append((user, reply))
        print(f"\n[{stats.new_tokens} tokens, {stats.tokens_per_second:.1f} tok/s, "
              f"{stats.acceleration_rate:.2f} tokens/step]")


if __name__ == "__main__":
    main()
