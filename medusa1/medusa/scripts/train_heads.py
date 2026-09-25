"""Train Medusa heads (Architecture.md §3.12).

  python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml --prepare-data-only
  python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml            # Medusa-1
  python scripts/train_heads.py --config configs/vicuna_7b_medusa.yaml --resume   # continue after a stop
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from medusa.model.utils import (build_medusa_model, get_device, gpu_memory_report,  # noqa: E402
                                load_config, load_tokenizer, set_seed)
from medusa.train.dataset import TokenizedConversations, prepare_dataset, split_dataset  # noqa: E402
from medusa.train.trainer import MedusaTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vicuna_7b_medusa.yaml")
    ap.add_argument("--stage", default=None, help="override train.stage: medusa1 | medusa2_joint | medusa2_distill")
    ap.add_argument("--data", default=None, help="ShareGPT-format .json/.jsonl (default: download ShareGPT)")
    ap.add_argument("--init-heads", default=None, help="start from trained heads (e.g. Medusa-1 -> Medusa-2)")
    ap.add_argument("--prepare-data-only", action="store_true")
    ap.add_argument("--inspect", type=int, default=0, help="print N tokenised samples with their loss mask")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--output-dir", default=None, help="override output_dir (e.g. for a short pilot run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_train_samples:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    tag = "sharegpt" if args.data is None else os.path.splitext(os.path.basename(args.data))[0]
    npz = prepare_dataset(cfg, tokenizer, source_path=args.data, tag=tag)

    if args.inspect:
        ds = TokenizedConversations(npz)
        for i in range(min(args.inspect, len(ds))):
            item = ds[i]
            keep = item["labels"] != -100
            print(f"--- sample {i}: {len(keep)} tokens, {int(keep.sum())} with loss")
            print("[FULL]  ", tokenizer.decode(item["input_ids"][:200]))
            print("[LOSS]  ", tokenizer.decode(item["input_ids"][keep][:200]))
    if args.prepare_data_only:
        return

    train_ds, eval_ds = split_dataset(npz, cfg["data"]["eval_samples"], cfg["data"]["max_train_samples"], cfg["seed"])
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = False            # Turing has no TF32; explicit for clarity
    model = build_medusa_model(cfg, heads_path=args.init_heads, device=device)
    print(gpu_memory_report("model loaded"))

    os.makedirs(cfg["output_dir"], exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg["output_dir"], "config_used.yaml"))
    trainer = MedusaTrainer(model, cfg, train_ds, eval_ds, pad_id=tokenizer.pad_token_id, stage=args.stage)
    if args.resume and not trainer.load_checkpoint():
        print("[resume] no checkpoint found, starting from scratch")
    print(f"[train] stage={trainer.stage} steps={trainer.total_steps} accum={trainer.accum} heads={trainer.K}")
    path = trainer.train()
    print(f"[done] heads saved to {path}")


if __name__ == "__main__":        # required on Windows (DataLoader workers use spawn)
    main()
