#!/usr/bin/env python3
"""Evaluate a released audio checkpoint on a local tokenized dataset.

The checkpoint supplies its architecture. Dataset options default to the
published DAC-9 configuration. The deterministic 80/20 split matches train.py.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import AudioTokenDataset
from src.model.checkpoints import architecture_label, load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a released MMC checkpoint on tokenized audio.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20, help="Set 0 to evaluate every split batch")
    parser.add_argument("--interleave-codebooks", type=int, default=9)
    parser.add_argument("--concat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sep-token", type=int, default=9218)
    parser.add_argument("--sep-gap-token", type=int, default=None)
    parser.add_argument("--score-file", type=Path, default=None)
    parser.add_argument("--min-detector-score", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.token_dir.is_dir():
        parser.error(f"Token directory not found: {args.token_dir}")
    if args.batch_size < 1 or args.max_batches < 0:
        parser.error("--batch-size must be >= 1 and --max-batches must be >= 0")

    checkpoint, config, model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    dataset = AudioTokenDataset(
        args.token_dir, max_seq_len=config.max_seq_len, vocab_size=config.vocab_size, augment=False,
        concat=args.concat, sep_token=args.sep_token if args.concat else None,
        sep_gap_token=args.sep_gap_token, interleave_codebooks=args.interleave_codebooks,
        score_file=args.score_file, min_detector_score=args.min_detector_score,
    )
    if not len(dataset):
        raise SystemExit("The dataset has no usable windows with the selected options")
    n_train = int(len(dataset) * 0.8)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    selected = indices[:n_train] if args.split == "train" else indices[n_train:]
    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    weighted_loss = 0.0
    valid_tokens = batches = 0
    with torch.inference_mode():
        for batch in loader:
            if args.max_batches and batches >= args.max_batches:
                break
            input_ids, targets = batch["input_ids"].to(args.device), batch["target_ids"].to(args.device)
            output = model(input_ids, attention_mask=batch["attention_mask"].to(args.device), targets=targets)
            count = int((targets != 0).sum().item())
            weighted_loss += float(output["loss"].item()) * count
            valid_tokens += count
            batches += 1
    if not valid_tokens:
        raise SystemExit("No non-padding target tokens were evaluated")

    loss = weighted_loss / valid_tokens
    result = {
        "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint.get("step"),
        "architecture": architecture_label(config), "token_dir": str(args.token_dir), "split": args.split,
        "context_tokens": config.max_seq_len, "dataset_windows": len(dataset),
        "evaluated_windows": min(len(selected), batches * args.batch_size), "evaluated_batches": batches,
        "evaluated_target_tokens": valid_tokens, "mean_cross_entropy": loss,
        "perplexity": math.exp(loss), "max_batches": args.max_batches,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
