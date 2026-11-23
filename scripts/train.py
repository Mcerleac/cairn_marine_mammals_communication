#!/usr/bin/env python3
"""Training script for marine mammal communication LLM.

Single GPU:
    python scripts/train.py configs/<config>.yaml

Several GPUs (any number; torchrun starts one process per GPU):
    torchrun --nproc_per_node=8 scripts/train.py configs/<config>.yaml
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import AudioTokenDataset, create_symbolic_datasets
from src.model.checkpoints import (
    architecture_label,
    checkpoint_config_to_dict,
    config_from_checkpoint,
    load_checkpoint,
    summarize_config,
)
from src.model.config import get_config
from src.model.transformer import CausalTransformer
from src.training.distributed import cleanup_distributed, init_distributed
from src.training.trainer import TrainConfig, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train marine mammal LLM")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for a single-process run. Under torchrun each process uses cuda:<LOCAL_RANK> "
             "(or cpu, for testing, if --device cpu).",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize model weights from a checkpoint (e.g., runs/.../best_model.pt)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume optimizer/model/training step from a full checkpoint_step*.pt file",
    )
    args = parser.parse_args()

    if args.init_from and args.resume_from:
        parser.error("--init-from and --resume-from are mutually exclusive")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Joins the process group when launched by torchrun; otherwise a no-op.
    dist_info, device = init_distributed(args.device)
    try:
        _train(args, cfg, dist_info, device)
    finally:
        cleanup_distributed()


def _train(args, cfg, dist_info, device):
    print(f"Config: {args.config}")
    print(f"Device: {device}" + (f" (+{dist_info.world_size - 1} more processes)" if dist_info.enabled else ""))

    print("\n=== Loading data ===")
    dataset_type = cfg["data"]["dataset_type"]

    if dataset_type == "audio":
        token_dir = cfg["data"]["token_dir"]
        augment = cfg["data"].get("augment", False)
        vocab_size = cfg["model"]["vocab_size"]
        concat = cfg["data"].get("concat", False)
        sep_token = cfg["data"].get("sep_token", None)

        codebook_index = cfg["data"].get("codebook_index", None)
        interleave_codebooks = cfg["data"].get("interleave_codebooks", None)
        ds_kwargs = dict(
            max_seq_len=cfg["data"].get("max_seq_len", 512),
            vocab_size=vocab_size,
            concat=concat,
            sep_token=sep_token,
            sep_gap_token=cfg["data"].get("sep_gap_token", None),
            adjacency_file=cfg["data"].get("adjacency_file", None),
            codebook_index=codebook_index,
            interleave_codebooks=interleave_codebooks,
            score_file=cfg["data"].get("score_file", None),
            min_detector_score=cfg["data"].get("min_detector_score", None),
        )

        full_ds_noaug = AudioTokenDataset(token_dir, augment=False, **ds_kwargs)
        n_total = len(full_ds_noaug)
        n_train = int(n_total * 0.8)
        n_val = n_total - n_train

        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_total, generator=g).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        if augment:
            full_ds_aug = AudioTokenDataset(
                token_dir,
                augment=True,
                token_noise_prob=cfg["data"].get("token_noise_prob", 0.05),
                token_mask_prob=cfg["data"].get("token_mask_prob", 0.02),
                time_stretch_prob=cfg["data"].get("time_stretch_prob", 0.3),
                **ds_kwargs,
            )
            train_ds = torch.utils.data.Subset(full_ds_aug, train_indices)
        else:
            train_ds = torch.utils.data.Subset(full_ds_noaug, train_indices)

        val_ds = torch.utils.data.Subset(full_ds_noaug, val_indices)

        print(f"Dataset: audio tokens from {token_dir}")
        print(f"Total windows: {n_total} (train: {n_train}, val: {n_val})")
        print(f"Augmentation: {augment}, Concat: {concat}")
        if codebook_index is not None:
            print(f"Codebook index: {codebook_index} (extracting single codebook from 2D files)")
        if interleave_codebooks:
            print(f"Interleave: {interleave_codebooks} codebooks (9CB interleaved from 2D files)")
        print(f"Vocab size: {vocab_size}")
    else:
        datasets = create_symbolic_datasets(
            cfg["data"]["codas_csv"],
            cfg["data"]["dialogues_csv"],
            max_seq_len=cfg["data"].get("max_seq_len", 128),
            dialogue_max_seq_len=cfg["data"].get("max_seq_len", 256),
        )
        if dataset_type == "coda":
            train_ds = datasets["coda_train"]
            val_ds = datasets["coda_val"]
        elif dataset_type == "dialogue":
            train_ds = datasets["dialogue_train"]
            val_ds = datasets["dialogue_val"]
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        vocab_size = datasets["vocab"].vocab_size
        print(f"Dataset: {dataset_type}")
        print(f"Train: {len(train_ds)} samples")
        print(f"Val: {len(val_ds)} samples")
        print(f"Vocab size: {vocab_size}")

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)  # per GPU process
    loader_kwargs = dict(pin_memory=True, num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    val_batch_size = cfg["training"].get("val_batch_size", batch_size * 4)
    if dist_info.enabled:
        # Each GPU gets its own, non-overlapping share of the data.
        train_sampler = DistributedSampler(
            train_ds, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=True)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, **loader_kwargs)

    print("\n=== Creating model ===")
    ckpt_path = args.resume_from or args.init_from
    ckpt = None
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        model_cfg = config_from_checkpoint(ckpt)
        print(f"Model architecture source: checkpoint ({ckpt_path})")
        print(f"Architecture: {architecture_label(model_cfg)}")
        print(f"Config summary: {summarize_config(model_cfg)}")

        yaml_model = dict(cfg.get("model", {}))
        yaml_model.pop("preset", None)
        ckpt_model = checkpoint_config_to_dict(model_cfg)
        mismatches = {}
        for key, yaml_value in yaml_model.items():
            if key in ckpt_model and ckpt_model[key] != yaml_value:
                mismatches[key] = {"yaml": yaml_value, "checkpoint": ckpt_model[key]}
        if mismatches:
            print("NOTE: ignoring YAML model overrides that differ from checkpoint config:")
            for key, values in mismatches.items():
                print(f"  {key}: yaml={values['yaml']} checkpoint={values['checkpoint']}")
    else:
        model_overrides = {k: v for k, v in cfg["model"].items() if k != "preset"}
        model_cfg = get_config(cfg["model"]["preset"], **model_overrides)
        print(f"Model architecture source: YAML preset {cfg['model']['preset']}")

    model = CausalTransformer(model_cfg)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")

    if ckpt is not None:
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain model_state_dict")
        model.load_state_dict(ckpt["model_state_dict"])
        mode = "resume" if args.resume_from else "init"
        print(f"Loaded checkpoint for {mode} (step {ckpt.get('step', '?')}, val_loss {ckpt.get('val_loss', '?')})")

    train_cfg = TrainConfig(
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=batch_size,
        grad_accumulation_steps=cfg["training"].get("grad_accumulation_steps", 1),
        num_epochs=cfg["training"]["num_epochs"],
        warmup_steps=cfg["training"].get("warmup_steps", 100),
        min_lr_ratio=cfg["training"].get("min_lr_ratio", 0.1),
        log_interval=cfg["training"].get("log_interval", 10),
        eval_interval=cfg["training"].get("eval_interval", 50),
        save_interval=cfg["training"].get("save_interval", 200),
        patience=cfg["training"].get("patience", 20),
        max_eval_batches=cfg["training"].get("max_eval_batches", 0),
        use_8bit_adam=cfg["training"].get("use_8bit_adam", False),
        use_muon=cfg["training"].get("use_muon", False),
        muon_lr=cfg["training"].get("muon_lr", 0.01),
        output_dir=cfg["training"]["output_dir"],
        keep_effective_batch=cfg["training"].get("keep_effective_batch", True),
        ddp_find_unused_parameters=cfg["training"].get("ddp_find_unused_parameters", None),
    )

    print("\n=== Training ===")
    print(f"Output: {train_cfg.output_dir}")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_cfg,
        device=device,
        resume_from=args.resume_from,
        dist_info=dist_info,
    )
    trainer.train()


if __name__ == "__main__":
    main()
