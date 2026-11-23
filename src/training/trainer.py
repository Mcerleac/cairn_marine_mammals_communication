"""Training loop for the causal transformer.

Runs on one GPU as a plain process, or on several GPUs with
DistributedDataParallel when launched by torchrun (see src/training/distributed.py).

Step counting: ``step`` is always measured in single-GPU units, i.e. one batch
on one GPU. A run on N GPUs advances ``step`` by N per iteration. The LR
schedule, warmup, log/eval/save intervals and the ``step`` stored in
checkpoints therefore mean the same thing whatever the GPU count, and a
checkpoint can be resumed on a different number of GPUs.
"""

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.model.transformer import CausalTransformer
from src.training.distributed import (
    DistInfo,
    all_reduce_mean,
    all_reduce_sum,
    barrier,
    interval_reached,
    local_grad_accumulation,
)
from src.training.muon import Muon


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    batch_size: int = 32            # per GPU
    grad_accumulation_steps: int = 1  # as for a single-GPU run; see keep_effective_batch

    # Schedule
    num_epochs: int = 100
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # min LR = lr * min_lr_ratio

    # Logging & checkpointing
    log_interval: int = 10       # Log every N steps
    eval_interval: int = 50      # Evaluate every N steps
    save_interval: int = 200     # Save checkpoint every N steps
    save_top_k: int = 2          # Keep N most recent checkpoint_step*.pt files (best_model.pt always kept separately)
    output_dir: str = "runs/default"

    # Early stopping
    patience: int = 20           # Stop after N evals without improvement
    patience_counter: int = 0

    # Evaluation
    max_eval_batches: int = 0    # 0 = full val set; >0 = cap eval to N batches in total (split across GPUs)

    # Memory optimization
    use_8bit_adam: bool = False   # Use bitsandbytes 8-bit Adam to reduce VRAM

    # Muon optimizer (DeepSeek V4-inspired)
    use_muon: bool = False        # Muon for 2D weight matrices + AdamW for the rest
    muon_lr: float = 0.01        # Constant LR for Muon (not cosine-annealed)

    # Multi-GPU (ignored on a single GPU)
    keep_effective_batch: bool = True  # split grad_accumulation_steps across GPUs so sequences/update match 1 GPU
    ddp_find_unused_parameters: Optional[bool] = None  # None = auto: on for MoE models (idle experts get no grad)


def get_lr(step: int, config: TrainConfig, total_steps: int = 0) -> float:
    """Cosine annealing with warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    # Cosine decay
    if total_steps <= 0:
        total_steps = config.num_epochs * 1000  # fallback estimate
    decay_ratio = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * (config.min_lr_ratio + (1 - config.min_lr_ratio) * coeff)


class Trainer:
    """Training loop with bf16, gradient accumulation, checkpointing and optional DDP."""

    def __init__(
        self,
        model: CausalTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainConfig,
        device: str = "cuda",
        resume_from: str | None = None,
        dist_info: DistInfo | None = None,
    ):
        self.dist = dist_info or DistInfo()
        self.world_size = self.dist.world_size
        self.is_main = self.dist.is_main

        # raw_model: the CausalTransformer itself (config, state_dict, eval).
        # model: what the training step calls; the DDP wrapper when distributed.
        self.raw_model = model.to(device)
        self.model = self.raw_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.device_type = torch.device(device).type   # autocast wants "cuda", not "cuda:3"

        self.local_accum = local_grad_accumulation(
            config.grad_accumulation_steps, self.world_size, config.keep_effective_batch)

        self.output_dir = Path(config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        # Optimizer setup
        # When use_muon=True: Muon handles 2D weight matrices (attention + FFN weights),
        # AdamW handles everything else (embeddings, norms, biases, expert_bias, lm_head).
        self.muon_optimizer: Optional[Muon] = None

        if getattr(config, 'use_muon', False):
            muon_params, adamw_wd_params, adamw_no_wd_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # 2D matrices that are not embeddings or the (weight-tied) lm_head
                if (param.ndim == 2
                        and 'token_emb' not in name
                        and 'lm_head' not in name):
                    muon_params.append(param)
                elif 'norm' in name or param.ndim < 2:
                    adamw_no_wd_params.append(param)
                else:
                    # token_emb, lm_head (weight-tied), any remaining 2D+ non-matrix
                    adamw_wd_params.append(param)

            print(f"  Muon params:  {sum(p.numel() for p in muon_params):,}")
            print(f"  AdamW params: {sum(p.numel() for p in adamw_wd_params + adamw_no_wd_params):,}")

            self.muon_optimizer = Muon(
                muon_params,
                lr=config.muon_lr,
                momentum=0.95,
                nesterov=True,
            )
            adamw_groups = [
                {"params": adamw_wd_params,    "weight_decay": config.weight_decay},
                {"params": adamw_no_wd_params, "weight_decay": 0.0},
            ]
        else:
            adamw_groups = [
                {"params": [p for n, p in model.named_parameters()
                            if "norm" not in n and p.requires_grad],
                 "weight_decay": config.weight_decay},
                {"params": [p for n, p in model.named_parameters()
                            if "norm" in n and p.requires_grad],
                 "weight_decay": 0.0},
            ]

        if getattr(config, 'use_8bit_adam', False):
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                adamw_groups, lr=config.learning_rate, betas=(0.9, 0.95))
        else:
            self.optimizer = torch.optim.AdamW(
                adamw_groups, lr=config.learning_rate, betas=(0.9, 0.95))

        # Logging
        self.log_file = self.output_dir / "training_log.jsonl"
        self.best_val_loss = float("inf")
        self.saved_checkpoints = []  # ordered list of paths, oldest first
        self._resume_step = 0

        # Load weights before wrapping in DDP. Every process loads the same
        # checkpoint, and DDP also broadcasts rank 0's weights when it wraps.
        if resume_from:
            self._resume_from_checkpoint(Path(resume_from))
        else:
            self._auto_resume_from_output_dir()

        if self.dist.enabled:
            find_unused = config.ddp_find_unused_parameters
            if find_unused is None:
                # MoE: experts that receive no tokens in a step have no gradient,
                # which plain DDP treats as an error.
                find_unused = getattr(self.raw_model.config, "n_experts", 0) > 1
            self.model = DDP(
                self.raw_model,
                device_ids=[self.dist.local_rank] if self.device_type == "cuda" else None,
                find_unused_parameters=find_unused,
                gradient_as_bucket_view=True,   # gradients share DDP's buffers: no extra copy in memory
                broadcast_buffers=False,        # only buffer is the constant RoPE table
            )

        eff = config.batch_size * self.local_accum * self.world_size
        print(f"  GPUs (processes): {self.world_size} | per-GPU batch: {config.batch_size} | "
              f"grad accumulation per GPU: {self.local_accum} | sequences per update: {eff}")

    def _empty_cache(self):
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

    def train(self):
        """Run the full training loop."""
        W = self.world_size
        step = self._resume_step
        self.model.train()

        log_entries = []
        start_time = time.time()
        steps_per_epoch = len(self.train_loader) * W     # single-GPU units
        total_steps = self.config.num_epochs * steps_per_epoch
        it = step // W   # this process's iteration count; drives gradient accumulation

        # Skip epochs already completed before the resume point
        start_epoch = step // max(steps_per_epoch, 1)
        if step > 0:
            print(f"  Resuming at step {step} (epoch {start_epoch}, LR {get_lr(step, self.config, total_steps):.2e})")

        for epoch in range(start_epoch, self.config.num_epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)   # DistributedSampler: new shuffle each epoch

            for batch in self.train_loader:
                # Update learning rate. This iteration covers single-GPU steps
                # [step, step + W); use the last one, as a 1-GPU run would at its update.
                lr = get_lr(step + W - 1, self.config, total_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                input_ids = batch["input_ids"].to(self.device)
                targets = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Only average gradients across GPUs on the micro-step that updates weights.
                update_now = (it + 1) % self.local_accum == 0
                sync_ctx = self.model.no_sync() if (self.dist.enabled and not update_now) else nullcontext()

                with sync_ctx:
                    # Forward pass with bf16
                    with torch.autocast(self.device_type, dtype=torch.bfloat16):
                        output = self.model(input_ids, attention_mask=attention_mask, targets=targets)
                        loss = output["loss"]
                        if "aux_loss" in output:
                            loss = loss + self.raw_model.config.moe_aux_weight * output["aux_loss"]
                        loss = loss / self.local_accum

                    loss.backward()

                if update_now:
                    nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if self.muon_optimizer is not None:
                        self.muon_optimizer.step()
                    self.optimizer.zero_grad()
                    if self.muon_optimizer is not None:
                        self.muon_optimizer.zero_grad()
                    # Flush allocator fragment cache to prevent long-run OOM from
                    # fragmentation accumulation (cuMemRetain blocks grow over many steps).
                    self._empty_cache()

                # Logging (loss averaged over all GPUs)
                if interval_reached(step, W, self.config.log_interval):
                    elapsed = time.time() - start_time
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": all_reduce_mean((loss * self.local_accum).item(), self.device),
                        "lr": lr,
                        "elapsed_s": round(elapsed, 1),
                    }
                    if "aux_loss" in output:
                        entry["aux_loss"] = all_reduce_mean(output["aux_loss"].item(), self.device)
                    log_entries.append(entry)
                    print(
                        f"Step {step:5d} | Epoch {epoch:3d} | "
                        f"Loss {entry['train_loss']:.4f} | LR {lr:.2e} | "
                        f"Time {elapsed:.0f}s"
                    )

                # Evaluation (every process evaluates its share; result is identical on all)
                if self.val_loader and interval_reached(step, W, self.config.eval_interval, skip_zero=True):
                    self._empty_cache()
                    val_loss = self.evaluate()
                    print(f"  -> Val loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")

                    if log_entries:
                        log_entries[-1]["val_loss"] = val_loss

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.config.patience_counter = 0
                        self.save_checkpoint(step, val_loss, is_best=True)
                    else:
                        self.config.patience_counter += 1

                    # Flush logs to disk after each eval
                    self._write_logs(log_entries)

                    if self.config.patience_counter >= self.config.patience:
                        print(f"Early stopping at step {step}")
                        barrier()
                        return

                    self.model.train()

                # Periodic save
                if interval_reached(step, W, self.config.save_interval, skip_zero=True):
                    self.save_checkpoint(step, self.best_val_loss)

                step += W
                it += 1

        # Final evaluation and save
        if self.val_loader:
            val_loss = self.evaluate()
            print(f"Final val loss: {val_loss:.4f}")

        self.save_checkpoint(step, self.best_val_loss, is_best=True)
        self._write_logs(log_entries)
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        barrier()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average loss on the validation set, across all GPUs."""
        self.raw_model.eval()
        total_loss = 0.0
        n_batches = 0
        max_batches = self.config.max_eval_batches
        if max_batches > 0:
            # The cap is a total; each GPU evaluates its share of it.
            max_batches = math.ceil(max_batches / self.world_size)

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.autocast(self.device_type, dtype=torch.bfloat16):
                output = self.raw_model(input_ids, attention_mask=attention_mask, targets=targets)

            total_loss += output["loss"].item()
            n_batches += 1

            if max_batches > 0 and n_batches >= max_batches:
                break

        total_loss, n_batches = all_reduce_sum([total_loss, float(n_batches)], self.device)
        return total_loss / max(n_batches, 1)

    def _auto_resume_from_output_dir(self):
        """Resume automatically from the most recent checkpoint in output_dir."""
        candidates = []  # (step_num, path, state_dict)
        for ckpt_file in sorted(self.output_dir.glob("checkpoint_step*.pt")):
            try:
                step_num = int(ckpt_file.stem.replace("checkpoint_step", ""))
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                candidates.append((step_num, ckpt_file, ckpt))
            except Exception:
                pass

        if not candidates:
            return

        print(f"Found {len(candidates)} existing checkpoints in {self.output_dir}")
        candidates.sort(key=lambda x: x[0])
        # Every process must finish reading before rank 0 deletes anything.
        barrier()
        while len(candidates) > self.config.save_top_k:
            old_step, old_path, _ = candidates.pop(0)
            if self.is_main and old_path.exists():
                old_path.unlink()
                print(f"  Pruned {old_path.name}")

        self.saved_checkpoints = [p for _, p, _ in candidates]
        resume_step, resume_path, ckpt = candidates[-1]
        self._load_checkpoint_state(ckpt, resume_path, resume_step)

    def _resume_from_checkpoint(self, ckpt_path: Path):
        """Resume explicitly from a checkpoint path."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" not in ckpt:
            raise ValueError(
                f"{ckpt_path} does not contain optimizer state. Use --init-from with best_model.pt, or --resume-from with checkpoint_step*.pt."
            )
        resume_step = int(ckpt.get("step", 0))
        self.saved_checkpoints = [ckpt_path]
        self._load_checkpoint_state(ckpt, ckpt_path, resume_step)

    def _load_checkpoint_state(self, ckpt: dict, ckpt_path: Path, resume_step: int):
        """Load model, optimizer, and trainer state from a full checkpoint."""
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.muon_optimizer is not None and "muon_state_dict" in ckpt:
            self.muon_optimizer.load_state_dict(ckpt["muon_state_dict"])
        self.best_val_loss = ckpt["val_loss"]
        self._resume_step = resume_step
        print(f"  Resuming from {ckpt_path.name} (step {resume_step}, val_loss {ckpt['val_loss']:.4f})")

    def save_checkpoint(self, step: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint, keeping only the N most recent on disk.

        Only rank 0 writes. The weights are saved without the DDP wrapper, so
        checkpoints have the same format as single-GPU (and released) ones.
        """
        if not self.is_main:
            return

        ckpt_path = self.output_dir / f"checkpoint_step{step}.pt"
        ckpt = {
            "step": step,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.raw_model.config,
            "val_loss": val_loss,
        }
        if self.muon_optimizer is not None:
            ckpt["muon_state_dict"] = self.muon_optimizer.state_dict()
        torch.save(ckpt, ckpt_path)

        self.saved_checkpoints.append(ckpt_path)

        # Delete oldest until we're at the limit
        while len(self.saved_checkpoints) > self.config.save_top_k:
            old_path = self.saved_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save({
                "step": step,
                "model_state_dict": self.raw_model.state_dict(),
                "config": self.raw_model.config,
                "val_loss": val_loss,
            }, best_path)

    def _write_logs(self, entries: list[dict]):
        if not self.is_main:
            return
        with open(self.log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
