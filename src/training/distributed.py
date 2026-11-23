"""Multi-GPU (DistributedDataParallel) helpers.

Nothing here hardcodes a GPU count. A launcher such as ``torchrun`` (directly,
or from an sbatch script via ``srun torchrun ...``) sets ``WORLD_SIZE``,
``RANK`` and ``LOCAL_RANK`` for every process it starts; these helpers read
them. When they are absent the script runs as a single ordinary process,
exactly as before.

Example launches:

    # one GPU (unchanged)
    python scripts/train.py configs/<config>.yaml

    # all GPUs on one node
    torchrun --nproc_per_node=8 scripts/train.py configs/<config>.yaml
"""

from __future__ import annotations

import builtins
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    """Where this process sits in the job."""

    world_size: int = 1   # total processes (one per GPU)
    rank: int = 0         # global index of this process, 0..world_size-1
    local_rank: int = 0   # index of this process on its own node (selects the GPU)

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def dist_info_from_env() -> DistInfo:
    """Read the launcher's environment. Defaults to a single process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(world_size=world_size, rank=rank, local_rank=local_rank)


def init_distributed(requested_device: str) -> tuple[DistInfo, str]:
    """Join the process group if launched by torchrun; return (info, device).

    With more than one process and a CUDA device requested, each process takes
    the GPU matching its LOCAL_RANK and uses the NCCL backend. A CPU device
    uses the gloo backend, which is only meant for testing.
    """
    info = dist_info_from_env()
    if not info.enabled:
        return info, requested_device

    use_cuda = torch.device(requested_device).type == "cuda"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed run requested CUDA but no GPU is visible")
        if info.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={info.local_rank} but only {torch.cuda.device_count()} GPU(s) are visible "
                f"on this node; start no more processes per node than there are GPUs"
            )
        torch.cuda.set_device(info.local_rank)
        device = f"cuda:{info.local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"

    dist.init_process_group(backend=backend)
    _quiet_non_main_ranks(info)
    return info, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _quiet_non_main_ranks(info: DistInfo) -> None:
    """Only rank 0 prints, so logs are not repeated once per GPU.

    Any rank can still print with ``print(..., force=True)``.
    """
    builtin_print = builtins.print

    def _print(*args, force: bool = False, **kwargs):
        if info.is_main or force:
            builtin_print(*args, **kwargs)

    builtins.print = _print


def local_grad_accumulation(target_steps: int, world_size: int, keep_effective_batch: bool) -> int:
    """Gradient-accumulation steps each process should run.

    ``target_steps`` is the YAML value, which describes a single-GPU run.

    keep_effective_batch=True (default): split the accumulation across GPUs so
    the number of sequences per optimizer update stays the same as on one GPU.
    For example 8 accumulation steps on 8 GPUs becomes 1 step each; on 4 GPUs,
    2 steps each.

    keep_effective_batch=False: every process keeps the YAML value, so the
    effective batch grows with the number of GPUs.
    """
    if target_steps < 1:
        raise ValueError(f"grad_accumulation_steps must be >= 1, got {target_steps}")
    if not keep_effective_batch or world_size == 1:
        return target_steps
    if target_steps % world_size != 0:
        raise ValueError(
            f"grad_accumulation_steps={target_steps} cannot be split evenly across {world_size} GPUs "
            f"while keeping the effective batch size. Use a GPU count that divides {target_steps}, "
            f"change grad_accumulation_steps, or set training.keep_effective_batch: false."
        )
    return target_steps // world_size


def interval_reached(step: int, step_increment: int, interval: int, skip_zero: bool = False) -> bool:
    """True if a multiple of ``interval`` falls in [step, step + step_increment).

    Training steps are counted in single-GPU units (one batch on one GPU), so a
    run on N GPUs advances N steps per iteration. This keeps log/eval/save
    intervals and the LR schedule meaning the same thing whatever the GPU
    count. With step_increment=1 it is the same as ``step % interval == 0``.
    """
    if interval <= 0:
        return False
    first_multiple = -(-step // interval) * interval   # ceil(step / interval) * interval
    if skip_zero and first_multiple == 0:
        first_multiple = interval
    return first_multiple < step + step_increment


def all_reduce_mean(value: float, device: str) -> float:
    """Average a Python float across all processes (no-op when not distributed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def all_reduce_sum(values: list[float], device: str) -> list[float]:
    """Sum a list of Python floats across all processes (no-op when not distributed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
