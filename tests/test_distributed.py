"""Multi-GPU (DDP) support, tested on CPU with the gloo backend.

The unit tests cover the step/accumulation bookkeeping. The `slow` tests start
real 2-process jobs: they check that splitting gradient accumulation across
processes gives the same weights as one process, that MoE models (where some
experts get no tokens) train under DDP, and that scripts/train.py runs under
torchrun.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.model.transformer import CausalTransformer
from src.training.distributed import (
    DistInfo,
    dist_info_from_env,
    init_distributed,
    interval_reached,
    local_grad_accumulation,
)
from src.training.trainer import TrainConfig, Trainer
from tests.helpers import ARCHS, make_tiny_cfg

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 64
SEQ = 16


# ---------------------------------------------------------------- unit tests

def test_dist_info_defaults_to_single_process(monkeypatch):
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    info = dist_info_from_env()
    assert info == DistInfo(world_size=1, rank=0, local_rank=0)
    assert not info.enabled and info.is_main


def test_init_distributed_is_noop_without_launcher(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info, device = init_distributed("cuda")
    assert not info.enabled and device == "cuda"
    assert not dist.is_initialized()


@pytest.mark.parametrize("target,world,keep,expected", [
    (8, 1, True, 8),
    (8, 8, True, 1),
    (8, 4, True, 2),
    (8, 2, True, 4),
    (8, 8, False, 8),
    (1, 1, True, 1),
])
def test_local_grad_accumulation(target, world, keep, expected):
    assert local_grad_accumulation(target, world, keep) == expected


def test_local_grad_accumulation_rejects_uneven_split():
    with pytest.raises(ValueError, match="cannot be split evenly"):
        local_grad_accumulation(8, 3, True)


@pytest.mark.parametrize("interval", [1, 2, 3, 10, 50])
def test_interval_reached_matches_modulo_on_one_gpu(interval):
    for step in range(200):
        assert interval_reached(step, 1, interval) == (step % interval == 0)
        assert interval_reached(step, 1, interval, skip_zero=True) == (step > 0 and step % interval == 0)


def test_interval_reached_fires_once_per_interval_on_many_gpus():
    # 8 GPUs, step advances by 8; an interval of 50 single-GPU steps fires once per 50.
    fired = [s for s in range(0, 400, 8) if interval_reached(s, 8, 50, skip_zero=True)]
    assert fired == [48, 96, 144, 200, 248, 296, 344]
    # Interval smaller than the step increment: fires every iteration.
    assert all(interval_reached(s, 8, 4) for s in range(0, 80, 8))


# ------------------------------------------------- multi-process (gloo / CPU)

class _FixedTokens(Dataset):
    """Deterministic tiny sequences, so every process sees identical data."""

    def __init__(self, n=8):
        g = torch.Generator().manual_seed(1234)
        self.x = torch.randint(1, VOCAB, (n, SEQ + 1), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        row = self.x[i]
        return {"input_ids": row[:-1], "target_ids": row[1:], "attention_mask": torch.ones(SEQ, dtype=torch.long)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _train_cfg(out_dir, **overrides):
    fields = dict(num_epochs=1, warmup_steps=0, learning_rate=1e-3, grad_accumulation_steps=2,
                  batch_size=1, log_interval=1, eval_interval=1000, save_interval=1000,
                  output_dir=str(out_dir))
    fields.update(overrides)
    return TrainConfig(**fields)


def _worker(rank, world, port, arch, out_dir, result_file):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world),
                      RANK=str(rank), LOCAL_RANK=str(rank), USE_LIBUV="0")
    info, device = init_distributed("cpu")
    try:
        torch.manual_seed(0)
        model = CausalTransformer(make_tiny_cfg(vocab_size=VOCAB, **ARCHS[arch]))
        if arch == "nsa_moe":
            # Make the last expert unreachable, so its weights get no gradient:
            # the case that crashes DDP unless find_unused_parameters is on.
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name.endswith("expert_bias"):
                        p[-1] = -1e4
        ds = _FixedTokens()
        # shuffle=False: rank r gets samples r, r+world, ... so each update covers
        # the same samples as consecutive batches in a 1-process run.
        loader = DataLoader(ds, batch_size=1, sampler=DistributedSampler(ds, world, rank, shuffle=False))
        trainer = Trainer(model, loader, loader, _train_cfg(out_dir), device=device, dist_info=info)
        trainer.train()

        # All ranks must end with identical weights.
        flat = torch.cat([p.detach().flatten() for p in trainer.raw_model.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        if rank == 0:
            max_rank_diff = max((g - gathered[0]).abs().max().item() for g in gathered)
            torch.save({"max_rank_diff": max_rank_diff}, result_file)
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, arch, world=2):
    out = tmp_path / f"ddp_{arch}"
    result_file = tmp_path / f"result_{arch}.pt"
    mp.spawn(_worker, args=(world, _free_port(), arch, str(out), str(result_file)), nprocs=world, join=True)
    return out, torch.load(result_file)


def _single_process_reference(tmp_path, arch):
    torch.manual_seed(0)
    model = CausalTransformer(make_tiny_cfg(vocab_size=VOCAB, **ARCHS[arch]))
    ds = _FixedTokens()
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    out = tmp_path / f"ref_{arch}"
    Trainer(model, loader, loader, _train_cfg(out), device="cpu").train()
    return out


@pytest.mark.slow
def test_two_processes_match_one_process_with_accumulation(tmp_path):
    """2 processes x 1 accumulation step == 1 process x 2 accumulation steps."""
    ref_out = _single_process_reference(tmp_path, "dense")
    ddp_out, result = _spawn(tmp_path, "dense")

    assert result["max_rank_diff"] == 0.0
    ref = torch.load(ref_out / "best_model.pt", weights_only=False)
    ddp = torch.load(ddp_out / "best_model.pt", weights_only=False)
    assert ref["step"] == ddp["step"] == len(_FixedTokens())
    # Saved without the DDP wrapper: same keys as a single-GPU checkpoint.
    assert ref["model_state_dict"].keys() == ddp["model_state_dict"].keys()
    for k, v in ref["model_state_dict"].items():
        assert torch.allclose(v, ddp["model_state_dict"][k], atol=1e-5), k

    ref_log = (ref_out / "training_log.jsonl").read_text().splitlines()
    ddp_log = (ddp_out / "training_log.jsonl").read_text().splitlines()
    assert ddp_log and len(ddp_log) == len(ref_log) // 2


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["swa_moe", "nsa_moe"])
def test_moe_trains_under_ddp(tmp_path, arch):
    """MoE experts that receive no tokens on one rank must not break DDP."""
    out, result = _spawn(tmp_path, arch)
    assert result["max_rank_diff"] == 0.0
    assert (out / "best_model.pt").exists()


@pytest.mark.slow
def test_train_script_under_torchrun(tmp_path, fake_token_dir):
    """scripts/train.py launched by torchrun with 2 CPU processes."""
    real = yaml.safe_load((ROOT / "configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml").read_text())
    cfg = {
        "model": {**real["model"], "n_layers": 2, "n_heads": 2, "d_model": 32, "d_ff": 64,
                  "max_seq_len": 64, "swa_window_size": 16, "expert_d_ff": 32, "n_experts": 4,
                  "compressed_attn_stride": 4, "compressed_attn_chunk": 8},
        "data": {**real["data"], "token_dir": str(fake_token_dir), "max_seq_len": 64, "score_file": None,
                 "adjacency_file": None},
        "training": {**real["training"], "use_8bit_adam": False, "batch_size": 1, "val_batch_size": 1,
                     "grad_accumulation_steps": 2, "num_epochs": 1, "warmup_steps": 1, "eval_interval": 2,
                     "log_interval": 1, "max_eval_batches": 2, "num_workers": 0,
                     "output_dir": str(tmp_path / "run")},
    }
    cfg_path = tmp_path / "toy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Start both processes with the environment torchrun would give them. (Calling
    # torchrun itself fails on Windows CPU builds of PyTorch, which lack libuv.)
    port = str(_free_port())
    procs = []
    for rank in range(2):
        env = {**os.environ, "PYTHONPATH": str(ROOT), "USE_LIBUV": "0", "MASTER_ADDR": "127.0.0.1",
               "MASTER_PORT": port, "WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}
        procs.append(subprocess.Popen(
            [sys.executable, "scripts/train.py", str(cfg_path), "--device", "cpu"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = [p.communicate(timeout=300) for p in procs]

    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, out[-3000:] + err[-3000:]
    rank0_out, rank1_out = outs[0][0], outs[1][0]
    assert "Training complete" in rank0_out and rank1_out == ""   # only rank 0 prints
    assert "sequences per update: 2" in rank0_out                 # 2 GPUs x 1 step = the YAML's 2
    assert (tmp_path / "run" / "best_model.pt").exists()
