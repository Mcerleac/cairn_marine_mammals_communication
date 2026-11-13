"""Train a toy model on CPU: exercises Trainer + checkpoints the same way scripts/train.py does."""

import json

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AudioTokenDataset
from src.model.checkpoints import architecture_label, load_model_from_checkpoint, summarize_config
from src.model.transformer import CausalTransformer
from src.training.trainer import TrainConfig, Trainer, get_lr
from tests.helpers import ARCHS, N_CODEBOOKS, SEP, VOCAB, make_tiny_cfg


def test_lr_schedule():
    cfg = TrainConfig(learning_rate=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert get_lr(0, cfg, 100) == 0.0
    assert get_lr(5, cfg, 100) == pytest.approx(0.5)
    assert get_lr(10, cfg, 100) == pytest.approx(1.0)
    assert get_lr(100, cfg, 100) == pytest.approx(0.1)
    assert get_lr(1000, cfg, 100) == pytest.approx(0.1)


def _run(tmp_path, token_dir, arch, **train_overrides):
    ds = AudioTokenDataset(token_dir, max_seq_len=32, vocab_size=VOCAB, concat=True,
                           sep_token=SEP, interleave_codebooks=N_CODEBOOKS)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = CausalTransformer(make_tiny_cfg(vocab_size=VOCAB, **ARCHS[arch]))
    cfg = TrainConfig(num_epochs=1, warmup_steps=1, log_interval=1, eval_interval=2, save_interval=3,
                      max_eval_batches=2, output_dir=str(tmp_path / "run"), **train_overrides)
    trainer = Trainer(model, loader, loader, cfg, device="cpu")
    trainer.train()
    return trainer, tmp_path / "run"


@pytest.mark.parametrize("arch,overrides", [
    ("dense", {}),
    ("swa_moe", {"grad_accumulation_steps": 2}),
    ("nsa_moe", {"use_muon": True}),
])
def test_smoke_train(tmp_path, fake_token_dir, arch, overrides):
    trainer, out = _run(tmp_path, fake_token_dir, arch, **overrides)
    logs = [json.loads(line) for line in (out / "training_log.jsonl").read_text().splitlines()]
    assert logs and all(torch.isfinite(torch.tensor(e["train_loss"])) for e in logs)
    assert (out / "best_model.pt").exists()
    assert len(list(out.glob("checkpoint_step*.pt"))) <= trainer.config.save_top_k


def test_best_model_reloads_identically(tmp_path, fake_token_dir):
    trainer, out = _run(tmp_path, fake_token_dir, "nsa_moe")
    _, cfg, reloaded = load_model_from_checkpoint(out / "best_model.pt")
    assert architecture_label(cfg) == "nsa+moe+bias-routing+split-qkv"
    assert summarize_config(cfg)["compressed_attn_stride"] == 4
    x = torch.randint(1, VOCAB, (1, 16))
    trainer.model.eval()
    assert torch.allclose(trainer.model(x)["logits"], reloaded(x)["logits"], atol=1e-5)


def test_resume_from_checkpoint(tmp_path, fake_token_dir):
    _, out = _run(tmp_path, fake_token_dir, "dense")
    ckpt = sorted(out.glob("checkpoint_step*.pt"))[-1]
    model = CausalTransformer(make_tiny_cfg(vocab_size=VOCAB))
    cfg = TrainConfig(output_dir=str(tmp_path / "resumed"))
    trainer = Trainer(model, [], None, cfg, device="cpu", resume_from=str(ckpt))
    assert trainer._resume_step == torch.load(ckpt, weights_only=False)["step"]
