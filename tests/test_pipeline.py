"""Automated CPU unit and integration tests for cairn-marine-mammals-communication."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.analyze_generation_quality import metrics as generation_audio_metrics
from scripts.evaluate_dac_roundtrip import comparison_metrics
from scripts.generate_dac_9cb_unified import generate_with_sep_stopping, interleave_2d
from scripts.grade_audio_quality import grade, quality_metrics, score
from src.data.dataset import AudioTokenDataset
from src.model.checkpoints import load_model_from_checkpoint
from src.model.config import TransformerConfig
from src.model.transformer import CausalTransformer
from src.model.weight_analysis import analyze_state_dict
from src.training.muon import Muon


def test_dac_codebook_interleave_roundtrip():
    """FR-1: Verify 9-codebook 2D (9, T) interleaving and de-interleaving preserves exact codes."""
    rng = np.random.default_rng(42)
    codes_2d = rng.integers(1, 1025, size=(9, 20), dtype=np.int32)
    flat = interleave_2d(codes_2d)
    assert flat.shape == (180,)
    assert flat.min() >= 1 and flat.max() <= 9216

    reconstructed = flat.reshape(20, 9).T - (np.arange(9).reshape(9, 1) * 1024)
    np.testing.assert_array_equal(codes_2d, reconstructed)


def test_audio_token_dataset_concat_and_filtering(tmp_path: Path):
    """FR-2: Verify AudioTokenDataset filters by detector_score and inserts SEP/SEP_GAP tokens."""
    rng = np.random.default_rng(7)
    for idx in range(3):
        arr = rng.integers(1, 1025, size=(9, 4), dtype=np.int32)
        np.save(tmp_path / f"sanctsound_hi01_01_00000{idx}.npy", arr)

    scores_csv = tmp_path / "chunk_scores.csv"
    pd.DataFrame([
        {"npy_file": "sanctsound_hi01_01_000000.npy", "flac_name": "a.flac", "chunk_idx_in_flac": 0, "detector_score": 0.9},
        {"npy_file": "sanctsound_hi01_01_000001.npy", "flac_name": "a.flac", "chunk_idx_in_flac": 1, "detector_score": 0.85},
        {"npy_file": "sanctsound_hi01_01_000002.npy", "flac_name": "a.flac", "chunk_idx_in_flac": 2, "detector_score": 0.2},
    ]).to_csv(scores_csv, index=False)

    ds = AudioTokenDataset(
        token_dir=tmp_path,
        max_seq_len=64,
        vocab_size=9219,
        concat=True,
        sep_token=9218,
        sep_gap_token=9217,
        adjacency_file=scores_csv,
        interleave_codebooks=9,
        score_file=scores_csv,
        min_detector_score=0.7,
    )
    assert len(ds) == 2
    sample = ds[0]
    assert sample["input_ids"].shape == (64,)
    assert sample["target_ids"].shape == (64,)
    assert sample["input_ids"][36].item() == 9218


def test_transformer_forward_and_kv_generation():
    """FR-3: Verify SWA+MoE CausalTransformer loss computation and KV-cached generation."""
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=128, max_seq_len=64, n_layers=2, n_heads=4,
        d_model=32, d_ff=64, swa_window_size=16, full_attention_every_n=2,
        n_experts=4, moe_top_k=2, expert_d_ff=32, dropout=0.0,
    )
    model = CausalTransformer(cfg)
    x = torch.randint(1, 120, (2, 24))
    y = torch.randint(1, 120, (2, 24))
    out = model(x, targets=y)
    assert out["loss"] is not None and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.token_emb.weight.grad is not None

    gen = model.generate(x[:1, :10], max_new_tokens=6, temperature=0.0, eos_token_id=999)
    assert gen.shape == (1, 16)


def test_nsa_moe_forward_and_sep_stopping():
    """FR-3 & NFR-2: Verify CompressedGlobalAttention (NSA) + split QKV + bias routing + SEP stopping."""
    torch.manual_seed(1)
    cfg = TransformerConfig(
        vocab_size=9219, max_seq_len=128, n_layers=2, n_heads=4,
        d_model=32, d_ff=64, swa_window_size=16, full_attention_every_n=2,
        compressed_attn_stride=9, compressed_attn_chunk=32,
        n_experts=4, moe_top_k=1, expert_d_ff=32,
        use_bias_routing=True, use_split_qkv=True, dropout=0.0,
    )
    model = CausalTransformer(cfg)
    assert hasattr(model.blocks[1].attn, "q_proj")
    x = torch.randint(1, 9200, (1, 27))
    out = model(x, targets=x)
    assert torch.isfinite(out["loss"])

    gen = generate_with_sep_stopping(model, x[:, :9], max_new_tokens=5, temperature=0.0)
    assert gen.shape[1] == 14


def test_checkpoint_roundtrip_and_weight_analysis(tmp_path: Path):
    """FR-4: Verify checkpoint save/load architecture reconstruction and deduplicated weight analysis."""
    cfg = TransformerConfig(vocab_size=64, max_seq_len=32, n_layers=2, n_heads=2, d_model=16, d_ff=32, dropout=0.0)
    model = CausalTransformer(cfg)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save({"step": 10, "val_loss": 2.5, "config": cfg, "model_state_dict": model.state_dict()}, ckpt_path)

    loaded_ckpt, loaded_cfg, loaded_model = load_model_from_checkpoint(ckpt_path, device="cpu")
    assert loaded_cfg.d_model == 16
    x = torch.randint(1, 60, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(x)["logits"], loaded_model(x)["logits"])

    report = analyze_state_dict(loaded_ckpt["model_state_dict"], loaded_cfg, include_tensors=True)
    rope_buffer_elems = model.rope.inv_freq.numel()
    assert report["overall"]["parameter_count"] == model.count_parameters() + rope_buffer_elems
    assert any(t["is_shared_alias"] for t in report["tensors"])


def test_muon_optimizer_decreases_loss():
    """FR-5: Verify Muon Newton-Schulz orthogonalized step decreases 2D matrix objective."""
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.eye(16)
    opt = Muon([w], lr=0.05)

    initial_loss = (w - target).pow(2).mean().item()
    for _ in range(5):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        opt.step()
    final_loss = (w - target).pow(2).mean().item()
    assert final_loss < initial_loss


def test_acoustic_and_roundtrip_metrics():
    """FR-6: Verify spectral convergence, SNR, generation descriptors, and quality grading."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    whale_tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    high_tone = 0.5 * np.sin(2 * np.pi * 10000 * t)

    identical = comparison_metrics(whale_tone, whale_tone, sr)
    perturbed = comparison_metrics(whale_tone, high_tone, sr)
    assert identical["snr_db"] >= 90.0
    assert identical["spectral_convergence"] < 1e-6
    assert perturbed["spectral_convergence"] > 0.5

    gen_desc = generation_audio_metrics(whale_tone, sr)
    assert 900.0 <= gen_desc["peak_frequency_hz"] <= 1100.0

    m = quality_metrics(whale_tone, sr)
    assert grade(score(m)) in {"A", "B", "C"}
