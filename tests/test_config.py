from pathlib import Path

import pytest
import yaml

from src.model.config import PRESETS, TransformerConfig, get_config

CONFIG_FILES = sorted(Path(__file__).resolve().parents[1].glob("configs/**/*.yaml"))


@pytest.mark.parametrize("name", PRESETS)
def test_presets_are_consistent(name):
    cfg = PRESETS[name]
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.d_head * cfg.n_heads == cfg.d_model
    assert cfg.param_count_estimate() > 0


def test_get_config_overrides_do_not_mutate_preset():
    before = PRESETS["small"].n_layers
    cfg = get_config("small", n_layers=1, vocab_size=10)
    assert (cfg.n_layers, cfg.vocab_size) == (1, 10)
    assert PRESETS["small"].n_layers == before


def test_get_config_rejects_unknown_field():
    with pytest.raises(TypeError):
        get_config("tiny", not_a_field=1)


def test_config_files_found():
    assert CONFIG_FILES, "no YAML configs found under configs/"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_training_config_is_valid(path):
    """Catch config typos before they cost GPU hours. Builds the config object only, not the model."""
    cfg = yaml.safe_load(path.read_text())
    model, data, train = cfg["model"], cfg["data"], cfg["training"]

    assert model["preset"] in PRESETS
    overrides = {k: v for k, v in model.items() if k != "preset"}
    mcfg = get_config(model["preset"], **overrides)  # raises on unknown keys
    assert isinstance(mcfg, TransformerConfig)
    assert mcfg.d_model % mcfg.n_heads == 0

    assert data["dataset_type"] in {"audio", "coda", "dialogue"}
    assert data.get("max_seq_len", 512) <= mcfg.max_seq_len
    for key in ("sep_token", "sep_gap_token"):
        if data.get(key) is not None:
            assert 0 < data[key] < mcfg.vocab_size, f"{key} outside vocab"
    if data.get("interleave_codebooks"):
        # CB offsets put tokens in 1..n*1024; SEP tokens must sit above that.
        assert data["interleave_codebooks"] * 1024 < mcfg.vocab_size

    for key in ("learning_rate", "weight_decay", "batch_size", "num_epochs", "output_dir"):
        assert key in train, f"training.{key} missing"
    if mcfg.compressed_attn_stride:
        assert mcfg.full_attention_every_n > 0, "compressed attention only applies to global layers"
