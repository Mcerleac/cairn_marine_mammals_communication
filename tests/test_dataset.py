import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import AudioTokenDataset
from src.data.symbolic_tokenizer import SymbolicVocab, decode_token_sequence, tokenize_coda_sequence
from tests.helpers import N_CODEBOOKS, SEP, VOCAB


def _check_items(ds, max_seq_len):
    assert len(ds) > 0
    for i in range(len(ds)):
        item = ds[i]
        for key in ("input_ids", "target_ids", "attention_mask"):
            assert item[key].shape == (max_seq_len,)
        assert item["input_ids"].min() >= 0 and item["input_ids"].max() < VOCAB
        assert torch.equal(item["input_ids"][1:], item["target_ids"][:-1]), "targets must be inputs shifted by 1"


def test_interleaved_concat_matches_training_config(fake_token_dir):
    """Same data options as configs/*: 9 interleaved codebooks, files concatenated with SEP."""
    ds = AudioTokenDataset(
        fake_token_dir, max_seq_len=64, vocab_size=VOCAB,
        concat=True, sep_token=SEP, interleave_codebooks=N_CODEBOOKS,
    )
    _check_items(ds, 64)
    # 3 files x 20 timesteps x 9 CBs + 2 SEPs per recording
    assert sum(g.total_length for g in ds._groups) == 2 * (3 * 20 * 9 + 2)
    assert any((ds[i]["input_ids"] == SEP).any() for i in range(len(ds)))


def test_interleave_order_and_offsets(fake_token_dir):
    ds = AudioTokenDataset(fake_token_dir, max_seq_len=1000, vocab_size=VOCAB, interleave_codebooks=N_CODEBOOKS)
    raw = np.load(sorted(fake_token_dir.glob("*.npy"))[0])
    tokens = ds[0]["input_ids"][:N_CODEBOOKS].numpy()
    # first timestep, codebook c is shifted into its own 1024-wide range
    assert list(tokens) == [raw[c, 0] + c * 1024 for c in range(N_CODEBOOKS)]


def test_simple_mode_pads_short_files(fake_token_dir):
    ds = AudioTokenDataset(fake_token_dir, max_seq_len=32, vocab_size=1025, codebook_index=0)
    _check_items(ds, 32)
    long_ds = AudioTokenDataset(fake_token_dir, max_seq_len=64, vocab_size=1025, codebook_index=0)
    item = long_ds[0]
    assert item["attention_mask"].sum() == 20  # 20 real tokens, rest padding
    assert item["target_ids"][19] == 0  # last real input predicts PAD (ignored by the loss)


def test_augment_keeps_tokens_in_vocab(fake_token_dir):
    ds = AudioTokenDataset(
        fake_token_dir, max_seq_len=64, vocab_size=VOCAB, augment=True,
        token_noise_prob=0.5, token_mask_prob=0.5, time_stretch_prob=1.0,
        concat=True, sep_token=SEP, interleave_codebooks=N_CODEBOOKS,
    )
    _check_items(ds, 64)


def test_empty_dir_gives_empty_dataset(tmp_path):
    assert len(AudioTokenDataset(tmp_path)) == 0


# --- symbolic tokenizer ---

def test_vocab_ranges_do_not_overlap():
    v = SymbolicVocab()
    assert v.whale_token(1) == v.WHALE_OFFSET
    assert v.whale_token(v.MAX_WHALES) < v.CODA_OFFSET
    assert min(v.coda_type_to_id.values()) == v.CODA_OFFSET
    assert v.vocab_size == max(v.coda_type_to_id.values()) + 1


@pytest.mark.parametrize("gap,expected", [(0.1, "PAUSE_VERY_SHORT"), (1.0, "PAUSE_SHORT"), (3, "PAUSE_MEDIUM"),
                                          (10, "PAUSE_LONG"), (60, "PAUSE_VERY_LONG")])
def test_pause_bins(gap, expected):
    v = SymbolicVocab()
    assert v.pause_token(gap) == getattr(v, expected)


def test_tokenize_and_decode_roundtrip():
    v = SymbolicVocab()
    df = pd.DataFrame({
        "CodaType": ["1+1+3", "5R1-NOISE", "4D"],
        "TsTo": [0.0, 1.0, 3.0],
        "Duration": [0.5, 0.5, 0.5],
        "Whale": [1, 1, 2],
    })
    tokens = tokenize_coda_sequence(df, v, include_whale_ids=True)
    assert tokens[0] == v.BOS and tokens[-1] == v.EOS
    decoded = decode_token_sequence(tokens, v)
    assert "1+1+3" in decoded and "4D" in decoded
    assert not any("NOISE" in t for t in decoded)
    assert "<whale:2>" in decoded
