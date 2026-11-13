"""Shared fixtures. Constants and builders live in tests/helpers.py."""

import numpy as np
import pytest
import torch

from tests.helpers import N_CODEBOOKS


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def fake_token_dir(tmp_path):
    """Two 'recordings' of three chunks each, shaped like real DAC output: (9, T) int16."""
    d = tmp_path / "tokens"
    d.mkdir()
    rng = np.random.default_rng(0)
    for rec in ("recA", "recB"):
        for i in range(3):
            arr = rng.integers(1, 1025, size=(N_CODEBOOKS, 20), dtype=np.int16)
            np.save(d / f"{rec}_{i:06d}.npy", arr)
    return d
