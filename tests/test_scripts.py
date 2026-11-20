"""Script-level checks: every CLI parses its args, and train.py runs end-to-end on a toy config."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
# Optional extras not installed in CI (audio-codec / dataset / training).
OPTIONAL_DEPS = ("google", "pyloudnorm", "dac", "tensorflow", "bitsandbytes", "audiotools", "lac")


def _run(args, timeout=120):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_help(script):
    r = _run([str(script), "--help"])
    missing = [d for d in OPTIONAL_DEPS if f"No module named '{d}" in r.stderr]
    if missing:
        pytest.skip(f"needs optional dependency: {missing[0]}")
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.slow
def test_train_script_end_to_end(tmp_path, fake_token_dir):
    """scripts/train.py with a toy config: no GPU, no real data."""
    real = yaml.safe_load((ROOT / "configs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml").read_text())
    cfg = {
        "model": {**real["model"], "n_layers": 2, "n_heads": 2, "d_model": 32, "d_ff": 64,
                  "max_seq_len": 64, "swa_window_size": 16, "expert_d_ff": 32, "n_experts": 4},
        "data": {**real["data"], "token_dir": str(fake_token_dir), "max_seq_len": 64, "score_file": None},
        "training": {**real["training"], "use_8bit_adam": False, "batch_size": 2, "val_batch_size": 2,
                     "num_epochs": 1, "warmup_steps": 1, "eval_interval": 2, "log_interval": 1,
                     "output_dir": str(tmp_path / "run")},
    }
    cfg_path = tmp_path / "toy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    r = _run(["scripts/train.py", str(cfg_path), "--device", "cpu"], timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "Training complete" in r.stdout
    assert (tmp_path / "run" / "best_model.pt").exists()

    r = _run(["scripts/describe_checkpoint.py", str(tmp_path / "run" / "best_model.pt")])
    assert r.returncode == 0, r.stderr[-2000:]
