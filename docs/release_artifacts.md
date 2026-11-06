# Release Artifacts

This document defines the public artifact layout expected by the curated MMC release repo.

## Goal

With the code in this repository plus the published Hugging Face artifacts, a third party should be able to:

- regenerate the SanctSound humpback tokenized dataset
- train the published audio models from scratch
- fine-tune from published model weights
- resume from full training checkpoints
- generate audio examples from the released checkpoints
- evaluate checkpoints and generation outputs with the supported public utilities

## Dataset Release

Recommended Hugging Face dataset repo contents:

- tokenized `.npy` files under a stable directory layout
- `chunk_scores.csv`
- any metadata files required by the training pipeline

Minimum required dataset files for the 9-codebook SanctSound training path:

- `data/tokenized/sanctsound_humpback_dac/*.npy`
- `data/tokenized/sanctsound_humpback_dac/chunk_scores.csv`

Why `chunk_scores.csv` matters:

- generation scripts use it to select high-quality prompts
- training configs use it for `score_file`
- adjacency-aware concat training uses it through `adjacency_file`

Expected token format:

- 2D DAC 44 kHz code arrays with 9 codebooks
- training interleaves them at load time using `interleave_codebooks: 9`

## Checkpoint Release

Recommended Hugging Face model repo contents for each released model family:

- `best_model.pt`
- at least one full `checkpoint_step*.pt` file for exact resume
- model card / README

Checkpoint semantics:

- `best_model.pt`
  - contains model weights, config, and validation loss
  - self-describes the model architecture, including dense vs SWA vs MoE vs NSA-style settings
  - use with `--init-from`
  - appropriate for inference and fine-tuning
- `checkpoint_step*.pt`
  - contains model weights, config, optimizer state, and step
  - may also contain `muon_state_dict`
  - self-describes the model architecture the same way as `best_model.pt`
  - use with `--resume-from`
  - appropriate for exact continuation of training

## Config Mapping

The curated repo currently includes the public configs most directly tied to the blog posts:

- `configs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml`
  - 10k context training / inference path
- `configs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml`
  - 32k Medium SWA+MoE release path
- `configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml`
  - 128k context path with `SEP` / `SEP_GAP` boundary support

Suggested public naming on Hugging Face:

- model repo: `cairninstitute/mmc-humpback-dac9-10k`
- model repo: `cairninstitute/mmc-humpback-dac9-32k`
- model repo: `cairninstitute/mmc-humpback-dac9-128k`
- dataset repo: `cairninstitute/mmc-sanctsound-humpback-dac9`

The 64k NSA config is retained at `configs/historical/` for historical reference and is not a release artifact.

These names are suggestions only. The important requirement is that the README or model card clearly maps each released artifact to one of the included config files.

## Supported Public Workflows

Dataset reconstruction:

1. Download raw SanctSound FLAC files with `scripts/download_sanctsound.py`
2. Process/tokenize them with `scripts/process_sanctsound_humpback.py --codec dac`

Training from scratch:

1. Download the released tokenized dataset to `data/tokenized/sanctsound_humpback_dac`
2. Run `scripts/train.py` with one of the included configs

Fine-tuning:

1. Download `best_model.pt`
2. Optionally inspect it with `scripts/describe_checkpoint.py`
3. Run `scripts/train.py ... --init-from best_model.pt`
4. The training script will reconstruct the model architecture from the checkpoint and ignore conflicting YAML model overrides

Exact resume:

1. Download `checkpoint_step*.pt`
2. Optionally inspect it with `scripts/describe_checkpoint.py`
3. Run `scripts/train.py ... --resume-from checkpoint_step*.pt`
4. The training script will reconstruct the model architecture from the checkpoint and ignore conflicting YAML model overrides

Prompted generation:

1. Download a released checkpoint
2. Download the tokenized dataset and `chunk_scores.csv`
3. Run `scripts/generate_dac_9cb_unified.py --checkpoint <checkpoint>`
4. The unified script reconstructs architecture from the checkpoint and chooses the appropriate generation behavior automatically
5. Prompt selection can be score-driven or explicit via `--prompt-file`, and output can be saved as prompt+continuation, generated-only, or both

## What Is Not In Git

This repo intentionally excludes:

- raw audio
- tokenized datasets
- checkpoints
- run outputs
- logs

Those artifacts should live on Hugging Face or other durable storage.
