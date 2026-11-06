# Cairn Marine Mammals Communication

Curated code release for the marine mammal communication work described in the Cairn Institute blog posts on dataset construction and model training.

## License

The code, configurations, and documentation in this repository are licensed
under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for required
attribution and source-data notices. The tokenized Project DoLittle dataset is
governed separately by the [Project DoLittle Dataset Terms](docs/DATASET_TERMS.md).

This repository is intentionally smaller than the active research repo. It contains the minimum code, configs, and documentation needed to:

- reproduce the SanctSound humpback tokenization pipeline described in the dataset post
- train the published 9-codebook audio language models
- run prompted generation from the released checkpoints

It does not include:

- raw audio
- tokenized datasets
- model checkpoints
- run artifacts
- local experiment logs

Those large artifacts should live in Hugging Face repositories or other durable storage.

## Release Status

This repository is intended to be sufficient, together with published Hugging Face artifacts, to:

- regenerate the released SanctSound humpback tokenized dataset
- train the released model families from scratch
- initialize from a published checkpoint and fine-tune
- resume from a published full training checkpoint
- inspect checkpoint architecture directly from the `.pt` file
- run prompted audio generation

The Hugging Face side still needs to provide the actual datasets and checkpoints.

## Scope

Included public paths:

- `scripts/download_sanctsound.py`
  - anonymous download helper for NOAA SanctSound FLAC files
- `scripts/process_sanctsound_humpback.py`
  - humpback preprocessing and tokenization pipeline
- `scripts/tokenize_audio.py`
  - generic tokenization entrypoint for WAV directories
- `scripts/train.py`
  - training entrypoint for released configs
- `scripts/generate_dac_9cb_unified.py`
  - unified prompted generation for released 10k, 32k, and 128k DAC 9-codebook checkpoints
- `scripts/generate_dac_9cb_prompted.py`
  - legacy prompted generation script for the 10k context checkpoint
- `scripts/generate_dac_9cb_prompted_128k.py`
  - legacy prompted generation script for the 128k context checkpoint

Included configs:

- `configs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml`
- `configs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml`
- `configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml`

Included implementation packages:

- `src/data`
- `src/model`
- `src/tokenizer`
- `src/training`

Included documentation:

- `docs/whale_audio_pipeline.md`
- `docs/model_diagrams.md` — regenerate architecture SVGs from training configs

## Blog Post Coverage

This repository is intended to support the public claims in:

- the dataset post describing the SanctSound humpback pipeline, DAC 44 kHz tokenization, interleaving, and `SEP` / `SEP_GAP` boundaries
- the training post describing the released 10k, 32k, and 128k model families

## External Artifacts

Expected external locations:

- tokenized dataset repo(s): Hugging Face dataset repos
- checkpoint repo(s): Hugging Face model repos
- large raw and processed archives: external storage or buckets

See `docs/release_artifacts.md` for the expected public artifact layout and checkpoint semantics.

The scripts assume local paths such as `data/tokenized/...` and `runs/...`. For release use, download the required artifacts into those paths or adapt the script arguments.

## Installation

Base install:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

For dataset processing and audio codecs:

```bash
pip install -e ".[dataset,audio-codec]"
```

## Training data prerequisite

The repository contains the Python training code, but not the tokenized dataset. Before running `scripts/train.py`, download the released Hugging Face dataset into `data/tokenized/sanctsound_humpback_dac/` so it includes the `.npy` token files and `chunk_scores.csv`. See `docs/release_artifacts.md` for the required artifact layout.

## Minimal Workflows

Download SanctSound:

```bash
PYTHONPATH=. python3 scripts/download_sanctsound.py --station hi01 --deployment 1
```

Process humpback FLAC files into tokenized DAC outputs:

```bash
PYTHONPATH=. python3 scripts/process_sanctsound_humpback.py --codec dac --station hi01
```

Train the 10k model:

```bash
PYTHONPATH=. python3 scripts/train.py configs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml
```

Generate audio from any released checkpoint with the unified entrypoint:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k_v2/best_model.pt
```

Generate audio from the 32k checkpoint:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt
```

Generate audio from the 128k checkpoint:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model.pt
```

Use an explicit tokenized prompt file and control prompt/output lengths:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt   --prompt-file sanctsound_hi05_01_000003.npy   --prompt-token-offset 0   --prompt-token-length 3072   --max-new-tokens 6144   --decode-audio both
```

Save generated continuation only, not prompt+continuation:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model.pt   --prompt-file sanctsound_hi05_01_000003.npy   --max-new-seconds 8   --decode-audio generated
```

Unified generation options:

- checkpoint and paths
  - `--checkpoint PATH`: required checkpoint file
  - `--token-dir PATH`: tokenized dataset directory, defaults to `data/tokenized/sanctsound_humpback_dac`
  - `--scores-csv PATH`: metadata CSV, defaults to `<token-dir>/chunk_scores.csv`
  - `--output-dir PATH`: output directory, defaults to `<checkpoint-dir>/prompted_unified`
  - `--device DEVICE`: generation device, defaults to `cuda`

- prompt selection
  - score-driven mode (default)
    - `--n-samples N`: number of ranked prompt files to use
    - `--min-detector F`: minimum detector score threshold
  - explicit mode
    - `--prompt-file PATH`: explicit `.npy` prompt file, repeatable
    - relative paths resolve under `--token-dir`

- prompt token controls
  - `--prompt-seconds S`: prompt duration in seconds
  - `--prompt-token-offset N`: start token offset inside each selected prompt file
  - `--prompt-token-length N`: exact prompt token count

- generation controls
  - `--max-new-tokens N`: max continuation length in tokens
  - `--max-new-seconds S`: max continuation length in seconds
  - `--mode auto|standard|sep-stop`: generation behavior override
  - `--temperature F`: sampling temperature
  - `--top-k N`: top-k sampling cutoff
  - `--top-p F`: nucleus sampling cutoff

- output controls
  - `--decode-audio full|generated|both`
    - `full`: save prompt + continuation audio only
    - `generated`: save continuation-only audio only
    - `both`: save both forms

Rules and defaults:

- if `--prompt-file` is omitted, prompt files are chosen from `chunk_scores.csv`
- if `--prompt-token-length` is omitted, prompt length is derived from `--prompt-seconds`
- `--max-new-tokens` and `--max-new-seconds` are mutually exclusive
- `--mode auto` uses checkpoint architecture to choose behavior automatically
- long-context 128k-style checkpoints default to SEP-stopping mode in `auto`

Canonical recipes:

Score-driven prompt selection with generated-only output:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k_v2/best_model.pt   --n-samples 3   --min-detector 0.85   --max-new-seconds 6   --decode-audio generated
```

Explicit prompt file with exact token slice:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt   --prompt-file sanctsound_hi05_01_000003.npy   --prompt-token-offset 2048   --prompt-token-length 4096   --max-new-tokens 8192   --decode-audio both
```

Force standard generation mode:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt   --prompt-file sanctsound_hi05_01_000003.npy   --mode standard   --temperature 0.8   --top-k 60
```

Force SEP-stopping mode:

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py   --checkpoint runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model.pt   --prompt-file sanctsound_hi05_01_000003.npy   --mode sep-stop   --max-new-seconds 12   --decode-audio generated
```

Resume training from a full checkpoint:

```bash
PYTHONPATH=. python3 scripts/train.py configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml \
  --resume-from runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/checkpoint_step109000.pt
```

Inspect a checkpoint and infer its architecture directly from the saved config:

```bash
PYTHONPATH=. python3 scripts/describe_checkpoint.py runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model.pt
```

Fine-tune from a released inference checkpoint:

### Checkpoint weight analysis

Analyze weights by transformer layer and architectural structure (attention,
MoE router, experts, norms, embeddings). Totals de-duplicate tied weights such
as the token embedding and language-model head:

```bash
PYTHONPATH=. python3 scripts/analyze_checkpoint_weights.py \
  runs/audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k_v2/best_model.pt \
  --json-out analysis/reports/model_weight_analysis.json
```

The command produces a self-contained `mmc-weight-report/v2` including tensor
and per-attention-head statistics. Use `--summary-only` to omit those detailed
records. The static explorer and its sanitized released reports are documented
in [`analysis/README.md`](analysis/README.md).

```bash
PYTHONPATH=. python3 scripts/train.py configs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k.yaml \
  --init-from runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model.pt
```

When `--init-from` or `--resume-from` is used, the training script reconstructs the model architecture from the checkpoint itself and ignores conflicting YAML model overrides. This makes checkpoint-based fine-tuning and resume safe across dense, SWA, SWA+MoE, and NSA+MoE variants.

## Notes

- The repository preserves the original `src.*` import layout so that the copied scripts continue to run without refactoring.
- Some scripts require optional dependencies such as TensorFlow Hub, Google Cloud Storage, DAC, or LAC.
- Checkpoints and tokenized datasets are intentionally excluded from Git and should be published separately.

- `--init-from` loads model weights only and is appropriate for fine-tuning from `best_model.pt`.
- `--resume-from` expects a full `checkpoint_step*.pt` file with optimizer state and is appropriate for exact training continuation.
