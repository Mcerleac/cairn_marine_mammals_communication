# Evaluation utilities

These scripts support inspection of new raw audio, DAC reconstruction, released
checkpoint loss, and WAV outputs from the public generator. Install the base
dependencies first:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

For DAC round-trip reconstruction, install the codec extra:

```bash
pip install -e ".[audio-codec]"
```

## Grade new raw audio

`scripts/grade_audio_quality.py` recursively reads WAV and FLAC files,
resamples them, divides them into deterministic fixed-duration segments, and
writes a CSV of spectral flatness, spectral centroid and bandwidth, RMS,
crest factor, energy variance, a composite 2È›y¯ßy”1 score, and BÈ›y¯ßy”F grade.

```bash
PYTHONPATH=. python3 scripts/grade_audio_quality.py \
  --audio-dir data/new_recordings \
  --source university_fieldwork \
  --output-csv data/new_recordings_quality.csv
```

The grade is an acoustic signal-quality heuristic. It is **not** a whale
detector, species classifier, biological-quality score, or replacement for
manual review. It is appropriate for ranking or filtering the teakßuÁ‚ùÁYs own raw
recordings; it does not reproduce the released SanctSound score file.

## Measure DAC round-trip reconstruction

`scripts/evaluate_dac_roundtrip.py` encodes and decodes one raw file with
DAC, then reports aligned-sample count, SNR, and spectral convergence. It can
also save the decoded WAV for listening.

```bash
PYTHONPATH=. python3 scripts/evaluate_dac_roundtrip.py recording.flac \
  --n-codebooks 9 --device cuda \
  --output-json runs/codec_roundtrip.json \
  --output-wav runs/codec_roundtrip.wav
```

This measures codec reconstruction only. It does not evaluate a language-model
checkpoint or a generated continuation.

## Evaluate a released checkpoint

`scripts/evaluate_checkpoint.py` reconstructs the architecture from a
checkpoint and evaluates next-token cross-entropy and perplexity over the
deterministic 80/20 split used by `scripts/train.py`. The default limit of 20
batches keeps long-context evaluation practical; use `--max-batches 0` only
when a full split evaluation is intentional.

```bash
PYTHONPATH=. python3 scripts/evaluate_checkpoint.py \
  models/best_model.pt \
  --token-dir data/tokenized/sanctsound_humpback_dac \
  --score-file data/tokenized/sanctsound_humpback_dac/chunk_scores.csv \
  --min-detector-score 0.7 \
  --max-batches 20 \
  --output-json runs/eval_32k.json
```

For the released DAC-9 datasets, the defaults use 9-codebook interleaving and
the `SEP` token (9218). Provide compatible overrides only when evaluating a
different token format.

## Analyze generated WAV files

`scripts/analyze_generation_quality.py` reads the
`generated_*.wav` files written by `scripts/generate_dac_9cb_unified.py`.
It writes JSON and CSV summaries containing duration, RMS, peak level, active
fraction, spectral centroid and bandwidth, spectral flatness, peak frequency,
and spectral novelty.

```bash
PYTHONPATH=. python3 scripts/generate_dac_9cb_unified.py \
  --checkpoint models/best_model.pt \
  --token-dir data/tokenized/sanctsound_humpback_dac \
  --output-dir runs/prompted \
  --decode-audio both

PYTHONPATH=. python3 scripts/analyze_generation_quality.py \
  --audio-dir runs/prompted --include-full
```

The analyzer reports acoustic descriptors; it does not determine whether a
continuation is whale-like, biologically meaningful, or preferable to another
sample. Compare files generated from the same prompt and decode mode, and use
listening plus domain review for substantive claims.
