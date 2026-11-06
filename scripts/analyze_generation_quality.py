#!/usr/bin/env python3
"""Summarize acoustic properties of WAV files made by the public generator.

Metrics support comparisons across prompts and sampling settings. They are not a
measure of whale-song quality, biological validity, or model safety.
"""

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def metrics(audio, sr):
    if len(audio) < 64:
        raise ValueError("audio is too short to analyze")
    rms, peak = float(np.sqrt(np.mean(np.square(audio)))), float(np.max(np.abs(audio)))
    n_fft = min(2048, len(audio))
    spectrum = np.abs(librosa.stft(audio, n_fft=n_fft))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_length = min(2048, len(audio))
    hop = min(512, max(1, len(audio) // 2))
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop)
    energy = np.sqrt(np.mean(np.square(frames), axis=0))
    active = float(np.mean(energy > max(np.max(energy) * 0.01, 1e-8))) if len(energy) else 0.0
    novelty = np.linalg.norm(np.diff(spectrum, axis=1), axis=0) if spectrum.shape[1] > 1 else np.array([0.0])
    power = np.square(spectrum)
    return {
        "duration_s": round(len(audio) / sr, 4), "sample_rate_hz": sr,
        "rms": round(rms, 8), "peak": round(peak, 8), "active_fraction": round(active, 6),
        "spectral_centroid_hz": round(float(np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sr))), 3),
        "spectral_bandwidth_hz": round(float(np.mean(librosa.feature.spectral_bandwidth(S=spectrum, sr=sr))), 3),
        "spectral_flatness": round(float(np.mean(librosa.feature.spectral_flatness(S=power))), 6),
        "peak_frequency_hz": round(float(frequencies[np.argmax(np.mean(power, axis=1))]), 3),
        "spectral_novelty": round(float(np.mean(novelty)), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure generated-audio properties from public MMC generator outputs.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--include-full", action="store_true", help="Also analyze full_*.wav prompt-plus-continuation files")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    if not args.audio_dir.is_dir():
        parser.error(f"--audio-dir does not exist or is not a directory: {args.audio_dir}")
    paths = sorted(args.audio_dir.glob("generated_*.wav"))
    if args.include_full:
        paths += sorted(args.audio_dir.glob("full_*.wav"))
    if not paths:
        parser.error("No generated_*.wav files found (or full_*.wav with --include-full)")

    rows = []
    for path in paths:
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            rows.append({"file": path.name, **metrics(audio, sr)})
        except Exception as error:
            print(f"Skipping {path.name}: {error}")
    if not rows:
        raise SystemExit("No files could be analyzed")

    output_json = args.output_json or args.audio_dir / "generation_quality_metrics.json"
    output_csv = args.output_csv or args.audio_dir / "generation_quality_metrics.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Analyzed {len(rows)} files")
    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
