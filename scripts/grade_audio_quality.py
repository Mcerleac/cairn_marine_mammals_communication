#!/usr/bin/env python3
"""Assign transparent, heuristic signal-quality grades to raw audio files.

This tool is intended for triaging new raw audio before a training pipeline. It
does not identify whale vocalizations, estimate biological relevance, or replace
an annotated quality-control protocol.
"""

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


GRADE_ORDER = ("A", "B", "C", "D", "F")


def segment_audio(audio, sr, max_duration, min_duration):
    """Split audio into deterministic, fixed-duration segments."""
    step, minimum = int(max_duration * sr), int(min_duration * sr)
    if step <= 0:
        raise ValueError("max_duration must be positive")
    return [audio[i:i + step] for i in range(0, len(audio), step)
            if len(audio[i:i + step]) >= minimum]


def quality_metrics(audio, sr):
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak_to_rms = float(np.max(np.abs(audio)) / max(rms, 1e-10))
    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return {"spectral_flatness": 1.0, "spectral_centroid_hz": 0.0,
                "spectral_bandwidth_hz": 0.0, "rms_energy": rms,
                "peak_to_rms": peak_to_rms, "energy_variance": 0.0}
    hop, frame = max(int(0.010 * sr), 1), min(int(0.025 * sr), len(audio))
    energies = [np.sqrt(np.mean(np.square(audio[i:i + frame])))
                for i in range(0, len(audio) - frame + 1, hop)]
    power = np.abs(librosa.stft(audio, n_fft=n_fft)) ** 2
    return {
        "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(S=power))),
        "spectral_centroid_hz": float(np.mean(librosa.feature.spectral_centroid(S=power, sr=sr))),
        "spectral_bandwidth_hz": float(np.mean(librosa.feature.spectral_bandwidth(S=power, sr=sr))),
        "rms_energy": rms, "peak_to_rms": peak_to_rms,
        "energy_variance": float(np.var(energies)) if energies else 0.0,
    }


def score(metrics):
    """Return a 0-1 acoustic-signal heuristic, not vocalization confidence."""
    tonality = 1.0 - min(metrics["spectral_flatness"], 1.0)
    dynamics = min(metrics["energy_variance"] / 0.005, 1.0)
    crest = max(0.0, min((metrics["peak_to_rms"] - 1.0) / 10.0, 1.0))
    loudness = min(metrics["rms_energy"] / 0.01, 1.0)
    return float(np.clip(0.35 * tonality + 0.25 * dynamics + 0.20 * crest + 0.20 * loudness, 0.0, 1.0))


def grade(value):
    for threshold, label in ((0.8, "A"), (0.6, "B"), (0.4, "C"), (0.2, "D")):
        if value >= threshold:
            return label
    return "F"


def main():
    parser = argparse.ArgumentParser(description="Grade raw audio with a transparent signal-quality heuristic.")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory searched recursively for WAV/FLAC files")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--source", default="unspecified", help="Label written to every output row")
    parser.add_argument("--target-sr", type=int, default=44100)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--min-duration", type=float, default=0.3)
    args = parser.parse_args()
    if not args.audio_dir.is_dir():
        parser.error(f"--audio-dir does not exist or is not a directory: {args.audio_dir}")
    files = sorted(p for suffix in ("*.wav", "*.flac") for p in args.audio_dir.rglob(suffix) if p.stat().st_size)
    if not files:
        parser.error("No non-empty WAV or FLAC files found")

    rows = []
    for path in tqdm(files, desc="Grading audio"):
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != args.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.target_sr)
            for index, segment in enumerate(segment_audio(audio, args.target_sr, args.max_duration, args.min_duration)):
                values = quality_metrics(segment, args.target_sr)
                value = score(values)
                rows.append({"source": args.source, "file": str(path.relative_to(args.audio_dir)),
                             "segment_idx": index, "duration_s": round(len(segment) / args.target_sr, 3),
                             **{key: round(metric, 6) for key, metric in values.items()},
                             "quality_score": round(value, 4), "grade": grade(value)})
        except Exception as error:
            tqdm.write(f"Skipping {path}: {error}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "file", "segment_idx", "duration_s", "spectral_flatness", "spectral_centroid_hz",
              "spectral_bandwidth_hz", "rms_energy", "peak_to_rms", "energy_variance", "quality_score", "grade"]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: sum(row["grade"] == label for row in rows) for label in GRADE_ORDER}
    print(f"Wrote {len(rows)} segments to {args.output_csv}")
    print("Grades: " + ", ".join(f"{label}={counts[label]}" for label in GRADE_ORDER))


if __name__ == "__main__":
    main()
