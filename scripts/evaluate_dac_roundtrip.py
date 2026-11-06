#!/usr/bin/env python3
"""Measure DAC encode/decode reconstruction for one raw audio file.

This evaluates codec reconstruction, not generated-audio quality. It requires
the audio-codec optional dependency and DAC's public 44 kHz model weights.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.tokenizer.dac_tokenizer import DACTokenizer


def comparison_metrics(original, reconstructed, sr):
    size = min(len(original), len(reconstructed))
    original, reconstructed = original[:size], reconstructed[:size]
    noise = original - reconstructed
    signal_power, noise_power = float(np.mean(original ** 2)), float(np.mean(noise ** 2))
    snr = 100.0 if noise_power < 1e-12 else 10 * np.log10(signal_power / noise_power)
    n_fft = min(2048, size)
    source, decoded = np.abs(librosa.stft(original, n_fft=n_fft)), np.abs(librosa.stft(reconstructed, n_fft=n_fft))
    frames = min(source.shape[1], decoded.shape[1])
    convergence = np.linalg.norm(source[:, :frames] - decoded[:, :frames]) / max(np.linalg.norm(source[:, :frames]), 1e-10)
    return {"aligned_samples": size, "duration_s": size / sr, "snr_db": float(snr), "spectral_convergence": float(convergence)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAC round-trip reconstruction for one audio file.")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--n-codebooks", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-wav", type=Path, default=None)
    args = parser.parse_args()
    if not args.audio_file.is_file():
        parser.error(f"Audio file not found: {args.audio_file}")
    audio, sr = sf.read(args.audio_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tokenizer = DACTokenizer(device=args.device, n_codebooks=args.n_codebooks)
    if sr != tokenizer.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=tokenizer.sample_rate)
        sr = tokenizer.sample_rate
    codes, _ = tokenizer.encode_file(args.audio_file)
    reconstructed = tokenizer.decode_2d_to_audio(tokenizer.codes_to_2d(codes), n_codebooks=args.n_codebooks)
    result = {"audio_file": str(args.audio_file), "n_codebooks": args.n_codebooks, **comparison_metrics(audio, reconstructed, sr)}
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_wav:
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_wav, reconstructed, sr)


if __name__ == "__main__":
    main()
