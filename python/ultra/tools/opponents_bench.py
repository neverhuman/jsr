"""Cold-boundary adapters for external whole-utterance CPU ASR opponents.

Writes one <stem>.txt hypothesis per staged 16 kHz mono PCM16 WAV so outputs are scored
by the exact same `paper_dev_audio.py score` + `ultra-score` pipeline as all other arms.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import wave

import numpy as np


def read_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        if sr != 16000 or ch != 1 or sw != 2:
            raise ValueError(f"expected 16kHz mono PCM16 WAV: {path} ({sr}Hz, {ch}ch, {sw}B)")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


def sorted_wavs(wav_dir: Path, limit: int | None) -> list[Path]:
    wavs = sorted(p for p in wav_dir.iterdir() if p.is_file() and p.suffix == ".wav")
    if not wavs:
        raise ValueError(f"no .wav files in {wav_dir}")
    return wavs[:limit] if limit else wavs


def run_parakeet_tdt_v3(model_dir: Path, wav_dir: Path, output_dir: Path, threads: int, limit: int | None) -> None:
    import sherpa_onnx

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=threads,
        provider="cpu",
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in sorted_wavs(wav_dir, limit):
        samples, sr = read_wav_f32(wav_path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        (output_dir / f"{wav_path.stem}.txt").write_text(text + "\n", encoding="utf-8")


def run_parakeet_redux(model_id: str, wav_dir: Path, output_dir: Path, threads: int, limit: int | None) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import moondream as md

    output_dir.mkdir(parents=True, exist_ok=True)
    with md.photon(model_id, device="cpu") as speech:
        for wav_path in sorted_wavs(wav_dir, limit):
            result = speech.transcribe(audio=str(wav_path))
            text = result["text"] if isinstance(result, dict) else str(result)
            (output_dir / f"{wav_path.stem}.txt").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tdt = sub.add_parser("parakeet-tdt-v3", help="Run sherpa-onnx Parakeet TDT 0.6B v3 INT8")
    p_tdt.add_argument("--model-dir", type=Path, required=True)
    p_tdt.add_argument("--wav-dir", type=Path, required=True)
    p_tdt.add_argument("--output-dir", type=Path, required=True)
    p_tdt.add_argument("--threads", type=int, default=1)
    p_tdt.add_argument("--limit", type=int, default=None)

    p_rdx = sub.add_parser("parakeet-redux", help="Run moondream/parakeet-redux via Photon CPU")
    p_rdx.add_argument("--model-id", default="moondream/parakeet-redux")
    p_rdx.add_argument("--wav-dir", type=Path, required=True)
    p_rdx.add_argument("--output-dir", type=Path, required=True)
    p_rdx.add_argument("--threads", type=int, default=1)
    p_rdx.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "parakeet-tdt-v3":
        run_parakeet_tdt_v3(args.model_dir, args.wav_dir, args.output_dir, args.threads, args.limit)
    elif args.cmd == "parakeet-redux":
        run_parakeet_redux(args.model_id, args.wav_dir, args.output_dir, args.threads, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
