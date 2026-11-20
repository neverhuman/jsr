"""Frozen evaluation slices (bench plan C4). Rules and thresholds are fixed on DEV only.

- long_form: utterance audio longer than LONG_FORM_S seconds.
- fast_speech: reference words per second at or above FAST_WPS (the 90th percentile of DEV,
  dev-clean + dev-other, computed once by `freeze_thresholds` and written here).
- silence probe: a deterministic 10-minute non-speech file (digital silence, then low-level
  noise); scored as inserted words per hour, never as WER.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import soundfile as sf

LONG_FORM_S = 20.0
FAST_WPS = 3.5682  # DEV p90 words/s (n=5567), frozen 2026-11-20 by freeze_thresholds(); see docs/ultra/SLICES.md
SR = 16000


def words_per_second(text: str, seconds: float) -> float:
    return len(text.split()) / max(seconds, 1e-9)


def slice_of(text: str, seconds: float) -> list[str]:
    out = []
    if seconds > LONG_FORM_S:
        out.append("long_form")
    if words_per_second(text, seconds) >= FAST_WPS:
        out.append("fast_speech")
    return out


def freeze_thresholds(dev_utts) -> dict:
    """p90 words/s over DEV utterances (objects with .text and .seconds)."""
    wps = np.array([words_per_second(u.text, u.seconds) for u in dev_utts])
    return {"fast_wps_p90": round(float(np.percentile(wps, 90)), 4), "n": int(len(wps))}


def silence_probe(path: str, seed: int = 20261120) -> str:
    """Write the 10-minute probe (5 min digital silence + 5 min noise at about -50 dBFS); return sha256."""
    rng = np.random.default_rng(seed)
    silence = np.zeros(5 * 60 * SR, dtype=np.int16)
    noise = np.clip(rng.normal(0.0, 32768 * 10 ** (-50 / 20), 5 * 60 * SR), -32767, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.concatenate([silence, noise]), SR, subtype="PCM_16")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
