"""H57 FACT Factor E: default-OFF TRAIN-only gain, noise, and causal RIR augmentation.

All probabilities default to 0.0 (no-op, bit-identical to baseline). Deterministic from
`(seed, epoch, uid)` so thread and process data loaders produce identical waveforms and features.
Real-audio `noise_dir` / `rir_dir` directories are validated fail-closed to prevent DEV/TEST leakage.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import soundfile as sf

from train.errors import DataError

_AUDIO_EXTS = (".wav", ".flac", ".ogg")
_FORBIDDEN_EVAL_TOKENS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "dev_clean",
    "dev_other",
    "test_clean",
    "test_other",
)


@dataclass(frozen=True)
class FactAugConfig:
    """Factor E waveform augmentation settings (all default-OFF)."""

    gain_p: float = 0.0
    gain_min_db: float = -35.0
    gain_max_db: float = 6.0
    noise_p: float = 0.0
    noise_dir: str = ""
    noise_snr_min_db: float = 5.0
    noise_snr_max_db: float = 25.0
    rir_p: float = 0.0
    rir_dir: str = ""

    @property
    def active(self) -> bool:
        return self.gain_p > 0.0 or self.noise_p > 0.0 or self.rir_p > 0.0

    def validate(self, splits: list[str] | tuple[str, ...] = ()) -> None:
        if not self.active:
            return
        for name, p in (("gain_p", self.gain_p), ("noise_p", self.noise_p), ("rir_p", self.rir_p)):
            if not 0.0 <= p <= 1.0:
                raise DataError(f"fact_aug {name} must be in [0, 1], got {p}")
        if self.gain_min_db > self.gain_max_db:
            raise DataError(f"gain_min_db {self.gain_min_db} > gain_max_db {self.gain_max_db}")
        if self.noise_snr_min_db > self.noise_snr_max_db:
            raise DataError(
                f"noise_snr_min_db {self.noise_snr_min_db} > noise_snr_max_db {self.noise_snr_max_db}"
            )
        for sp in splits:
            low = sp.lower()
            if low.startswith("dev") or low.startswith("test"):
                raise DataError(f"TRAIN-only Factor E augmentation refused on eval split {sp!r}")
        if self.noise_p > 0.0 and self.noise_dir:
            list_train_only_audio(self.noise_dir, "noise_dir")
        if self.rir_p > 0.0 and self.rir_dir:
            list_train_only_audio(self.rir_dir, "rir_dir")


def _check_not_eval_path(path: str, label: str, root: str | None = None) -> None:
    rel = os.path.basename(os.path.abspath(path)) if root is None else os.path.relpath(path, root)
    norm = rel.replace("\\", "/").lower()
    parts = [p for p in norm.split("/") if p and p != "."]
    for tok in _FORBIDDEN_EVAL_TOKENS:
        if tok in norm:
            raise DataError(f"{label} {path!r} contains forbidden eval token {tok!r} (TRAIN-only)")
    for part in parts:
        stem = os.path.splitext(part)[0]
        if stem in ("dev", "test", "eval") or stem.startswith(("dev-", "test-", "dev_", "test_")):
            raise DataError(f"{label} {path!r} contains forbidden eval path segment {part!r}")


@lru_cache(maxsize=16)
def list_train_only_audio(directory: str, label: str = "audio_dir") -> tuple[str, ...]:
    """Return sorted audio file paths under `directory`, failing closed on eval/missing dirs."""
    if not directory or not os.path.isdir(directory):
        raise DataError(f"{label} {directory!r} does not exist or is not a directory")
    _check_not_eval_path(directory, label)
    found: list[str] = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fn in sorted(files):
            if fn.lower().endswith(_AUDIO_EXTS):
                fp = os.path.join(root, fn)
                _check_not_eval_path(fp, label, root=directory)
                found.append(fp)
    if not found:
        raise DataError(f"{label} {directory!r} contains no audio files ({_AUDIO_EXTS})")
    return tuple(found)


@lru_cache(maxsize=512)
def _load_mono_16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000 and len(y) > 0:
        from fractions import Fraction
        from scipy.signal import resample_poly

        r = Fraction(16000, int(sr)).limit_denominator(256)
        y = resample_poly(y, r.numerator, r.denominator).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


def _rng_for(tag: str, seed: int, epoch: int, uid: str) -> np.random.Generator:
    h = hashlib.sha256(f"{tag}:{seed}:{epoch}:{uid}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _causal_rir(y: np.ndarray, cfg: FactAugConfig, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve

    if cfg.rir_dir:
        paths = list_train_only_audio(cfg.rir_dir, "rir_dir")
        ir = _load_mono_16k(paths[int(rng.integers(0, len(paths)))])
        if len(ir) > 0:
            peak = int(np.argmax(np.abs(ir)))
            ir = ir[peak:].copy()  # direct path at tap 0 preserves strict causal onset alignment
            if len(ir) > 8000:
                ir = ir[:8000]
        else:
            ir = np.array([1.0], dtype=np.float32)
    else:
        rt60 = float(rng.uniform(0.10, 0.45))
        n_rir = int(rng.integers(512, 1921))
        t = np.arange(n_rir, dtype=np.float32) / 16000.0
        ir = rng.standard_normal(n_rir).astype(np.float32) * np.exp(-6.9077553 * t / rt60) * 0.25
        ir[0] = 1.0
    norm = float(np.sqrt(np.sum(ir * ir)) + 1e-12)
    ir = ir / norm
    return fftconvolve(y, ir, mode="full")[: len(y)].astype(np.float32)


def _noise_segment(
    n_samples: int,
    cfg: FactAugConfig,
    rng: np.random.Generator,
    donor_pcm: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.noise_dir:
        paths = list_train_only_audio(cfg.noise_dir, "noise_dir")
        clip = _load_mono_16k(paths[int(rng.integers(0, len(paths)))])
        if len(clip) > 0:
            if len(clip) >= n_samples:
                off = int(rng.integers(0, len(clip) - n_samples + 1))
                return clip[off : off + n_samples]
            reps = (n_samples + len(clip) - 1) // len(clip) + 1
            tiled = np.tile(clip, reps)
            off = int(rng.integers(0, len(clip)))
            return tiled[off : off + n_samples]
    if donor_pcm is not None and len(donor_pcm) >= 64:
        clip = donor_pcm.astype(np.float32) / 32768.0
        if len(clip) >= n_samples:
            off = int(rng.integers(0, len(clip) - n_samples + 1))
            return clip[off : off + n_samples]
        reps = (n_samples + len(clip) - 1) // len(clip)
        return np.tile(clip, reps)[:n_samples]
    # Deterministic pink/white fallback when neither noise_dir nor TRAIN donor is supplied.
    n = rng.standard_normal(n_samples).astype(np.float32)
    if int(rng.integers(0, 2)) == 0:
        spec = np.fft.rfft(n)
        freqs = np.arange(1, len(spec) + 1, dtype=np.float32)
        spec /= np.sqrt(freqs)
        n = np.fft.irfft(spec, n=n_samples).astype(np.float32)
    return n


def apply_fact_aug(
    pcm: np.ndarray,
    uid: str,
    epoch: int,
    seed: int,
    cfg: FactAugConfig | None,
    donor_pcm: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Factor E TRAIN-only causal RIR + noise + random input gain (exact length preserved).

    When `cfg is None` or `not cfg.active`, returns `pcm` untouched with zero RNG draws.
    """
    if cfg is None or not cfg.active or len(pcm) < 64:
        return pcm
    rng = _rng_for("fact_e", seed, epoch, uid)
    do_rir = cfg.rir_p > 0.0 and float(rng.random()) < cfg.rir_p
    do_noise = cfg.noise_p > 0.0 and float(rng.random()) < cfg.noise_p
    do_gain = cfg.gain_p > 0.0 and float(rng.random()) < cfg.gain_p
    if not (do_rir or do_noise or do_gain):
        return pcm
    y = pcm.astype(np.float32) / 32768.0
    if do_rir:
        y = _causal_rir(y, cfg, rng)
    if do_noise:
        snr_db = float(rng.uniform(cfg.noise_snr_min_db, cfg.noise_snr_max_db))
        sig_rms = float(np.sqrt(np.mean(y * y)) + 1e-8)
        n = _noise_segment(len(pcm), cfg, rng, donor_pcm=donor_pcm)
        n_rms = float(np.sqrt(np.mean(n * n)) + 1e-8)
        y = y + n * (sig_rms / n_rms * (10.0 ** (-snr_db / 20.0)))
    if do_gain:
        gain_db = float(rng.uniform(cfg.gain_min_db, cfg.gain_max_db))
        y = y * (10.0 ** (gain_db / 20.0))
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
