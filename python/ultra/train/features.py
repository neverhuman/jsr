"""80-dim log-mel features, 25 ms window / 10 ms hop at 16 kHz (float control frontend).

Pure torch (no torchaudio). Framing is causal: frame i covers samples [160 i, 160 i + 400), so a
frame never depends on audio after its own window. Normalisation uses fixed global statistics
(estimated once from training audio and stored with the checkpoint), never per-utterance
statistics, so the streaming path computes the same numbers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

SAMPLE_RATE = 16000
WIN = 400
HOP = 160
N_FFT = 512
N_MELS = 80
F_MIN = 20.0
F_MAX = 7600.0
LOG_FLOOR = 1e-6


def _hz_to_mel(f: float) -> float:
    return 1127.0 * math.log(1.0 + f / 700.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE,
                   fmin: float = F_MIN, fmax: float = F_MAX) -> torch.Tensor:
    """Triangular HTK-mel filterbank, shape (n_fft // 2 + 1, n_mels)."""
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_bins, dtype=torch.float64)
    mel = 1127.0 * torch.log1p(freqs / 700.0)
    edges = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2, dtype=torch.float64)
    lo, mid, hi = edges[:-2], edges[1:-1], edges[2:]
    up = (mel[:, None] - lo[None]) / (mid - lo)[None]
    down = (hi[None] - mel[:, None]) / (hi - mid)[None]
    return torch.clamp(torch.minimum(up, down), min=0.0).to(torch.float32)


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN, periodic=False), persistent=False)
        self.register_buffer("fbank", mel_filterbank(), persistent=False)

    def num_frames(self, n_samples: int) -> int:
        return 0 if n_samples < WIN else 1 + (n_samples - WIN) // HOP

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (n,) float in [-1, 1]. Returns (frames, 80) float32 log-mel."""
        n = self.num_frames(wav.shape[-1])
        if n == 0:
            return torch.zeros(0, N_MELS)
        frames = wav[: WIN + (n - 1) * HOP].unfold(0, WIN, HOP)  # (n, WIN)
        frames = frames - frames.mean(dim=1, keepdim=True)  # per-frame DC removal (causal)
        spec = torch.fft.rfft(frames * self.window, n=N_FFT).abs().pow(2)
        return torch.log(spec @ self.fbank + LOG_FLOOR)


def pcm_to_float(pcm: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


class GlobalCMVN(torch.nn.Module):
    """Fixed affine normalisation (x - mean) / std with statistics stored as buffers."""

    def __init__(self, dim: int = N_MELS):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("istd", torch.ones(dim))

    def set_stats(self, feats: list[torch.Tensor]) -> None:
        x = torch.cat(feats, 0).double()
        self.mean.copy_(x.mean(0).float())
        self.istd.copy_((1.0 / x.std(0).clamp(min=1e-3)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.istd
