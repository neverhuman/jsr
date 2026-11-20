"""Causal chunked streaming encoder (DENSE family, spec v1.0-B §17.2-17.4, §17.6).

Layout
  log-mel (100 Hz, 80) -> causal Conv2d subsampling x4 -> 25 Hz, width d
  -> n_blocks Conformer blocks (macaron FFN, chunk-masked RoPE attention with bounded left context,
     causal depthwise convolution with LayerNorm, FFN, LayerNorm)
  -> dense 25 Hz encoder states (CTC head and transducer joiner read them).

Causality contract (tested in tests/test_train_model.py):
  * Encoder frame t depends on log-mel frames <= 4 t + 3 through the stem, at most.
  * Attention lets frame t (chunk c = t // C) see frames in [c C - L, (c + 1) C); no other future.
  * The depthwise convolution sees only past and current frames.
  Hence perturbing log-mel frames >= 4 (c + 1) C leaves every encoder frame < (c + 1) C unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from train.errors import ShapeError


@dataclass(frozen=True)
class EncoderConfig:
    n_mels: int = 80
    d_model: int = 384
    n_blocks: int = 16
    n_heads: int = 6
    ffn_mult: int = 4
    conv_kernel: int = 15
    stem_channels: int = 128
    chunk: int = 4  # 25 Hz frames per chunk: 4 = 160 ms
    left_context: int = 128  # 25 Hz frames of attention history: 128 = 5.12 s
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def stem_out_len(n: torch.Tensor | int):
    """Frames after two stride-2 causal convolutions: ceil(ceil(n / 2) / 2)."""
    if isinstance(n, torch.Tensor):
        return torch.div(torch.div(n + 1, 2, rounding_mode="floor") + 1, 2, rounding_mode="floor")
    return ((n + 1) // 2 + 1) // 2


class CausalStem(nn.Module):
    """Two 3x3 stride-2 Conv2d layers, padded only on the past side of the time axis."""

    def __init__(self, n_mels: int, channels: int, d_model: int):
        super().__init__()
        self.c1 = nn.Conv2d(1, channels, 3, stride=2)
        self.c2 = nn.Conv2d(channels, channels, 3, stride=2)
        f = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.out = nn.Linear(channels * f, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Pad 2 frames of past before each conv so output s sees inputs <= 2 s.
        x = x.unsqueeze(1)
        x = F.silu(self.c1(F.pad(x, (0, 0, 2, 0))))
        x = F.silu(self.c2(F.pad(x, (0, 0, 2, 0))))
        b, c, t, f = x.shape
        return self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))


def chunk_mask(t: int, chunk: int, left: int, device=None) -> torch.Tensor:
    """Boolean (t, t) mask, True where query i may attend key j."""
    i = torch.arange(t, device=device)[:, None]
    j = torch.arange(t, device=device)[None, :]
    c_start = (i // chunk) * chunk
    return (j < c_start + chunk) & (j >= c_start - left)


def rope_tables(t: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    ang = torch.arange(t, device=device, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, p: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d * mult)
        self.w2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.w2(self.drop(F.silu(self.w1(self.norm(x))))))


class ChunkAttention(nn.Module):
    def __init__(self, d: int, heads: int, p: float):
        super().__init__()
        if d % heads:
            raise ShapeError(f"d_model {d} not divisible by heads {heads}")
        self.h, self.dh = heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.p = p
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))


class CausalConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, p: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.k = kernel
        self.dnorm = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x, pad):
        y = F.glu(self.pw1(self.norm(x)), dim=-1)
        y = y.masked_fill(pad[..., None], 0.0)
        y = self.dw(F.pad(y.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        return self.drop(self.pw2(F.silu(self.dnorm(y))))


class ConformerBlock(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.ff1 = FeedForward(c.d_model, c.ffn_mult, c.dropout)
        self.att = ChunkAttention(c.d_model, c.n_heads, c.dropout)
        self.conv = CausalConvModule(c.d_model, c.conv_kernel, c.dropout)
        self.ff2 = FeedForward(c.d_model, c.ffn_mult, c.dropout)
        self.norm = nn.LayerNorm(c.d_model)

    def forward(self, x, mask, pad, cos, sin):
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, mask, cos, sin)
        x = x + self.conv(x, pad)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.cfg = c
        self.stem = CausalStem(c.n_mels, c.stem_channels, c.d_model)
        self.blocks = nn.ModuleList(ConformerBlock(c) for _ in range(c.n_blocks))

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: int | None = None):
        """feats (B, T, 80) normalised log-mel; returns (states (B, T', d), lens (B,))."""
        x = self.stem(feats)
        lens = stem_out_len(feat_lens)
        b, t, _ = x.shape
        pad = torch.arange(t, device=x.device)[None] >= lens[:, None]
        mask = chunk_mask(t, chunk or self.cfg.chunk, self.cfg.left_context, x.device)
        # Padded keys are excluded; a padded query keeps its own key so softmax stays finite.
        mask = (mask[None] & ~pad[:, None, :]) | torch.eye(t, dtype=torch.bool, device=x.device)[None]
        mask = mask[:, None]  # (B, 1, T, T)
        cos, sin = rope_tables(t, self.cfg.d_model // self.cfg.n_heads, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, mask, pad, cos, sin)
        return x, lens
