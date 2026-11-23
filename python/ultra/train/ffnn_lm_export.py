"""Exporter, loader, and numerical parity verifier for `ultra.lm-ffnn/v1`.

Matches `crates/ultra-engine/src/lm/ffnn.rs` byte-for-byte: 32-byte little-endian header
(`ULTRFF01`, version 1, vocab, ctx, d_e, d_h, flags), per-row symmetric INT8 weights (`|q| <= 127`)
with `f32` per-row scales/biases, and a 32-byte trailing SHA-256 digest over `header || payload`.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from train.errors import DataError, ShapeError

FFNN_MAGIC = b"ULTRFF01"
FFNN_VERSION = 1
FLAG_TIED = 1 << 0
FLAG_SILU = 1 << 1
FLAG_KNOWN_MASK = FLAG_TIED | FLAG_SILU


@dataclass(frozen=True)
class FfnnLmConfig:
    vocab: int = 512
    ctx: int = 6
    d_e: int = 128
    d_h: int = 512
    activation: str = "silu"
    tied: bool = False

    def validate(self) -> None:
        if not (1 <= self.vocab < 65535) or not (1 <= self.ctx <= 16):
            raise ShapeError(f"invalid vocab={self.vocab} or ctx={self.ctx}")
        if self.d_e <= 0 or self.d_h <= 0 or self.activation not in ("relu", "silu"):
            raise ShapeError(f"invalid d_e={self.d_e}, d_h={self.d_h}, act={self.activation!r}")
        if self.tied and self.d_h != self.d_e:
            raise ShapeError(f"tied FFNN requires d_h == d_e (got {self.d_h} != {self.d_e})")

    def weight_bytes(self) -> int:
        w2 = 0 if self.tied else self.vocab * (self.d_h + 4)
        return self.vocab * (self.d_e + 8) + self.d_h * (self.ctx * self.d_e + 8) + w2

    def file_bytes(self) -> int:
        return self.weight_bytes() + 64


PRESETS: dict[str, FfnnLmConfig] = {
    "0.5m": FfnnLmConfig(vocab=512, ctx=6, d_e=96, d_h=384, activation="relu", tied=False),
    "l2": FfnnLmConfig(vocab=512, ctx=6, d_e=128, d_h=512, activation="silu", tied=False),
    "1.4m": FfnnLmConfig(vocab=512, ctx=6, d_e=160, d_h=896, activation="silu", tied=False),
    "2.0m": FfnnLmConfig(vocab=512, ctx=6, d_e=192, d_h=1120, activation="silu", tied=False),
}


def preset_config(name: str, vocab: int | None = None) -> FfnnLmConfig:
    if name not in PRESETS:
        raise ShapeError(f"unknown FFNN preset {name!r}; expected {sorted(PRESETS)}")
    cfg = PRESETS[name]
    if vocab is not None:
        cfg = FfnnLmConfig(vocab, cfg.ctx, cfg.d_e, cfg.d_h, cfg.activation, cfg.tied)
    cfg.validate()
    return cfg


def _to_f32_np(x: Any) -> np.ndarray:
    arr = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise DataError("FFNN LM tensor contains non-finite values")
    return arr


def quantize_rows_i8(weight: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization (`|q| <= 127`, matches `WeightRule::Engine`)."""
    w = _to_f32_np(weight)
    if w.ndim != 2:
        raise ShapeError(f"expected 2D weight matrix, got {w.shape}")
    mx = np.max(np.abs(w), axis=1)
    scale = (mx / np.float32(127.0)).astype(np.float32)
    safe_mx = np.where(mx > 0.0, mx, np.float32(1.0))
    inv = np.where(mx > 0.0, np.float32(127.0) / safe_mx, np.float32(0.0)).astype(np.float32)
    return np.clip(np.rint(w * inv[:, None]), -127.0, 127.0).astype(np.int8), scale


def pack_ffnn_v1(
    cfg: FfnnLmConfig, emb_q: np.ndarray, emb_scale: np.ndarray,
    w1_q: np.ndarray, w1_scale: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    w2_q: np.ndarray | None = None, w2_scale: np.ndarray | None = None,
) -> bytes:
    """Serialize pre-quantized arrays to SHA-256 verified `ultra.lm-ffnn/v1` bytes."""
    cfg.validate()
    flags = (FLAG_TIED if cfg.tied else 0) | (FLAG_SILU if cfg.activation == "silu" else 0)
    parts = [
        FFNN_MAGIC,
        struct.pack("<6I", FFNN_VERSION, cfg.vocab, cfg.ctx, cfg.d_e, cfg.d_h, flags),
        np.ascontiguousarray(emb_q, dtype=np.int8).tobytes(),
        np.ascontiguousarray(emb_scale, dtype="<f4").tobytes(),
        np.ascontiguousarray(w1_q, dtype=np.int8).tobytes(),
        np.ascontiguousarray(w1_scale, dtype="<f4").tobytes(),
        np.ascontiguousarray(b1, dtype="<f4").tobytes(),
    ]
    if not cfg.tied:
        if w2_q is None or w2_scale is None:
            raise ShapeError("untied FFNN requires w2_q and w2_scale")
        parts.extend([
            np.ascontiguousarray(w2_q, dtype=np.int8).tobytes(),
            np.ascontiguousarray(w2_scale, dtype="<f4").tobytes(),
        ])
    parts.append(np.ascontiguousarray(b2, dtype="<f4").tobytes())
    payload = b"".join(parts)
    return payload + hashlib.sha256(payload).digest()


def export_ffnn_tensors(
    emb_w: Any, w1_w: Any, b1: Any, b2: Any, w2_w: Any | None = None,
    *, ctx: int = 6, activation: str = "silu", path: str | Path | None = None,
) -> bytes:
    """Quantize raw FP32 weight tensors/arrays and serialize to `ultra.lm-ffnn/v1`."""
    emb_np, w1_np, b1_np, b2_np = map(_to_f32_np, (emb_w, w1_w, b1, b2))
    if emb_np.ndim != 2 or w1_np.ndim != 2:
        raise ShapeError(f"emb_w {emb_np.shape} and w1_w {w1_np.shape} must be 2D")
    vocab, d_e = emb_np.shape
    d_h, w1_in = w1_np.shape
    if w1_in != ctx * d_e or b1_np.shape != (d_h,) or b2_np.shape != (vocab,):
        raise ShapeError(f"shape mismatch w1={w1_np.shape} b1={b1_np.shape} b2={b2_np.shape}")
    cfg = FfnnLmConfig(vocab, ctx, d_e, d_h, activation, tied=w2_w is None)
    cfg.validate()
    emb_q, emb_scale = quantize_rows_i8(emb_np)
    w1_q, w1_scale = quantize_rows_i8(w1_np)
    if cfg.tied:
        raw = pack_ffnn_v1(cfg, emb_q, emb_scale, w1_q, w1_scale, b1_np, b2_np)
    else:
        w2_np = _to_f32_np(w2_w)
        if w2_np.shape != (vocab, d_h):
            raise ShapeError(f"w2_w shape {w2_np.shape} != ({vocab}, {d_h})")
        w2_q, w2_scale = quantize_rows_i8(w2_np)
        raw = pack_ffnn_v1(cfg, emb_q, emb_scale, w1_q, w1_scale, b1_np, b2_np, w2_q, w2_scale)
    if path is not None:
        Path(path).write_bytes(raw)
    return raw


def export_ffnn_v1(
    model: Any, path: str | Path | None = None,
    *, ctx: int | None = None, activation: str | None = None,
) -> bytes:
    """Export a PyTorch FFNN LM module with `.emb`, `.fc1`, and `.fc2` (or tied `.out_bias`)."""
    cfg = getattr(model, "cfg", None)
    eff_ctx = ctx if ctx is not None else int(getattr(cfg, "ctx", getattr(model, "ctx", 6)))
    eff_act = activation if activation is not None else str(getattr(cfg, "activation", "silu"))
    fc2 = getattr(model, "fc2", None)
    b2 = getattr(model, "out_bias", None) if fc2 is None else fc2.bias
    if b2 is None:
        raise ShapeError("FFNN module must expose `fc2.bias` or tied `out_bias`")
    return export_ffnn_tensors(
        model.emb.weight, model.fc1.weight, model.fc1.bias, b2, None if fc2 is None else fc2.weight,
        ctx=eff_ctx, activation=eff_act, path=path,
    )


@dataclass(frozen=True)
class QuantizedFfnnLm:
    """Python reference runtime matching `crates/ultra-engine/src/lm/ffnn.rs`."""

    cfg: FfnnLmConfig
    emb_q: np.ndarray
    emb_scale: np.ndarray
    w1_q: np.ndarray
    w1_scale: np.ndarray
    b1: np.ndarray
    w2_q: np.ndarray | None
    w2_scale: np.ndarray | None
    b2: np.ndarray

    def score_all(self, history: list[int] | tuple[int, ...]) -> np.ndarray:
        c = self.cfg
        take = min(len(history), c.ctx)
        tail = [t if 0 <= t < c.vocab else 0 for t in history[-take:]] if take else []
        slots = [0] * (c.ctx - take) + tail
        w1 = self.w1_q.reshape(c.d_h, c.ctx, c.d_e)
        acc = np.zeros(c.d_h, dtype=np.float32)
        for s, tok in enumerate(slots):
            dots = (w1[:, s, :].astype(np.int32) * self.emb_q[tok].astype(np.int32)[None, :]).sum(axis=1)
            acc = (acc + dots.astype(np.float32) * self.emb_scale[tok]).astype(np.float32)
        pre = (acc * self.w1_scale + self.b1).astype(np.float32)
        h = (pre / (np.float32(1.0) + np.exp(-pre))).astype(np.float32) if c.activation == "silu" else np.maximum(pre, np.float32(0.0))
        out_q, out_s = (self.emb_q, self.emb_scale) if c.tied else (self.w2_q, self.w2_scale)
        assert out_q is not None and out_s is not None
        logits = ((out_q.astype(np.float32) @ h) * out_s + self.b2).astype(np.float32)
        mx = np.max(logits)
        lse = mx + np.log(np.sum(np.exp(logits - mx), dtype=np.float32), dtype=np.float32)
        return (logits - lse).astype(np.float32)

    def score_next(self, history: list[int] | tuple[int, ...], token: int) -> float:
        return float(self.score_all(history)[token]) if 0 <= token < self.cfg.vocab else float("-inf")


def load_ffnn_v1(raw_or_path: bytes | str | Path) -> QuantizedFfnnLm:
    raw = raw_or_path if isinstance(raw_or_path, (bytes, bytearray)) else Path(raw_or_path).read_bytes()
    if len(raw) < 64 or hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise DataError("FFNN LM truncated or SHA-256 verification failed")
    payload = raw[:-32]
    if payload[:8] != FFNN_MAGIC:
        raise DataError("bad FFNN LM magic header")
    ver, vocab, ctx, d_e, d_h, flags = struct.unpack_from("<6I", payload, 8)
    if ver != FFNN_VERSION or (flags & ~FLAG_KNOWN_MASK):
        raise DataError(f"unsupported FFNN LM version={ver} flags=0x{flags:x}")
    cfg = FfnnLmConfig(vocab, ctx, d_e, d_h, "silu" if (flags & FLAG_SILU) else "relu", bool(flags & FLAG_TIED))
    cfg.validate()
    pos = 32

    def take_i8(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal pos
        n = int(np.prod(shape))
        if pos + n > len(payload):
            raise DataError("unexpected EOF in FFNN LM i8 payload")
        arr = np.frombuffer(payload, dtype=np.int8, count=n, offset=pos).copy().reshape(shape)
        pos += n
        if (arr == -128).any():
            raise DataError("FFNN LM contains disallowed INT8 code -128")
        return arr

    def take_f32(n: int) -> np.ndarray:
        nonlocal pos
        if pos + n * 4 > len(payload):
            raise DataError("unexpected EOF in FFNN LM f32 payload")
        arr = np.frombuffer(payload, dtype="<f4", count=n, offset=pos).copy()
        pos += n * 4
        if not np.isfinite(arr).all():
            raise DataError("FFNN LM contains non-finite scale/bias")
        return arr

    emb_q, emb_scale = take_i8((vocab, d_e)), take_f32(vocab)
    w1_q, w1_scale, b1 = take_i8((d_h, ctx * d_e)), take_f32(d_h), take_f32(d_h)
    w2_q, w2_scale = (None, None) if cfg.tied else (take_i8((vocab, d_h)), take_f32(vocab))
    b2 = take_f32(vocab)
    if pos != len(payload):
        raise DataError(f"trailing {len(payload) - pos} bytes in FFNN LM")
    return QuantizedFfnnLm(cfg, emb_q, emb_scale, w1_q, w1_scale, b1, w2_q, w2_scale, b2)


def verify_parity(
    fp32_scorer: Callable[[list[int]], np.ndarray],
    raw: bytes,
    histories: list[list[int]],
    max_abs_tol: float = 0.25,
) -> dict[str, float]:
    """Compare FP32 `fp32_scorer(history) -> log_probs[V]` against exported INT8 `ultra.lm-ffnn/v1`."""
    qlm = load_ffnn_v1(raw)
    diffs, top1_hits = [], 0
    for hist in histories:
        fp32, q = np.asarray(fp32_scorer(hist), dtype=np.float32), qlm.score_all(hist)
        diffs.append(np.abs(fp32 - q))
        top1_hits += int(int(np.argmax(fp32)) == int(np.argmax(q)))
    all_d = np.concatenate(diffs) if diffs else np.zeros(1, dtype=np.float32)
    max_abs = float(np.max(all_d))
    if max_abs > max_abs_tol:
        raise DataError(f"FFNN INT8 parity max_abs_diff={max_abs:.4f} exceeds tol={max_abs_tol}")
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean(all_d)),
        "top1_agreement": top1_hits / max(1, len(histories)),
        "size_bytes": float(len(raw)),
    }
