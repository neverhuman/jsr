"""Exporter, loader, and NumPy reference verifier for `ultra.tf-rescore/v1` (`.utfr`).

Packs any causal L2R/R2L `rescore_tf` PyTorch checkpoint (`best.pt`) into a compact,
SHA-256 verified binary file with FP32 embeddings/RMSNorm weights and per-row symmetric
INT8 (`|q| <= 127`) linear weights + FP32 scales. Torch-free at module import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TF_MAGIC = b"ULTRTF01"
TF_VERSION = 1
FLAG_TIED = 1 << 0
FLAG_KNOWN_MASK = FLAG_TIED
DIR_TO_CODE = {"l2r": 0, "r2l": 1}
CODE_TO_DIR = {0: "l2r", 1: "r2l"}
HEADER_FMT = "<11I"
HEADER_BYTES = 8 + struct.calcsize(HEADER_FMT)


@dataclass(frozen=True)
class TfRescoreConfig:
    vocab: int = 501
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    ctx: int = 128
    direction: str = "l2r"
    tied: bool = True
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6

    def validate(self) -> None:
        if not (2 <= self.vocab <= 65535) or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError(f"invalid vocab/d_model/n_layers: {self}")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.d_ff <= 0 or self.ctx < 2 or self.direction not in DIR_TO_CODE:
            raise ValueError(f"invalid d_ff/ctx/direction: {self}")
        if not (math.isfinite(self.rope_theta) and self.rope_theta > 0.0):
            raise ValueError(f"invalid rope_theta={self.rope_theta}")
        if not (math.isfinite(self.rms_eps) and self.rms_eps > 0.0):
            raise ValueError(f"invalid rms_eps={self.rms_eps}")

    def file_bytes(self) -> int:
        d, f, v = self.d_model, self.d_ff, self.vocab
        layer_bytes = 2 * d * 4 + (4 * d * d + 3 * f * d) + (4 * d + 2 * f + d) * 4
        head_bytes = 0 if self.tied else (v * d + v * 4)
        return HEADER_BYTES + v * d * 4 + self.n_layers * layer_bytes + d * 4 + head_bytes + 32


@dataclass(frozen=True)
class QuantizedTfLayer:
    n1_w: np.ndarray; qkv_q: np.ndarray; qkv_s: np.ndarray; wo_q: np.ndarray; wo_s: np.ndarray  # noqa: E702
    n2_w: np.ndarray; w12_q: np.ndarray; w12_s: np.ndarray; w3_q: np.ndarray; w3_s: np.ndarray  # noqa: E702


@dataclass(frozen=True)
class QuantizedTfRescorer:
    cfg: TfRescoreConfig
    emb: np.ndarray
    layers: tuple[QuantizedTfLayer, ...]
    norm_w: np.ndarray
    head_q: np.ndarray | None = None
    head_s: np.ndarray | None = None

    def score_sequence(self, tokens: list[int]) -> dict[str, float]:
        """Pure-NumPy dequantized forward pass matching `rescore_tf.score_sequences`."""
        c = self.cfg
        seq_toks = list(reversed(tokens)) if c.direction == "r2l" else list(tokens)
        seq = [0] + [t if 0 <= t < c.vocab else 0 for t in seq_toks[: c.ctx - 1]] + [0]
        x_ids, y_ids = seq[:-1], seq[1:]
        t_len, nh, hd = len(x_ids), c.n_heads, c.d_model // c.n_heads
        h = self.emb[x_ids].astype(np.float32, copy=True)
        inv_freq = 1.0 / (c.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        ang = np.arange(t_len, dtype=np.float32)[:, None] * inv_freq[None, :]
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        scale = np.float32(1.0 / math.sqrt(hd))
        causal = np.triu(np.full((t_len, t_len), -1e9, dtype=np.float32), k=1)

        def rms_norm(z: np.ndarray, w: np.ndarray) -> np.ndarray:
            inv = 1.0 / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + np.float32(c.rms_eps))
            return (z * inv) * w

        def i8_lin(z: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
            return (z @ q.astype(np.float32).T) * s[None, :]

        for ly in self.layers:
            qkv = i8_lin(rms_norm(h, ly.n1_w), ly.qkv_q, ly.qkv_s).reshape(t_len, 3, nh, hd)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            qe, qo = q[..., ::2], q[..., 1::2]
            ke, ko = k[..., ::2], k[..., 1::2]
            q_rot = np.empty_like(q)
            k_rot = np.empty_like(k)
            q_rot[..., ::2], q_rot[..., 1::2] = qe * cos - qo * sin, qe * sin + qo * cos
            k_rot[..., ::2], k_rot[..., 1::2] = ke * cos - ko * sin, ke * sin + ko * cos
            att_out = np.empty((t_len, nh, hd), dtype=np.float32)
            for hh in range(nh):
                sc = (q_rot[:, hh, :] @ k_rot[:, hh, :].T) * scale + causal
                sc -= np.max(sc, axis=-1, keepdims=True)
                p = np.exp(sc)
                p /= np.sum(p, axis=-1, keepdims=True)
                att_out[:, hh, :] = p @ v[:, hh, :]
            h = h + i8_lin(att_out.reshape(t_len, c.d_model), ly.wo_q, ly.wo_s)
            ff = i8_lin(rms_norm(h, ly.n2_w), ly.w12_q, ly.w12_s)
            g, u = ff[:, : c.d_ff], ff[:, c.d_ff :]
            h = h + i8_lin((g / (1.0 + np.exp(-g))) * u, ly.w3_q, ly.w3_s)

        hn = rms_norm(h, self.norm_w)
        if c.tied:
            logits = hn @ self.emb.T
        else:
            assert self.head_q is not None and self.head_s is not None
            logits = i8_lin(hn, self.head_q, self.head_s)
        mx = np.max(logits, axis=-1, keepdims=True)
        lse = mx + np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True))
        lp = logits[np.arange(t_len), y_ids] - lse[:, 0]
        tok_lp = float(np.sum(lp[:-1])) if t_len > 1 else 0.0
        eos_lp = float(lp[-1]) if t_len > 0 else 0.0
        return {"tok_logp": tok_lp, "eos_logp": eos_lp, "total_logp": tok_lp + eos_lp, "n_tok": max(0, t_len - 1)}


def _to_f32_np(x: Any) -> np.ndarray:
    arr = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("tensor contains non-finite values")
    return np.ascontiguousarray(arr, dtype="<f4")


def quantize_rows_i8(weight: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization (`|q| <= 127`, matches `WeightRule::Engine`)."""
    w = _to_f32_np(weight)
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight matrix, got {w.shape}")
    mx = np.max(np.abs(w), axis=1)
    scale = (mx / np.float32(127.0)).astype("<f4")
    safe_mx = np.where(mx > 0.0, mx, np.float32(1.0))
    inv = np.where(mx > 0.0, np.float32(127.0) / safe_mx, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(w * inv[:, None]), -127.0, 127.0).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)


def _f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def export_tf_tensors(cfg: TfRescoreConfig, state: dict[str, Any], path: str | Path | None = None) -> bytes:
    """Quantize a `rescore_tf` state dict (`emb.weight`, `blocks.i.*`, `norm.weight`) to `.utfr`."""
    cfg.validate()
    d, f, v = cfg.d_model, cfg.d_ff, cfg.vocab
    emb = _to_f32_np(state["emb.weight"])
    if emb.shape != (v, d):
        raise ValueError(f"emb.weight shape {emb.shape} != {(v, d)}")
    parts: list[bytes] = [
        TF_MAGIC,
        struct.pack(
            HEADER_FMT, TF_VERSION, v, d, cfg.n_layers, cfg.n_heads, f, cfg.ctx,
            DIR_TO_CODE[cfg.direction], FLAG_TIED if cfg.tied else 0,
            _f32_bits(cfg.rope_theta), _f32_bits(cfg.rms_eps),
        ),
        emb.tobytes(),
    ]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        n1, n2 = _to_f32_np(state[f"{p}n1.weight"]), _to_f32_np(state[f"{p}n2.weight"])
        qkv_q, qkv_s = quantize_rows_i8(state[f"{p}qkv.weight"])
        wo_q, wo_s = quantize_rows_i8(state[f"{p}wo.weight"])
        w12_q, w12_s = quantize_rows_i8(state[f"{p}w12.weight"])
        w3_q, w3_s = quantize_rows_i8(state[f"{p}w3.weight"])
        if n1.shape != (d,) or n2.shape != (d,) or qkv_q.shape != (3 * d, d) or wo_q.shape != (d, d) or w12_q.shape != (2 * f, d) or w3_q.shape != (d, f):
            raise ValueError(f"layer {i} shape mismatch")
        parts.extend([
            n1.tobytes(), qkv_q.tobytes(), qkv_s.tobytes(), wo_q.tobytes(), wo_s.tobytes(),
            n2.tobytes(), w12_q.tobytes(), w12_s.tobytes(), w3_q.tobytes(), w3_s.tobytes(),
        ])
    norm_w = _to_f32_np(state["norm.weight"])
    if norm_w.shape != (d,):
        raise ValueError(f"norm.weight shape {norm_w.shape} != {(d,)}")
    parts.append(norm_w.tobytes())
    if not cfg.tied:
        hq, hs = quantize_rows_i8(state["head.weight"])
        if hq.shape != (v, d):
            raise ValueError(f"head.weight shape {hq.shape} != {(v, d)}")
        parts.extend([hq.tobytes(), hs.tobytes()])
    payload = b"".join(parts)
    raw = payload + hashlib.sha256(payload).digest()
    assert len(raw) == cfg.file_bytes()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
    return raw


def load_tf_v1(raw_or_path: bytes | str | Path) -> QuantizedTfRescorer:
    raw = bytes(raw_or_path) if isinstance(raw_or_path, (bytes, bytearray)) else Path(raw_or_path).read_bytes()
    if len(raw) < HEADER_BYTES + 32 or hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise ValueError("UTFR truncated or SHA-256 verification failed")
    payload = raw[:-32]
    if payload[:8] != TF_MAGIC:
        raise ValueError("bad UTFR magic")
    ver, v, d, nl, nh, f, ctx, dcode, flags, theta_b, eps_b = struct.unpack_from(HEADER_FMT, payload, 8)
    if ver != TF_VERSION or (flags & ~FLAG_KNOWN_MASK) or dcode not in CODE_TO_DIR:
        raise ValueError(f"unsupported UTFR header ver={ver} dir={dcode} flags=0x{flags:x}")
    cfg = TfRescoreConfig(
        v, d, nl, nh, f, ctx, CODE_TO_DIR[dcode], bool(flags & FLAG_TIED), _bits_f32(theta_b), _bits_f32(eps_b),
    )
    cfg.validate()
    pos = HEADER_BYTES

    def take_f32(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal pos
        n = int(np.prod(shape))
        if pos + n * 4 > len(payload):
            raise ValueError("unexpected EOF in f32 tensor")
        arr = np.frombuffer(payload, dtype="<f4", count=n, offset=pos).copy().reshape(shape)
        pos += n * 4
        if not np.isfinite(arr).all():
            raise ValueError("non-finite f32 value in UTFR")
        return arr

    def take_i8(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal pos
        n = int(np.prod(shape))
        if pos + n > len(payload):
            raise ValueError("unexpected EOF in i8 tensor")
        arr = np.frombuffer(payload, dtype=np.int8, count=n, offset=pos).copy().reshape(shape)
        pos += n
        if (arr == -128).any():
            raise ValueError("disallowed INT8 code -128 in UTFR")
        return arr

    emb = take_f32((v, d))
    layers = tuple(
        QuantizedTfLayer(
            take_f32((d,)), take_i8((3 * d, d)), take_f32((3 * d,)), take_i8((d, d)), take_f32((d,)),
            take_f32((d,)), take_i8((2 * f, d)), take_f32((2 * f,)), take_i8((d, f)), take_f32((d,)),
        )
        for _ in range(nl)
    )
    norm_w = take_f32((d,))
    hq, hs = (None, None) if cfg.tied else (take_i8((v, d)), take_f32((v,)))
    if pos != len(payload):
        raise ValueError(f"trailing {len(payload) - pos} bytes in UTFR")
    return QuantizedTfRescorer(cfg, emb, layers, norm_w, hq, hs)


def export_tf_checkpoint(ckpt_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Load a `rescore_tf` PyTorch checkpoint (`best.pt`) and write a `.utfr` artifact."""
    import torch

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    c = ck["cfg"]
    vocab = int(sd["emb.weight"].shape[0])
    tied = "head.weight" not in sd
    cfg = TfRescoreConfig(
        vocab=vocab, d_model=int(c["d_model"]), n_layers=int(c["n_layers"]),
        n_heads=int(c["n_heads"]), d_ff=int(c["d_ff"]), ctx=int(c["ctx"]),
        direction=str(ck["direction"]), tied=tied,
    )
    raw = export_tf_tensors(cfg, sd, out_path)
    return {
        "out": str(out_path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "direction": cfg.direction, "vocab": cfg.vocab,
        "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff, "ctx": cfg.ctx, "tied": cfg.tied,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Path to rescore_tf best.pt checkpoint")
    ap.add_argument("--out", required=True, help="Destination .utfr path")
    args = ap.parse_args(argv)
    info = export_tf_checkpoint(args.ckpt, args.out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
