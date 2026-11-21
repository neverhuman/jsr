"""Ternary quantisation-aware training (R4 pilot; spec v1.0-B §26.7 step 5, master spec T7).

Weights of the selected Linear layers are trained through a ternary projection
    s_n = mean_k |w[n, k]|                (per output channel, "absmean")
    t   = clamp(round(w / s_n), -1, 1)    in {-1, 0, +1}
    w'  = t * s_n
with a straight-through estimator: the forward pass uses w', the gradient updates the latent
fp32 w as if the projection were the identity.

The inputs of the same layers are fake-quantised per row with the centred symmetric int8 scheme
of `crates/ultra-engine/src/quant.rs`, so training sees the arithmetic the CPU engine runs:
    c = (max + min) / 2,  h = (max - min) / 2,  q = round_half_even((x - c) * 127 / h)
    clamped to [-127, 127],   x' = q * h / 127 + c      (a constant row stays exact)
also with a straight-through estimator. Everything is computed in fp32, also under autocast.

Selected layers: every nn.Linear of the encoder except the stem's output projection, and the
joiner's encoder projection and output layer. Stem convolutions, norms, embeddings, the
predictor projection and the training-only CTC head stay fp32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

QMAX = 127.0
EPS = 1e-8


def ternary_project(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(trits in {-1, 0, 1} as float, per-output-channel scale (n, 1)) of a (n, k) weight."""
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=EPS)
    return torch.clamp(torch.round(w / s), -1.0, 1.0), s


def ternary_weight_ste(w: torch.Tensor) -> torch.Tensor:
    t, s = ternary_project(w.float())
    return w + (t * s - w).detach()


def fake_quant_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-row (last axis) centred symmetric int8 round trip, with the engine's exact float32
    arithmetic (ultra-engine kernels::scalar::row_params / quantize_row):
        half = 0.5*hi - 0.5*lo,  c = 0.5*hi + 0.5*lo,  inv = 127 / half,
        q = clamp(round_half_even((x - c) * inv), -127, 127),  x' = q * (half / 127) + c.
    A row whose half range is 0, or whose inverse step is not finite, is treated as constant."""
    xf = x.float()
    hi = xf.amax(dim=-1, keepdim=True)
    lo = xf.amin(dim=-1, keepdim=True)
    half = 0.5 * hi - 0.5 * lo
    c = 0.5 * hi + 0.5 * lo
    inv = QMAX / half
    ok = (half > 0) & torch.isfinite(inv)
    q = torch.clamp(torch.round((xf - c) * torch.where(ok, inv, torch.zeros_like(inv))),
                    -QMAX, QMAX)
    xq = torch.where(ok, q * (half / QMAX) + c, c)
    return xf + (xq - xf).detach()


class TernaryLinear(nn.Linear):
    """nn.Linear with ternary STE weights and per-row int8 STE inputs. Same parameters and
    state-dict keys as nn.Linear, so fp32 checkpoints load unchanged (warm start)."""

    def effective_weight(self) -> torch.Tensor:
        return ternary_weight_ste(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            b = None if self.bias is None else self.bias.float()
            return F.linear(fake_quant_rows(x), self.effective_weight().float(), b)


def ternary_targets(model: nn.Module) -> list[str]:
    """Names of the Linear modules the ternary pilot quantises."""
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.startswith("encoder.") and not name.startswith("encoder.stem."):
            out.append(name)
        elif name in ("joiner.enc_proj", "joiner.out"):
            out.append(name)
    return out


def apply_ternary(model: nn.Module) -> list[str]:
    """Turn the target Linears into TernaryLinear in place; returns their names."""
    names = ternary_targets(model)
    mods = dict(model.named_modules())
    for n in names:
        mods[n].__class__ = TernaryLinear
    return names


def int8_weight_ste(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 weight round trip (ultra-engine quant.rs:
    s_w[n] = max_k |w[n][k]| / 127, w_q = round(w / s_w)), straight-through."""
    wf = w.float()
    flat = wf.reshape(wf.shape[0], -1)
    s = (flat.abs().amax(dim=1, keepdim=True) / QMAX).clamp(min=EPS)
    wq = (torch.clamp(torch.round(flat / s), -QMAX, QMAX) * s).reshape(wf.shape)
    return w + (wq - w).detach()


class Int8Linear(nn.Linear):
    """nn.Linear trained as the engine's int8 GEMM: int8 per-channel weights, per-row centred
    int8 inputs (both straight-through). Same state-dict keys as nn.Linear."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            b = None if self.bias is None else self.bias.float()
            return F.linear(fake_quant_rows(x), int8_weight_ste(self.weight).float(), b)


def int8_conv2d_as_gemm(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """A 2-D convolution computed the way the engine runs stem layer 2: im2col rows (one per
    output position, K = c_in * kh * kw), each row quantised per-row centred int8, times int8
    per-output-channel weights. Order of K inside a row does not change its quantisation."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        b, _, h, w = xf.shape
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        cols = F.unfold(xf, (kh, kw), stride=(sh, sw))  # (B, C*kh*kw, L)
        rows = fake_quant_rows(cols.transpose(1, 2))  # (B, L, K)
        wq = int8_weight_ste(conv.weight).float().reshape(conv.out_channels, -1)
        y = rows @ wq.t() + conv.bias.float()
        ho, wo = (h - kh) // sh + 1, (w - kw) // sw + 1
        return y.transpose(1, 2).reshape(b, conv.out_channels, ho, wo)


def apply_int8_stem(model: nn.Module) -> list[str]:
    """Train the stem's second convolution and its output projection as the engine's int8 path
    (layer 1, a 1-channel 3x3 conv, stays fp32 as in the engine)."""
    stem = model.encoder.stem
    stem.int8_qat = True
    stem.out.__class__ = Int8Linear
    return ["encoder.stem.c2", "encoder.stem.out"]
