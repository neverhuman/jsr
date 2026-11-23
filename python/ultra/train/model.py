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
    # True: LayerNorm closing every block (original Conformer, post-norm-like residual stream).
    # False: pre-norm modules only plus one final LayerNorm. D1 (2026-11-20) showed the 16-block
    # post-block-norm stack collapsing to near time-invariant output (CTC flat at 6.18/token).
    post_block_norm: bool = True
    # "dense": n_blocks at 25 Hz. "multirate" (spec v1.0-B §17.2-17.4): lower_blocks at 25 Hz,
    # a trained projection of each pair of 25 Hz positions to one 12.5 Hz position, middle_blocks
    # at 12.5 Hz (chunk/2, left_context/2), a trained restoration to two 25 Hz positions plus the
    # lower-block skip, then final_blocks at 25 Hz. "unet3" (JSR v2 Wave H28): 0-lookahead 3-scale
    # U-Net (25 -> 12.5 -> 6.25 -> 12.5 -> 25 Hz) with lower/middle/deep/up/final blocks.
    family: str = "dense"
    lower_blocks: int = 4
    middle_blocks: int = 10
    final_blocks: int = 2
    deep_blocks: int = 8  # unet3 6.25 Hz bottleneck blocks (ignored for dense/multirate)
    up_blocks: int = 2  # unet3 12.5 Hz upsample blocks (ignored for dense/multirate)
    kv_heads: int = 0  # 0 = standard MHA (kv_heads == n_heads); >0 = GQA (n_heads % kv_heads == 0)
    ffn_act: str = "silu"  # "silu" (v1 default) | "relu2" (v2 squared ReLU for sparse pqscan)
    # JSR v2 Stage-1A upgrades (all default-off for 100% v1 bit-exactness):
    stem_type: str = "conv2d"  # "conv2d" (v1 dense) | "dwsep" (5.2x cheaper causal DW-Sep 2D stem)
    qk_rmsnorm: bool = False  # Pre-RoPE per-head Q/K RMSNorm with learnable scale
    norm_affine: bool = True  # False removes redundant pre-norm LN.g <-> W scale freedom
    layer_scale_init: float = 0.0  # >0: residual LayerScale (folds into w2/out/pw2 at export)
    stage_left: tuple[int, ...] = ()  # optional (25Hz, 12.5Hz, 6.25Hz) left contexts; () = default
    inter_ctc_vocab: int = 0  # >0: Self-Conditioned InterCTC head+feedback after lower_blocks
    layer_drop: float = 0.0  # H57 Factor B stochastic depth max drop-path prob (0 = off, eval identity)

    def to_dict(self) -> dict:
        return asdict(self)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def activation_balancer_loss(x: torch.Tensor, min_rms: float = 0.25,
                             max_rms: float = 4.0) -> torch.Tensor:
    """Zipformer-style channel RMS/mean balancer (train-only, 0 inference ops)."""
    xf = x.float()
    rms = xf.pow(2).mean(dim=(0, 1)).add(1e-6).sqrt()
    mean = xf.mean(dim=(0, 1)).abs()
    hi = F.relu(rms - max_rms).pow(2).mean()
    lo = F.relu(min_rms - rms).pow(2).mean()
    m = F.relu(mean - 0.5).pow(2).mean()
    return hi + lo + 0.1 * m


def stem_out_len(n: torch.Tensor | int):
    """Frames after two stride-2 causal convolutions: ceil(ceil(n / 2) / 2)."""
    if isinstance(n, torch.Tensor):
        return torch.div(torch.div(n + 1, 2, rounding_mode="floor") + 1, 2, rounding_mode="floor")
    return ((n + 1) // 2 + 1) // 2


class CausalStem(nn.Module):
    """Two stride-2 causal Conv2d layers (`conv2d` dense or `dwsep` 5.2x cheaper depthwise-sep)."""

    def __init__(self, n_mels: int, channels: int, d_model: int, stem_type: str = "conv2d"):
        super().__init__()
        if stem_type not in ("conv2d", "dwsep"):
            raise ShapeError(f"unknown stem_type {stem_type}")
        self.stem_type = stem_type
        f = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        if stem_type == "conv2d":
            self.c1 = nn.Conv2d(1, channels, 3, stride=2)
            self.c2 = nn.Conv2d(channels, channels, 3, stride=2)
        else:
            mid = min(32, channels)
            self.c1 = nn.Conv2d(1, mid, 3, stride=2)
            self.c2_dw = nn.Conv2d(mid, mid, 3, stride=2, groups=mid)
            self.c2_pw = nn.Conv2d(mid, channels, 1)
        self.out = nn.Linear(channels * f, d_model)

    int8_qat = False  # set by train.qat.apply_int8_stem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). Pad 2 frames of past before each conv so output s sees inputs <= 2 s.
        x = x.unsqueeze(1)
        x = F.silu(self.c1(F.pad(x, (0, 0, 2, 0))))
        if self.stem_type == "dwsep":
            x = F.silu(self.c2_pw(self.c2_dw(F.pad(x, (0, 0, 2, 0)))))
        elif self.int8_qat:
            from train.qat import int8_conv2d_as_gemm

            x = F.silu(int8_conv2d_as_gemm(F.pad(x, (0, 0, 2, 0)), self.c2))
        else:
            x = F.silu(self.c2(F.pad(x, (0, 0, 2, 0))))
        b, c, t, f = x.shape
        return self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))


def chunk_mask(t: int, chunk: int, left: int, device=None) -> torch.Tensor:
    """Boolean (t, t) mask, True where query i may attend key j. chunk <= 0 means full utterance."""
    if chunk <= 0:
        return torch.ones((t, t), dtype=torch.bool, device=device)
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
    def __init__(self, d: int, mult: int, p: float, act: str = "silu", norm_affine: bool = True):
        super().__init__()
        if act not in ("silu", "relu2"):
            raise ShapeError(f"unknown ffn_act {act}")
        self.act = act
        self.norm = nn.LayerNorm(d, elementwise_affine=norm_affine)
        self.w1 = nn.Linear(d, d * mult)
        self.w2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.w1(self.norm(x))
        h = F.silu(h) if self.act == "silu" else F.relu(h).square()
        return self.drop(self.w2(self.drop(h)))


class ChunkAttention(nn.Module):
    def __init__(self, d: int, heads: int, p: float, kv_heads: int = 0,
                 qk_rmsnorm: bool = False, norm_affine: bool = True):
        super().__init__()
        if d % heads:
            raise ShapeError(f"d_model {d} not divisible by heads {heads}")
        kvh = heads if kv_heads <= 0 else kv_heads
        if heads % kvh:
            raise ShapeError(f"n_heads {heads} not divisible by kv_heads {kvh}")
        self.h, self.kvh, self.dh = heads, kvh, d // heads
        self.qk_rmsnorm = qk_rmsnorm
        self.norm = nn.LayerNorm(d, elementwise_affine=norm_affine)
        self.qkv = nn.Linear(d, (heads + 2 * kvh) * self.dh)
        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(1, heads, 1, self.dh))
            self.k_scale = nn.Parameter(torch.ones(1, kvh, 1, self.dh))
        self.out = nn.Linear(d, d)
        self.p = p
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, cos, sin):
        b, t, d = x.shape
        proj = self.qkv(self.norm(x))
        if self.kvh == self.h:
            q, k, v = proj.view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        else:
            q_dim, kv_dim = self.h * self.dh, self.kvh * self.dh
            q, k, v = proj.split((q_dim, kv_dim, kv_dim), dim=-1)
            q = q.view(b, t, self.h, self.dh).transpose(1, 2)
            k = k.view(b, t, self.kvh, self.dh).transpose(1, 2)
            v = v.view(b, t, self.kvh, self.dh).transpose(1, 2)
        if self.qk_rmsnorm:
            q = _rms_norm(q) * self.q_scale.to(q.dtype)
            k = _rms_norm(k) * self.k_scale.to(k.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kvh != self.h:
            rep = self.h // self.kvh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))


class CausalConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, p: float, norm_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=norm_affine)
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


def _drop_path(y: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (drop-path) with inverted scaling; training-only identity at p=0."""
    if p <= 0.0 or not training:
        return y
    keep = 1.0 - p
    mask = torch.empty((y.shape[0],) + (1,) * (y.ndim - 1), dtype=y.dtype, device=y.device).bernoulli_(keep)
    return y * (mask / keep)


class ConformerBlock(nn.Module):
    def __init__(self, c: EncoderConfig, drop_path: float = 0.0):
        super().__init__()
        self.ff1 = FeedForward(c.d_model, c.ffn_mult, c.dropout, c.ffn_act, c.norm_affine)
        self.att = ChunkAttention(c.d_model, c.n_heads, c.dropout, c.kv_heads,
                                  c.qk_rmsnorm, c.norm_affine)
        self.conv = CausalConvModule(c.d_model, c.conv_kernel, c.dropout, c.norm_affine)
        self.ff2 = FeedForward(c.d_model, c.ffn_mult, c.dropout, c.ffn_act, c.norm_affine)
        self.norm = nn.LayerNorm(c.d_model) if c.post_block_norm else nn.Identity()
        self.drop_path = float(drop_path)
        self.has_ls = c.layer_scale_init > 0.0
        if self.has_ls:
            v = c.layer_scale_init
            self.ls_ff1 = nn.Parameter(torch.full((c.d_model,), v))
            self.ls_att = nn.Parameter(torch.full((c.d_model,), v))
            self.ls_conv = nn.Parameter(torch.full((c.d_model,), v))
            self.ls_ff2 = nn.Parameter(torch.full((c.d_model,), v))

    def forward(self, x, mask, pad, cos, sin):
        dp = self.drop_path if self.training else 0.0
        if self.has_ls:
            x = x + _drop_path((0.5 * self.ls_ff1.to(x.dtype)) * self.ff1(x), dp, self.training)
            x = x + _drop_path(self.ls_att.to(x.dtype) * self.att(x, mask, cos, sin), dp, self.training)
            x = x + _drop_path(self.ls_conv.to(x.dtype) * self.conv(x, pad), dp, self.training)
            x = x + _drop_path((0.5 * self.ls_ff2.to(x.dtype)) * self.ff2(x), dp, self.training)
        else:
            x = x + _drop_path(0.5 * self.ff1(x), dp, self.training)
            x = x + _drop_path(self.att(x, mask, cos, sin), dp, self.training)
            x = x + _drop_path(self.conv(x, pad), dp, self.training)
            x = x + _drop_path(0.5 * self.ff2(x), dp, self.training)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        if not 0.0 <= c.layer_drop < 1.0:
            raise ShapeError(f"layer_drop must be in [0, 1), got {c.layer_drop}")
        self.cfg = c
        self.stem = CausalStem(c.n_mels, c.stem_channels, c.d_model, c.stem_type)
        if c.chunk < 0:
            raise ShapeError(f"chunk must be >= 0 (0 = full utterance), got {c.chunk}")

        def _make_blocks(n: int, offset: int, total: int) -> nn.ModuleList:
            return nn.ModuleList(
                ConformerBlock(c, drop_path=c.layer_drop * (offset + i + 1) / max(1, total))
                for i in range(n)
            )

        if c.family == "dense":
            self.blocks = _make_blocks(c.n_blocks, 0, c.n_blocks)
        elif c.family == "multirate":
            if (c.chunk > 0 and c.chunk % 2) or c.left_context % 2:
                raise ShapeError("multirate needs chunk == 0 or an even chunk, and even left context")
            tot = c.lower_blocks + c.middle_blocks + c.final_blocks
            self.blocks = _make_blocks(c.lower_blocks, 0, tot)
            self.pair = nn.Linear(2 * c.d_model, c.d_model)
            self.mid_blocks = _make_blocks(c.middle_blocks, c.lower_blocks, tot)
            self.restore = nn.Linear(c.d_model, 2 * c.d_model)
            self.final_blocks = _make_blocks(c.final_blocks, c.lower_blocks + c.middle_blocks, tot)
        elif c.family == "unet3":
            if (c.chunk > 0 and c.chunk % 4) or c.left_context % 4:
                raise ShapeError("unet3 needs chunk == 0 or chunk divisible by 4, and left_context divisible by 4")
            tot = c.lower_blocks + c.middle_blocks + c.deep_blocks + c.up_blocks + c.final_blocks
            off = 0
            self.blocks = _make_blocks(c.lower_blocks, off, tot)
            off += c.lower_blocks
            if c.inter_ctc_vocab > 0:
                self.inter_ctc = nn.Linear(c.d_model, c.inter_ctc_vocab)
                self.inter_proj = nn.Linear(c.inter_ctc_vocab, c.d_model)
            self.pair = nn.Linear(2 * c.d_model, c.d_model)
            self.mid_blocks = _make_blocks(c.middle_blocks, off, tot)
            off += c.middle_blocks
            self.pair2 = nn.Linear(2 * c.d_model, c.d_model)
            self.deep_blocks = _make_blocks(c.deep_blocks, off, tot)
            off += c.deep_blocks
            self.restore2 = nn.Linear(c.d_model, 2 * c.d_model)
            self.up_blocks = _make_blocks(c.up_blocks, off, tot)
            off += c.up_blocks
            self.restore = nn.Linear(c.d_model, 2 * c.d_model)
            self.final_blocks = _make_blocks(c.final_blocks, off, tot)
        else:
            raise ShapeError(f"unknown encoder family {c.family}")
        self.final_norm = nn.Identity() if c.post_block_norm else nn.LayerNorm(c.d_model)

    def _stage_lefts(self) -> tuple[int, int, int]:
        c = self.cfg
        if c.stage_left:
            if len(c.stage_left) != 3:
                raise ShapeError(f"stage_left must have 3 entries (25Hz, 12.5Hz, 6.25Hz), got {c.stage_left}")
            return int(c.stage_left[0]), int(c.stage_left[1]), int(c.stage_left[2])
        return c.left_context, c.left_context // 2, c.left_context // 4

    def _ctx(self, lens: torch.Tensor, t: int, chunk: int, left: int, dtype, device):
        pad = torch.arange(t, device=device)[None] >= lens[:, None]
        mask = chunk_mask(t, chunk, left, device)
        # Padded keys are excluded; a padded query keeps its own key so softmax stays finite.
        mask = (mask[None] & ~pad[:, None, :]) | torch.eye(t, dtype=torch.bool, device=device)[None]
        cos, sin = rope_tables(t, self.cfg.d_model // self.cfg.n_heads, device, dtype)
        return mask[:, None], pad, cos, sin

    def _downsample2(self, x: torch.Tensor, pad: torch.Tensor, lens: torch.Tensor,
                     proj: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        x = x.masked_fill(pad[..., None], 0.0)
        if t % 2:
            x = F.pad(x, (0, 0, 0, 1))
        m = proj(x.reshape(b, -1, 2 * d))
        m_lens = torch.div(lens + 1, 2, rounding_mode="floor")
        return m, m_lens

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: int | None = None,
                return_aux: bool = False):
        """feats (B, T, 80) normalised log-mel; returns (states (B, T', d), lens (B,)),
        or (states, lens, aux_dict) when `return_aux=True`."""
        c = self.cfg
        chunk = c.chunk if chunk is None else chunk
        if chunk < 0:
            raise ValueError(f"chunk must be >= 0 (0 = full utterance bidirectional), got {chunk}")
        l25, l12, l6 = self._stage_lefts()
        x = self.stem(feats)
        lens = stem_out_len(feat_lens)
        b, t, d = x.shape
        mask, pad, cos, sin = self._ctx(lens, t, chunk, l25, x.dtype, x.device)
        for blk in self.blocks:
            x = blk(x, mask, pad, cos, sin)
        aux: dict[str, torch.Tensor] = {}
        bal_terms: list[torch.Tensor] = [activation_balancer_loss(x)] if return_aux else []
        if c.family == "unet3" and c.inter_ctc_vocab > 0:
            inter_logits = self.inter_ctc(x.float())
            x = x + self.inter_proj(torch.softmax(inter_logits, dim=-1).to(x.dtype))
            if return_aux:
                aux["inter_logits"] = inter_logits
        if c.family in ("multirate", "unet3"):
            if chunk > 0 and chunk % 2 != 0:
                raise ValueError(
                    f"{c.family} encoder requires an even positive chunk (got {chunk}); "
                    "chunk // 2 == 0 would silently remove causality"
                )
            skip = x
            m, m_lens = self._downsample2(x, pad, lens, self.pair)
            c2 = chunk // 2 if chunk > 0 else chunk
            mm, mp, mc, ms = self._ctx(m_lens, m.shape[1], c2, l12, m.dtype, m.device)
            for blk in self.mid_blocks:
                m = blk(m, mm, mp, mc, ms)
            if return_aux:
                bal_terms.append(activation_balancer_loss(m))
            if c.family == "unet3":
                if chunk > 0 and chunk % 4 != 0:
                    raise ValueError(
                        f"unet3 encoder requires a positive chunk divisible by 4 (got {chunk}); "
                        "chunk // 4 == 0 would silently remove causality"
                    )
                skip12 = m
                t12 = m.shape[1]
                deep, d_lens = self._downsample2(m, mp, m_lens, self.pair2)
                c4 = chunk // 4 if chunk > 0 else chunk
                dm, dp, dc, ds = self._ctx(d_lens, deep.shape[1], c4, l6,
                                           deep.dtype, deep.device)
                for blk in self.deep_blocks:
                    deep = blk(deep, dm, dp, dc, ds)
                if return_aux:
                    bal_terms.append(activation_balancer_loss(deep))
                m = self.restore2(deep).reshape(b, -1, d)[:, :t12] + skip12
                for blk in self.up_blocks:
                    m = blk(m, mm, mp, mc, ms)
                if return_aux:
                    bal_terms.append(activation_balancer_loss(m))
            x = self.restore(m).reshape(b, -1, d)[:, :t] + skip
            for blk in self.final_blocks:
                x = blk(x, mask, pad, cos, sin)
            if return_aux:
                bal_terms.append(activation_balancer_loss(x))
        out = self.final_norm(x)
        if return_aux:
            aux["balancer"] = torch.stack(bal_terms).mean()
            return out, lens, aux
        return out, lens
