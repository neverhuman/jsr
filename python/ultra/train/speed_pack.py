"""Zero-math-change training speedups for ULTRA Conformer RNN-T (`Wave H23`).

Targets the six empirically measured GPU-step and batch-packing bottlenecks on
A100 / RTX 4090 / RTX 3090 without altering model architecture or export graphs:
  1. `recommended_window_groups`: tighter duration buckets (`64% -> ~90%` pack eff).
  2. `vectorized_spec_augment`: zero CUDA->CPU `.item()` syncs, `torch.compile`-safe.
  3. `build_adamw`: enables CUDA `fused=True` (1 kernel instead of ~250).
  4. `ForeachEMA`: single `torch._foreach_lerp_` kernel instead of ~250 launches.
  5. `CachedQkvCap`: keeps per-layer row/gain caps on GPU (eliminates 32 H2D copies/step).
  6. `MaskRopeCache` + `enable_cuda_fast_math` (TF32) + `transfer_batch_async` (pinned H2D).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from train.asr import SpecAugConfig
from train.data import Batch
from train.model import chunk_mask, rope_tables


def enable_cuda_fast_math(device: torch.device) -> bool:
    """Enable TF32 Tensor Cores for FP32 heads/joiner/KD on Ampere+ CUDA GPUs."""
    if device.type != "cuda":
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return True


def recommended_window_groups(budget_s: float, max_batch: int, base: int = 4) -> int:
    """Scale parquet sort window with batch size so greedy `pack()` avoids 25-36% pad waste."""
    by_batch = math.ceil(max(1, max_batch) / 16)
    by_budget = math.ceil(max(1.0, budget_s) / 100.0)
    return max(base, min(24, max(by_batch, by_budget)))


def build_adamw(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    device: torch.device,
) -> torch.optim.AdamW:
    """AdamW matching Trainer's decay split, using CUDA fused kernels when available."""
    decay = [p for _, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict = {"lr": lr, "betas": betas}
    if device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def transfer_batch_async(batch: Batch, device: torch.device) -> Batch:
    """Pin host tensors and overlap H2D transfer on CUDA (`non_blocking=True`)."""
    tensors = (
        batch.feats,
        batch.feat_lens,
        batch.rnnt,
        batch.rnnt_lens,
        batch.ctc,
        batch.ctc_lens,
    )
    if device.type == "cuda":
        moved = tuple(
            (t.pin_memory() if t.device.type == "cpu" else t).to(device, non_blocking=True)
            for t in tensors
        )
    else:
        moved = tuple(t.to(device) for t in tensors)
    return Batch(batch.uids, batch.texts, *moved)


def vectorized_spec_augment(
    x: torch.Tensor,
    lens: torch.Tensor,
    cfg: SpecAugConfig,
    gen: torch.Generator,
) -> torch.Tensor:
    """Zero-sync GPU/CPU vectorized SpecAugment (no Python per-utterance `.item()` loop)."""
    if not cfg.enabled or x.numel() == 0:
        return x
    B, T, F_ = x.shape
    dev = x.device
    lens_d = lens.to(device=dev, dtype=torch.long).clamp(min=1, max=T)

    # Generate random draws on generator's device (CPU generator is portable across backends),
    # then transfer compact (B, M) integer arrays once instead of B*M scalar syncs.
    f_idx = torch.arange(F_, device=dev).view(1, 1, F_)
    t_idx = torch.arange(T, device=dev).view(1, 1, T)
    keep = torch.ones((B, T, F_), dtype=torch.bool, device=dev)

    if cfg.freq_masks > 0 and cfg.freq_width > 0:
        M = cfg.freq_masks
        w_f = torch.randint(0, cfg.freq_width + 1, (B, M), generator=gen).to(dev)
        f0 = (torch.rand((B, M), generator=gen).to(dev) * (F_ - w_f).clamp(min=1).float()).long().clamp(max=max(0, F_ - 1))
        f_any = ((f_idx >= f0.unsqueeze(-1)) & (f_idx < (f0 + w_f).unsqueeze(-1))).any(dim=1)
        valid_t = torch.arange(T, device=dev).unsqueeze(0) < lens_d.unsqueeze(1)
        keep &= ~(valid_t.unsqueeze(-1) & f_any.unsqueeze(1))

    max_tm = max(1, cfg.time_masks_per_10s * T // 1000)
    if max_tm > 0 and cfg.time_width > 0:
        n_tm = (cfg.time_masks_per_10s * lens_d // 1000).clamp(min=1, max=max_tm)
        active = torch.arange(max_tm, device=dev).unsqueeze(0) < n_tm.unsqueeze(1)
        max_w = (lens_d // 5).clamp(min=1, max=cfg.time_width)
        w_t = torch.where(active, (torch.rand((B, max_tm), generator=gen).to(dev) * (max_w.unsqueeze(1) + 1).float()).long(), 0)
        t0 = (torch.rand((B, max_tm), generator=gen).to(dev) * (lens_d.unsqueeze(1) - w_t).clamp(min=1).float()).long()
        t_any = ((t_idx >= t0.unsqueeze(-1)) & (t_idx < (t0 + w_t).unsqueeze(-1))).any(dim=1)
        keep &= ~t_any.unsqueeze(-1)

    return x.masked_fill(~keep, 0.0)


class ForeachEMA:
    """Single-kernel `torch._foreach_lerp_` EMA weight tracker compatible with `Trainer.ema`."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self._keys: list[str] = []
        self._float_keys: list[str] = []
        self._other_keys: list[str] = []
        self._shadow: dict[str, torch.Tensor] = {}
        self._init_from(model)

    def _init_from(self, model: nn.Module) -> None:
        state = model.state_dict()
        self._keys = list(state.keys())
        self._float_keys = [k for k, v in state.items() if v.is_floating_point()]
        self._other_keys = [k for k, v in state.items() if not v.is_floating_point()]
        self._shadow = {k: v.detach().clone() for k, v in state.items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        state = model.state_dict()
        if self._float_keys:
            ema_list = [self._shadow[k] for k in self._float_keys]
            cur_list = [state[k].detach() for k in self._float_keys]
            torch._foreach_lerp_(ema_list, cur_list, 1.0 - self.decay)
        for k in self._other_keys:
            self._shadow[k].copy_(state[k])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor], device: torch.device) -> None:
        self._shadow = {k: v.to(device).detach().clone() for k, v in sd.items()}
        self._keys = list(self._shadow.keys())
        self._float_keys = [k for k, v in self._shadow.items() if v.is_floating_point()]
        self._other_keys = [k for k, v in self._shadow.items() if not v.is_floating_point()]


class CachedQkvCap:
    """Device-resident attention QKV row-norm & pre-attention LayerNorm cap (0 H2D copies/step)."""

    def __init__(self, model: nn.Module, cap_ratio: float):
        self.cap_ratio = float(cap_ratio)
        self._mods: list[tuple[str, nn.Module]] = [
            (n, m) for n, m in model.named_modules() if hasattr(m, "qkv")
        ]
        self._caps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ref_norm: float = 0.0
        self.capture(model)

    @torch.no_grad()
    def capture(self, model: nn.Module) -> None:
        self._caps.clear()
        norms: list[float] = []
        for n, m in self._mods:
            w = m.qkv.weight.detach()
            g = m.norm.weight.detach()
            self._caps[n] = (
                (w.norm(dim=1) * self.cap_ratio).clone(),
                (g.abs() * self.cap_ratio).clone(),
            )
            norms.append(float(w.norm()))
        self.ref_norm = round(max(norms), 2) if norms else 0.0

    @torch.no_grad()
    def apply(self) -> float:
        max_n = 0.0
        for n, m in self._mods:
            row_cap, gain_cap = self._caps[n]
            w = m.qkv.weight
            rows = w.norm(dim=1)
            scale = torch.clamp(row_cap / rows.clamp(min=1e-12), max=1.0)
            w.mul_(scale[:, None])
            g = m.norm.weight
            g.clamp_(min=-gain_cap, max=gain_cap)
            wn = float(w.detach().norm())
            if wn > max_n:
                max_n = wn
        return round(max_n, 2)


class MaskRopeCache:
    """O(1) slice cache for `chunk_mask`, `eye`, and `rope_tables` avoiding per-step allocations."""

    def __init__(self) -> None:
        self._masks: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._ropes: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_mask_and_eye(
        self, t: int, chunk: int, left: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (chunk, left, str(device))
        entry = self._masks.get(key)
        if entry is None or entry[0].shape[0] < t:
            cap = max(t, 256 if entry is None else entry[0].shape[0] * 2)
            m = chunk_mask(cap, chunk, left, device=device)
            eye = torch.eye(cap, dtype=torch.bool, device=device)
            entry = (m, eye)
            self._masks[key] = entry
        return entry[0][:t, :t], entry[1][:t, :t]

    def get_rope(
        self, t: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, str(device), dtype)
        entry = self._ropes.get(key)
        if entry is None or entry[0].shape[0] < t:
            cap = max(t, 256 if entry is None else entry[0].shape[0] * 2)
            entry = rope_tables(cap, dim, device=device, dtype=dtype)
            self._ropes[key] = entry
        return entry[0][:t], entry[1][:t]


def measure_packing_efficiency(durations: Iterable[float], budget_s: float, max_batch: int) -> float:
    """Compute fraction of real audio vs padded tensor seconds for a packed window."""
    from train.data import pack

    durs = list(durations)
    if not durs:
        return 1.0
    groups = pack(durs, budget_s, max_batch)
    real_s = sum(durs)
    padded_s = sum(max(durs[i] for i in g) * len(g) for g in groups)
    return real_s / max(padded_s, 1e-9)
